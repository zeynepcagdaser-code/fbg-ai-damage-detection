"""
FBG Sensör Analiz - PyTorch Eğitim Script'i
Veri seti oluşturma, ön işleme, model eğitimi ve değerlendirme
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import os
import sys

# Custom modüller
from utils.data_generator import FBGDataGenerator
from utils.preprocessor import FBGPreprocessor
from models.fbg_models import (
    create_model, CNN1DModel, DenoisingAutoencoder, AttentiveSpecExLSTM
)


class FBGTrainer:
    """FBG modeli eğitim sınıfı"""
    
    def __init__(self, device='cpu', seed=42):
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Dizinleri oluştur
        os.makedirs('data/fbg_dataset', exist_ok=True)
        os.makedirs('models/saved', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def generate_dataset(self, n_samples_per_class=400):
        """Veri seti oluştur"""
        print("\n" + "="*60)
        print("AŞAMA 1: FBG VERİ SETİ OLUŞTURMA")
        print("="*60)
        
        generator = FBGDataGenerator(wavelength_start=1549.0, 
                                     wavelength_end=1551.0, 
                                     num_points=512)
        X, y = generator.generate_dataset(n_samples_per_class=n_samples_per_class, seed=self.seed)
        
        print(f"✓ Veri seti oluşturuldu: {X.shape}")
        print(f"  Sınıf dağılımı: {np.bincount(y)}")
        
        return X, y, generator
    
    def preprocess_data(self, X, y):
        """Veri ön işleme"""
        print("\n" + "="*60)
        print("AŞAMA 2: VERİ ÖN İŞLEME")
        print("="*60)
        
        preprocessor = FBGPreprocessor()
        # Hızlı denoising için Gaussian kullan (Wavelet yerine)
        X_processed = np.array([preprocessor.preprocess(spectrum, denoising_method='gaussian') 
                               for spectrum in X])
        
        print(f"✓ Veri ön işlemesi tamamlandı")
        print(f"  İşlenmiş veri şekli: {X_processed.shape}")
        print(f"  Min/Max: {X_processed.min():.4f} / {X_processed.max():.4f}")
        
        return X_processed, preprocessor
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15):
        """Eğitim/Validasyon/Test setine böl"""
        # Test seti ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        # Validasyon seti ayır
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adj, random_state=self.seed, stratify=y_train
        )
        
        print(f"\n✓ Veri seti bölme:")
        print(f"  Eğitim: {X_train.shape[0]} örnek ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Validasyon: {X_val.shape[0]} örnek ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test: {X_test.shape[0]} örnek ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_dataloaders(self, train_data, val_data, test_data, batch_size=32):
        """PyTorch DataLoader oluştur"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Numpy -> Torch tensor
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)
        
        # Dataset ve DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model, train_loader, val_loader, 
                   epochs=100, learning_rate=0.001, patience=15):
        """Modeli eğit"""
        print("\n" + "="*60)
        print("AŞAMA 3: MODEL EĞİTİMİ")
        print("="*60)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Eğitim
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validasyon
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == y_batch).sum().item()
                    val_total += y_batch.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Scheduler güncelle
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # En iyi model kaydet
                torch.save(model.state_dict(), 'models/saved/best_model.pt')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"\nErken durdurma etkinleştirildi (Epoch {epoch+1})")
                break
        
        print(f"✓ Eğitim tamamlandı")
        
        return history
    
    def evaluate_model(self, model, test_loader):
        """Modeli test setinde değerlendir"""
        print("\n" + "="*60)
        print("AŞAMA 4: MODEL DEĞERLENDİRME")
        print("="*60)
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Metrikler
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        print(f"\n✓ Test Metrikleri:")
        print(f"  Doğruluk (Accuracy): {accuracy:.4f}")
        print(f"  Hassasiyet (Precision): {precision:.4f}")
        print(f"  Geri Çağırma (Recall): {recall:.4f}")
        print(f"  F1 Skoru: {f1:.4f}")
        
        print(f"\n✓ Sınıf Bazlı Rapor:")
        print(classification_report(all_targets, all_preds, 
                                   target_names=['Normal', 'Gürültülü', 'Hafif Hasar', 'Ağır Hasar']))
        
        cm = confusion_matrix(all_targets, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def plot_results(self, history, metrics):
        """Sonuçları görselleştir"""
        print("\n" + "="*60)
        print("AŞAMA 5: SONUÇLAR GÖRSELLEŞTIRME")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('FBG Model Eğitim Sonuçları', fontsize=16, fontweight='bold')
        
        # Kayıp
        axes[0, 0].plot(history['train_loss'], label='Eğitim Kaybı', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validasyon Kaybı', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Kayıp')
        axes[0, 0].set_title('Eğitim Kaybı')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Doğruluk
        axes[0, 1].plot(history['train_acc'], label='Eğitim Doğruluğu', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Validasyon Doğruluğu', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Doğruluk')
        axes[0, 1].set_title('Eğitim Doğruluğu')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        im = axes[1, 0].imshow(cm, cmap='Blues', aspect='auto')
        axes[1, 0].set_xlabel('Tahmin Edilen')
        axes[1, 0].set_ylabel('Gerçek')
        axes[1, 0].set_title('Karışıklık Matrisi')
        axes[1, 0].set_xticks([0, 1, 2, 3])
        axes[1, 0].set_yticks([0, 1, 2, 3])
        axes[1, 0].set_xticklabels(['Normal', 'Gürültülü', 'Hafif', 'Ağır'])
        axes[1, 0].set_yticklabels(['Normal', 'Gürültülü', 'Hafif', 'Ağır'])
        plt.colorbar(im, ax=axes[1, 0])
        
        # Metrikleri göster
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = (
            f"Test Metrikleri\n"
            f"─" * 30 + "\n"
            f"Doğruluk: {metrics['accuracy']:.4f}\n"
            f"Hassasiyet: {metrics['precision']:.4f}\n"
            f"Geri Çağırma: {metrics['recall']:.4f}\n"
            f"F1 Skoru: {metrics['f1']:.4f}"
        )
        ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('results/model_training_results.png', dpi=150, bbox_inches='tight')
        print("✓ Görselleştirme kaydedildi: results/model_training_results.png")
        plt.close()


def main():
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + "  FBG SENSÖR HASAR TESPİTİ - PYTORCH EĞİTİM  ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    # Device ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nKullanılan Device: {device}")
    
    # Trainer oluştur
    trainer = FBGTrainer(device=device, seed=42)
    
    # 1. Veri seti oluştur
    X, y, generator = trainer.generate_dataset(n_samples_per_class=400)
    
    # 2. Ön işleme
    X_processed, preprocessor = trainer.preprocess_data(X, y)
    
    # 3. Veri bölme
    train_data, val_data, test_data = trainer.split_data(X_processed, y)
    
    # 4. DataLoader oluştur
    train_loader, val_loader, test_loader = trainer.create_dataloaders(
        train_data, val_data, test_data, batch_size=32
    )
    
    # 5. Model oluştur ve eğit
    print("\n" + "="*60)
    print("AŞAMA 3: MODEL EĞİTİMİ")
    print("="*60)
    
    model = create_model('cnn1d', input_size=512, num_classes=4, device=device)
    print(f"\nSeçilen Model: CNN1D")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parametre sayısı: {param_count:,}")
    
    history = trainer.train_model(model, train_loader, val_loader, 
                                 epochs=100, learning_rate=0.001, patience=15)
    
    # En iyi modeli yükle
    model.load_state_dict(torch.load('models/saved/best_model.pt'))
    
    # 6. Değerlendirme
    metrics = trainer.evaluate_model(model, test_loader)
    
    # 7. Görselleştirme
    trainer.plot_results(history, metrics)
    
    print("\n" + "="*60)
    print("PROJE TAMAMLANDI!")
    print("="*60)
    print("Sonuçlar: results/ ve models/saved/ dizinlerinde kaydedildi")


if __name__ == '__main__':
    main()
