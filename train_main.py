"""
FBG Spektrumu Analiz - Ana Eğitim Script'i
Veri üretimi, model eğitimi, doğrulama ve karşılaştırma
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Proje klasörlerini sys.path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "utils"))
sys.path.insert(0, str(project_root / "models"))

from data_generator import FBGDataGenerator
from preprocessor import FBGPreprocessor, SignalMetrics
from fbg_models import (
    CNN1DModel, LSTMModel, DenoisingAutoencoder,
    CNNLSTMHybrid, AttentiveSpecExLSTM, EnsembleModel, create_callbacks
)
from classical_methods_fixed import ClassicalSpectralMethods, SpectrumClassifier


def main():
    """Ana program akışı"""
    
    print("=" * 80)
    print("FBG SENSÖR SPEKTRUMU ANALIZ VE HASAR TESPİT SİSTEMİ")
    print("=" * 80)
    
    # 1. VERI SETI OLUŞTURMA
    print("\n[1/7] Veri seti oluşturuluyor...")
    print("-" * 80)
    
    generator = FBGDataGenerator(
        wavelength_start=1549.0,
        wavelength_end=1551.0,
        num_points=512
    )
    
    X, y = generator.generate_dataset(n_samples_per_class=400, seed=42)
    wavelengths = generator.wavelengths
    
    print(f"✓ Veri seti oluşturuldu: {X.shape}")
    print(f"  Etiket sayıları: {np.bincount(y)}")
    print(f"  Sınıflar: Normal(0), Gürültülü(1), Hafif Hasar(2), Ağır Hasar(3)")
    
    # Veri setini kaydet
    data_path = project_root / "data" / "fbg_dataset.pkl"
    data_path.parent.mkdir(exist_ok=True)
    generator.save_dataset(X, y, str(data_path))
    
    # 2. VERİ AYRIMI VE ÖN İŞLEME
    print("\n[2/7] Veriler ayrılıyor ve ön işleme yapılıyor...")
    print("-" * 80)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Veri bölünmesi tamamlandı:")
    print(f"  Eğitim: {X_train.shape[0]} örnek")
    print(f"  Validasyon: {X_val.shape[0]} örnek")
    print(f"  Test: {X_test.shape[0]} örnek")
    
    # Ön işleme (wavelet denoising)
    preprocessor = FBGPreprocessor()
    print("\n  Ön işleme uygulanıyor (Wavelet denoising)...")
    
    X_train_processed = preprocessor.process_batch(X_train, denoising_method='wavelet', normalize=True)
    X_val_processed = preprocessor.process_batch(X_val, denoising_method='wavelet', normalize=True)
    X_test_processed = preprocessor.process_batch(X_test, denoising_method='wavelet', normalize=True)
    
    print(f"  ✓ Ön işleme tamamlandı")
    
    # 3. DERIN ÖĞRENME MODELİ EĞİTİMİ
    print("\n[3/7] Derin öğrenme modeli eğitiliyor...")
    print("-" * 80)
    
    input_shape = (512,)
    num_classes = 4
    
    print("  Modeller oluşturuluyor...")
    models_to_train = {
        'CNN1D': CNN1DModel.build(input_shape, num_classes),
        'LSTM': LSTMModel.build(input_shape, num_classes),
        'CNN-LSTM': CNNLSTMHybrid.build(input_shape, num_classes),
        'Attention-LSTM': AttentiveSpecExLSTM.build(input_shape, num_classes),
        'Ensemble': EnsembleModel.build(input_shape, num_classes)
    }
    
    # Modelleri eğit
    history_dict = {}
    trained_models = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n  [{model_name}] Eğitiliyor...")
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train_processed, y_train,
            validation_data=(X_val_processed, y_val),
            epochs=50,
            batch_size=32,
            callbacks=create_callbacks(patience=15),
            verbose=0
        )
        
        history_dict[model_name] = history
        trained_models[model_name] = model
        print(f"    ✓ {model_name} eğitimi tamamlandı (Final val loss: {history.history['val_loss'][-1]:.4f})")
    
    # 4. MODEL DEĞERLENDİRMESİ
    print("\n[4/7] Modeller değerlendiriliyor...")
    print("-" * 80)
    
    results_dict = {}
    
    for model_name, model in trained_models.items():
        print(f"\n  [{model_name}] Değerlendiriliyor...")
        
        # Test seti üzerinde tahmin
        y_pred_probs = model.predict(X_test_processed, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # RMSE ve MAE (tahmin confidence üzerinde)
        max_probs = np.max(y_pred_probs, axis=1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results_dict[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'rmse': rmse,
            'mae': mae,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
        
        print(f"    Doğruluk: {accuracy:.4f}")
        print(f"    Hassasiyet: {precision:.4f}")
        print(f"    Geri Çağırma: {recall:.4f}")
        print(f"    F1 Skoru: {f1:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE: {mae:.4f}")
        
        # Detaylı sınıflandırma raporu
        print(f"\n    Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred,
              target_names=['Normal', 'Gürültülü', 'Hafif Hasar', 'Ağır Hasar'],
              digits=4))
    
    # 5. KLASIK YÖNTEMLERLE KARŞILAŞTIRMA
    print("\n[5/7] Klasik yöntemlerle karşılaştırma yapılıyor...")
    print("-" * 80)
    
    classifier = SpectrumClassifier()
    
    # Eğitim verileri üzerinde özellik çıkar
    print("  Özellikler çıkarılıyor...")
    X_train_features = classifier.extract_features_batch(X_train_processed, wavelengths)
    X_test_features = classifier.extract_features_batch(X_test_processed, wavelengths)
    
    # Scaler'ı uydur
    classifier.fit_scaler(X_train_features)
    X_train_features_scaled = classifier.scale_features(X_train_features)
    X_test_features_scaled = classifier.scale_features(X_test_features)
    
    # Threshold-tabanlı sınıflandırma
    print("  Threshold-tabanlı sınıflandırma yapılıyor...")
    y_pred_classical = np.array([
        classifier.classify_by_thresholds(spectrum, wavelengths)
        for spectrum in X_test_processed
    ])
    
    # Metrikleri hesapla
    accuracy_classical = accuracy_score(y_test, y_pred_classical)
    f1_classical = f1_score(y_test, y_pred_classical, average='weighted', zero_division=0)
    
    results_dict['Klasik Yöntem'] = {
        'accuracy': accuracy_classical,
        'f1': f1_classical,
        'y_pred': y_pred_classical
    }
    
    print(f"  ✓ Klasik Yöntem Doğruluğu: {accuracy_classical:.4f}")
    print(f"  ✓ Klasik Yöntem F1: {f1_classical:.4f}")
    
    # 6. SONUÇLAR VE KARŞILAŞTıRMA
    print("\n[6/7] Sonuçlar karşılaştırılıyor...")
    print("-" * 80)
    
    # DataFrame oluştur
    results_df = pd.DataFrame({
        model_name: {
            'Doğruluk': results['accuracy'],
            'F1': results['f1'],
            'Precision': results.get('precision', '-'),
            'Recall': results.get('recall', '-')
        }
        for model_name, results in results_dict.items()
    }).T
    
    print("\nOzet Sonuçlar:")
    print(results_df.to_string())
    
    # En iyi model
    best_model = max(results_dict.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✓ En iyi model: {best_model[0]} (Doğruluk: {best_model[1]['accuracy']:.4f})")
    
    # Sonuçları kaydet
    results_path = project_root / "results" / "model_results.pkl"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results_dict,
            'history': history_dict,
            'X_test': X_test_processed,
            'y_test': y_test,
            'wavelengths': wavelengths
        }, f)
    print(f"\n  Sonuçlar kaydedildi: {results_path}")
    
    # 7. GÖRSELLEŞTIRME
    print("\n[7/7] Sonuçlar görselleştiriliyor...")
    print("-" * 80)
    
    create_visualizations(history_dict, results_dict, y_test, wavelengths, project_root)
    
    print("\n✓ Tüm işlemler tamamlandı!")
    print("=" * 80)


def create_visualizations(history_dict, results_dict, y_test, wavelengths, project_root):
    """Görselleştirmeler oluştur"""
    
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Eğitim Kaybı Grafikleri
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, history) in enumerate(history_dict.items()):
        ax = axes[idx]
        ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_title(f'{model_name} - Eğitim Kaybı', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Son subplot'u gizle
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir / '01_training_loss.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Kaydedildi: 01_training_loss.png")
    plt.close()
    
    # 2. Doğruluk Karşılaştırması
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(results_dict.keys())
    accuracies = [results_dict[name]['accuracy'] for name in model_names]
    f1_scores = [results_dict[name]['f1'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, accuracies, width, label='Doğruluk', alpha=0.8)
    ax.bar(x + width/2, f1_scores, width, label='F1 Skoru', alpha=0.8)
    
    ax.set_xlabel('Modeller', fontsize=12, fontweight='bold')
    ax.set_ylabel('Skor', fontsize=12, fontweight='bold')
    ax.set_title('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(plots_dir / '02_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Kaydedildi: 02_model_comparison.png")
    plt.close()
    
    # 3. Karışıklık Matrisleri
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    class_names = ['Normal', 'Gürültülü', 'Hafif Hasar', 'Ağır Hasar']
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        y_pred = results['y_pred']
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(4))
        
        # Normalize et
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{model_name} - Karışıklık Matrisi', fontsize=11, fontweight='bold')
        ax.set_xlabel('Tahmin')
        ax.set_ylabel('Gerçek')
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        
        # Sayıları ekle
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, f'{cm[i, j]}\n{cm_normalized[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir / '03_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Kaydedildi: 03_confusion_matrices.png")
    plt.close()
    
    print(f"\n  Tüm görselleştirmeler {plots_dir} klasörüne kaydedildi")


if __name__ == "__main__":
    main()
