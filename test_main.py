"""
FBG Spektrumu Analiz - Basit Test Script'i
(TensorFlow olmadan, klasik yöntemler ve NumPy tabanlı test)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Proje klasörlerini sys.path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "utils"))

from data_generator import FBGDataGenerator
from preprocessor import FBGPreprocessor, SignalMetrics
from classical_methods_fixed import ClassicalSpectralMethods, SpectrumClassifier


def test_data_generation():
    """Veri seti oluşturmayı test et"""
    print("\n" + "="*80)
    print("TEST 1: VERİ SETİ OLUŞTURMA")
    print("="*80)
    
    generator = FBGDataGenerator(
        wavelength_start=1549.0,
        wavelength_end=1551.0,
        num_points=512
    )
    
    print("\n✓ FBGDataGenerator oluşturuldu")
    
    # Spektrumları görselleştir
    print("\nÖrnek spektrumlar oluşturuluyor...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    wavelengths = generator.wavelengths
    
    # Normal
    spec_normal = generator.generate_normal_spectrum()
    axes[0, 0].plot(wavelengths, spec_normal, linewidth=2, color='green')
    axes[0, 0].set_title('Normal Spektrum', fontweight='bold')
    axes[0, 0].set_ylabel('Yansıtma')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gürültülü
    spec_noisy = generator.generate_noisy_spectrum()
    axes[0, 1].plot(wavelengths, spec_noisy, linewidth=2, color='orange')
    axes[0, 1].set_title('Gürültülü Spektrum', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hafif Hasar
    spec_mild = generator.generate_mild_damage_spectrum()
    axes[1, 0].plot(wavelengths, spec_mild, linewidth=2, color='red')
    axes[1, 0].set_title('Hafif Hasar Spektrumu', fontweight='bold')
    axes[1, 0].set_xlabel('Dalga Boyu (nm)')
    axes[1, 0].set_ylabel('Yansıtma')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Ağır Hasar
    spec_severe = generator.generate_severe_damage_spectrum()
    axes[1, 1].plot(wavelengths, spec_severe, linewidth=2, color='darkred')
    axes[1, 1].set_title('Ağır Hasar Spektrumu', fontweight='bold')
    axes[1, 1].set_xlabel('Dalga Boyu (nm)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(project_root / "plots" / "00_sample_spectra.png", dpi=300, bbox_inches='tight')
    print("✓ Örnek spektrumlar kaydedildi: plots/00_sample_spectra.png")
    plt.close()
    
    # Veri seti oluştur
    print("\nVeri seti oluşturuluyor...")
    X, y = generator.generate_dataset(n_samples_per_class=200, seed=42)
    print(f"✓ Veri seti oluşturuldu: {X.shape}")
    print(f"  Sınıf dağılımı: {np.bincount(y)}")
    
    return X, y, wavelengths


def test_preprocessing(X, wavelengths):
    """Ön işlemeyi test et"""
    print("\n" + "="*80)
    print("TEST 2: ÖN İŞLEME")
    print("="*80)
    
    preprocessor = FBGPreprocessor()
    
    print("\nÖn işleme yöntemleri test ediliyor...")
    
    # Test spektrumu
    test_spec = X[0]
    
    # Farklı denoising yöntemleri
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Orijinal
    axes[0, 0].plot(wavelengths, test_spec, linewidth=2, color='black', label='Orijinal')
    axes[0, 0].set_title('Orijinal Spektrum', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gaussian filtresi
    filtered_gaussian = preprocessor.gaussian_blur(test_spec, sigma=1.0)
    axes[0, 1].plot(wavelengths, filtered_gaussian, linewidth=2, color='blue', label='Gaussian')
    axes[0, 1].set_title('Gaussian Filtresi', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hareketli Ortalama
    filtered_ma = preprocessor.moving_average(test_spec, window_size=5)
    axes[0, 2].plot(wavelengths, filtered_ma, linewidth=2, color='green', label='Moving Avg')
    axes[0, 2].set_title('Hareketli Ortalama', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Savitzky-Golay
    filtered_savgol = preprocessor.savitzky_golay_filter(test_spec)
    axes[1, 0].plot(wavelengths, filtered_savgol, linewidth=2, color='red', label='Savitzky-Golay')
    axes[1, 0].set_title('Savitzky-Golay Filtresi', fontweight='bold')
    axes[1, 0].set_xlabel('Dalga Boyu (nm)')
    axes[1, 0].set_ylabel('Yansıtma')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dalgacık Denoising
    filtered_wavelet = preprocessor.wavelet_denoising(test_spec)
    axes[1, 1].plot(wavelengths, filtered_wavelet, linewidth=2, color='purple', label='Wavelet')
    axes[1, 1].set_title('Dalgacık Denoising', fontweight='bold')
    axes[1, 1].set_xlabel('Dalga Boyu (nm)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Baseline Düzeltmesi
    corrected_baseline = preprocessor.baseline_correction(test_spec)
    axes[1, 2].plot(wavelengths, corrected_baseline, linewidth=2, color='brown', label='Baseline Corrected')
    axes[1, 2].set_title('Baseline Düzeltmesi', fontweight='bold')
    axes[1, 2].set_xlabel('Dalga Boyu (nm)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(project_root / "plots" / "01_preprocessing_methods.png", dpi=300, bbox_inches='tight')
    print("✓ Ön işleme yöntemleri kaydedildi: plots/01_preprocessing_methods.png")
    plt.close()
    
    # Toplu işleme
    print("\nToplu ön işleme uygulanıyor...")
    X_processed = preprocessor.process_batch(X, denoising_method='wavelet', normalize=True)
    print(f"✓ İşleme tamamlandı: {X_processed.shape}")
    
    # Spektral özellikler
    print("\nSpektral özellikler hesaplanıyor...")
    flatness = preprocessor.spectral_flatness(test_spec)
    centroid = preprocessor.spectral_centroid(test_spec, wavelengths)
    width, left, right = preprocessor.estimate_spectral_width(test_spec, wavelengths)
    print(f"  Spektral düzlük: {flatness:.4f}")
    print(f"  Spektral ağırlık merkezi: {centroid:.4f} nm")
    print(f"  Spektrum genişliği (FWHM): {width:.4f} nm")
    
    return X_processed


def test_classical_methods(X, y, wavelengths):
    """Klasik yöntemleri test et"""
    print("\n" + "="*80)
    print("TEST 3: KLASİK YÖNTEMLER")
    print("="*80)
    
    classifier = SpectrumClassifier()
    methods = ClassicalSpectralMethods()
    
    print("\nKlasik analiz yöntemleri test ediliyor...")
    
    # Örnek spektrumdan özellikler çıkar
    test_spectrum = X[0]
    
    # Spektral özellikler
    flatness = methods.spectral_flatness(test_spectrum)
    centroid = methods.spectral_centroid(test_spectrum, wavelengths)
    spread = methods.spectral_spread(test_spectrum, wavelengths)
    snr = methods.signal_to_noise_ratio(test_spectrum)
    
    print(f"\n  Spektral Analiz Sonuçları:")
    print(f"    Spektral Düzlük: {flatness:.4f}")
    print(f"    Spektral Ağırlık Merkezi: {centroid:.4f} nm")
    print(f"    Spektral Yayılma: {spread:.4f} nm")
    print(f"    SNR: {snr:.4f} dB")
    
    # Pik Tespiti
    peaks, props = methods.peak_detection_simple(test_spectrum)
    print(f"    Pik Sayısı: {len(peaks)}")
    
    # CUSUM
    anomaly_score, is_anomaly, cusum = methods.cumulative_sum_demodulation(test_spectrum)
    print(f"    CUSUM Anomali Skoru: {anomaly_score:.4f}")
    
    # Sınıflandırma
    print("\nKlasik Yöntem Sınıflandırması Test Ediliyor...")
    
    # Özellik çıkarma
    X_features = classifier.extract_features_batch(X[:100], wavelengths)
    y_subset = y[:100]
    
    # Ölçeklendirici uydur
    classifier.fit_scaler(X_features)
    
    # Threshold-tabanlı sınıflandırma
    y_pred = np.array([
        classifier.classify_by_thresholds(spectrum, wavelengths)
        for spectrum in X[:100]
    ])
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_subset, y_pred)
    f1 = f1_score(y_subset, y_pred, average='weighted', zero_division=0)
    
    print(f"  ✓ Doğruluk: {accuracy:.4f}")
    print(f"  ✓ F1 Skoru: {f1:.4f}")
    
    # Sınıflandırma raporu
    print(f"\nSınıflandırma Raporu:")
    print(classification_report(y_subset, y_pred,
          target_names=['Normal', 'Gürültülü', 'Hafif Hasar', 'Ağır Hasar'],
          digits=4, zero_division=0))
    
    # Karışıklık Matrisi
    cm = confusion_matrix(y_subset, y_pred, labels=[0, 1, 2, 3])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Klasik Yöntem - Karışıklık Matrisi', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    
    classes = ['Normal', 'Gürültülü', 'Hafif Hasar', 'Ağır Hasar']
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black",
                         fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(project_root / "plots" / "02_classical_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print("✓ Karışıklık Matrisi kaydedildi: plots/02_classical_confusion_matrix.png")
    plt.close()


def test_spectral_analysis(X, wavelengths):
    """Spektral analizi görselleştir"""
    print("\n" + "="*80)
    print("TEST 4: SPEKTRAL ANALİZ")
    print("="*80)
    
    methods = ClassicalSpectralMethods()
    
    print("\nFFT ve Dalgacık Analizi yapılıyor...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    test_spec = X[0]
    
    # Time domain
    axes[0, 0].plot(wavelengths, test_spec, linewidth=2, color='blue')
    axes[0, 0].set_title('Zaman Alanı Spektrumu', fontweight='bold')
    axes[0, 0].set_ylabel('Yansıtma')
    axes[0, 0].grid(True, alpha=0.3)
    
    # FFT
    frequencies, magnitudes, phases = methods.fft_analysis(test_spec)
    axes[0, 1].plot(frequencies[:len(frequencies)//2], magnitudes[:len(magnitudes)//2], 
                    linewidth=2, color='red')
    axes[0, 1].set_title('Frekans Alanı (FFT)', fontweight='bold')
    axes[0, 1].set_ylabel('Büyüklük')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dalgacık Ayrışması
    import pywt
    coeffs = pywt.wavedec(test_spec, 'db4', level=3)
    
    # Yaklaşım ve detayları görselleştir
    ca = coeffs[0]
    cd1, cd2, cd3 = coeffs[1:]
    
    axes[1, 0].plot(ca, linewidth=2, color='green', label='Yaklaşım (cA)')
    axes[1, 0].set_title('Dalgacık Yaklaşım', fontweight='bold')
    axes[1, 0].set_ylabel('Katsayı')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Detayları
    axes[1, 1].plot(cd1, linewidth=1, alpha=0.7, label='Detay 1 (cD1)', color='orange')
    axes[1, 1].plot(cd2, linewidth=1, alpha=0.7, label='Detay 2 (cD2)', color='red')
    axes[1, 1].plot(cd3, linewidth=1, alpha=0.7, label='Detay 3 (cD3)', color='darkred')
    axes[1, 1].set_title('Dalgacık Detayları', fontweight='bold')
    axes[1, 1].set_ylabel('Katsayı')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(project_root / "plots" / "03_spectral_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Spektral analiz kaydedildi: plots/03_spectral_analysis.png")
    plt.close()


def main():
    """Ana test fonksiyonu"""
    print("\n" + "="*80)
    print("FBG SPEKTRUMU ANALIZ - KAPSAMLI TEST")
    print("="*80)
    print(f"Proje Klasörü: {project_root}")
    print(f"Python Sürümü: {sys.version.split()[0]}")
    
    try:
        # Test 1: Veri Seti
        X, y, wavelengths = test_data_generation()
        
        # Test 2: Ön İşleme
        X_processed = test_preprocessing(X, wavelengths)
        
        # Test 3: Klasik Yöntemler
        test_classical_methods(X_processed, y, wavelengths)
        
        # Test 4: Spektral Analiz
        test_spectral_analysis(X_processed, wavelengths)
        
        print("\n" + "="*80)
        print("✓ TÜM TESTLER BAŞARILI İLE TAMAMLANDI!")
        print("="*80)
        
        print("\nÖnemli Notlar:")
        print("1. Derin öğrenme modelleri Python 3.14 ile TensorFlow uyumluluğu nedeniyle")
        print("   çalıştırılamadı. Python 3.10 veya 3.11 ile çalıştırılması önerilir.")
        print("2. Klasik yöntemler ve ön işleme başarıyla çalışmaktadır.")
        print("3. Tüm görselleştirmeler 'plots' klasörüne kaydedilmiştir.")
        
    except Exception as e:
        print(f"\n✗ HATA: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
