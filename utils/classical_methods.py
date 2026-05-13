"""
FBG Spektrumu Klasik İşleme Yöntemleri
Cumulative Sum, Wavelet Filtering, Demodülasyon vb.
"""

import numpy as np
from scipy import signal, ndimage
from scipy.fftpack import fft, fftfreq, ifft
import pywt
from sklearn.preprocessing import StandardScaler


class ClassicalSpectralMethods:
    """Klasik spektrum işleme yöntemleri"""
    
    @staticmethod
    def cumulative_sum_demodulation(spectrum, threshold=0.1):
        """
        Cumulative Sum (CUSUM) ile spektrum demodülasyonu
        
        Anomali tespiti için kümülatif toplam kullan
        
        Args:
            spectrum: Giriş spektrumu
            threshold: Anomali eşiği
            
        Returns:
            anomaly_score: Anomali skoru
            is_anomaly: Bool değer
        """
        # Baseline hesapla
        baseline = np.mean(spectrum)
        
        # Sapmaları hesapla
        deviations = spectrum - baseline
        
        # Kümülatif toplam
        cusum = np.cumsum(deviations)
        
        # Anomali skoru: kümülatif toplamın varyansı
        anomaly_score = np.std(cusum)
        
        # Eşik ile karşılaştır
        is_anomaly = anomaly_score > threshold
        
        return anomaly_score, is_anomaly, cusum
    
    @staticmethod
    def matched_filter(spectrum, template=None):
        """
        Eşleştirilmiş Filtre (Matched Filter) - 
        Bilinen sinyal desenini bulma
        
        Args:
            spectrum: Giriş spektrumu
            template: Şablon (None ise Gaussian kullan)
            
        Returns:
            filtered: Filtrelenmiş spektrum
            correlation: Korelasyon sonuçları
        """
        if template is None:
            # Gauss şablonu oluştur
            center = len(spectrum) // 2
            width = len(spectrum) // 8
            template = np.exp(-((np.arange(len(spectrum)) - center) ** 2) / (2 * width ** 2))
        
        # Korelasyon hesapla
        correlation = signal.correlate(spectrum, template, mode='same')
        
        # Normalize et
        filtered = correlation / np.max(np.abs(correlation) + 1e-8)
        
        return filtered, correlation
    
    @staticmethod
    def spectral_edge_detection(spectrum, order=5):
        """
        Spektral kenarları tespit et
        
        Args:
            spectrum: Giriş spektrumu
            order: Sobel filtresi derecesi
            
        Returns:
            edges: Kenar haritası
        """
        # Sobel operatörü ile türev al
        sx = ndimage.sobel(spectrum, axis=0)
        
        # Kenar gücü
        edges = np.abs(sx)
        
        return edges
    
    @staticmethod
    def peak_detection_simple(spectrum, height=0.1, distance=10, prominence=0.05):
        """
        Basit pik tespiti
        
        Args:
            spectrum: Giriş spektrumu
            height: Minimum pik yüksekliği
            distance: Pikler arası minimum mesafe
            prominence: Pik belirginliği
            
        Returns:
            peaks: Pik indisleri
            properties: Pik özellikleri
        """
        peaks, properties = signal.find_peaks(
            spectrum,
            height=height,
            distance=distance,
            prominence=prominence
        )
        
        return peaks, properties
    
    @staticmethod
    def wavelet_decomposition(spectrum, wavelet='db4', level=4):
        """
        Dalgacık ayrışması ile spektrum analizi
        
        Args:
            spectrum: Giriş spektrumu
            wavelet: Dalgacık tipi
            level: Ayrışma seviyesi
            
        Returns:
            coeffs: Dalgacık katsayıları
        """
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        
        return coeffs
    
    @staticmethod
    def multiresolution_analysis(spectrum, wavelet='db4', level=3):
        """
        Çok-çözünürlük analizi (Multi-resolution Analysis)
        Farklı frekans bileşenlerini ayrı ayrı analiz et
        
        Args:
            spectrum: Giriş spektrumu
            wavelet: Dalgacık tipi
            level: Ayrışma seviyesi
            
        Returns:
            approximation: Yaklaşım (düşük frekans)
            details: Detaylar (yüksek frekanslar listesi)
        """
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        
        approximation = coeffs[0]
        details = coeffs[1:]
        
        return approximation, details
    
    @staticmethod
    def spectral_flatness(spectrum):
        """
        Spektral Düzlüğü Hesapla (Wiener Entropy)
        
        Spektrumun ne kadar "düz" olduğunu ölçer.
        Normal = yüksek flatness, Gürültü = düşük flatness
        
        Args:
            spectrum: Giriş spektrumu
            
        Returns:
            flatness: Spektral düzlük (0-1)
        """
        # Power Spectrum Density hesapla
        psd = spectrum ** 2
        
        # Geometrik ortalama
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        
        # Aritmetik ortalama
        arithmetic_mean = np.mean(psd)
        
        # Spektral düzlük
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        return flatness
    
    @staticmethod
    def spectral_centroid(spectrum, wavelengths):
        """
        Spektral Ağırlık Merkezi (Spectral Centroid)
        
        Args:
            spectrum: Giriş spektrumu
            wavelengths: Dalga boyu değerleri
            
        Returns:
            centroid: Spektral ağırlık merkezi
        """
        centroid = np.sum(wavelengths * spectrum) / (np.sum(spectrum) + 1e-10)
        
        return centroid
    
    @staticmethod
    def spectral_spread(spectrum, wavelengths, centroid=None):
        """
        Spektral Yayılma (Spectral Spread)
        
        Args:
            spectrum: Giriş spektrumu
            wavelengths: Dalga boyu değerleri
            centroid: Önceden hesaplanmış ağırlık merkezi (opsiyonel)
            
        Returns:
            spread: Spektral yayılma
        """
        if centroid is None:
            centroid = ClassicalSpectralMethods.spectral_centroid(spectrum, wavelengths)
        
        spread = np.sqrt(
            np.sum(spectrum * (wavelengths - centroid) ** 2) / 
            (np.sum(spectrum) + 1e-10)
        )
        
        return spread
    
    @staticmethod
    def fft_analysis(spectrum):
        """
        FFT ile Frekans Domain Analizi
        
        Args:
            spectrum: Giriş spektrumu
            
        Returns:
            frequencies: Frekans değerleri
            magnitudes: Frekans büyüklükleri
            phases: Faz bilgileri
        """
        N = len(spectrum)
        
        # FFT hesapla
        fft_result = fft(spectrum)
        
        # Frekansları hesapla
        frequencies = fftfreq(N)
        
        # Büyüklük ve faz
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        return frequencies, magnitudes, phases
    
    @staticmethod
    def signal_to_noise_ratio(spectrum, noise_estimate=None):
        """
        Sinyal-Gürültü Oranını klasik yolla tahmin et
        
        Args:
            spectrum: Giriş spektrumu
            noise_estimate: Gürültü tahmini (opsiyonel)
            
        Returns:
            snr_db: SNR (dB cinsinden)
        """
        if noise_estimate is None:
            # Gürültüyü düşük amplitüdlü kısımlardan tahmin et
            sorted_spec = np.sort(spectrum)
            noise_estimate = np.mean(sorted_spec[:len(sorted_spec)//4])
        
        signal_power = np.max(spectrum) - noise_estimate
        noise_power = noise_estimate
        
        if noise_power <= 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear + 1e-10)
        
        return snr_db


class SpectrumClassifier:
    """Klasik yöntemlerle spektrum sınıflandırma"""
    
    def __init__(self):
        """Sınıflandırıcı başlat"""
        self.feature_scaler = StandardScaler()
        self.features_fitted = False
    
    def extract_features(self, spectrum, wavelengths):
        """
        Spektrumdan özellikler çıkar
        
        Args:
            spectrum: Giriş spektrumu
            wavelengths: Dalga boyu değerleri
            
        Returns:
            features: Özellik vektörü
        """
        # Temel istatistikler
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        skew = np.abs(np.mean((spectrum - mean) ** 3) / (std ** 3 + 1e-10))
        kurtosis = np.mean((spectrum - mean) ** 4) / (std ** 4 + 1e-10)
        
        # Spektral özellikler
        centroid = ClassicalSpectralMethods.spectral_centroid(spectrum, wavelengths)
        spread = ClassicalSpectralMethods.spectral_spread(spectrum, wavelengths, centroid)
        flatness = ClassicalSpectralMethods.spectral_flatness(spectrum)
        
        # Pik özellikleri
        peaks, _ = ClassicalSpectralMethods.peak_detection_simple(spectrum)
        num_peaks = len(peaks)
        max_peak_height = np.max(spectrum) if len(spectrum) > 0 else 0
        
        # CUSUM anomali skoru
        anomaly_score, _, _ = ClassicalSpectralMethods.cumulative_sum_demodulation(spectrum)
        
        # FFT analizi
        frequencies, magnitudes, _ = ClassicalSpectralMethods.fft_analysis(spectrum)
        fft_energy = np.sum(magnitudes ** 2)
        
        # Kenar tespiti
        edges = ClassicalSpectralMethods.spectral_edge_detection(spectrum)
        edge_sum = np.sum(edges)
        
        # Dalgacık analizi
        coeffs = ClassicalSpectralMethods.wavelet_decomposition(spectrum)
        wavelet_energy = np.sum([np.sum(c ** 2) for c in coeffs])\n        
        features = np.array([\n            mean, std, skew, kurtosis,\n            centroid, spread, flatness,\n            num_peaks, max_peak_height,\n            anomaly_score, fft_energy, edge_sum, wavelet_energy\n        ])\n        \n        return features\n    \n    def extract_features_batch(self, X, wavelengths):\n        \"\"\"\n        Toplu özellik çıkarma\n        \n        Args:\n            X: Spektrum toplusu (n_samples, n_points)\n            wavelengths: Dalga boyu değerleri\n            \n        Returns:\n            features: Özellik matrisi (n_samples, n_features)\n        \"\"\"\n        features = []\n        for spectrum in X:\n            feat = self.extract_features(spectrum, wavelengths)\n            features.append(feat)\n        \n        return np.array(features)\n    \n    def fit_scaler(self, X_features):\n        \"\"\"\n        Özellik ölçekleyicisini uydur\n        \n        Args:\n            X_features: Özellik matrisi\n        \"\"\"\n        self.feature_scaler.fit(X_features)\n        self.features_fitted = True\n    \n    def scale_features(self, X_features):\n        \"\"\"\n        Özellikleri ölçekle\n        \n        Args:\n            X_features: Özellik matrisi\n            \n        Returns:\n            scaled: Ölçeklenmiş özellikler\n        \"\"\"\n        if not self.features_fitted:\n            raise ValueError(\"Scaler önce fit edilmeli\")\n        \n        return self.feature_scaler.transform(X_features)\n    \n    @staticmethod\n    def classify_by_thresholds(spectrum, wavelengths):\n        \"\"\"\n        Eşik-tabanlı sınıflandırma\n        \n        Args:\n            spectrum: Spektrum\n            wavelengths: Dalga boyu değerleri\n            \n        Returns:\n            class_label: Sınıf (0: normal, 1: gürültülü, 2: hafif hasar, 3: ağır hasar)\n        \"\"\"\n        # Spektral düzlüğü hesapla\n        flatness = ClassicalSpectralMethods.spectral_flatness(spectrum)\n        \n        # Anomali skoru\n        anomaly_score, _, _ = ClassicalSpectralMethods.cumulative_sum_demodulation(spectrum)\n        \n        # Pik sayısı\n        peaks, _ = ClassicalSpectralMethods.peak_detection_simple(spectrum)\n        num_peaks = len(peaks)\n        \n        # SNR tahmin\n        snr = ClassicalSpectralMethods.signal_to_noise_ratio(spectrum)\n        \n        # Sınıflandırma kuralları\n        if snr < 5:  # Düşük SNR\n            return 3  # Ağır hasar\n        elif num_peaks > 3 or anomaly_score > 2.0:\n            return 3  # Ağır hasar\n        elif flatness < 0.5 or num_peaks == 2:\n            return 2  # Hafif hasar\n        elif snr < 15:\n            return 1  # Gürültülü\n        else:\n            return 0  # Normal\n\n\nif __name__ == \"__main__\":\n    print(\"Klasik yöntemler test ediliyor...\")\n    \n    # Test spektrumu\n    wavelengths = np.linspace(1549, 1551, 512)\n    spectrum = np.exp(-((wavelengths - 1550) ** 2) / (2 * 0.5 ** 2))\n    spectrum = spectrum + np.random.normal(0, 0.01, len(spectrum))\n    spectrum = np.clip(spectrum, 0, None)\n    spectrum = spectrum / np.max(spectrum)\n    \n    # Test yöntemleri\n    methods = ClassicalSpectralMethods()\n    \n    print(f\"Spektral düzlük: {methods.spectral_flatness(spectrum):.4f}\")\n    print(f\"Spektral ağırlık merkezi: {methods.spectral_centroid(spectrum, wavelengths):.4f}\")\n    print(f\"Spektral yayılma: {methods.spectral_spread(spectrum, wavelengths):.4f}\")\n    print(f\"SNR: {methods.signal_to_noise_ratio(spectrum):.4f} dB\")\n    \n    anomaly_score, is_anomaly, _ = methods.cumulative_sum_demodulation(spectrum)\n    print(f\"CUSUM Anomali Skoru: {anomaly_score:.4f}\")\n