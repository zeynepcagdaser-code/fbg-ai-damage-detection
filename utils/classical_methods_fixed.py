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
        """Cumulative Sum (CUSUM) ile spektrum demodülasyonu"""
        baseline = np.mean(spectrum)
        deviations = spectrum - baseline
        cusum = np.cumsum(deviations)
        anomaly_score = np.std(cusum)
        is_anomaly = anomaly_score > threshold
        return anomaly_score, is_anomaly, cusum
    
    @staticmethod
    def matched_filter(spectrum, template=None):
        """Eşleştirilmiş Filtre"""
        if template is None:
            center = len(spectrum) // 2
            width = len(spectrum) // 8
            template = np.exp(-((np.arange(len(spectrum)) - center) ** 2) / (2 * width ** 2))
        
        correlation = signal.correlate(spectrum, template, mode='same')
        filtered = correlation / np.max(np.abs(correlation) + 1e-8)
        return filtered, correlation
    
    @staticmethod
    def spectral_edge_detection(spectrum, order=5):
        """Spektral kenarları tespit et"""
        sx = ndimage.sobel(spectrum, axis=0)
        edges = np.abs(sx)
        return edges
    
    @staticmethod
    def peak_detection_simple(spectrum, height=0.1, distance=10, prominence=0.05):
        """Basit pik tespiti"""
        peaks, properties = signal.find_peaks(
            spectrum,
            height=height,
            distance=distance,
            prominence=prominence
        )
        return peaks, properties
    
    @staticmethod
    def wavelet_decomposition(spectrum, wavelet='db4', level=4):
        """Dalgacık ayrışması"""
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        return coeffs
    
    @staticmethod
    def multiresolution_analysis(spectrum, wavelet='db4', level=3):
        """Çok-çözünürlük analizi"""
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        approximation = coeffs[0]
        details = coeffs[1:]
        return approximation, details
    
    @staticmethod
    def spectral_flatness(spectrum):
        """Spektral Düzlüğü Hesapla"""
        psd = spectrum ** 2
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        return flatness
    
    @staticmethod
    def spectral_centroid(spectrum, wavelengths):
        """Spektral Ağırlık Merkezi"""
        centroid = np.sum(wavelengths * spectrum) / (np.sum(spectrum) + 1e-10)
        return centroid
    
    @staticmethod
    def spectral_spread(spectrum, wavelengths, centroid=None):
        """Spektral Yayılma"""
        if centroid is None:
            centroid = ClassicalSpectralMethods.spectral_centroid(spectrum, wavelengths)
        
        spread = np.sqrt(
            np.sum(spectrum * (wavelengths - centroid) ** 2) / 
            (np.sum(spectrum) + 1e-10)
        )
        return spread
    
    @staticmethod
    def fft_analysis(spectrum):
        """FFT ile Frekans Domain Analizi"""
        N = len(spectrum)
        fft_result = fft(spectrum)
        frequencies = fftfreq(N)
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        return frequencies, magnitudes, phases
    
    @staticmethod
    def signal_to_noise_ratio(spectrum, noise_estimate=None):
        """Sinyal-Gürültü Oranını klasik yolla tahmin et"""
        if noise_estimate is None:
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
        """Spektrumdan özellikler çıkar"""
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        skew = np.abs(np.mean((spectrum - mean) ** 3) / (std ** 3 + 1e-10))
        kurtosis = np.mean((spectrum - mean) ** 4) / (std ** 4 + 1e-10)
        
        centroid = ClassicalSpectralMethods.spectral_centroid(spectrum, wavelengths)
        spread = ClassicalSpectralMethods.spectral_spread(spectrum, wavelengths, centroid)
        flatness = ClassicalSpectralMethods.spectral_flatness(spectrum)
        
        peaks, _ = ClassicalSpectralMethods.peak_detection_simple(spectrum)
        num_peaks = len(peaks)
        max_peak_height = np.max(spectrum) if len(spectrum) > 0 else 0
        
        anomaly_score, _, _ = ClassicalSpectralMethods.cumulative_sum_demodulation(spectrum)
        
        frequencies, magnitudes, _ = ClassicalSpectralMethods.fft_analysis(spectrum)
        fft_energy = np.sum(magnitudes ** 2)
        
        edges = ClassicalSpectralMethods.spectral_edge_detection(spectrum)
        edge_sum = np.sum(edges)
        
        coeffs = ClassicalSpectralMethods.wavelet_decomposition(spectrum)
        wavelet_energy = np.sum([np.sum(c ** 2) for c in coeffs])
        
        features = np.array([
            mean, std, skew, kurtosis,
            centroid, spread, flatness,
            num_peaks, max_peak_height,
            anomaly_score, fft_energy, edge_sum, wavelet_energy
        ])
        
        return features
    
    def extract_features_batch(self, X, wavelengths):
        """Toplu özellik çıkarma"""
        features = []
        for spectrum in X:
            feat = self.extract_features(spectrum, wavelengths)
            features.append(feat)
        return np.array(features)
    
    def fit_scaler(self, X_features):
        """Özellik ölçekleyicisini uydur"""
        self.feature_scaler.fit(X_features)
        self.features_fitted = True
    
    def scale_features(self, X_features):
        """Özellikleri ölçekle"""
        if not self.features_fitted:
            raise ValueError("Scaler önce fit edilmeli")
        return self.feature_scaler.transform(X_features)
    
    @staticmethod
    def classify_by_thresholds(spectrum, wavelengths):
        """Eşik-tabanlı sınıflandırma"""
        flatness = ClassicalSpectralMethods.spectral_flatness(spectrum)
        anomaly_score, _, _ = ClassicalSpectralMethods.cumulative_sum_demodulation(spectrum)
        peaks, _ = ClassicalSpectralMethods.peak_detection_simple(spectrum)
        num_peaks = len(peaks)
        snr = ClassicalSpectralMethods.signal_to_noise_ratio(spectrum)
        
        if snr < 5:
            return 3
        elif num_peaks > 3 or anomaly_score > 2.0:
            return 3
        elif flatness < 0.5 or num_peaks == 2:
            return 2
        elif snr < 15:
            return 1
        else:
            return 0


if __name__ == "__main__":
    print("Klasik yöntemler test ediliyor...")
    wavelengths = np.linspace(1549, 1551, 512)
    spectrum = np.exp(-((wavelengths - 1550) ** 2) / (2 * 0.5 ** 2))
    spectrum = spectrum + np.random.normal(0, 0.01, len(spectrum))
    spectrum = np.clip(spectrum, 0, None)
    spectrum = spectrum / np.max(spectrum)
    
    methods = ClassicalSpectralMethods()
    print(f"Spektral düzlük: {methods.spectral_flatness(spectrum):.4f}")
    print(f"Spektral ağırlık merkezi: {methods.spectral_centroid(spectrum, wavelengths):.4f}")
    print(f"SNR: {methods.signal_to_noise_ratio(spectrum):.4f} dB")
