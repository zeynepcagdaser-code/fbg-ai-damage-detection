"""
FBG Spektrumu Ön İşleme Modülü
Sinyal temizleme, normalizasyon ve demodülasyon teknikleri
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import fft, ifft, fftfreq
import pywt  # PyWavelets - dalgacık dönüşümü


class FBGPreprocessor:
    """FBG spektrumlarını ön işleme ve temizleme"""
    
    @staticmethod
    def moving_average(spectrum, window_size=5):
        """
        Hareketli ortalama filtresi ile gürültü azalt
        
        Args:
            spectrum: Giriş spektrumu
            window_size: Pencere boyutu
            
        Returns:
            filtered: Filtrelenmiş spektrum
        """
        return np.convolve(spectrum, np.ones(window_size)/window_size, mode='same')
    
    @staticmethod
    def gaussian_blur(spectrum, sigma=1.0):
        """Gaussian bulanıklığı ile gürültü azalt"""
        return gaussian_filter1d(spectrum, sigma=sigma)
    
    @staticmethod
    def savitzky_golay_filter(spectrum, window_length=11, polyorder=3):
        """
        Savitzky-Golay filtresi - spektral yapıyı koruyarak gürültü azalt
        
        Args:
            spectrum: Giriş spektrumu
            window_length: Pencere uzunluğu (tek sayı olmalı)
            polyorder: Polinom derecesi
            
        Returns:
            filtered: Filtrelenmiş spektrum
        """
        if window_length > len(spectrum):
            window_length = len(spectrum) - 1 if len(spectrum) % 2 == 0 else len(spectrum)
        if window_length % 2 == 0:
            window_length -= 1
        
        return signal.savgol_filter(spectrum, window_length=window_length, polyorder=polyorder)
    
    @staticmethod
    def wavelet_denoising(spectrum, wavelet='db4', level=1, threshold_scale=1.0):
        """
        Dalgacık dönüşümü ile gürültü azalt
        
        Args:
            spectrum: Giriş spektrumu
            wavelet: Dalgacık tipi (db4, db8, sym5 vb.)
            level: Ayrışma seviyesi
            threshold_scale: Eşik ölçek faktörü
            
        Returns:
            denoised: Denoised spektrum
        """
        # Dalgacık ayrışması
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        
        # Yumuşak eşik (soft thresholding) uygula
        # Eşik = sigma * sqrt(2 * log(N))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(spectrum)))
        
        # Tüm katsayılara eşik uygula (cA hariç)
        coeffs_thresholded = [coeffs[0]]  # Yaklaşım katsayılarını koru
        for coeff in coeffs[1:]:
            coeffs_thresholded.append(pywt.threshold(coeff, thresh, mode='soft'))
        
        # Ters dalgacık dönüşümü
        denoised = pywt.waverec(coeffs_thresholded, wavelet)
        
        # Orijinal uzunluğa getir
        if len(denoised) > len(spectrum):
            denoised = denoised[:len(spectrum)]
        elif len(denoised) < len(spectrum):
            denoised = np.pad(denoised, (0, len(spectrum) - len(denoised)))
        
        return denoised
    
    @staticmethod
    def minmax_scaling(spectrum, feature_min=0, feature_max=1):
        """
        Min-Max normalizasyon (0-1 aralığına getirme)
        
        Args:
            spectrum: Giriş spektrumu
            feature_min: Hedef minimum
            feature_max: Hedef maksimum
            
        Returns:
            scaled: Ölçeklendirilmiş spektrum
        """
        x_min = np.min(spectrum)
        x_max = np.max(spectrum)
        
        if x_max - x_min == 0:
            return np.ones_like(spectrum) * feature_min
        
        scaled = (spectrum - x_min) / (x_max - x_min) * (feature_max - feature_min) + feature_min
        return scaled
    
    @staticmethod
    def standardization(spectrum):
        """
        Standardizasyon (z-score normalizasyon)
        
        Args:
            spectrum: Giriş spektrumu
            
        Returns:
            standardized: Standardize edilmiş spektrum
        """
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        
        if std == 0:
            return np.zeros_like(spectrum)
        
        return (spectrum - mean) / std
    
    @staticmethod
    def baseline_correction(spectrum, method='polynomial'):
        """
        Baseline düzeltmesi - drift ve baseline offset'i çıkar
        
        Args:
            spectrum: Giriş spektrumu
            method: 'ALS' (Asymmetric Least Squares) veya 'polynomial'
            
        Returns:
            corrected: Baseline düzeltilmiş spektrum
        """
        if method == 'polynomial':
            # Polinom uydurması ile baseline tahmin et
            x = np.arange(len(spectrum))
            coeffs = np.polyfit(x, spectrum, 3)
            baseline = np.polyval(coeffs, x)
        else:  # ALS
            # Basit ALS implementasyonu
            baseline = FBGPreprocessor._als_baseline(spectrum, lam=1000, p=0.01, niter=10)
        
        return spectrum - baseline
    
    @staticmethod
    def _als_baseline(spectrum, lam=1000, p=0.01, niter=10):
        """
        Asymmetric Least Squares baseline
        
        Args:
            spectrum: Giriş spektrumu
            lam: Smoothness parameter
            p: Asymmetry parameter
            niter: İterasyon sayısı
            
        Returns:
            baseline: Tahmini baseline
        """
        N = len(spectrum)
        w = np.ones(N)
        
        # Difference matrix (N x N)
        D = np.eye(N)
        D = np.diff(D, n=2, axis=0)  # (N-2) x N
        
        # H matrisini (N x N) yapmak için padding ekle
        H_temp = lam * np.dot(D.T, D)  # (N x (N-2)) @ ((N-2) x N) = N x N (ama hatalı)
        
        # Doğru şekilde N x N matrix oluştur
        D2 = np.zeros((N, N))
        for i in range(N-2):
            D2[i, i] = 1
            D2[i, i+1] = -2
            D2[i, i+2] = 1
        
        H = lam * np.dot(D2.T, D2)
        
        for _ in range(niter):
            W = np.diag(w)
            Z = W + H
            
            try:
                baseline = np.linalg.solve(Z, w * spectrum)
            except:
                # Singular matrix ise inverse kullan
                baseline = spectrum.copy()
                break
            
            # Ağırlıkları güncelle
            residual = spectrum - baseline
            w = p / (1 - p + p * np.exp(2 * residual) + 1e-10)
        
        return baseline
    
    @staticmethod
    def find_bragg_wavelength(spectrum, wavelengths):
        """
        Bragg dalga boyunu (pik konumunu) bul
        
        Args:
            spectrum: Spektrum
            wavelengths: Dalga boyu değerleri
            
        Returns:
            bragg_wavelength: Tahmini Bragg dalga boyu
        """
        # Maksimum piki bul
        peak_idx = np.argmax(spectrum)
        bragg_wavelength = wavelengths[peak_idx]
        
        return bragg_wavelength
    
    @staticmethod
    def estimate_spectral_width(spectrum, wavelengths, threshold=0.5):
        """
        Spektrum genişliğini tahmin et (FWHM - Full Width at Half Maximum)
        
        Args:
            spectrum: Spektrum
            wavelengths: Dalga boyu değerleri
            threshold: Maksimumun yüzde kaçında genişlik ölçülecek (0.5 = 50%)
            
        Returns:
            fwhm: Spektrumun genişliği
            left_edge: Sol kenar dalga boyu
            right_edge: Sağ kenar dalga boyu
        """
        normalized = spectrum / np.max(spectrum)
        
        # Eşik seviyesinde kenarları bul
        above_threshold = normalized > threshold
        
        if not np.any(above_threshold):
            return 0, 0, 0
        
        # İlk ve son True indeksleri bul
        indices = np.where(above_threshold)[0]
        left_idx = indices[0]
        right_idx = indices[-1]
        
        left_edge = wavelengths[left_idx]
        right_edge = wavelengths[right_idx]
        fwhm = right_edge - left_edge
        
        return fwhm, left_edge, right_edge
    
    @staticmethod
    def power_spectral_density(spectrum, method='periodogram'):
        """
        Güç spektral yoğunluğunu hesapla
        
        Args:
            spectrum: Zaman veya mekan alanı spektrumu
            method: 'periodogram' veya 'welch'
            
        Returns:
            frequencies: Frekans değerleri
            psd: Güç spektral yoğunluğu
        """
        if method == 'welch':
            frequencies, psd = signal.welch(spectrum)
        else:  # periodogram
            frequencies, psd = signal.periodogram(spectrum)
        
        return frequencies, psd
    
    @staticmethod
    def process_batch(X_batch, denoising_method='wavelet', normalize=True):
        """
        Toplu ön işleme uygula
        
        Args:
            X_batch: Spektrum toplusu (n_samples, n_points)
            denoising_method: 'wavelet', 'moving_average', 'savitzky_golay', 'gaussian'
            normalize: Min-max normalizasyon uygula mı?
            
        Returns:
            processed: Ön işlemli spektrumlar
        """
        processed = np.zeros_like(X_batch)
        
        for i, spectrum in enumerate(X_batch):
            # Denoising
            if denoising_method == 'wavelet':
                denoised = FBGPreprocessor.wavelet_denoising(spectrum)
            elif denoising_method == 'moving_average':
                denoised = FBGPreprocessor.moving_average(spectrum)
            elif denoising_method == 'savitzky_golay':
                denoised = FBGPreprocessor.savitzky_golay_filter(spectrum)
            elif denoising_method == 'gaussian':
                denoised = FBGPreprocessor.gaussian_blur(spectrum)
            else:
                denoised = spectrum
            
            # Baseline düzeltmesi
            corrected = FBGPreprocessor.baseline_correction(denoised, method='polynomial')
            
            # Normalizasyon
            if normalize:
                final = FBGPreprocessor.minmax_scaling(corrected)
            else:
                final = corrected
            
            processed[i] = final
        
        return processed
    
    def preprocess(self, spectrum, denoising_method='wavelet', normalize=True):
        """
        Tek spektrumu ön işle (ana metod)
        
        Args:
            spectrum: Giriş spektrumu
            denoising_method: Denoising yöntemi
            normalize: Min-max normalizasyon uygula mı?
            
        Returns:
            processed: Ön işlemli spektrum
        """
        # Denoising
        if denoising_method == 'wavelet':
            denoised = self.wavelet_denoising(spectrum)
        elif denoising_method == 'moving_average':
            denoised = self.moving_average(spectrum)
        elif denoising_method == 'savitzky_golay':
            denoised = self.savitzky_golay_filter(spectrum)
        elif denoising_method == 'gaussian':
            denoised = self.gaussian_blur(spectrum)
        else:
            denoised = spectrum.copy()
        
        # Baseline düzeltmesi
        corrected = self.baseline_correction(denoised, method='polynomial')
        
        # Normalizasyon
        if normalize:
            final = self.minmax_scaling(corrected)
        else:
            final = corrected
        
        return np.clip(final, 0, 1)  # 0-1 aralığına tuttur


class SignalMetrics:
    """Sinyal kalitesi metriklerini hesapla"""
    
    @staticmethod
    def snr(signal_clean, signal_noisy):
        """
        Sinyal-Gürültü Oranı (dB cinsinden)
        
        Args:
            signal_clean: Temiz sinyal
            signal_noisy: Gürültülü sinyal
            
        Returns:
            snr_db: SNR (dB)
        """
        noise = signal_noisy - signal_clean
        signal_power = np.mean(signal_clean ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(SignalMetrics.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def psnr(y_true, y_pred, max_value=1.0):
        """
        Pik Sinyal-Gürültü Oranı (dB)
        
        Args:
            y_true: Gerçek değer
            y_pred: Tahmin
            max_value: Maksimum sinyal değeri
            
        Returns:
            psnr_db: PSNR (dB)
        """
        mse = SignalMetrics.mse(y_true, y_pred)
        if mse == 0:
            return float('inf')
        
        psnr_db = 20 * np.log10(max_value) - 10 * np.log10(mse)
        return psnr_db


if __name__ == "__main__":
    # Test
    preprocessor = FBGPreprocessor()
    
    # Test spektrumu oluştur
    wavelengths = np.linspace(1549, 1551, 512)
    noise = np.random.normal(0, 0.1, 512)
    test_spectrum = np.exp(-((wavelengths - 1550) ** 2) / (2 * 0.5 ** 2)) + noise
    test_spectrum = np.clip(test_spectrum, 0, None)
    
    # Farklı filtreleri test et
    print("Ön işleme yöntemleri test ediliyor...")
    
    filtered_wavelet = preprocessor.wavelet_denoising(test_spectrum)
    filtered_savgol = preprocessor.savitzky_golay_filter(test_spectrum)
    filtered_gaussian = preprocessor.gaussian_blur(test_spectrum)
    
    # Metrikleri hesapla
    clean_spectrum = np.exp(-((wavelengths - 1550) ** 2) / (2 * 0.5 ** 2))
    
    print(f"Dalgacık denoising RMSE: {SignalMetrics.rmse(clean_spectrum, filtered_wavelet):.4f}")
    print(f"Savitzky-Golay RMSE: {SignalMetrics.rmse(clean_spectrum, filtered_savgol):.4f}")
    print(f"Gaussian RMSE: {SignalMetrics.rmse(clean_spectrum, filtered_gaussian):.4f}")
