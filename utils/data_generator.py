"""
FBG Sensör Veri Seti Oluşturucu (PyTorch uyumlu)
FBG (Fiber Bragg Grating) spektrumu simülasyonu ve veri augmentasyon
"""

import numpy as np
import os
from scipy import signal
import pickle

class FBGDataGenerator:
    """FBG sensör verilerini simüle eden sınıf"""
    
    def __init__(self, wavelength_start=1549.0, wavelength_end=1551.0, num_points=512):
        """
        Args:
            wavelength_start: Dalga boyu başlangıcı (nm)
            wavelength_end: Dalga boyu sonu (nm)
            num_points: Spektrum nokta sayısı
        """
        self.wavelength_start = wavelength_start
        self.wavelength_end = wavelength_end
        self.num_points = num_points
        self.wavelengths = np.linspace(wavelength_start, wavelength_end, num_points)
        
    def _gaussian_peak(self, center, width=0.5, amplitude=1.0):
        """Gaussian pik oluştur"""
        return amplitude * np.exp(-((self.wavelengths - center) ** 2) / (2 * width ** 2))
    
    def _lorentzian_peak(self, center, width=0.1, amplitude=1.0):
        """Lorentzian pik oluştur"""
        return amplitude * (width ** 2) / ((self.wavelengths - center) ** 2 + width ** 2)
    
    def generate_normal_spectrum(self, bragg_wavelength=1550.0, peak_width=0.5, 
                                 amplitude=1.0, noise_level=0.02):
        """
        Normal (sağlıklı) FBG spektrumu oluştur
        
        Args:
            bragg_wavelength: Bragg dalga boyu (nm)
            peak_width: Pik genişliği
            amplitude: Pik genliği
            noise_level: Gürültü seviyesi (0-1)
        
        Returns:
            normalized_spectrum: 0-1 aralığında normalize edilmiş spektrum
        """
        spectrum = self._gaussian_peak(bragg_wavelength, peak_width, amplitude)
        
        # Gaussian gürültü ekle
        noise = np.random.normal(0, noise_level, self.num_points)
        spectrum = spectrum + noise
        spectrum = np.clip(spectrum, 0, None)
        
        # Normalizasyon
        normalized = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
        return normalized
    
    def generate_noisy_spectrum(self, bragg_wavelength=1550.0, peak_width=0.5,
                               amplitude=1.0, noise_level=0.1, snr_db=10):
        """
        Gürültülü FBG spektrumu oluştur (ortam gürültüsü etkileri)
        """
        spectrum = self._gaussian_peak(bragg_wavelength, peak_width, amplitude)
        
        # Yüksek seviye Gaussian gürültü
        noise = np.random.normal(0, noise_level, self.num_points)
        
        # Sıcaklık kayması simülasyonu (baseline kayması)
        temp_shift = np.random.uniform(-0.02, 0.02) * np.ones(self.num_points)
        
        spectrum = spectrum + noise + temp_shift
        spectrum = np.clip(spectrum, 0, None)
        
        normalized = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
        return normalized
    
    def generate_mild_damage_spectrum(self, bragg_wavelength=1550.0, peak_width=0.5,
                                     amplitude=1.0, damage_factor=0.3):
        """
        Hafif hasar durumu (spektrum genişlemesi ve pik azalması)
        """
        # Pik genişler ve şekli değişir
        broadened_width = peak_width * (1 + damage_factor * 0.5)
        spectrum = self._gaussian_peak(bragg_wavelength, broadened_width, 
                                      amplitude * (1 - damage_factor * 0.2))
        
        # Yan loblar (side lobes) oluştur - hasar göstergesi
        side_lobe_amplitude = amplitude * 0.15
        spectrum += 0.3 * self._gaussian_peak(bragg_wavelength - 0.3, peak_width * 0.7, 
                                             side_lobe_amplitude)
        spectrum += 0.3 * self._gaussian_peak(bragg_wavelength + 0.3, peak_width * 0.7, 
                                             side_lobe_amplitude)
        
        # Orta seviye gürültü
        noise = np.random.normal(0, 0.04, self.num_points)
        spectrum = spectrum + noise
        spectrum = np.clip(spectrum, 0, None)
        
        normalized = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
        return normalized
    
    def generate_severe_damage_spectrum(self, bragg_wavelength=1550.0, peak_width=0.5,
                                       amplitude=1.0, damage_factor=0.6):
        """
        Ağır hasar durumu (spektrum çarpılması, çoklu pikar, düşük SNR)
        """
        # Ana pik önemli ölçüde zayıflar ve genişler
        broadened_width = peak_width * (1 + damage_factor)
        spectrum = self._gaussian_peak(bragg_wavelength, broadened_width,
                                      amplitude * (1 - damage_factor * 0.5))
        
        # Çoklu yan loblar - ağır hasar
        for offset in [-0.5, -0.25, 0.25, 0.5]:
            side_lobe_amp = amplitude * 0.25 * (1 - abs(offset) / 1.0)
            spectrum += self._gaussian_peak(bragg_wavelength + offset, 
                                           peak_width * 0.5, side_lobe_amp)
        
        # Yüksek seviye gürültü
        noise = np.random.normal(0, 0.08, self.num_points)
        
        # Titreşim etkisi - baseline titreşimi
        vibration = 0.05 * np.sin(np.linspace(0, 4*np.pi, self.num_points))
        
        spectrum = spectrum + noise + vibration
        spectrum = np.clip(spectrum, 0, None)
        
        normalized = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
        return normalized
    
    def generate_thermal_drift(self, base_spectrum, drift_amount=0.5):
        """Bragg dalga boyuna sıcaklık kayması ekle"""
        # Dalga boyu kayması (nm olarak)
        shift_points = int((drift_amount / (self.wavelength_end - self.wavelength_start)) * self.num_points)
        
        if shift_points > 0:
            shifted = np.roll(base_spectrum, shift_points)
        else:
            shifted = np.roll(base_spectrum, shift_points)
        
        return shifted
    
    def generate_dataset(self, n_samples_per_class=400, seed=42):
        """
        Tam veri seti oluştur
        
        Args:
            n_samples_per_class: Her sınıf için örnek sayısı
            seed: Rastgelelik tohumu
            
        Returns:
            X: Spektrum verisi (n_samples, num_points)
            y: Etiketler (0: normal, 1: gürültülü, 2: hafif hasar, 3: ağır hasar)
        """
        np.random.seed(seed)
        
        X_data = []
        y_data = []
        
        # Sınıf 0: Normal
        for _ in range(n_samples_per_class):
            bragg_wl = np.random.uniform(1549.8, 1550.2)
            spectrum = self.generate_normal_spectrum(
                bragg_wavelength=bragg_wl,
                peak_width=np.random.uniform(0.4, 0.6),
                amplitude=np.random.uniform(0.9, 1.1),
                noise_level=np.random.uniform(0.01, 0.03)
            )
            X_data.append(spectrum)
            y_data.append(0)
        
        # Sınıf 1: Gürültülü
        for _ in range(n_samples_per_class):
            bragg_wl = np.random.uniform(1549.5, 1550.5)
            spectrum = self.generate_noisy_spectrum(
                bragg_wavelength=bragg_wl,
                peak_width=np.random.uniform(0.4, 0.7),
                amplitude=np.random.uniform(0.8, 1.1),
                noise_level=np.random.uniform(0.08, 0.15)
            )
            X_data.append(spectrum)
            y_data.append(1)
        
        # Sınıf 2: Hafif Hasar
        for _ in range(n_samples_per_class):
            bragg_wl = np.random.uniform(1549.7, 1550.3)
            spectrum = self.generate_mild_damage_spectrum(
                bragg_wavelength=bragg_wl,
                peak_width=np.random.uniform(0.4, 0.6),
                amplitude=np.random.uniform(0.8, 1.0),
                damage_factor=np.random.uniform(0.2, 0.4)
            )
            X_data.append(spectrum)
            y_data.append(2)
        
        # Sınıf 3: Ağır Hasar
        for _ in range(n_samples_per_class):
            bragg_wl = np.random.uniform(1549.5, 1550.5)
            spectrum = self.generate_severe_damage_spectrum(
                bragg_wavelength=bragg_wl,
                peak_width=np.random.uniform(0.4, 0.6),
                amplitude=np.random.uniform(0.7, 0.9),
                damage_factor=np.random.uniform(0.5, 0.8)
            )
            X_data.append(spectrum)
            y_data.append(3)
        
        X = np.array(X_data, dtype=np.float32)
        y = np.array(y_data, dtype=np.int32)
        
        return X, y
    
    def save_dataset(self, X, y, filepath):
        """Veri setini kaydet"""
        data = {'X': X, 'y': y, 'wavelengths': self.wavelengths}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Veri seti kaydedildi: {filepath}")
        print(f"  Veri boyutu: {X.shape}")
        print(f"  Etiketler: {np.unique(y)}")
        print(f"  Sınıf dağılımı: {np.bincount(y)}")
    
    @staticmethod
    def load_dataset(filepath):
        """Veri setini yükle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y'], data['wavelengths']


if __name__ == "__main__":
    # Test
    generator = FBGDataGenerator()
    X, y = generator.generate_dataset(n_samples_per_class=400)
    print(f"Veri seti oluşturuldu: {X.shape}")
    print(f"Etiket dağılımı: {np.bincount(y)}")
