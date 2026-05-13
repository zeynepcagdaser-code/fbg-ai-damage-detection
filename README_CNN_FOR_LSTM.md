# CNN Pipeline Referansı (Emine için LSTM geliştirme)

Bu doküman, Zeynep’in çalışan **CNN** eğitim pipeline’ını Emine’nin **LSTM** çalışması için referans olarak paketler. Amaç; LSTM’nin CNN ile **adil** kıyaslanabilmesi için aynı veri hazırlama adımlarının ve aynı train/test split ayarlarının kullanılmasıdır.

## Veri

- Veri dosyası (ana): `data/fbg_filtered_dataset (1).csv`
- Kullanılan giriş sütunu: `delta_lambda_filtered`
- Etiket sütunu: `label`

## Sınıflar (3 sınıf)

Ham etiketler:
- `normal`
- `mild_damage`
- `severe_damage`

Türkçe karşılıkları:
- Normal
- Hafif Hasar
- Ağır Hasar

## Pencereleme (Windowing)

`train_team_1dcnn.py` içinde varsayılan parametreler:
- `window_size = 32`
- `stride = 8`

Çıktı tensörü:
- `X` shape: `(num_windows, 32, 1)`
- `y` shape: `(num_windows,)`

Etiketleme mantığı:
- Her pencerenin etiketi, pencere içindeki `label` değerlerinin **çoğunluk oyu** ile belirlenir.

## Normalizasyon

- `delta_lambda_filtered` sinyali için **Min-Max Scaling** uygulanır:
  - `normalize_signal()` fonksiyonu
  - çıktı aralığı: `[0, 1]`

## Train/Test Split

Adil karşılaştırma için **aynı** split ayarları kullanılmalı:
- `test_size = 0.2`
- `random_state = 42`
- `stratify = y`

Not: Eğitim setinin içinden ayrıca validasyon ayrılır:
- `val_size = 0.2` (train içinden)
- `random_state = 42`
- `stratify = y_train`

Bu mantık `prepare_train_test_split()` fonksiyonunda bulunur.

## Model Input Shape

- CNN input shape: `(32, 1)`
- LSTM için de aynı input shape kullanılmalıdır: `(32, 1)`

## CNN Nasıl Eğitiliyor?

`train_team_1dcnn.py` şu adımları izler:
1. `load_dataset()` → CSV okuma (`delta_lambda_filtered`, `label`)
2. `normalize_signal()` → Min-Max scaling
3. `create_windows()` → overlap’li pencereleme (`window_size=32`, `stride=8`)
4. `LabelEncoder` → etiketleri sayısal hale getirme
5. `prepare_train_test_split()` → stratified train/val/test
6. `compute_class_weight()` → sınıf dengesizliği için `class_weight`
7. CNN modeli eğitimi:
   - `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`
   - `ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5)`
8. En iyi model kaydı: `models/fbg_team_1dcnn.keras`
9. Değerlendirme çıktıları:
   - `results/cnn_training_history.png`
   - `results/cnn_confusion_matrix.png`
   - `results/cnn_classification_report.txt`

## Emine için LSTM Uygulama Notu

LSTM ile adil kıyas için Emine’nin şu kurala uyması gerekir:
- Aynı veri dosyası: `data/fbg_filtered_dataset (1).csv`
- Aynı normalizasyon: `normalize_signal()`
- Aynı pencereleme: `window_size=32`, `stride=8`
- Aynı split: `random_state=42`, `stratify=y`, `test_size=0.2`

En pratik yol:
- `train_team_1dcnn.py` içindeki şu fonksiyonları LSTM eğitim dosyana **aynı şekilde** kopyalamak / import etmek:
  - `load_dataset()`
  - `normalize_signal()`
  - `create_windows()`
  - `prepare_train_test_split()`

## Çalıştırma

CNN eğitimi:
```bash
python train_team_1dcnn.py
```

## Gerekli paketler

`requirements.txt` içinde en az şu paketler bulunmalı:
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow==2.15.1
- keras==2.15.0
