# FBG Sensörlerde Yapay Zekâ ile Hasar Tespiti (Streamlit)

Bu repo, FBG sensör zaman serisi verilerinden **3 sınıflı** hasar tespiti yapan Streamlit dashboard’udur:

- Normal
- Hafif Hasar
- Ağır Hasar

## Çalıştırma (Lokal)

```bash
pip install -r requirements.txt
streamlit run panel.py
```

## Streamlit Cloud Deploy (Kalıcı Link)

1. Streamlit Community Cloud → **New app**
2. Repo: `zeynepcagdaser-code/fbg-ai-damage-detection`
3. Branch: `main`
4. Main file path: `panel.py`
5. Deploy

`runtime.txt` içinde `python-3.11` tanımlıdır (TensorFlow uyumu için).

## Beklenen Veri Formatı

Yükleme sırasında CSV/XLSX kabul edilir.

Zorunlu sütunlar:
- `time` veya `zaman`
- `delta_lambda_noisy`

Opsiyonel sütunlar:
- `delta_lambda_filtered` (yoksa panel `delta_lambda_noisy` üzerinden moving average üretir)
- `label` veya `etiket` (varsa metrikler hesaplanır)

## Model

- Model dosyası: `models/fbg_team_1dcnn.keras`
- Model yoksa panel çökmez; kullanıcıya uyarı verir ve sadece veri görselleştirir.

## Ekip Görev Dağılımı (Özet)

- Simay: Fiziksel modelleme (Simulink) → ham zaman serisi
- Aleyna: Gürültü temizleme → `delta_lambda_filtered`
- Gizem: Feature engineering → ölçekleme + pencereleme
- Zeynep: 1D CNN modeli
- Emine: LSTM modeli (planlı)
- Çağla: Hasar analizi / yorum
- Zeynep & Emine: Entegrasyon ve dashboard
