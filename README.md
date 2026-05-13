# FBG Sensörlerde Yapay Zekâ ile Hasar Tespiti (Streamlit)

Bu proje, Simay ve Aleyna’dan gelen FBG zaman serisi verilerini kullanarak **CNN/LSTM tabanlı hasar tespiti** yapan Streamlit panelidir.

## Lokal çalıştırma

```bash
pip install -r requirements.txt
streamlit run panel.py
```

## Kalıcı paylaşım (Streamlit Community Cloud)

1. Bu projeyi GitHub’a push et.
2. Streamlit Community Cloud’da “New app” → repo’yu seç.
3. **Main file path**: `panel.py`
4. Deploy et; oluşan URL kalıcıdır. Repo’ya yeni commit push edince otomatik güncellenir.

## Notlar

- CNN modeli dosyası: `models/fbg_team_1dcnn.keras`
- Model yoksa panel çökmez; panel içinden eğitim başlatılabilir.
