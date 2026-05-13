"""
Ngrok tüneli açmak için yardımcı script.

Öncelikle paneli çalıştır:
    streamlit run panel.py

Sonra bu script'i başka bir terminalde çalıştır:
    python run_ngrok.py

Ardından terminalde oluşan URL'yi arkadaşlarınla paylaşabilirsin.
"""

import time
from pyngrok import ngrok


def main():
    print("Ngrok tüneli başlatılıyor...")
    tunnel = ngrok.connect(8501, "http")
    print("Ngrok URL:", tunnel.public_url)
    print("Panelin çalıştığı bilgisayarda bu script açık kaldıkça bağlantı aktif olacaktır.")
    print("Kapatmak için CTRL+C basın.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Ngrok kapatılıyor...")
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        print("Ngrok durduruldu.")


if __name__ == '__main__':
    main()
