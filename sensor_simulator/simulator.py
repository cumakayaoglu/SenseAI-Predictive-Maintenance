import time
import requests
import pandas as pd
import logging
from pathlib import Path

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SIMULATOR - %(message)s')

# Yollar
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = BASE_DIR / "ml_pipeline" / "data" / "cmapss" / "processed_test.csv"
API_URL = "http://localhost:8000/ingest"

def run_simulation():
    if not TEST_DATA_PATH.exists():
        logging.error(f"[!] Test veri seti bulunamadı: {TEST_DATA_PATH}")
        return

    logging.info(f"Test veri seti yükleniyor... ({TEST_DATA_PATH})")
    
    # NASA sütun isimlerini ayarlıyoruz (Train ile birebir aynı)
    columns = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
    columns += [f'sensor_{i}' for i in range(1, 22)]
    
    # Pandas ile oku (Başlığı (header) yok, kendi isimlerimizi atıyoruz)
    try:
        df = pd.read_csv(TEST_DATA_PATH)
    except Exception as e:
        # Eğer Pandas ile okunurken hata verirse, varsayılan formata dön
        df = pd.read_csv(TEST_DATA_PATH, sep='\s+', header=None, names=columns)

    logging.info(f"Simülasyon Başlıyor... Toplam Test Satırı: {len(df)}")
    logging.info("Motorlar uçağa bağlandı. Veri akışı saniyede 1 okuma olacak şekilde başlatılıyor.\n")
    
    # API'ye çok hızlı veri atıp boğmamak (ve canlı görmeni sağlamak) için saniyede 1 veri göndereceğiz
    for index, row in df.iterrows():
        unit = int(row['unit_number'])
        cycle = int(row['time_cycles'])
        
        # 3 Ayar (Setting)
        set_1, set_2, set_3 = row['setting_1'], row['setting_2'], row['setting_3']
        
        # 21 Sensör (Sensor)
        sensors = [row[f'sensor_{i}'] for i in range(1, 22)]

        # FastAPI'nin beklediği Pydantic şemasına tam uyumlu JSON objesi
        payload = {
            "unit_number": unit,
            "time_cycles": cycle,
            "setting_1": set_1,
            "setting_2": set_2,
            "setting_3": set_3,
            "sensors": sensors
        }

        try:
            # Veriyi ateşle
            response = requests.post(API_URL, json=payload)
            
            # API'den dönen yanıtı (Risk, Normal, Buffering) yakala ve konsola bas
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "buffering":
                    logging.info(f"[MOTOR {unit}] {result.get('msg')} (Geçmiş veri toplanıyor...)")
                elif result.get("status") == "CRITICAL_RISK":
                    logging.error(f"!!! [MOTOR {unit} - Cycle {cycle}] KRİTİK ARIZA RİSKİ !!! (Güven: %{result.get('confidence_pct')})")
                else:
                    logging.info(f"[MOTOR {unit} - Cycle {cycle}] Durum Normal. (Güven: %{result.get('confidence_pct')})")
            else:
                logging.warning(f"API Yanıt Hatası: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            logging.error("API sunucusuna ulaşılamıyor! (uvicorn çalışıyor mu?)")
            break # Sunucu yoksa döngüyü kır
            
        time.sleep(0.5) # Gerçekçi bir izleme için her döngü arası yarım saniye bekle

if __name__ == "__main__":
    run_simulation()