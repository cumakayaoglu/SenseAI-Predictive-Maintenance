from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import logging
import torch
import torch.nn.functional as F
from collections import deque
import joblib
from pathlib import Path
import warnings
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
warnings.filterwarnings("ignore")

from ml_model.model import load_production_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Yollar
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "ml_model" / "artifacts"


INFLUX_URL = "http://influxdb:8086"
INFLUX_TOKEN = "senseai-super-secret-token"
INFLUX_ORG = "senseai_org"
INFLUX_BUCKET = "senseai_bucket"

# Global Nesneler
ml_models = {}
scaler = None
device_buffers = {}
SEQ_LENGTH = 30
influx_client = None
write_api = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scaler, influx_client, write_api
    logging.info("Model ve Scaler belleğe yükleniyor (Cold Start önlemi)...")
    ml_models["senseai"] = load_production_model(ARTIFACT_DIR / "best_model.pth")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    
    # InfluxDB Bağlantısını Başlat
    try:
        influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        logging.info("InfluxDB bağlantısı BAŞARILI!")
    except Exception as e:
        logging.error(f"InfluxDB bağlantı hatası: {e}")

    logging.info("Sistem hazır!")
    yield
    
    ml_models.clear()
    if influx_client:
        influx_client.close()

app = FastAPI(title="SenseAI PdM Inference API", lifespan=lifespan)

class SensorData(BaseModel):
    unit_number: int
    time_cycles: int
    setting_1: float
    setting_2: float
    setting_3: float
    sensors: list[float]

# Arka planda InfluxDB'ye veri yazan fonksiyon
def write_to_influx(unit: str, cycle: int, predicted_class: int, confidence: float, status: str):
    if write_api:
        try:
            point = (
                Point("engine_health")
                .tag("unit", unit)
                .field("cycle", cycle)
                .field("risk_class", predicted_class)
                .field("confidence", confidence)
                .field("status", status)
            )
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        except Exception as e:
            logging.error(f"InfluxDB yazma hatası: {e}")

@app.post("/ingest")
async def ingest_data(data: SensorData, background_tasks: BackgroundTasks):
    global scaler
    
    if len(data.sensors) != 21:
        return {"error": "Tam olarak 21 sensör verisi gönderilmelidir."}

    unit = str(data.unit_number)
    if unit not in device_buffers:
        device_buffers[unit] = deque(maxlen=SEQ_LENGTH)
        
    raw_features = [data.setting_1, data.setting_2, data.setting_3] + data.sensors
    scaled_features = scaler.transform([raw_features])[0]
    device_buffers[unit].append(scaled_features.tolist())
    
    if len(device_buffers[unit]) < SEQ_LENGTH:
        return {
            "unit": unit, 
            "status": "buffering", 
            "cycle": data.time_cycles,
            "msg": f"Pencere doluyor: {len(device_buffers[unit])}/{SEQ_LENGTH}"
        }

    sequence_tensor = torch.tensor([list(device_buffers[unit])], dtype=torch.float32)

    with torch.no_grad():
        logits = ml_models["senseai"](sequence_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    if predicted_class == 1:
        log_msg = f"[MOTOR {unit} - Cycle {data.time_cycles}] DİKKAT! KRİTİK ARIZA RİSKİ - Güven: %{confidence:.1f}"
        logging.error(log_msg)
        status = "CRITICAL_RISK"
    else:
        log_msg = f"[MOTOR {unit} - Cycle {data.time_cycles}] Sistem Normal - Güven: %{confidence:.1f}"
        logging.info(log_msg)
        status = "NORMAL"

    
    background_tasks.add_task(
        write_to_influx, 
        unit=unit, 
        cycle=data.time_cycles, 
        predicted_class=predicted_class, 
        confidence=round(confidence, 1), 
        status=status
    )

    return {
        "unit": unit,
        "cycle": data.time_cycles,
        "status": status,
        "confidence_pct": round(confidence, 1)
    }