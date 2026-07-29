import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from pathlib import Path
import joblib
from train_lstm import TimeSeriesPdMLSTM

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = BASE_DIR / "ml_pipeline" / "data" / "cmapss" / "processed_test.csv"
RUL_PATH = BASE_DIR / "ml_pipeline" / "data" / "cmapss" / "RUL_FD001.txt"
MODEL_PATH = BASE_DIR / "api" / "ml_model" / "artifacts" / "best_model.pth"
SCALER_PATH = BASE_DIR / "api" / "ml_model" / "artifacts" / "scaler.pkl"

def evaluate_model():
    print("Test verisi, NASA RUL cevap anahtarı ve model yükleniyor...")
    df = pd.read_csv(TEST_DATA_PATH)
    
    # NASA RUL (Kalan Ömür) cevap anahtarını oku
    with open(RUL_PATH, 'r') as f:
        true_ruls = [int(line.strip()) for line in f if line.strip()]
        
    scaler = joblib.load(SCALER_PATH)
    
    model = TimeSeriesPdMLSTM(input_size=24)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    all_preds = []
    all_targets = []

    # NASA FD001 test veri setinde arıza eşiği genelde RUL <= 30'dur (Binary Classification için)
    FAILURE_THRESHOLD = 30

    grouped = df.groupby('unit_number')
    for idx, (unit_id, group) in enumerate(grouped):
        # Bu motorun NASA cevap anahtarındaki gerçek kalan ömrü
        if idx >= len(true_ruls):
            break
        final_rul = true_ruls[idx]
        
        scaled_data = scaler.transform(group[feature_cols])
        
        # Motorun son anındaki gerçek durumu: Kalan ömrü 30'dan küçükse 1 (Kritik), değilse 0 (Normal)
        # Ancak test dosyasının satır bazlı etiketini simüle etmek için her döngüdeki kalan ömrü hesaplıyoruz
        total_cycles_in_test = len(group)
        
        for i in range(30, total_cycles_in_test):
            window = scaled_data[i-30:i]
            
            # Bu satırdaki anlık gerçek RUL = (Toplam satır - o anki döngü) + NASA'nın verdiği final RUL
            current_rul = final_rul + (total_cycles_in_test - i)
            target = 1 if current_rul <= FAILURE_THRESHOLD else 0
            
            tensor_in = torch.tensor([window], dtype=torch.float32)
            with torch.no_grad():
                output = model(tensor_in)
                pred = torch.argmax(output, dim=1).item()
                
            all_preds.append(pred)
            all_targets.append(target)

    # Metrikleri Hesapla
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    
    print("\n" + "="*45)
    print(" 📊 SENSEAI MODEL NASA DOĞRULAMA RAPORU")
    print("="*45)
    print(f" Test Edilen Toplam Pencere : {len(all_targets)}")
    print(f" Doğruluk (Accuracy)       : %{acc * 100:.2f}")
    print(f" Kesinlik (Precision)      : %{precision * 100:.2f}")
    print(f" Duyarlılık (Recall)       : %{recall * 100:.2f}")
    print(f" F1-Skoru                  : %{f1 * 100:.2f}")
    print("="*45)
    print("Doğrulama başarıyla tamamlandı!")

if __name__ == "__main__":
    evaluate_model()