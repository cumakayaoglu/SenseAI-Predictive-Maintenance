import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
from pathlib import Path

# --- 1. Yollar ve Yapılandırma ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cmapss" / "processed_train.csv"
ARTIFACT_DIR = BASE_DIR.parent / "api" / "ml_model" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparametreler
SEQ_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 20 
LEARNING_RATE = 0.001
FEATURES_COUNT = 24 
PATIENCE = 5 

# --- 2. Motor Bazlı Sequence Oluşturma ---
def create_sequences(df, feature_cols, seq_length):
    xs, ys = [], []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        X_unit = unit_data[feature_cols].values
        y_unit = unit_data['label'].values
        
        for i in range(len(unit_data) - seq_length):
            xs.append(X_unit[i:(i + seq_length)])
            ys.append(y_unit[i + seq_length])
            
    return np.array(xs), np.array(ys)

class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- 3. LSTM Mimarisi ---
class TimeSeriesPdMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        # Bidirectional=True ekledik. Artık model geçmişi ve geleceği harmanlıyor!
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        # Çift yönlü olduğu için gizli katman boyutu 2 katına çıkar (hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.dropout = nn.Dropout(0.2) # Fazladan bir unutma katmanı (ezberi önler)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.dropout(out)
        out = self.relu(out)
        return self.fc2(out)

# --- 4. Eğitim Döngüsü ---
def train_model():
    print(f"NASA CMAPSS verisi yükleniyor... ({DATA_PATH})")
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # Dinamik ve güvenli Motor Bazlı Ayrım (Train/Val Split)
    units = sorted(df["unit_number"].unique())
    split = int(len(units) * 0.8)
    train_units = units[:split]
    val_units = units[split:]
    
    train_df = df[df["unit_number"].isin(train_units)].copy()
    val_df = df[df["unit_number"].isin(val_units)].copy()

    # Normalizasyon
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
    print("Scaler kaydedildi.")

    # Sequence'leri Oluştur
    X_train_seq, y_train_seq = create_sequences(train_df, feature_cols, SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(val_df, feature_cols, SEQ_LENGTH)

    train_loader = DataLoader(MotorDataset(X_train_seq, y_train_seq), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MotorDataset(X_val_seq, y_val_seq), batch_size=BATCH_SIZE, shuffle=False)

    model = TimeSeriesPdMLSTM(input_size=FEATURES_COUNT)
    
    # Label Smoothing eklendi! (0 ve 1 yerine, hedefleri yumuşatır, %100 güveni engeller)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Weight Decay eklendi! (Modelin ağırlıklarını kısıtlayarak ezberlemeyi zorlaştırır)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_path = ARTIFACT_DIR / "best_model.pth"

    print("\n--- Model Eğitimi Başlıyor ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= PATIENCE:
                print(f"\n[!] Early Stopping tetiklendi.")
                break

    print("\n--- En İyi Model İle Nihai Değerlendirme ---")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    
    print(f"Final Val Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print(f"\nEğitim tamamlandı. Model ve scaler kaydedildi: {ARTIFACT_DIR}")

if __name__ == "__main__":
    train_model()