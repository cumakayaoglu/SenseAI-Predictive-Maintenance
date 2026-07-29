import torch
import torch.nn as nn

# NASA Verisine Uygun (24 Özellik, 2 Sınıf)
class TimeSeriesPdMLSTM(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        # Bidirectional=True yapıldı. Geçmişi ve geleceği harmanlar.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        # Çift yönlü olduğu için gizli katman boyutu 2 katına çıkar (hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.dropout = nn.Dropout(0.2) # Ezberi önleyen unutma katmanı
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes) # 0: Normal, 1: Kritik

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.dropout(out)
        out = self.relu(out)
        return self.fc2(out)

def load_production_model(model_path):
    model = TimeSeriesPdMLSTM(input_size=24)
    # Sadece ağırlıkları yükle, CPU'da çalışmaya zorla
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval() # Geri yayılımı kapat (Tahmin modu)
    return model