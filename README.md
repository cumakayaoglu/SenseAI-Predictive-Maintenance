#  SenseAI: Uçtan Uca Kestirimci Bakım (PdM) Sistemi

SenseAI, turbofan motor arızalarını gerçekleşmeden önce tahmin etmek üzere tasarlanmış, üretime hazır (production-ready) bir Makine Öğrenmesi Operasyonları (MLOps) mimarisidir. NASA CMAPSS (FD001) veri setini temel alan bu sistem, fiziksel uç cihaz (edge device) sensörleri ile yönetici seviyesi izleme (observability) ekranları arasında asenkron ve gerçek zamanlı bir köprü kurar.

---

##  Mimari ve Temel Özellikler

* **Durumlu (Stateful) Kayan Pencere Çıkarımı:** Gerçek zamanlı IoT sensör telemetri akışlarını RAM üzerinde tamponlamak için `collections.deque` kullanır. 30 döngülük zaman serisi pencerelerini veritabanı darboğazı yaşamadan anlık olarak işler.
* **Kalibre Edilmiş Derin Öğrenme:** Modelin aşırı özgüven (overconfidence) problemini ve ezberlemesini önlemek amacıyla, **Label Smoothing** ve **L2 Regularization (Weight Decay)** entegreli **Çift Yönlü (Bidirectional) LSTM** mimarisi kullanılmıştır. Kesin logit değerleri yerine, endüstriyel kararlara uygun olasılıksal güven skorları (örn. %94.5) üretir.
* **Asenkron Gözlemlenebilirlik Akışı:** Çıkarım sonuçlarını (Risk Sınıfı ve Güven Skoru) **InfluxDB**'ye asenkron olarak yazmak için FastAPI `BackgroundTasks` yapısını kullanır. Bu sayede simülatör ve API arasındaki haberleşmede sıfır gecikme garanti edilir.
* **Gerçek Zamanlı Dashboard:** Motor bozulma eğilimlerini, anomali eşiklerini ve sensör bazlı risk durumlarını görselleştirmek için Flux sorguları aracılığıyla **Grafana** entegrasyonu sağlanmıştır.
* **Donanım-Sunucu Simülasyonu:** Endüstriyel donanımların veri iletim protokollerini taklit eden, asenkron veri basan özel bir Python simülatörü içerir.

---

##  Model Performans Doğrulaması

Model, eğitimde hiç kullanılmayan `test_FD001.txt` veri seti ve NASA'nın gerçek `RUL_FD001.txt` (Kalan Faydalı Ömür) hedef değişkenleri (ground-truth) üzerinden doğrulanmıştır.

| Metrik | Skor | Endüstriyel Karşılığı |
| :--- | :--- | :--- |
| **Doğruluk (Accuracy)** | **%98.90** | 10.096 farklı zaman penceresi boyunca sistemin genel doğruluk oranı. |
| **F1-Skoru (F1-Score)** | **%81.95** | Sınıf dengesizliği durumlarında modelin gerçek performansını yansıtan harmonik ortalama. |
| **Duyarlılık (Recall)** | **%82.08** | Arızalanacak motorların %82'sini, operasyonel bir çöküş yaşanmadan önce yakalama oranı. |
| **Kesinlik (Precision)** | **%81.82** | Maliyetli "Yanlış Alarm (False Positive)" operasyonlarını minimize eden yüksek güvenilirlik oranı. |

---

##  Hızlı Başlangıç (Docker Compose)

Tüm altyapı (API, Simülatör, Veritabanı ve Dashboard) mikroservis mantığıyla konteynerize edilmiştir.

### 1. Klonla ve Çalıştır
```bash
git clone [https://github.com/KULLANICI_ADIN/SenseAI-Predictive-Maintenance.git](https://github.com/KULLANICI_ADIN/SenseAI-Predictive-Maintenance.git)
cd SenseAI-Predictive-Maintenance
docker-compose up --build -d
2. Servislere Erişim
Inference API: http://localhost:8000 (Swagger UI: http://localhost:8000/docs)

Grafana Dashboard: http://localhost:3000 (Kullanıcı adı: admin / Şifre: admin)

InfluxDB: http://localhost:8086

3. Grafana - InfluxDB Bağlantısı
Grafana arayüzünden Data Sources bölümüne gidin.

InfluxDB seçeneğini ekleyip sorgu dilini (Query Language) Flux olarak ayarlayın.

URL: http://influxdb:8086

Organization: senseai_org | Token: senseai-super-secret-token | Bucket: senseai_bucket

📂 Proje Yapısı
SenseAI-Predictive-Maintenance/
├── api/                     # FastAPI çıkarım sunucusu
│   ├── main.py              # Uç noktalar (Endpoints) ve InfluxDB asenkron görevleri
│   ├── Dockerfile
│   └── ml_model/            # Ağırlıklar ve standardizasyon dosyaları (.pth, .pkl)
├── ml_pipeline/             # Makine öğrenmesi eğitim ve test modülleri
│   ├── train_lstm.py        # Bi-LSTM eğitim betiği (Label Smoothing + L2)
│   ├── evaluate.py          # NASA RUL doğrulama ve metrik hesaplama betiği
│   └── data/                # CMAPSS veri seti
├── sensor_simulator/        # IoT donanım veri iletim simülatörü
│   ├── simulator.py         # Asenkron HTTP POST jeneratörü
│   └── Dockerfile
└── docker-compose.yml       # Tüm servislerin orkestrasyonu
