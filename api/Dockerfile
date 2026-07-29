# Aşama 1: Builder (Derleyici - İmaj boyutunu küçültmek için)
FROM python:3.10-slim as builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Mülakat Detayı: Docker imaj boyutunu küçültmek için sadece CPU odaklı PyTorch indiriyoruz.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Aşama 2: Runner (Çalıştırıcı)
FROM python:3.10-slim
# Güvenlik kuralı: Uygulamayı root olarak DEĞİL, kısıtlı kullanıcı ile çalıştır
RUN useradd -m -r appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY api/ .
USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]