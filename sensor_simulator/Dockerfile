FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir requests
COPY sensor_simulator/ .
CMD ["python", "simulator.py"]