FROM python:3.11-slim

WORKDIR /app

# System deps for TensorFlow + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure saved_models directory exists in the image
RUN mkdir -p saved_models

EXPOSE 5000

# 2 workers; increase for higher traffic (each worker is ~1–2 GB RAM with TF loaded)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "--log-level", "info", "app:app"]
