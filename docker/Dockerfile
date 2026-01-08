FROM python:3.10-slim

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (TensorFlow needs these)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    numpy

# Copy training script
COPY train.py /app/train.py

ENTRYPOINT ["python", "/app/train.py"]