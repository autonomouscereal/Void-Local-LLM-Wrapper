FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.7.1+cu118" "torchaudio==2.7.1+cu118" "torchvision==0.22.1+cu118" \
 && pip install --no-cache-dir fastapi uvicorn soundfile numpy huggingface_hub

WORKDIR /app
COPY app /app/app

EXPOSE 7863
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7863"]


