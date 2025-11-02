FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Pre-install CUDA torch so coqui-tts uses GPU if available (falls back to CPU otherwise)
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]


