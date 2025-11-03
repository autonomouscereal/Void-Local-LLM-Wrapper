FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Pre-install CUDA torch cu118 compatible with P40 and RTX 8000
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]


