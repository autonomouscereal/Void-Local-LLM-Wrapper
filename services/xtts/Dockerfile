FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
# Compose controls PyTorch wheel channel (e.g., cu126 for Pascal P40)
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Pre-install CUDA torch cu126 compatible with P40 and RTX 8000
RUN pip install --no-cache-dir torch==2.8.* torchaudio torchvision
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]


