FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

# Torch cu118 (P40 + RTX 8000)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.7.1+cu118" "torchaudio==2.7.1+cu118" "torchvision==0.22.1+cu118" \
 && pip install --no-cache-dir fastapi uvicorn soundfile librosa numpy scipy

ENV HF_HOME=/models/hf TRANSFORMERS_CACHE=/models/hf TORCH_HOME=/models/torch
ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9"

WORKDIR /app

COPY app /app/app

EXPOSE 7860

CMD ["python","-m","app.main"]


