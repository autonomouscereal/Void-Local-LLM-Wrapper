FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox rubberband-cli && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.7.1+cu118" "torchaudio==2.7.1+cu118" \
 && pip install --no-cache-dir crepe pyloudnorm librosa soundfile numpy scipy fastapi uvicorn soxr fastdtw

ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9"

WORKDIR /app
COPY app /app/app

EXPOSE 7864
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7864","--log-level","info","--access-log","--use-colors"]


