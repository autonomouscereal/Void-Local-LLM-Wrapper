FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Force CPU-only execution and prevent CUDA probing
ENV CUDA_VISIBLE_DEVICES="" \
    PYTORCH_FORCE_CPU=1

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8188

# Run ComfyUI in CPU mode
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--cpu"]


