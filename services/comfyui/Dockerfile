FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Enable CUDA wheels channel for PyTorch cu126 (Pascal sm_61 P40 + RTX8000)
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN apt-get update && apt-get install -y git ffmpeg build-essential libgl1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git .

# Pre-install CUDA-enabled torch (cu126) compatible with P40 (sm_61) and RTX 8000
RUN pip install --no-cache-dir torch==2.8.* torchvision torchaudio
# Install ComfyUI requirements and common deps
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir numpy==1.26.4 opencv-python-headless==4.9.0.80 imageio-ffmpeg==0.4.9 pillow==10.4.0 \
    onnxruntime==1.16.3 scikit-image==0.22.0 scikit-learn==1.3.2 diffusers==0.30.2 insightface==0.7.3
## execstack not available on Debian slim; rely on onnxruntime==1.14.1 CPU wheel which avoids execstack

EXPOSE 8188

# Entrypoint ensures required models exist, then starts ComfyUI in CPU mode
COPY docker-entrypoint.sh /usr/local/bin/comfyui-entrypoint.sh
RUN chmod +x /usr/local/bin/comfyui-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/comfyui-entrypoint.sh"]


