FROM python:3.10-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Force CPU-only execution and prevent CUDA probing
ENV CUDA_VISIBLE_DEVICES="" \
    PYTORCH_FORCE_CPU=1

RUN apt-get update && apt-get install -y git ffmpeg build-essential libgl1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git .

RUN pip install --no-cache-dir -r requirements.txt
# Pin CPU-safe wheels for Python 3.10 to avoid NumPy 2 breakage and source builds
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80 imageio-ffmpeg==0.4.9 pillow==10.4.0 onnxruntime==1.16.3
RUN pip install --no-cache-dir scikit-image==0.22.0 scikit-learn==1.3.2 diffusers==0.30.2 insightface==0.7.3
## execstack not available on Debian slim; rely on onnxruntime==1.14.1 CPU wheel which avoids execstack

EXPOSE 8188

# Entrypoint ensures required models exist, then starts ComfyUI in CPU mode
COPY docker-entrypoint.sh /usr/local/bin/comfyui-entrypoint.sh
RUN chmod +x /usr/local/bin/comfyui-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/comfyui-entrypoint.sh"]


