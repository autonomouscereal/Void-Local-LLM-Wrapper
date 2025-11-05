FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Compose controls CUDA wheels channel for PyTorch (e.g., cu118 for Pascal P40)
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

RUN apt-get update && apt-get install -y git ffmpeg build-essential libgl1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git .

# Refresh or install common custom nodes (stable API levels) and remove deprecated frontend widget
RUN set -eux; \
    mkdir -p /comfyui/custom_nodes; \
    if [ -d /comfyui/custom_nodes/ComfyUI-IPAdapter_plus ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-IPAdapter_plus pull --rebase; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git ComfyUI-IPAdapter_plus; \
    fi; \
    if [ -d /comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved pull --rebase; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git; \
    fi; \
    if [ -d /comfyui/custom_nodes/ComfyUI-VideoHelperSuite ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite pull --rebase; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git; \
    fi; \
    if [ -d /comfyui/custom_nodes/ComfyUI-InstantID ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-InstantID pull --rebase; \
    elif [ -d /comfyui/custom_nodes/ComfyUI-InstantIDContext ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-InstantIDContext pull --rebase; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/cubiq/ComfyUI_InstantID.git ComfyUI-InstantID; \
    fi; \
    # Install/refresh ADetailer and SVDiffusion Relight nodes (best-effort; ignore failures)
    if [ -d /comfyui/custom_nodes/ComfyUI-ADetailer ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-ADetailer pull --rebase || true; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/Okipol/ComfyUI-ADetailer.git || true; \
    fi; \
    if [ -d /comfyui/custom_nodes/ComfyUI-SVDiffusion ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-SVDiffusion pull --rebase || true; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/kijai/ComfyUI-SVDiffusion.git ComfyUI-SVDiffusion || true; \
    fi; \
    if [ -d /comfyui/custom_nodes/ComfyUI-SVDiffusion-Relight ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-SVDiffusion-Relight pull --rebase || true; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/kijai/ComfyUI-SVDiffusion-Relight.git ComfyUI-SVDiffusion-Relight || true; \
    fi; \
    # Real-ESRGAN custom node(s)
    if [ -d /comfyui/custom_nodes/ComfyUI-Real-ESRGAN ]; then \
      git -C /comfyui/custom_nodes/ComfyUI-Real-ESRGAN pull --rebase || true; \
    else \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/pythongosssss/ComfyUI-Real-ESRGAN.git ComfyUI-Real-ESRGAN || true; \
    fi; \
    if [ ! -d /comfyui/custom_nodes/ComfyUI-Real-ESRGAN ] && [ ! -d /comfyui/custom_nodes/comfyui-RealESRGAN ]; then \
      git -C /comfyui/custom_nodes clone --depth 1 https://github.com/Acly/comfyui-RealESRGAN.git comfyui-RealESRGAN || true; \
    fi; \
    rm -f /comfyui/web/extensions/core/widgetInputs.js || true

# Pre-install CUDA-enabled torch (cu118) compatible with P40 (sm_61) and RTX 8000.
# This ensures Pascal coverage; Maxwell nodes are not primary targets for torch wheels.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118"
# Pin NumPy ABI FIRST, then install ComfyUI requirements without upgrading pins
RUN pip install --no-cache-dir --upgrade "numpy==1.26.4"
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt
# Pinned runtime libs compatible with NumPy 1.26.x and py3.10/3.11
RUN pip install --no-cache-dir \
    opencv-python-headless==4.9.0.80 imageio-ffmpeg==0.4.9 pillow==10.4.0 \
    onnxruntime==1.17.3 scikit-image==0.22.0 scikit-learn==1.4.2 \
    diffusers==0.30.2 transformers==4.44.2 accelerate==0.34.2 safetensors>=0.4.3 \
    kornia==0.7.2 insightface==0.7.3
# Real-ESRGAN runtime deps (GPU optional)
RUN pip install --no-cache-dir realesrgan==0.3.0 basicsr==1.4.2 facexlib==0.3.0 filterpy
RUN mkdir -p /comfyui/models/upscale
## execstack not available on Debian slim; rely on onnxruntime==1.14.1 CPU wheel which avoids execstack

EXPOSE 8188

# Entrypoint ensures required models exist, then starts ComfyUI in CPU mode
COPY docker-entrypoint.sh /usr/local/bin/comfyui-entrypoint.sh
RUN chmod +x /usr/local/bin/comfyui-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/comfyui-entrypoint.sh"]


