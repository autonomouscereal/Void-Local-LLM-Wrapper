FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tar ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Constraints (lock numpy<2 and friends)
COPY constraints.txt /constraints.txt
ENV PIP_CONSTRAINT=/constraints.txt
ENV GIT_TERMINAL_PROMPT=0

# Core: NumPy 1.x + Torch cu118 (GPU optional)
RUN python -V && pip -V && \
    pip install --no-cache-dir --upgrade "numpy==1.26.4" && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
      "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118"

# Core libs under constraints
RUN pip install --no-cache-dir \
    kornia onnxruntime scikit-learn opencv-python-headless \
    safetensors accelerate transformers diffusers insightface && \
    pip uninstall -y opencv-python || true

# ComfyUI via codeload tarball (no git)
WORKDIR /comfyui
RUN curl -fL https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/master \
    | tar -xzf - --strip-components=1 -C /comfyui

# No custom node tarball pulls; rely on ComfyUI core only

RUN mkdir -p /comfyui/models/upscale /comfyui/input /comfyui/output

EXPOSE 8188
CMD ["python","-u","main.py","--listen","--port","8188"]


