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

# Custom nodes via codeload tarballs
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-Real-ESRGAN && \
    curl -fL https://codeload.github.com/pythongosssss/ComfyUI-Real-ESRGAN/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-Real-ESRGAN || \
    ( mkdir -p /comfyui/custom_nodes/comfyui-RealESRGAN && \
      curl -fL https://codeload.github.com/Acly/comfyui-RealESRGAN/tar.gz/main \
        | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/comfyui-RealESRGAN )

RUN mkdir -p /comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    curl -fL https://codeload.github.com/cubiq/ComfyUI_IPAdapter_plus/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

RUN mkdir -p /comfyui/custom_nodes/ComfyUI-InstantID && \
    curl -fL https://codeload.github.com/cubiq/ComfyUI-InstantID/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-InstantID

RUN mkdir -p /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    curl -fL https://codeload.github.com/Kosinkadink/ComfyUI-VideoHelperSuite/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite

RUN mkdir -p /comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved && \
    curl -fL https://codeload.github.com/ArtVentureX/comfyui-animatediff/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved

# Real-ESRGAN runtime deps (respect constraints)
RUN pip install --no-cache-dir realesrgan basicsr facexlib filterpy

# Make any self-upgrading node respect constraints
RUN find /comfyui/custom_nodes -type f -name "*.py" \
    -exec sed -i 's/pip install -U/pip install -U -c \/constraints.txt/g' {} \; || true

RUN mkdir -p /comfyui/models/upscale /comfyui/input /comfyui/output

EXPOSE 8188
CMD ["python","-m","ComfyUI.main","--listen","--port","8188"]


