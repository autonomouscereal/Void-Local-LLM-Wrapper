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
    pip install --no-cache-dir --upgrade "pip<24" && \
    pip install --no-cache-dir --upgrade "numpy==1.26.4" && \
    pip install --no-cache-dir --upgrade "PyYAML==6.0.1" && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
      "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118"

# Explicitly install modules required by custom nodes (constrained by PIP_CONSTRAINT)
RUN pip install --no-cache-dir \
      opencv-python-headless \
      diffusers \
      imageio-ffmpeg \
      insightface \
      onnx \
      scikit-image \
      matplotlib \
      easydict \
      prettytable \
      requests

# Core libs under constraints
RUN pip install --no-cache-dir --no-deps -r /constraints.txt && \
    pip install --no-cache-dir --requirement /constraints.txt && \
    pip uninstall -y opencv-python || true

# ComfyUI via codeload tarball (no git)
WORKDIR /comfyui
RUN curl -fL https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/master \
    | tar -xzf - --strip-components=1 -C /comfyui

# Ensure no leftover Real-ESRGAN custom node remains in the image (two common names)
RUN rm -rf /comfyui/custom_nodes/ComfyUI-Real-ESRGAN /comfyui/custom_nodes/comfyui-RealESRGAN || true && \
    find /comfyui/custom_nodes -maxdepth 1 -type d -iname "*real*esrgan*" -exec rm -rf {} + || true

# Install only the custom nodes we actually use (no Real-ESRGAN)
RUN mkdir -p /comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    curl -fL https://codeload.github.com/cubiq/ComfyUI_IPAdapter_plus/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI_IPAdapter_plus
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-InstantID && \
    curl -fL https://codeload.github.com/cubiq/ComfyUI-InstantID/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-InstantID || true
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    curl -fL https://codeload.github.com/Kosinkadink/ComfyUI-VideoHelperSuite/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite

# Ensure PyYAML is in ComfyUI requirements and install via requirements.txt (respects constraints)
RUN if [ -f /comfyui/requirements.txt ]; then \
      (grep -qi '^PyYAML' /comfyui/requirements.txt || echo 'PyYAML==6.0.1' >> /comfyui/requirements.txt); \
      pip install --no-cache-dir -r /comfyui/requirements.txt; \
    else \
      echo 'PyYAML==6.0.1' > /comfyui/requirements.txt && pip install --no-cache-dir -r /comfyui/requirements.txt; \
    fi

# No custom node tarball pulls; rely on ComfyUI core only

RUN mkdir -p /comfyui/models/upscale /comfyui/input /comfyui/output

EXPOSE 8188
CMD ["python","-u","main.py","--listen","--port","8188"]


