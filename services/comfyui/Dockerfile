FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tar ffmpeg build-essential g++ && \
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

# Build helpers for packages that may fall back to sdist (e.g., insightface)
RUN pip install --no-cache-dir "setuptools<70" wheel "cython==3.0.10" "pybind11>=2.12"

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
    (curl -fL https://codeload.github.com/cubiq/ComfyUI-InstantID/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-InstantID || true) && \
    if [ ! -f /comfyui/custom_nodes/ComfyUI-InstantID/__init__.py ]; then \
      (curl -fL https://codeload.github.com/InstantX/ComfyUI-InstantID/tar.gz/main \
        | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-InstantID || true); \
    fi && \
    if [ ! -f /comfyui/custom_nodes/ComfyUI-InstantID/__init__.py ] && [ -f /comfyui/custom_nodes/ComfyUI-InstantID/InstantIDNode.py ]; then \
      echo 'from .InstantIDNode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS' > /comfyui/custom_nodes/ComfyUI-InstantID/__init__.py; \
    fi
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    curl -fL https://codeload.github.com/Kosinkadink/ComfyUI-VideoHelperSuite/tar.gz/main \
      | tar -xzf - --strip-components=1 -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite

# Preload InsightFace models so InstantID works out-of-the-box
ENV INSIGHTFACE_HOME=/comfyui/models/insightface
RUN mkdir -p "$INSIGHTFACE_HOME" && \
    python - <<'PY'
import os
os.environ.setdefault('INSIGHTFACE_HOME', '/comfyui/models/insightface')
try:
    from insightface.app import FaceAnalysis  # type: ignore
    app = FaceAnalysis(name='buffalo_l')
    # Use CPU at build; runtime may switch to GPU
    app.prepare(ctx_id=-1)
except Exception as e:
    print('[insightface preload] skipped:', e)
PY

# Ensure ComfyUI-InstantID has a package __init__.py pointing to the node module that defines NODE_CLASS_MAPPINGS
RUN python - <<'PY'
import os, sys
root = "/comfyui/custom_nodes/ComfyUI-InstantID"
init_path = os.path.join(root, "__init__.py")
if not os.path.isdir(root):
    sys.exit(0)
def find_node_module() -> str | None:
    for d, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith('.py'): continue
            p = os.path.join(d, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                    txt = fh.read()
                if ('NODE_CLASS_MAPPINGS' in txt) and ('NODE_DISPLAY_NAME_MAPPINGS' in txt):
                    rel = os.path.relpath(p, root).replace('\\', '/').removesuffix('.py')
                    return rel
            except Exception:
                continue
    return None
mod = find_node_module()
if (not os.path.isfile(init_path)) and mod:
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(f"from .{mod.replace('/', '.')} import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\n")
    print('[instantid] created __init__.py ->', mod)
elif not os.path.isfile(init_path):
    # Fallback to common filename
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write("from .InstantIDNode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\n")
    print('[instantid] created fallback __init__.py')
PY

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


