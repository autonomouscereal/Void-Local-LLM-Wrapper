FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Force CPU-only execution and prevent CUDA probing
ENV CUDA_VISIBLE_DEVICES="" \
    PYTORCH_FORCE_CPU=1

RUN apt-get update && apt-get install -y git ffmpeg build-essential libgl1 execstack && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git .

RUN pip install --no-cache-dir -r requirements.txt
# Pin CPU-safe wheels for Python 3.10 to avoid NumPy 2 breakage and source builds
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80 imageio-ffmpeg==0.4.9 pillow==10.4.0 onnxruntime==1.16.3
RUN pip install --no-cache-dir scikit-image==0.22.0 scikit-learn==1.3.2 diffusers==0.30.2 insightface==0.7.3
\
 && execstack -c /usr/local/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so || true

EXPOSE 8188

# Run ComfyUI in CPU mode
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--cpu"]


