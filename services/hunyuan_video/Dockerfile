# Devel image is required to compile SageAttention and SGL-Kernel
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System dependencies for compilation and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg ca-certificates build-essential libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install merged requirements
COPY services/hunyuan_video/requirements.txt /app/requirements.txt
RUN python -m pip install -U pip && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# Clone the specific HunyuanVideo-1.5 repository
ARG HUNYUAN_REPO_URL=https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
ARG HUNYUAN_REPO_REF=main
RUN git clone --depth=1 --branch "${HUNYUAN_REPO_REF}" "${HUNYUAN_REPO_URL}" /opt/hunyuan15_src

# Set up environment paths
ENV HYVIDEO_PORT=8094
ENV HYVIDEO_CODE_PATH=/opt/hunyuan15_src
ENV PYTHONPATH="${PYTHONPATH}:/opt/hunyuan15_src"
ENV FILM2_MODELS=/opt/models

# Shared packages and service code
COPY void_json /app/void_json
COPY services/hunyuan_video/app.py /app/app.py

EXPOSE 8094
CMD ["python","-u","/app/app.py"]
