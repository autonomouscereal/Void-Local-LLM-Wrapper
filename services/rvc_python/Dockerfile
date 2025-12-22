FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# System deps (include build tools and Python headers for fairseq/pyworld, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-venv \
        python3-pip \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps: pin pip to a version compatible with rvc-python's fairseq/omegaconf
# stack, then install rvc-python and the canonical Torch stack (cu118).
RUN python3 -m pip install --no-cache-dir "pip<24.1" && \
    python3 -m pip install --no-cache-dir rvc-python==0.1.5 && \
    python3 -m pip install --no-cache-dir \
        "torch==2.7.1+cu118" \
        "torchaudio==2.7.1+cu118" \
        --index-url https://download.pytorch.org/whl/cu118

# Optional but commonly required by RVC training/inference stacks.
# Installed here so it is always available inside the container.
RUN python3 -m pip install --no-cache-dir "tensorboardX>=2.6"

# Models directory; mount a volume here
ENV RVC_MODELS_DIR=/srv/rvc_models
RUN mkdir -p ${RVC_MODELS_DIR}

# Copy entrypoint script
COPY entrypoint.sh /srv/entrypoint.sh
RUN chmod +x /srv/entrypoint.sh

EXPOSE 5050

# Use entrypoint to set up Titan model from bootstrap, then start API server
ENTRYPOINT ["/srv/entrypoint.sh"]

