FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone RVC WebUI repo at build time (server has internet)
RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git rvc_webui

WORKDIR /srv/rvc_webui

# Optional: pin to a specific commit for reproducibility
# RUN git checkout <COMMIT_HASH>

# Install WebUI requirements
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir "speechbrain==0.5.16"

# Wire Titan pretrain from host-mounted /opt/models/rvc_titan into WebUI's expected pretrained paths.
# Bootstrap downloads directly to /opt/models/rvc_titan/ with files:
#   /opt/models/rvc_titan/G-f048k-TITAN-Medium.pth
#   /opt/models/rvc_titan/D-f048k-TITAN-Medium.pth
# Note: This runs at build time, but the actual files will be available at runtime via volume mount
RUN mkdir -p ./pretrained && \
    echo "Titan pretrain will be symlinked at runtime from /opt/models/rvc_titan"

# Back to /srv for our trainer API
WORKDIR /srv

# Copy your trainer API implementation from your codebase
COPY trainer_api.py /srv/trainer_api.py

# Copy entrypoint script
COPY entrypoint.sh /srv/entrypoint.sh
RUN chmod +x /srv/entrypoint.sh

# Shared dirs: models + training data + logs
RUN mkdir -p /srv/rvc_models /srv/data /srv/logs

EXPOSE 7070

# Use entrypoint to set up Titan pretrain symlinks, then start API
ENTRYPOINT ["/srv/entrypoint.sh"]

