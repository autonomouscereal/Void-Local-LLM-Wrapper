#!/bin/bash
set -euo pipefail

# Duplicate all stdout/stderr to shared log volume while still streaming realtime.
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" || true
LOG_FILE="${LOG_FILE:-$LOG_DIR/rvc_trainer_entrypoint.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[rvc_trainer] entrypoint logging to $LOG_FILE"

# Set up Titan pretrain symlinks at runtime (files are available via volume mount)
TITAN_SOURCE="/opt/models/rvc_titan"
PRETRAINED_DIR="/srv/rvc_webui/pretrained"

if [ -d "$TITAN_SOURCE" ] && [ -n "$(ls -A "$TITAN_SOURCE" 2>/dev/null)" ]; then
    echo "[rvc_trainer] Setting up Titan pretrain symlinks..."
    mkdir -p "$PRETRAINED_DIR"
    
    # Check for files directly in rvc_titan (bootstrap downloads them there)
    G_FILE=""
    D_FILE=""
    
    # Try direct path first
    if [ -f "$TITAN_SOURCE/G-f048k-TITAN-Medium.pth" ] && [ -f "$TITAN_SOURCE/D-f048k-TITAN-Medium.pth" ]; then
        G_FILE="$TITAN_SOURCE/G-f048k-TITAN-Medium.pth"
        D_FILE="$TITAN_SOURCE/D-f048k-TITAN-Medium.pth"
    # Try nested path (if hf_hub_download preserved structure)
    elif [ -f "$TITAN_SOURCE/models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth" ] && \
         [ -f "$TITAN_SOURCE/models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth" ]; then
        G_FILE="$TITAN_SOURCE/models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth"
        D_FILE="$TITAN_SOURCE/models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth"
    fi
    
    if [ -n "$G_FILE" ] && [ -n "$D_FILE" ]; then
        ln -sf "$G_FILE" "$PRETRAINED_DIR/f0G48k.pth"
        ln -sf "$D_FILE" "$PRETRAINED_DIR/f0D48k.pth"
        echo "[rvc_trainer] Titan pretrain symlinks created"
    else
        echo "[rvc_trainer] Warning: Titan pretrain files not found in $TITAN_SOURCE"
    fi
else
    echo "[rvc_trainer] Titan source directory not found, skipping pretrain setup"
fi

# Start trainer API
exec python3 /srv/trainer_api.py "$@"

