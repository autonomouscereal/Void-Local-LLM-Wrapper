#!/bin/bash
set -euo pipefail

# Duplicate all stdout/stderr to shared log volume while still streaming realtime.
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" || true
LOG_FILE="${LOG_FILE:-$LOG_DIR/rvc_python_entrypoint.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[rvc_python] entrypoint logging to $LOG_FILE"

# Create rvc_models directory if it doesn't exist
mkdir -p /srv/rvc_models

# Set up Titan model from bootstrap if it exists
TITAN_SOURCE="/opt/models/rvc_titan"
TITAN_TARGET="/srv/rvc_models/TITAN"

if [ -d "$TITAN_SOURCE" ] && [ -n "$(ls -A "$TITAN_SOURCE" 2>/dev/null)" ]; then
    echo "[rvc_python] Setting up Titan model from bootstrap..."
    mkdir -p "$TITAN_TARGET"

    # Locate Titan G/D checkpoints (handle both flat and nested layouts)
    G_FILE=""
    D_FILE=""

    # Direct layout (bootstrap may drop files here):
    if [ -f "$TITAN_SOURCE/G-f048k-TITAN-Medium.pth" ] && [ -f "$TITAN_SOURCE/D-f048k-TITAN-Medium.pth" ]; then
        G_FILE="$TITAN_SOURCE/G-f048k-TITAN-Medium.pth"
        D_FILE="$TITAN_SOURCE/D-f048k-TITAN-Medium.pth"
    # Nested layout (hf_hub_download preserving repo structure):
    elif [ -f "$TITAN_SOURCE/models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth" ] && \
         [ -f "$TITAN_SOURCE/models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth" ]; then
        G_FILE="$TITAN_SOURCE/models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth"
        D_FILE="$TITAN_SOURCE/models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth"
    fi

    if [ -n "$G_FILE" ] && [ -n "$D_FILE" ]; then
        # rvc-python expects: /srv/rvc_models/<model_name>/<model_name>.pth
        # For Titan, we'll use the G (generator) checkpoint as the model file
        if [ ! -f "$TITAN_TARGET/TITAN.pth" ]; then
            # Try to copy first (works across volumes), fall back to symlink
            if cp "$G_FILE" "$TITAN_TARGET/TITAN.pth" 2>/dev/null; then
                echo "[rvc_python] Titan model copied to $TITAN_TARGET/TITAN.pth"
            else
                ln -sf "$G_FILE" "$TITAN_TARGET/TITAN.pth"
                echo "[rvc_python] Titan model symlinked to $TITAN_TARGET/TITAN.pth"
            fi
        else
            echo "[rvc_python] Titan model already exists at $TITAN_TARGET/TITAN.pth"
        fi

        # Copy/symlink D checkpoint if needed (some RVC setups use both)
        if [ ! -f "$TITAN_TARGET/D-f048k-TITAN-Medium.pth" ]; then
            if cp "$D_FILE" "$TITAN_TARGET/" 2>/dev/null; then
                echo "[rvc_python] Titan D checkpoint copied"
            else
                ln -sf "$D_FILE" "$TITAN_TARGET/"
            fi
        fi
    else
        echo "[rvc_python] Warning: Titan model files not found in $TITAN_SOURCE (checked flat and nested layouts)"
    fi
else
    echo "[rvc_python] Titan source directory not found or empty, skipping setup"
fi

# Start rvc-python API server
exec python3 -m rvc_python api -p 5050 -l -md /srv/rvc_models "$@"

