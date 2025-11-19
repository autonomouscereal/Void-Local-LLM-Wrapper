#!/bin/bash
set -euo pipefail

# Create rvc_models directory if it doesn't exist
mkdir -p /srv/rvc_models

# Set up Titan model from bootstrap if it exists
TITAN_SOURCE="/opt/models/rvc_titan"
TITAN_TARGET="/srv/rvc_models/TITAN"

if [ -d "$TITAN_SOURCE" ] && [ -n "$(ls -A "$TITAN_SOURCE" 2>/dev/null)" ]; then
    echo "[rvc_python] Setting up Titan model from bootstrap..."
    mkdir -p "$TITAN_TARGET"
    
    # Check if Titan model files exist in source
    if [ -f "$TITAN_SOURCE/D-f048k-TITAN-Medium.pth" ] && [ -f "$TITAN_SOURCE/G-f048k-TITAN-Medium.pth" ]; then
        # rvc-python expects: /srv/rvc_models/<model_name>/<model_name>.pth
        # For Titan, we'll use the G (generator) checkpoint as the model file
        if [ ! -f "$TITAN_TARGET/TITAN.pth" ]; then
            # Try to copy first (works across volumes), fall back to symlink
            if cp "$TITAN_SOURCE/G-f048k-TITAN-Medium.pth" "$TITAN_TARGET/TITAN.pth" 2>/dev/null; then
                echo "[rvc_python] Titan model copied to $TITAN_TARGET/TITAN.pth"
            else
                # If copy fails (different volumes), create symlink
                ln -sf "$TITAN_SOURCE/G-f048k-TITAN-Medium.pth" "$TITAN_TARGET/TITAN.pth"
                echo "[rvc_python] Titan model symlinked to $TITAN_TARGET/TITAN.pth"
            fi
        else
            echo "[rvc_python] Titan model already exists at $TITAN_TARGET/TITAN.pth"
        fi
        
        # Copy/symlink D checkpoint if needed (some RVC setups use both)
        if [ ! -f "$TITAN_TARGET/D-f048k-TITAN-Medium.pth" ]; then
            if cp "$TITAN_SOURCE/D-f048k-TITAN-Medium.pth" "$TITAN_TARGET/" 2>/dev/null; then
                echo "[rvc_python] Titan D checkpoint copied"
            else
                ln -sf "$TITAN_SOURCE/D-f048k-TITAN-Medium.pth" "$TITAN_TARGET/"
            fi
        fi
    else
        echo "[rvc_python] Warning: Titan model files not found in $TITAN_SOURCE"
    fi
else
    echo "[rvc_python] Titan source directory not found or empty, skipping setup"
fi

# Start rvc-python API server
exec python3 -m rvc_python api -p 5050 -l -md /srv/rvc_models "$@"

