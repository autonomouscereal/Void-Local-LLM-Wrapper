#!/usr/bin/env bash
set -euo pipefail

# Ensure upscaler model is present; optionally download from provided URL
UPSCALE_DIR="/comfyui/models/upscale"
MODEL_NAME="${COMFY_UPSCALE_MODEL:-4x-UltraSharp.pth}"
MODEL_URL="${COMFY_UPSCALE_MODEL_URL:-}"
MODEL_SHA256="${COMFY_UPSCALE_MODEL_SHA256:-}"

mkdir -p "$UPSCALE_DIR"
TARGET="$UPSCALE_DIR/$MODEL_NAME"

if [ -n "$MODEL_URL" ] && [ ! -s "$TARGET" ]; then
  echo "[comfyui] downloading upscaler model: $MODEL_NAME"
  curl -fL "$MODEL_URL" -o "$TARGET"
  if [ -n "$MODEL_SHA256" ]; then
    echo "$MODEL_SHA256  $TARGET" | sha256sum -c -
  fi
fi

exec python -m ComfyUI.main --listen --port 8188

