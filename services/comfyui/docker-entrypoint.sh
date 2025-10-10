#!/bin/sh
set -e

# Ensure AnimateDiff motion models exist (mounted via shared volume or fetch fallback)
MODELS_DIR="/comfyui/models/animatediff_models"
ALT_DIR="/comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved/models"
mkdir -p "$MODELS_DIR" "$ALT_DIR"

# Mirror models between the two known scan paths if present in either
for fn in mm_sd_v15_v2.ckpt mm_sdxl_v10.ckpt; do
  if [ -f "$MODELS_DIR/$fn" ] && [ ! -f "$ALT_DIR/$fn" ]; then
    cp -f "$MODELS_DIR/$fn" "$ALT_DIR/$fn" || true
  fi
  if [ -f "$ALT_DIR/$fn" ] && [ ! -f "$MODELS_DIR/$fn" ]; then
    cp -f "$ALT_DIR/$fn" "$MODELS_DIR/$fn" || true
  fi
done

exec python main.py --listen 0.0.0.0 --port 8188 --cpu

