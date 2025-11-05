#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${FILM2_MODELS:-/opt/models}"
HF_HOME="${HF_HOME:-$MODELS_DIR/.hf}"
export HF_HOME TRANSFORMERS_CACHE="$HF_HOME" TORCH_HOME="$MODELS_DIR/.torch"

mkdir -p "$MODELS_DIR" "$HF_HOME" "$TORCH_HOME" "$MODELS_DIR/manifests"

log() { echo "[bootstrap] $*"; }

need_cli=0
command -v huggingface-cli >/dev/null || need_cli=1
if [ "$need_cli" -eq 1 ]; then
  pip install -U "huggingface_hub[cli]" git-lfs >/dev/null 2>&1 || true
fi

hf_snap() { # $1=repo  $2=subdir
  local repo="$1"; local sub="$2"; local tgt="$MODELS_DIR/$sub"
  if [ -d "$tgt" ]; then log "exists $tgt"; return 0; fi
  log "download $repo -> $tgt"
  huggingface-cli download "$repo" --local-dir "$tgt" --resume-download --exclude "*.md" || true
}

# Core video gen/edit
hf_snap "tencent/HunyuanVideo"                          "hunyuan"
hf_snap "OpenMotionLab/Open-Sora"                        "opensora"
hf_snap "NUS-Tim/LTX-Video"                              "ltx_video"
hf_snap "stabilityai/stable-video-diffusion-img2vid"     "svd"

# Controls/locks
hf_snap "InstantX/InstantID"                             "instantid"
hf_snap "h94/IP-Adapter"                                 "ip_adapter"
hf_snap "lllyasviel/ControlNet"                          "controlnet"
hf_snap "facebook/sam2-hiera-large"                      "sam"
hf_snap "tryon/tryondiffusion"                           "tryon"

# Temporal/geometry
hf_snap "princeton-vl/RAFT"                              "raft"
hf_snap "LiheYoung/Depth-Anything-V2-base"               "depth_anything"
hf_snap "CMU/openpose"                                   "openpose"

# Relight/restore/SR/interp
hf_snap "lllyasviel/IC-Light"                            "ic_light"
hf_snap "ckpt/BasicVSR-PP"                               "basicvsrpp" || true
hf_snap "ckpt/EDVR"                                      "edvr" || true
hf_snap "TencentARC/GFPGAN"                              "gfpgan"
hf_snap "sczhou/CodeFormer"                              "codeformer"
hf_snap "megvii-research/ECCV2022-RIFE"                  "rife"

# Caption/score
hf_snap "laion/CLIP-ViT-L-14-336"                        "clip"
hf_snap "laion/clap-htsat-unfused"                       "clap"
hf_snap "Salesforce/blip2-flan-t5-xl"                    "blip2"
hf_snap "openai/whisper-large-v3"                        "whisper"

# LUTs
mkdir -p "$MODELS_DIR/luts"

cat > "$MODELS_DIR/manifests/model_manifest.json" <<'JSON'
{
  "hunyuan":"snapshot",
  "opensora":"snapshot",
  "ltx_video":"snapshot",
  "svd":"snapshot",
  "instantid":"snapshot",
  "ip_adapter":"snapshot",
  "controlnet":"snapshot",
  "sam":"snapshot",
  "tryon":"snapshot",
  "raft":"snapshot",
  "depth_anything":"snapshot",
  "openpose":"snapshot",
  "ic_light":"snapshot",
  "basicvsrpp":"snapshot",
  "edvr":"snapshot",
  "gfpgan":"snapshot",
  "codeformer":"snapshot",
  "rife":"snapshot",
  "clip":"snapshot",
  "clap":"snapshot",
  "blip2":"snapshot",
  "whisper":"snapshot",
  "luts":"v1"
}
JSON

log "✅ Models bootstrapped into $MODELS_DIR"

