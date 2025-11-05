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

# Ensure huggingface_hub is importable for Python snapshot_download
python3 - <<'PY' >/dev/null 2>&1 || (
  pip install -U huggingface_hub >/dev/null 2>&1 || true
)
import sys
try:
    import huggingface_hub  # noqa: F401
except Exception:
    sys.exit(1)
PY

py_snapshot() { # $1=repo  $2=local_dir
  python3 - "$1" "$2" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"[dl-ok] {repo_id} -> {local_dir}")
PY
}

git_clone() { # $1=url $2=dir
  local url="$1"; local dir="$2"
  if [ -d "$dir/.git" ] || [ -d "$dir" ]; then log "exists $dir"; return 0; fi
  git clone --depth=1 "$url" "$dir" >/dev/null 2>&1 || return 1
  log "[git-ok] $url -> $dir"
}

hf_snap() { # $1=repo  $2=subdir
  local repo="$1"; local sub="$2"; local tgt="$MODELS_DIR/$sub"
  if [ -d "$tgt" ]; then log "exists $tgt"; return 0; fi
  log "download $repo -> $tgt"
  if ! py_snapshot "$repo" "$tgt"; then
    log "FAILED to download $repo (see above)."
    echo "$repo" >> "$MODELS_DIR/manifests/failed.txt"
    return 0
  fi
}

# Core video gen/edit
hf_snap "tencent/HunyuanVideo"                          "hunyuan"
# Prefer GitHub clone for Open-Sora code
git_clone "https://github.com/hpcaitech/Open-Sora.git"   "$MODELS_DIR/opensora" || echo "hpcaitech/Open-Sora" >> "$MODELS_DIR/manifests/failed.txt"
hf_snap "NUS-Tim/LTX-Video"                              "ltx_video"
# Optional: Stable Video Diffusion (disabled when SKIP_SVD=1)
if [ "${SKIP_SVD:-0}" != "1" ]; then
  hf_snap "stabilityai/stable-video-diffusion-img2vid"     "svd"
fi

# Controls/locks
hf_snap "InstantX/InstantID"                             "instantid"
hf_snap "h94/IP-Adapter"                                 "ip_adapter"
hf_snap "lllyasviel/ControlNet"                          "controlnet"
hf_snap "facebook/sam2-hiera-large"                      "sam"
hf_snap "tryon/tryondiffusion"                           "tryon"

# Temporal/geometry
# Prefer GitHub clone for RAFT code
git_clone "https://github.com/princeton-vl/RAFT.git"     "$MODELS_DIR/raft" || echo "princeton-vl/RAFT" >> "$MODELS_DIR/manifests/failed.txt"
hf_snap "LiheYoung/Depth-Anything-V2-base"               "depth_anything"
# Prefer GitHub clone for OpenPose code
git_clone "https://github.com/CMU-Perceptual-Computing-Lab/openpose.git" "$MODELS_DIR/openpose" || echo "CMU/openpose" >> "$MODELS_DIR/manifests/failed.txt"

# Relight/restore/SR/interp
hf_snap "lllyasviel/IC-Light"                            "ic_light"
# Prefer GitHub clones for SR/restore/interp libs
git_clone "https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git" "$MODELS_DIR/basicvsrpp" || echo "ckkelvinchan/BasicVSR_PlusPlus" >> "$MODELS_DIR/manifests/failed.txt"
git_clone "https://github.com/xinntao/EDVR.git"                   "$MODELS_DIR/edvr"       || echo "xinntao/EDVR" >> "$MODELS_DIR/manifests/failed.txt"
git_clone "https://github.com/TencentARC/GFPGAN.git"              "$MODELS_DIR/gfpgan"     || echo "TencentARC/GFPGAN" >> "$MODELS_DIR/manifests/failed.txt"
git_clone "https://github.com/sczhou/CodeFormer.git"              "$MODELS_DIR/codeformer"  || echo "sczhou/CodeFormer" >> "$MODELS_DIR/manifests/failed.txt"
git_clone "https://github.com/MegEngine/ECCV2022-RIFE.git"        "$MODELS_DIR/rife"        || echo "MegEngine/ECCV2022-RIFE" >> "$MODELS_DIR/manifests/failed.txt"

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

if [ -f "$MODELS_DIR/manifests/failed.txt" ]; then
  log "FAILED downloads:"
  cat "$MODELS_DIR/manifests/failed.txt" | sed 's/^/[bootstrap]   - /'
fi

log "✅ Models bootstrapped into $MODELS_DIR"

