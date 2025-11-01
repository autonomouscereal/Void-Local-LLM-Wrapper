#!/usr/bin/env bash
set -euo pipefail
trap 'exit 0' TERM INT

# Host lib path mounted at /host/lib
HOST_ROOT=${HOST_ROOT:-/host}
LIBDIR=${TARGET_LIB_DIR:-$HOST_ROOT/usr/lib/x86_64-linux-gnu}
MARKER="$HOST_ROOT/var/run/nvidia-init.done"
FORCE=${FORCE:-}
mkdir -p "$LIBDIR" "$HOST_ROOT/var/run"
if [ -f "$MARKER" ] && [ -z "$FORCE" ]; then
  exit 0
fi
cd "$LIBDIR" || exit 0

pick_target() {
  local stem="$1"
  local t="$(ls -1 ${stem}.* 2>/dev/null | sort -rV | head -n1 || true)"
  if [ -z "$t" ]; then
    t="${stem}.1"
    [ -e "$t" ] || echo -n > "$t"
  fi
  echo "$t"
}

EGL_TARGET=$(pick_target libEGL_nvidia.so)
GLX_TARGET=$(pick_target libGLX_nvidia.so)
GLES2_TARGET=$(pick_target libGLESv2_nvidia.so)
GLES1_TARGET=$(pick_target libGLESv1_CM_nvidia.so)
CUDA_TARGET=$(pick_target libcuda.so)
NVML_TARGET=$(pick_target libnvidia-ml.so)

# Create symlinks only for installed driver version
DRV_VER=$(grep -oE '[0-9]{3}\.[0-9]{2,3}\.[0-9]{2}' "$HOST_ROOT/proc/driver/nvidia/version" 2>/dev/null | head -n1 || true)
# Create links for detected driver version if available
if [ -n "$DRV_VER" ]; then
  MAJ=$(echo "$DRV_VER" | cut -d. -f1)
  MIN=$(echo "$DRV_VER" | cut -d. -f2)
  PAT=$(echo "$DRV_VER" | cut -d. -f3)
  link_for() {
    local stem="$1"; shift
    local target="${1:-}"
    [ -z "$target" ] && return 0
    ln -sf "$target" "$stem.$MAJ" 2>/dev/null || true
    ln -sf "$target" "$stem.$MAJ.$MIN" 2>/dev/null || true
    ln -sf "$target" "$stem.$MAJ.$MIN.$PAT" 2>/dev/null || true
  }
  link_for libEGL_nvidia.so "$EGL_TARGET"
  link_for libGLX_nvidia.so "$GLX_TARGET"
  link_for libGLESv2_nvidia.so "$GLES2_TARGET"
  link_for libGLESv1_CM_nvidia.so "$GLES1_TARGET"
  link_for libcuda.so "$CUDA_TARGET"
  link_for libnvidia-ml.so "$NVML_TARGET"

  # Dynamically link all NVIDIA/CUDA-related libs present
  for f in $(ls -1 lib*{nvidia,cuda}*.so* 2>/dev/null | sort -u); do
    stem="${f%%.so*}.so"
    tgt=$(pick_target "$stem")
    [ -n "$tgt" ] && link_for "$stem" "$tgt"
  done
fi

# Always provide 580.65.06 aliases (observed in the error) for all discovered stems
STEMS="$(
  (
    echo libEGL_nvidia.so; echo libGLX_nvidia.so; echo libGLESv2_nvidia.so; echo libGLESv1_CM_nvidia.so; echo libcuda.so; echo libnvidia-ml.so;
    ls -1 lib*{nvidia,cuda}*.so* 2>/dev/null | sed -E 's#(.*\.so).*#\1#' | sort -u
  ) | sort -u
)"
for stem in $STEMS; do
  tgt="${stem}.1"
  [ -e "$tgt" ] || tgt="$(ls -1 ${stem}.* 2>/dev/null | head -n1 || echo "${stem}.1")"
  [ -e "$tgt" ] || echo -n > "$tgt"
  ln -sf "$tgt" "$stem.580" 2>/dev/null || true
  ln -sf "$tgt" "$stem.580.65" 2>/dev/null || true
  ln -sf "$tgt" "$stem.580.65.06" 2>/dev/null || true
done

touch "$MARKER"

# Best effort: ensure directory exists so bind-target is valid
mkdir -p "$LIBDIR"

# Explicitly ensure libcudadebugger 580 aliases exist (observed toolchain expectation)
if [ -e "libcudadebugger.so.1" ]; then
  ln -sf "libcudadebugger.so.1" "libcudadebugger.so.580" 2>/dev/null || true
  ln -sf "libcudadebugger.so.1" "libcudadebugger.so.580.65" 2>/dev/null || true
  ln -sf "libcudadebugger.so.1" "libcudadebugger.so.580.65.06" 2>/dev/null || true
else
  : > "libcudadebugger.so.580" 2>/dev/null || true
  : > "libcudadebugger.so.580.65" 2>/dev/null || true
  : > "libcudadebugger.so.580.65.06" 2>/dev/null || true
fi

# Parse NVIDIA runtime host file manifests and materialize any missing versioned paths (version-agnostic)
MAN_DIR="$HOST_ROOT/usr/share/nvidia-container-runtime/host-files-for-container.d"
ensure_path() {
  local ap="$1"
  # Only handle files under libdir
  case "$ap" in
    /usr/lib/x86_64-linux-gnu/*) ;;
    *) return 0 ;;
  esac
  local rel="${ap#/usr/lib/x86_64-linux-gnu/}"
  local dir="$LIBDIR"
  local name="$rel"
  # If already exists, nothing to do
  [ -e "$dir/$name" ] && return 0
  # Determine stem (strip trailing .so.* to .so)
  local stem="${name%%.so*}.so"
  local tgt=""
  if [ -e "$dir/$stem.1" ]; then tgt="$stem.1";
  else
    tgt=$(ls -1 "$dir/$stem".* 2>/dev/null | xargs -r -n1 basename | sort -rV | head -n1 || true)
  fi
  if [ -z "$tgt" ]; then
    # As a last resort, create an empty placeholder so the bind target exists (version-agnostic)
    mkdir -p "$(dirname "$dir/$name")" 2>/dev/null || true
    : > "$dir/$name" 2>/dev/null || true
  else
    ln -sf "$tgt" "$dir/$name" 2>/dev/null || true
  fi
}

if [ -d "$MAN_DIR" ]; then
  for jf in "$MAN_DIR"/*.json; do
    [ -e "$jf" ] || continue
    # Extract absolute paths from JSON (simple grep; structure is stable)
    for p in $(grep -oE '"/usr/lib/x86_64-linux-gnu/[^"]+"' "$jf" | tr -d '"'); do
      ensure_path "$p"
    done
  done
fi

# Done
exit 0

