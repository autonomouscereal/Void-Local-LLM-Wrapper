#!/usr/bin/env bash
set -euo pipefail
trap 'exit 0' TERM INT

# Host lib path mounted at /host/lib
HOST_ROOT=${HOST_ROOT:-/host}
LIBDIR=${TARGET_LIB_DIR:-$HOST_ROOT/usr/lib/x86_64-linux-gnu}
MARKER="$HOST_ROOT/var/run/nvidia-init.done"
mkdir -p "$LIBDIR" "$HOST_ROOT/var/run"
if [ -f "$MARKER" ]; then
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

# Create symlinks only for installed driver version
DRV_VER=$(grep -oE '[0-9]{3}\.[0-9]{2,3}\.[0-9]{2}' /proc/driver/nvidia/version 2>/dev/null | head -n1 || true)
# Create links for detected driver version if available
if [ -n "$DRV_VER" ]; then
  MAJ=$(echo "$DRV_VER" | cut -d. -f1)
  MIN=$(echo "$DRV_VER" | cut -d. -f2)
  PAT=$(echo "$DRV_VER" | cut -d. -f3)
  link_for() {
    local stem="$1"; shift
    local target="$2"
    ln -sf "$target" "$stem.$MAJ" 2>/dev/null || true
    ln -sf "$target" "$stem.$MAJ.$MIN" 2>/dev/null || true
    ln -sf "$target" "$stem.$MAJ.$MIN.$PAT" 2>/dev/null || true
  }
  link_for libEGL_nvidia.so "$EGL_TARGET"
  link_for libGLX_nvidia.so "$GLX_TARGET"
  link_for libGLESv2_nvidia.so "$GLES2_TARGET"
  link_for libGLESv1_CM_nvidia.so "$GLES1_TARGET"
fi

# Always provide 580.65.06 aliases (observed in the error)
for stem in libEGL_nvidia.so libGLX_nvidia.so libGLESv2_nvidia.so libGLESv1_CM_nvidia.so; do
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

# Done
exit 0

