#!/usr/bin/env bash
set -euo pipefail

# Host lib path mounted at /host/lib
LIBDIR=${TARGET_LIB_DIR:-/host/lib}
cd "$LIBDIR" || exit 0

# If the base lib doesn't exist, nothing to do
BASE_EGL="$(ls -1 libEGL_nvidia.so.* 2>/dev/null | head -n1 || true)"
[ -z "$BASE_EGL" ] && BASE_EGL="libEGL_nvidia.so.1"
if [ ! -e "$BASE_EGL" ]; then
  # create a tiny placeholder that will be overridden by NVIDIA hook
  echo -n > "$BASE_EGL"
fi

# Create symlinks only for installed driver version
DRV_VER=$(grep -oE '[0-9]{3}\.[0-9]{2,3}\.[0-9]{2}' /proc/driver/nvidia/version 2>/dev/null | head -n1 || true)
if [ -n "$DRV_VER" ]; then
  MAJ=$(echo "$DRV_VER" | cut -d. -f1)
  MIN=$(echo "$DRV_VER" | cut -d. -f2)
  PAT=$(echo "$DRV_VER" | cut -d. -f3)
  link_for() {
    local stem="$1"; shift
    local target=""
    if   [ -e "$stem.1" ]; then target="$stem.1";
    elif [ -e "$stem.0" ]; then target="$stem.0";
    else return 0; fi
    ln -sf "$target" "$stem.$MAJ" 2>/dev/null || true
    ln -sf "$target" "$stem.$MAJ.$MIN" 2>/dev/null || true
    ln -sf "$target" "$stem.$MAJ.$MIN.$PAT" 2>/dev/null || true
  }
  link_for libEGL_nvidia.so
  link_for libGLX_nvidia.so
  link_for libGLESv2_nvidia.so
  link_for libGLESv1_CM_nvidia.so
fi

# Best effort: ensure directory exists so bind-target is valid
mkdir -p "$LIBDIR"

# Done
exit 0

