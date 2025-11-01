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

# Generate version-agnostic symlinks for common patterns
make_links() {
  local stem="$1"; shift
  for major in $(seq 450 650); do
    ln -sf "$stem.1" "$stem.$major" 2>/dev/null || true
    for minor in $(seq 0 200); do
      # create stem.major.minor
      ln -sf "$stem.1" "$stem.$major.$minor" 2>/dev/null || true
      for patch in $(seq 0 200); do
        # zero-pad patch to 2 digits and also create non-padded variant
        zp=$(printf "%02d" "$patch")
        ln -sf "$stem.1" "$stem.$major.$minor.$zp" 2>/dev/null || true
        ln -sf "$stem.1" "$stem.$major.$minor.$patch" 2>/dev/null || true
      done
    done
  done
}

make_links libEGL_nvidia.so
make_links libGLX_nvidia.so
make_links libGLESv2_nvidia.so
make_links libGLESv1_CM_nvidia.so

# Best effort: ensure directory exists so bind-target is valid
mkdir -p "$LIBDIR"

# Done
exit 0

