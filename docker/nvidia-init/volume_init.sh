#!/usr/bin/env bash
set -euo pipefail

LIBVOL=${LIBVOL:-/libdir}
MAN_DIR=/host-man
DRV_VER=$(grep -oE '[0-9]{3}\.[0-9]{2,3}\.[0-9]{2}' /host-proc/driver/nvidia/version 2>/dev/null | head -n1 || true)
MAJ=""; MIN=""; PAT=""
if [ -n "$DRV_VER" ]; then
  MAJ=$(echo "$DRV_VER" | cut -d. -f1)
  MIN=$(echo "$DRV_VER" | cut -d. -f2)
  PAT=$(echo "$DRV_VER" | cut -d. -f3)
fi

mkdir -p "$LIBVOL"

ensure_file() {
  local p="$1"
  [ -e "$p" ] || { mkdir -p "$(dirname "$p")"; : > "$p"; }
}

# 1) Materialize targets from NVIDIA runtime manifests
if [ -d "$MAN_DIR" ]; then
  for jf in "$MAN_DIR"/*.json; do
    [ -e "$jf" ] || continue
    grep -oE '"/usr/lib/x86_64-linux-gnu/[^"]+"' "$jf" | tr -d '"' | while read -r ap; do
      rel="${ap#/usr/lib/x86_64-linux-gnu/}"
      ensure_file "$LIBVOL/$rel"
    done
  done
fi

# 2) Create driver-version variants for common stems (so mount targets always exist)
stems=(libcudadebugger.so libcuda.so libnvidia-ml.so libEGL_nvidia.so libGLX_nvidia.so libGLESv2_nvidia.so libGLESv1_CM_nvidia.so)
for s in "${stems[@]}"; do
  [ -n "$MAJ" ] && ensure_file "$LIBVOL/$s.$MAJ"
  [ -n "$MIN" ] && ensure_file "$LIBVOL/$s.$MAJ.$MIN"
  [ -n "$PAT" ] && ensure_file "$LIBVOL/$s.$MAJ.$MIN.$PAT"
done

# 3) Known 580.* aliases some toolchains expect
for s in "${stems[@]}"; do
  ensure_file "$LIBVOL/$s.580"
  ensure_file "$LIBVOL/$s.580.65"
  ensure_file "$LIBVOL/$s.580.65.06"
done

exit 0

