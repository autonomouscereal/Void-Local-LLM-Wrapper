#!/usr/bin/env bash
set -euo pipefail
trap 'exit 0' TERM INT

# When run inside the init container, we operate on the host via chroot /host.
HOST_ROOT="${HOST_ROOT:-/host}"

# 1) Regenerate CDI with the actual host libraries and configure NVIDIA runtime for Docker
if chroot "$HOST_ROOT" /usr/bin/which nvidia-ctk >/dev/null 2>&1; then
  chroot "$HOST_ROOT" /usr/bin/nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml || true
  chroot "$HOST_ROOT" /usr/bin/nvidia-ctk runtime configure --runtime=docker || true
fi

# 2) Do NOT create or modify any files under /usr/lib/x86_64-linux-gnu on the host.
#    We rely solely on CDI/runtime to mount the correct libs.

# 3) Terse summary for logs
DRV_VER="$(chroot "$HOST_ROOT" /usr/bin/bash -lc 'nvidia-smi --query-gpu=driver_version --format=csv,noheader' 2>/dev/null || echo n/a)"
echo "nvidia-init-safe: driver=${DRV_VER}"
echo "nvidia-init-safe: cdi=/etc/cdi/nvidia.yaml $(chroot "$HOST_ROOT" /usr/bin/test -s /etc/cdi/nvidia.yaml && echo present || echo missing)"

exit 0


