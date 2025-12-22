# bootstrap models script

#!/usr/bin/env bash
set -euo pipefail

export FILM2_MODELS=${FILM2_MODELS:-/opt/models}
export HF_HOME=${HF_HOME:-$FILM2_MODELS/.hf}

# Ensure apt/depconf run in fully non-interactive mode inside the container.
# This avoids noisy Term::ReadLine/TTY warnings while still allowing installs.
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends git git-lfs ca-certificates python3 python3-pip
git lfs install || true
python3 -m pip install -U pip

python3 /bootstrap/bootstrap_models.py

