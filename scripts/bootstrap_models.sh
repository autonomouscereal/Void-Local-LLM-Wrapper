#!/bin/bash
# InsightFace model bootstrap (ArcFace)
INSIGHTFACE_DIR="./assets/insightface"

mkdir -p "${INSIGHTFACE_DIR}"

# Use a tiny Python script to trigger model download into INSIGHTFACE_DIR
python3 - << 'EOF'
import os
import sys

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("insightface not installed, skipping model download")
    sys.exit(0)

os.environ["INSIGHTFACE_HOME"] = os.path.abspath("assets/insightface")

# This will auto-download the 'buffalo_l' model into INSIGHTFACE_HOME
app = FaceAnalysis(name="buffalo_l", root=os.environ["INSIGHTFACE_HOME"])
app.prepare(ctx_id=0, det_size=(640, 640))
print(f"InsightFace models checked/downloaded to {os.environ['INSIGHTFACE_HOME']}")
EOF

