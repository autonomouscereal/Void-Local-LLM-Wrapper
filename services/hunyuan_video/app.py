from __future__ import annotations

import os
import sys
import time
import uuid
import logging
from typing import Any, Dict, Optional

# 2026 Optimization Kernels
import sageattention
import angelslim
import sgl_kernel
import flex_block_attn

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from void_json.json_parser import JSONParser

# ---------------------------
# Config & Environment
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("hunyuan_1.5_service")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
MODELS_DIR = os.getenv("FILM2_MODELS", "/opt/models").strip()
MODEL_PATH = os.getenv("HYVIDEO_MODEL_PATH", os.path.join(MODELS_DIR, "hunyuan")).strip()
CODE_PATH = os.getenv("HYVIDEO_CODE_PATH", "/opt/hunyuan15_src").strip()
PORT = int(os.getenv("HYVIDEO_PORT", "8094"))

# Defaults for 720p base generation
WIDTH = 1280
HEIGHT = 720
FRAMES = 129
STEPS = 50
FPS = 24

# Offload & Multi-GPU Logic
# Set to '1' to enable multi-gpu distribution + cpu fallback
ENABLE_OFFLOADING = os.getenv("HYVIDEO_OFFLOADING", "1") == "1"
ENABLE_SR = os.getenv("HYVIDEO_SR_ENABLE", "1") == "1"

# Optimize CUDA memory for 2026
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

if CODE_PATH and os.path.isdir(CODE_PATH) and CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

try:
    # 1.5 Specific Pipeline Imports
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
    from hyvideo.commons.infer_state import initialize_infer_state
except Exception as e:
    log.error("Failed to import Hunyuan 1.5 core: %s", e)
    raise

APP = FastAPI(title="HunyuanVideo-1.5 Service")
PIPE: Optional[Any] = None
_PIPE_LOCK = torch.multiprocessing.Lock()

def _ensure_dirs():
    os.makedirs(os.path.join(UPLOAD_DIR, "artifacts", "video"), exist_ok=True)

def _load_pipeline():
    t0 = time.monotonic()
    initialize_infer_state()
    
    # 2026 Multi-GPU Implementation: 
    # Use device_map="auto" to balance across all visible GPUs.
    # enable_offloading=True triggers the CPU fallback for text encoders and VAE.
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=MODEL_PATH,
        transformer_version="720p_t2v",
        device_map="auto",             # Enables Multi-GPU distribution
        enable_offloading=ENABLE_OFFLOADING, 
        enable_group_offloading=True,
        transformer_dtype=torch.bfloat16,
        # 1.5 uses a dedicated SR (Super Resolution) path
        use_sr_module=ENABLE_SR 
    )

    log.info(f"Pipeline Loaded: Multi-GPU Auto-Distribute Enabled. Time: {int(time.monotonic()-t0)}s")
    return pipe

@APP.on_event("startup")
def _startup():
    global PIPE
    _ensure_dirs()
    PIPE = _load_pipeline()

@APP.post("/v1/video/generate")
async def generate(req: Request):
    body = (await req.body()).decode("utf-8")
    data = JSONParser().parse(body, {})
    prompt = data.get("prompt")
    
    if not prompt:
        return JSONResponse(status_code=422, content={"error": "Prompt required"})

    trace_id = data.get("trace_id", uuid.uuid4().hex)
    out_dir = os.path.join(UPLOAD_DIR, "artifacts", "video", trace_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.mp4")

    t0 = time.monotonic()
    with _PIPE_LOCK:
        try:
            # 1.5 Pipeline handles 720p -> 1080p internally if enable_sr=True
            # It triggers the specialized Video Super-Resolution network
            PIPE(
                prompt=prompt.strip(),
                height=HEIGHT,
                width=WIDTH,
                video_length=FRAMES,
                infer_steps=STEPS,
                fps=FPS,
                save_path=out_path,
                enable_sr=ENABLE_SR,        # Triggers auto-upscale to 1080p
                sr_max_batch_size=1,       # Lower if you OOM during upscale
                use_sage_attn=True         # 2026 performance flag
            )
        except Exception as e:
            log.error(f"Trace: {trace_id} | Error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "ok": True,
        "trace_id": trace_id,
        "output_url": f"{PUBLIC_BASE_URL}/uploads/artifacts/video/{trace_id}/output.mp4",
        "resolution": "1920x1080" if ENABLE_SR else "1280x720",
        "dur_ms": int((time.monotonic() - t0) * 1000)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=PORT)
