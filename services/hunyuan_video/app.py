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

# Force enable for maximum VRAM savings and multi-GPU stability
ENABLE_GROUP_OFFLOADING = os.getenv("HYVIDEO_GROUP_OFFLOADING", "1").strip().lower() in ("1", "true", "yes", "on")

# Additionally, consider enabling 'overlap_group_offloading' for a 2026 performance boost
ENABLE_OVERLAP_OFFLOADING = True # Speeds up inference by overlapping data transfer with computation


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

import argparse

def _load_pipeline() -> Any:
    t0 = time.monotonic()
    
    # Create the mock args object with ALL required 1.5 attributes
    args = argparse.Namespace()
    
    # 1. Essential Performance/Attention Flags
    args.use_sageattn = True
    args.sparse_attn = False
    args.sage_blocks_range = "0-53"  # The missing attribute causing your crash
    args.no_cache_block_id = None    # Required for the internal caching logic
    args.resolution = "720p"         # Used for transformer versioning
    
    # 2. Precision and Offloading Flags
    args.dtype = "bf16"
    args.offloading = ENABLE_OFFLOADING
    args.group_offloading = ENABLE_GROUP_OFFLOADING
    args.overlap_group_offloading = ENABLE_OVERLAP_OFFLOADING
    
    # 3. Model Logic Flags
    args.cfg_distilled = True        # Default for 1.5
    args.enable_step_distill = False
    args.image_path = None           # Default to T2V mode
    args.sr = ENABLE_SR              # Super-resolution flag
    args.model_path = MODEL_PATH

    # Initialize the state (This returns the object needed for apply_infer_optimization)
    # The code you provided calls: infer_state = initialize_infer_state(args)
    infer_state = initialize_infer_state(args)
    
    # Determine devices as per generate.py logic
    # If offloading is on, they init on CPU to save VRAM
    device = torch.device('cpu') if args.offloading else torch.device('cuda')
    transformer_init_device = torch.device('cpu') if args.group_offloading else device

    # Create the pipeline using the 1.5 Factory
    # Note: 'create_sr_pipeline' is the arg name in the code you provided
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="720p_t2v",
        create_sr_pipeline=args.sr,
        transformer_dtype=torch.bfloat16,
        device=device,
        transformer_init_device=transformer_init_device,
    )

    # Apply the optimizations using the infer_state we just created
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )

    log.info("Hunyuan 1.5 Pipeline & SR optimized and loaded. dur_ms=%d", 
             int((time.monotonic() - t0) * 1000))
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
