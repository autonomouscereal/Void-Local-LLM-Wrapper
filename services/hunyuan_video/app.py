from __future__ import annotations
import os
import sys
import time
import uuid
import logging
import argparse
from typing import Any, Dict, Optional

# 2026 Optimization Kernels
import sageattention
import angelslim
import torch
import torch.distributed as dist
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

# Offload & Multi-GPU Logic
ENABLE_OFFLOADING = os.getenv("HYVIDEO_OFFLOADING", "1") == "1"
ENABLE_GROUP_OFFLOADING = os.getenv("HYVIDEO_GROUP_OFFLOADING", "1") == "1"
ENABLE_OVERLAP_OFFLOADING = True 
ENABLE_SR = os.getenv("HYVIDEO_SR_ENABLE", "1") == "1"

# Defaults
WIDTH, HEIGHT, FRAMES, STEPS, FPS = 1280, 720, 129, 50, 24

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

if CODE_PATH and os.path.isdir(CODE_PATH) and CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

try:
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.commons.parallel_states import initialize_parallel_state
except Exception as e:
    log.error("Failed to import Hunyuan 1.5 core: %s", e)
    raise

APP = FastAPI(title="HunyuanVideo-1.5 Service")
PIPE: Optional[Any] = None
_PIPE_LOCK = torch.multiprocessing.Lock()

def _ensure_dirs():
    os.makedirs(os.path.join(UPLOAD_DIR, "artifacts", "video"), exist_ok=True)

def _load_pipeline() -> Any:
    t0 = time.monotonic()
    
    # 1. Create the mock args object with ALL attributes required by 1.5
    args = argparse.Namespace()
    
    # --- Required by initialize_infer_state ---
    args.enable_cache = False
    args.cache_type = "standard"        # FIX: The attribute missing in your error
    args.cache_device = "cuda"          # Required for cache config
    args.enable_torch_compile = False
    args.use_fp8_gemm = False
    args.quant_type = "none"
    args.enable_step_distill = False
    
    # --- Attention & Caching Logic ---
    args.use_sageattn = True
    args.sparse_attn = False
    args.sage_blocks_range = "0-53"     # Full range for 8.3B model
    args.no_cache_block_id = "54"       # Valid index string to avoid parser crash
    
    # --- Path & Model Logic ---
    args.resolution = "720p"
    args.dtype = "bf16"
    args.cfg_distilled = True
    args.image_path = None
    args.sr = ENABLE_SR
    args.model_path = MODEL_PATH
    args.checkpoint_path = None
    args.lora_path = None

    # --- Offloading Configuration ---
    args.offloading = ENABLE_OFFLOADING
    args.group_offloading = ENABLE_GROUP_OFFLOADING
    args.overlap_group_offloading = ENABLE_OVERLAP_OFFLOADING
    
    # 2. Initialize Parallel State for Multi-GPU
    WORLD_SIZE = torch.cuda.device_count()
    if not dist.is_initialized():
        # sp=1 is standard for single-process inference distribution
        initialize_parallel_state(sp=1)

    # 3. Initialize Infer State (Hunyuan 1.5 Core)
    infer_state = initialize_infer_state(args)
    
    # 4. Device Placement (Logic from 1.5 generate.py)
    device = torch.device('cpu') if args.offloading else torch.device('cuda')
    transformer_init_device = torch.device('cpu') if args.group_offloading else device

    # 5. Create Pipeline via 1.5 Factory
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="720p_t2v",
        create_sr_pipeline=args.sr,
        transformer_dtype=torch.bfloat16,
        device=device,
        transformer_init_device=transformer_init_device,
    )

    # 6. Apply Multi-GPU & CPU Offload optimizations
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )

    log.info(f"Hunyuan 1.5 service optimized for 2026. Loaded on {WORLD_SIZE} GPUs. Time: {int(time.monotonic()-t0)}s")
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
            # HunyuanVideo 1.5 call
            PIPE(
                prompt=prompt.strip(),
                height=HEIGHT,
                width=WIDTH,
                video_length=FRAMES,
                infer_steps=STEPS,
                fps=FPS,
                save_path=out_path,
                enable_sr=ENABLE_SR,        # 1080p Upscale
                sr_max_batch_size=1,        # Prevents OOM during upscale
                use_sage_attn=True
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
