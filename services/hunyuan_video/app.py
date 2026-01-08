from __future__ import annotations
import os
import sys
import time
import uuid
import logging
import argparse
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from void_json.json_parser import JSONParser

# 2026 Optimization Kernels
import sageattention
import angelslim

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

# Offload & Multi-GPU Constants
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

def get_exhaustive_args(task_mode="t2v", image_path=None) -> argparse.Namespace:
    """Returns a Namespace containing EVERY attribute required by Hunyuan 1.5."""
    args = argparse.Namespace()
    
    # --- Generation / Sampling Params ---
    args.prompt = ""
    args.video_length = FRAMES
    args.total_steps = STEPS
    args.infer_steps = STEPS
    args.fps = FPS
    args.num_videos = 1
    args.seed = 42
    args.neg_prompt = None
    args.cfg_scale = 1.0
    args.embedded_cfg_scale = 6.0
    args.denoise_type = "flow"
    
    # --- Attention & Performance Params ---
    args.use_sageattn = True
    args.sparse_attn = False
    args.sage_blocks_range = "0-53"
    args.no_cache_block_id = "54"
    args.enable_torch_compile = False
    
    # --- Caching System Params ---
    args.enable_cache = False
    args.cache_type = "standard"
    args.cache_device = "cuda"
    args.cache_start_step = 0
    args.cache_end_step = STEPS
    args.cache_step_interval = 1 
    
    # --- Precision, Quantization & Distillation ---
    args.dtype = "bf16"
    args.use_fp8_gemm = False
    args.quant_type = "none"
    args.cfg_distilled = True
    args.enable_step_distill = False
    
    # --- Path & Logic Params ---
    args.resolution = "720p"
    args.image_path = image_path
    args.sr = ENABLE_SR
    args.model_path = MODEL_PATH
    args.checkpoint_path = None
    args.lora_path = None
    args.save_path = "output.mp4"

    # --- Offloading & Distribution ---
    args.offloading = ENABLE_OFFLOADING
    args.group_offloading = ENABLE_GROUP_OFFLOADING
    args.overlap_group_offloading = ENABLE_OVERLAP_OFFLOADING
    args.sp = 1 
    
    return args

def _load_pipeline() -> Any:
    t0 = time.monotonic()
    
    # 1. Parallelism Guard
    WORLD_SIZE = torch.cuda.device_count()
    if not dist.is_initialized():
        initialize_parallel_state(sp=1)

    # 2. Get full args for initialization
    args = get_exhaustive_args(task_mode="t2v")
    infer_state = initialize_infer_state(args)
    
    # 3. Device placement logic
    device = torch.device('cpu') if args.offloading else torch.device('cuda')
    transformer_init_device = torch.device('cpu') if args.group_offloading else device

    # 4. Create Pipeline
    # Note: 1.5 create_pipeline internally handles loading 720p_t2v and 720p_i2v folders
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="720p_t2v",
        create_sr_pipeline=args.sr,
        transformer_dtype=torch.bfloat16,
        device=device,
        transformer_init_device=transformer_init_device,
        use_safetensors=True
    )

    # 5. Apply Optimizations
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )

    log.info(f"Hunyuan 1.5 fully loaded. SR={ENABLE_SR}. GPUs={WORLD_SIZE}. Time: {int(time.monotonic()-t0)}s")
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
    image_path = data.get("image_path") 
    
    if not prompt:
        return JSONResponse(status_code=422, content={"error": "Prompt required"})

    # Check for I2V vs T2V
    is_i2v = bool(image_path and os.path.exists(image_path))
    task_mode = "i2v" if is_i2v else "t2v"
    
    trace_id = data.get("trace_id", uuid.uuid4().hex)
    out_dir = os.path.join(UPLOAD_DIR, "artifacts", "video", trace_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.mp4")

    t0 = time.monotonic()
    with _PIPE_LOCK:
        try:
            # We must pass the exhaustive args for every call to satisfy internal lookups
            # especially for the tile-based attention buffers in 1.5
            current_args = get_exhaustive_args(task_mode=task_mode, image_path=image_path if is_i2v else None)
            
            PIPE(
                prompt=prompt.strip(),
                image_path=current_args.image_path,
                height=HEIGHT,
                width=WIDTH,
                video_length=FRAMES,
                infer_steps=STEPS,
                fps=FPS,
                save_path=out_path,
                enable_sr=ENABLE_SR,
                sr_max_batch_size=1,
                use_sage_attn=True
            )
        except Exception as e:
            log.error(f"Trace: {trace_id} | Task: {task_mode} | Error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "ok": True,
        "trace_id": trace_id,
        "task": task_mode,
        "output_url": f"{PUBLIC_BASE_URL}/uploads/artifacts/video/{trace_id}/output.mp4",
        "resolution": "1920x1080" if ENABLE_SR else "1280x720",
        "dur_ms": int((time.monotonic() - t0) * 1000)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=PORT)
