from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import argparse
import shutil
from typing import Any, Optional, Dict

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from void_json.json_parser import JSONParser

# Optional perf libs (do not hard-fail if missing)
try:
    import sageattention  # noqa: F401
except Exception:
    sageattention = None

try:
    import angelslim  # noqa: F401
except Exception:
    angelslim = None

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("hunyuan_1.5_service")

# ---------------------------
# Config & Environment
# ---------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
MODELS_DIR = os.getenv("FILM2_MODELS", "/opt/models").strip()
MODEL_PATH = os.getenv("HYVIDEO_MODEL_PATH", os.path.join(MODELS_DIR, "hunyuan")).strip()
CODE_PATH = os.getenv("HYVIDEO_CODE_PATH", "/opt/hunyuan15_src").strip()
PORT = int(os.getenv("HYVIDEO_PORT", "8094"))

ENABLE_OFFLOADING = os.getenv("HYVIDEO_OFFLOADING", "1") == "1"
ENABLE_GROUP_OFFLOADING = os.getenv("HYVIDEO_GROUP_OFFLOADING", "1") == "1"
ENABLE_OVERLAP_OFFLOADING = os.getenv("HYVIDEO_OVERLAP_GROUP_OFFLOADING", "1") == "1"
ENABLE_SR = os.getenv("HYVIDEO_SR_ENABLE", "1") == "1"

# Defaults
WIDTH, HEIGHT, FRAMES, STEPS, FPS = 1280, 720, 129, 50, 24

# Recommended allocator setting for fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# ---------------------------
# Torch load compatibility shim (PyTorch 2.6+ weights_only default)
# and safetensors fallback for "torch.load" calls inside Tencent repo.
# ---------------------------
try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None

_torch_load_orig = torch.load

def _torch_load_compat(f, *args, **kwargs):
    path = f if isinstance(f, str) else getattr(f, "name", None)
    if isinstance(path, str) and os.path.isfile(path):
        # If extension is safetensors, load via safetensors
        if path.endswith(".safetensors") and safetensors_load_file is not None:
            device = kwargs.get("map_location", "cpu")
            return safetensors_load_file(path, device=device)

        # If file is named .pt/.bin but is actually safetensors, try safetensors first
        if safetensors_load_file is not None:
            try:
                with open(path, "rb") as fh:
                    head = fh.read(64)
                # heuristic: safetensors header often contains b"safetensors" early
                if b"safetensors" in head.lower():
                    device = kwargs.get("map_location", "cpu")
                    return safetensors_load_file(path, device=device)
            except Exception:
                pass

    # Force old behavior for torch checkpoints
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _torch_load_orig(f, *args, **kwargs)

torch.load = _torch_load_compat

# ---------------------------
# Import Tencent repo code
# ---------------------------
if CODE_PATH and os.path.isdir(CODE_PATH) and CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

try:
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
    from hyvideo.commons.infer_state import initialize_infer_state
except Exception as e:
    log.error("Failed to import Hunyuan 1.5 core from CODE_PATH=%s: %s", CODE_PATH, e)
    raise

APP = FastAPI(title="HunyuanVideo-1.5 Service")
PIPE: Optional[Any] = None
_PIPE_LOCK = torch.multiprocessing.Lock()


def _ensure_dirs():
    os.makedirs(os.path.join(UPLOAD_DIR, "artifacts", "video"), exist_ok=True)


def get_exhaustive_args(task_mode: str = "t2v", image_path: Optional[str] = None) -> argparse.Namespace:
    """Returns a Namespace containing every attribute the Tencent repo expects in infer_state."""
    args = argparse.Namespace()

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

    args.use_sageattn = True
    args.sparse_attn = False
    args.sage_blocks_range = "0-53"
    args.no_cache_block_id = "54"
    args.enable_torch_compile = False

    args.enable_cache = False
    args.cache_type = "standard"
    args.cache_device = "cuda"
    args.cache_start_step = 0
    args.cache_end_step = STEPS
    args.cache_step_interval = 1

    args.dtype = "bf16"
    args.use_fp8_gemm = False
    args.quant_type = "none"
    args.cfg_distilled = True
    args.enable_step_distill = False

    args.resolution = "720p"
    args.image_path = image_path
    args.sr = ENABLE_SR
    args.model_path = MODEL_PATH
    args.checkpoint_path = None
    args.lora_path = None
    args.save_path = "output.mp4"

    args.offloading = ENABLE_OFFLOADING
    args.group_offloading = ENABLE_GROUP_OFFLOADING
    args.overlap_group_offloading = ENABLE_OVERLAP_OFFLOADING

    # Sequence parallel setting (keep 1 for single-process service)
    args.sp = 1

    return args


def _repair_glyph_layout_non_destructive():
    """
    Fix expected glyph layout WITHOUT renaming/moving the originals.
    This prevents you from feeding safetensors bytes into torch.load via wrong rename.
    """
    glyph_root = os.path.join(MODEL_PATH, "text_encoder", "Glyph-SDXL-v2")
    assets_dir = os.path.join(glyph_root, "assets")
    checkpoints_dir = os.path.join(glyph_root, "checkpoints")

    if not os.path.isdir(glyph_root):
        log.warning("glyph_root missing: %s", glyph_root)
        return

    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Copy assets into assets/ if they are at root
    for asset_file in ["multilingual_10-lang_idx.json", "color_idx.json"]:
        src_path = os.path.join(glyph_root, asset_file)
        dst_path = os.path.join(assets_dir, asset_file)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            try:
                shutil.copy2(src_path, dst_path)
                log.info("Glyph fix: copied %s -> assets/", asset_file)
            except Exception as e:
                log.warning("Glyph fix: failed copying %s: %s", asset_file, e)

    # Ensure a checkpoint exists at checkpoints/byt5_model.pt by COPYING a candidate
    target_ckpt = os.path.join(checkpoints_dir, "byt5_model.pt")
    if not os.path.exists(target_ckpt):
        candidates = [
            os.path.join(glyph_root, "pytorch_model.bin"),     # torch
            os.path.join(glyph_root, "pytorch_model.pt"),      # torch
            os.path.join(glyph_root, "byt5_model.pt"),         # torch
            os.path.join(glyph_root, "model.safetensors"),     # safetensors
            os.path.join(glyph_root, "pytorch_model.safetensors"),
        ]
        for src in candidates:
            if os.path.exists(src) and os.path.getsize(src) > 0:
                try:
                    shutil.copy2(src, target_ckpt)
                    log.info("Glyph fix: copied %s -> checkpoints/byt5_model.pt", os.path.basename(src))
                    break
                except Exception as e:
                    log.warning("Glyph fix: failed copying %s: %s", src, e)


def _load_pipeline() -> Any:
    t0 = time.monotonic()

    # Non-destructive glyph layout repair before pipeline loads
    _repair_glyph_layout_non_destructive()

    args = get_exhaustive_args(task_mode="t2v")
    infer_state = initialize_infer_state(args)

    device = torch.device("cpu") if args.offloading else torch.device("cuda")
    transformer_init_device = torch.device("cpu") if args.group_offloading else device

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="720p_t2v",
        create_sr_pipeline=args.sr,
        transformer_dtype=torch.bfloat16,
        device=device,
        transformer_init_device=transformer_init_device,
        use_safetensors=True,
    )

    # Apply Tencent optimizations
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )

    log.info(
        "Hunyuan 1.5 loaded. SR=%s offloading=%s group_offloading=%s overlap=%s GPUs=%d dur_s=%d",
        ENABLE_SR,
        ENABLE_OFFLOADING,
        ENABLE_GROUP_OFFLOADING,
        ENABLE_OVERLAP_OFFLOADING,
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        int(time.monotonic() - t0),
    )
    return pipe


@APP.on_event("startup")
def _startup():
    global PIPE
    _ensure_dirs()
    PIPE = _load_pipeline()


@APP.post("/v1/video/generate")
async def generate(req: Request):
    body_txt = (await req.body()).decode("utf-8", errors="replace")
    data = JSONParser().parse(body_txt, {}) if body_txt.strip() else {}
    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid_json"})

    prompt = data.get("prompt")
    image_path = data.get("image_path")

    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse(status_code=422, content={"ok": False, "error": "prompt_required"})

    is_i2v = isinstance(image_path, str) and image_path.strip() and os.path.exists(image_path.strip())
    task_mode = "i2v" if is_i2v else "t2v"

    trace_id = data.get("trace_id")
    trace_id = trace_id.strip() if isinstance(trace_id, str) and trace_id.strip() else uuid.uuid4().hex

    out_dir = os.path.join(UPLOAD_DIR, "artifacts", "video", trace_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.mp4")

    if PIPE is None:
        return JSONResponse(status_code=500, content={"ok": False, "error": "pipe_not_ready"})

    t0 = time.monotonic()
    with _PIPE_LOCK:
        try:
            PIPE(
                prompt=prompt.strip(),
                image_path=image_path.strip() if is_i2v else None,
                height=HEIGHT,
                width=WIDTH,
                video_length=FRAMES,
                infer_steps=STEPS,
                fps=FPS,
                save_path=out_path,
                enable_sr=ENABLE_SR,
                sr_max_batch_size=1,
                use_sage_attn=True,
            )
        except Exception as e:
            log.error("Trace=%s task=%s error=%s", trace_id, task_mode, e, exc_info=True)
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

    dur_ms = int((time.monotonic() - t0) * 1000)
    url = f"{PUBLIC_BASE_URL}/uploads/artifacts/video/{trace_id}/output.mp4" if PUBLIC_BASE_URL else None

    return {
        "ok": True,
        "trace_id": trace_id,
        "task": task_mode,
        "output_path": out_path,
        "output_url": url,
        "resolution": "1920x1080" if ENABLE_SR else "1280x720",
        "dur_ms": dur_ms,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=PORT)
