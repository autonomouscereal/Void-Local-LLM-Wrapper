from __future__ import annotations

import json
import logging
import os
import time
import uuid
import inspect
from threading import Lock
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from PIL import Image


APP = FastAPI(title="HunyuanVideo Service", version="1.0")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
LOG_DIR = os.getenv("LOG_DIR", "/workspace/logs")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

# OFFLINE ONLY: model must exist locally under /opt/models (bootstrap pre-download).
DEFAULT_MODEL_ID = os.getenv("HYVIDEO_MODEL_ID", "/opt/models/hunyuan_diffusers")
DEFAULT_DEVICE = os.getenv("HYVIDEO_DEVICE", "cuda:0")
DEFAULT_PORT = int(os.getenv("HYVIDEO_PORT", "8094"))

# DType controls
DEFAULT_TRANSFORMER_DTYPE = os.getenv("HYVIDEO_TRANSFORMER_DTYPE", "bf16")  # bf16|fp16|fp32
DEFAULT_PIPE_DTYPE = os.getenv("HYVIDEO_PIPE_DTYPE", "fp16")  # bf16|fp16|fp32

# Default memory toggles
DEFAULT_CPU_OFFLOAD = os.getenv("HYVIDEO_CPU_OFFLOAD", "1").strip().lower() in ("1", "true", "yes", "on")
DEFAULT_VAE_TILING = os.getenv("HYVIDEO_VAE_TILING", "1").strip().lower() in ("1", "true", "yes", "on")

PIPE = None
PIPE_MODEL_ID = None
_PIPE_LOCK = Lock()


def _configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    level = (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _dtype_from_str(s: str):
    v = (s or "").strip().lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _ensure_dirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_DIR, "videos"), exist_ok=True)


def _pick_seed(seed: Optional[int]) -> int:
    if isinstance(seed, int) and seed >= 0:
        return int(seed)
    return int(time.time() * 1000) ^ (uuid.uuid4().int & 0x7FFFFFFF)


def _device() -> str:
    if torch.cuda.is_available():
        return DEFAULT_DEVICE
    return "cpu"


def _load_pipe(model_id: str) -> None:
    global PIPE, PIPE_MODEL_ID

    _ensure_dirs()

    dev = _device()
    t_dtype = _dtype_from_str(DEFAULT_TRANSFORMER_DTYPE)
    p_dtype = _dtype_from_str(DEFAULT_PIPE_DTYPE)

    t0 = time.monotonic()
    logging.getLogger(__name__).info(
        "hunyuan_video.load start model_id=%s device=%s transformer_dtype=%s pipe_dtype=%s cpu_offload=%s vae_tiling=%s",
        model_id,
        dev,
        str(t_dtype),
        str(p_dtype),
        bool(DEFAULT_CPU_OFFLOAD),
        bool(DEFAULT_VAE_TILING),
    )

    # Service policy: use the single pre-downloaded local model directory.
    if not isinstance(model_id, str) or not model_id.strip() or (not model_id.startswith("/opt/models/")):
        raise RuntimeError(f"HYVIDEO_MODEL_ID must be a local /opt/models path; got: {model_id!r}")
    if not os.path.isdir(model_id) or not os.listdir(model_id):
        raise RuntimeError(f"HYVIDEO_MODEL_ID directory missing/empty (bootstrap did not download?): {model_id!r}")

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=t_dtype,
        local_files_only=True,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=p_dtype,
        local_files_only=True,
    )

    if DEFAULT_VAE_TILING and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    if DEFAULT_CPU_OFFLOAD and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)

    PIPE = pipe
    PIPE_MODEL_ID = model_id
    logging.getLogger(__name__).info("hunyuan_video.load done model_id=%s dur_ms=%d", model_id, int((time.monotonic() - t0) * 1000))


def _resolve_local_image(image_ref: str) -> Optional[Image.Image]:
    """
    Resolve a local image reference into a PIL.Image.

    Supported forms:
    - /workspace/...
    - /uploads/... (mapped to /workspace/uploads/...)
    """
    if not isinstance(image_ref, str) or not image_ref.strip():
        return None
    p = image_ref.strip()
    if p.startswith("/uploads/"):
        p = "/workspace" + p
    if not p.startswith("/workspace/"):
        return None
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("RGB")
    return im


def _pipe_accepts_image(pipe: Any) -> str | None:
    """
    Return the keyword to pass an init image to this pipeline call, if supported.
    """
    try:
        sig = inspect.signature(pipe.__call__)
        params = sig.parameters
        for key in ("image", "init_image", "images"):
            if key in params:
                return key
    except Exception:
        return None
    return None


def _coerce_int(v: Any, default: int) -> int:
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    return int(default)


def _coerce_float(v: Any, default: float) -> float:
    if isinstance(v, bool):
        return float(default)
    if isinstance(v, (int, float)):
        return float(v)
    return float(default)


def _run_generate_dict(data: Dict[str, Any]) -> Dict[str, Any] | JSONResponse:
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse(status_code=422, content={"ok": False, "error": {"code": "missing_prompt", "message": "prompt (string) is required"}})

    # Back-compat aliases
    negative_prompt = data.get("negative_prompt")
    if negative_prompt is None and isinstance(data.get("negative"), str):
        negative_prompt = data.get("negative")

    height = data.get("height")
    width = data.get("width")
    num_frames = data.get("num_frames")
    num_inference_steps = data.get("num_inference_steps")
    guidance_scale = data.get("guidance_scale")
    fps = data.get("fps")
    seed = data.get("seed")
    num_videos_per_prompt = data.get("num_videos_per_prompt")
    init_image_ref = data.get("init_image") if isinstance(data.get("init_image"), str) else None
    request_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    # Ignore any per-request model_id; this service only runs one pinned model.
    # Do NOT error if callers include it (we simply don't honor it).
    _ignored_model_id = data.get("model_id")
    cpu_offload = data.get("cpu_offload")
    vae_tiling = data.get("vae_tiling")
    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}

    used_seed = _pick_seed(seed=seed if isinstance(seed, int) else None)
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(used_seed) if dev != "cpu" else torch.Generator().manual_seed(used_seed)

    job_id = data.get("job_id")
    job = str(job_id).strip() if isinstance(job_id, str) and job_id.strip() else uuid.uuid4().hex
    out_dir = os.path.join(UPLOAD_DIR, "videos", job)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.mp4")

    with _PIPE_LOCK:
        global PIPE, PIPE_MODEL_ID
        pipe = PIPE
        if pipe is None:
            return JSONResponse(status_code=500, content={"ok": False, "error": {"code": "pipe_not_ready", "message": "pipeline not loaded"}})

        if isinstance(vae_tiling, bool) and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            if vae_tiling:
                pipe.vae.enable_tiling()

        if isinstance(cpu_offload, bool) and hasattr(pipe, "enable_model_cpu_offload"):
            if cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(_device())

        kwargs: Dict[str, Any] = {"prompt": prompt.strip(), "generator": gen}
        if isinstance(negative_prompt, str) and negative_prompt.strip():
            kwargs["negative_prompt"] = negative_prompt.strip()
        if isinstance(height, int) and height > 0:
            kwargs["height"] = int(height)
        if isinstance(width, int) and width > 0:
            kwargs["width"] = int(width)
        if isinstance(num_frames, int) and num_frames > 0:
            kwargs["num_frames"] = int(num_frames)
        if isinstance(num_inference_steps, int) and num_inference_steps > 0:
            kwargs["num_inference_steps"] = int(num_inference_steps)
        if isinstance(guidance_scale, (int, float)):
            kwargs["guidance_scale"] = float(guidance_scale)
        if isinstance(num_videos_per_prompt, int) and num_videos_per_prompt > 0:
            kwargs["num_videos_per_prompt"] = int(num_videos_per_prompt)

        # Optional i2v: if init_image is provided, we REQUIRE it to be usable.
        if isinstance(init_image_ref, str) and init_image_ref.strip():
            kw = _pipe_accepts_image(pipe)
            img = _resolve_local_image(init_image_ref)
            if not kw:
                return JSONResponse(status_code=400, content={"ok": False, "error": {"code": "init_image_unsupported", "message": "pipeline does not accept init_image"}})
            if img is None:
                return JSONResponse(status_code=422, content={"ok": False, "error": {"code": "invalid_init_image", "message": "init_image must be a local /uploads/... or /workspace/... path"}})
            kwargs[kw] = img

        for k, v in extra.items():
            if k not in kwargs:
                kwargs[k] = v

        t0 = time.monotonic()
        logging.getLogger(__name__).info(
            "hunyuan_video.generate start job_id=%s model_id=%s device=%s seed=%d prompt_len=%d frames=%s steps=%s size=%sx%s fps=%s init_image=%s",
            job,
            PIPE_MODEL_ID,
            dev,
            used_seed,
            len(prompt.strip()),
            (kwargs.get("num_frames")),
            (kwargs.get("num_inference_steps")),
            (kwargs.get("width")),
            (kwargs.get("height")),
            fps,
            (init_image_ref[:120] + "..." if isinstance(init_image_ref, str) and len(init_image_ref) > 120 else init_image_ref),
        )
        if request_meta:
            logging.getLogger(__name__).info("hunyuan_video.generate request_meta keys=%s", sorted(list(request_meta.keys()))[:64])
        out = pipe(**kwargs)
        frames0 = out.frames[0]
        export_to_video(frames0, out_path, fps=int(fps) if isinstance(fps, int) and fps > 0 else 15)
        dur_ms = int((time.monotonic() - t0) * 1000.0)

    url = (PUBLIC_BASE_URL.rstrip("/") + f"/uploads/videos/{job}/output.mp4") if PUBLIC_BASE_URL else None
    logging.getLogger(__name__).info("hunyuan_video.generate done job_id=%s dur_ms=%d out=%s", job, dur_ms, out_path)

    return {
        "ok": True,
        "result": {
            "job_id": job,
            "model_id": PIPE_MODEL_ID,
            "device": dev,
            "seed": used_seed,
            "duration_ms": dur_ms,
            "output_path": out_path,
            "output_url": url,
            "request_meta": request_meta,
        },
    }


@APP.on_event("startup")
def _startup():
    _configure_logging()
    _load_pipe(model_id=DEFAULT_MODEL_ID)


@APP.get("/healthz")
def healthz():
    return {"ok": True, "model_id": PIPE_MODEL_ID, "cuda": bool(torch.cuda.is_available()), "device": _device()}


@APP.post("/v1/video/generate")
async def generate(req: Request):
    body_bytes = await req.body()
    body_txt = body_bytes.decode("utf-8", errors="replace")
    if not body_txt.strip():
        data: Dict[str, Any] = {}
    else:
        try:
            data = json.loads(body_txt)
        except Exception:
            return JSONResponse(status_code=400, content={"ok": False, "error": {"code": "invalid_json", "message": "request body must be JSON"}})
    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"ok": False, "error": {"code": "invalid_json", "message": "request body must be a JSON object"}})
    out = _run_generate_dict(data)
    return out




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(APP, host="0.0.0.0", port=DEFAULT_PORT, log_level="info")


