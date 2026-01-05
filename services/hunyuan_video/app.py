from __future__ import annotations

import inspect
import json
import logging
import os
import time
import traceback
import uuid
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from diffusers import (
    AutoModel,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideo15Pipeline,
    HunyuanVideo15Transformer3DModel,
)
from diffusers.utils import export_to_video

from void_json.json_parser import JSONParser


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _configure_logging()
    _ensure_dirs()

    global PIPE, PIPE_MODEL_ID, SR_TRANSFORMER, SR_UPSAMPLER, SR_SCHEDULER

    PIPE = _load_base_pipe(model_id=DEFAULT_MODEL_ID)
    PIPE_MODEL_ID = DEFAULT_MODEL_ID if PIPE is not None else None

    # Load SR components
    SR_TRANSFORMER, SR_UPSAMPLER, SR_SCHEDULER = _load_sr_components(SR_ROOT)

    logging.getLogger(__name__).info(
        "startup complete base_loaded=%s sr_loaded=%s base_model=%s sr_root=%s",
        bool(PIPE),
        bool(SR_TRANSFORMER and SR_UPSAMPLER and SR_SCHEDULER),
        PIPE_MODEL_ID,
        SR_ROOT,
    )
    
    yield
    
    # Shutdown (if needed)
    pass


APP = FastAPI(title="HunyuanVideo Service", version="1.5+sr", lifespan=lifespan)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
LOG_DIR = os.getenv("LOG_DIR", "/workspace/logs")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

# Base (720p) diffusers checkpoint (downloaded by bootstrap)
DEFAULT_MODEL_ID = os.getenv("HYVIDEO_MODEL_ID", "/opt/models/hunyuan_diffusers").strip()
DEFAULT_DEVICE = os.getenv("HYVIDEO_DEVICE", "cuda:0")
DEFAULT_PORT = int(os.getenv("HYVIDEO_PORT", "8094"))

# DType controls (keep your envs)
DEFAULT_TRANSFORMER_DTYPE = os.getenv("HYVIDEO_TRANSFORMER_DTYPE", "bf16")  # bf16|fp16|fp32
DEFAULT_PIPE_DTYPE = os.getenv("HYVIDEO_PIPE_DTYPE", "fp16")  # bf16|fp16|fp32

# Base memory toggles
DEFAULT_CPU_OFFLOAD = os.getenv("HYVIDEO_CPU_OFFLOAD", "1").strip().lower() in ("1", "true", "yes", "on")
DEFAULT_VAE_TILING = os.getenv("HYVIDEO_VAE_TILING", "1").strip().lower() in ("1", "true", "yes", "on")

# Attention backend (Diffusers recommends setting attention backend for HunyuanVideo1.5)
# Examples: flash_hub, flash_varlen_hub, sage_hub (depends on GPU + installed kernels)
DEFAULT_ATTENTION_BACKEND = os.getenv("HYVIDEO_ATTENTION_BACKEND", "").strip()

# --- SR (Hunyuan official 720->1080 super-resolution) ---
SR_ENABLE = os.getenv("HYVIDEO_SR_ENABLE", "1").strip().lower() in ("1", "true", "yes", "on")
SR_ROOT = os.getenv("HYVIDEO_SR_ROOT", "/opt/models/hunyuan").strip()  # bootstrap downloads Tencent SR subset here
SR_TRANSFORMER_SUBFOLDER = os.getenv("HYVIDEO_SR_TRANSFORMER_SUBFOLDER", "transformer/1080p_sr_distilled").strip()
SR_UPSAMPLER_SUBFOLDER = os.getenv("HYVIDEO_SR_UPSAMPLER_SUBFOLDER", "upsampler/1080p_sr_distilled").strip()
SR_SCHEDULER_SUBFOLDER = os.getenv("HYVIDEO_SR_SCHEDULER_SUBFOLDER", "scheduler").strip()

SR_TARGET_W = int(os.getenv("HYVIDEO_SR_TARGET_WIDTH", "1920"))
SR_TARGET_H = int(os.getenv("HYVIDEO_SR_TARGET_HEIGHT", "1080"))

# SR recommended defaults from Tencent model card:
# 720->1080 SR step distilled: CFG=1, flow shift=2, steps=8
SR_NUM_INFERENCE_STEPS = int(os.getenv("HYVIDEO_SR_NUM_INFERENCE_STEPS", "8"))
SR_GUIDANCE_SCALE = float(os.getenv("HYVIDEO_SR_GUIDANCE_SCALE", "1.0"))

# We'll generate at a "safe internal size" and crop to exact 1080p if needed.
# Many diffusion video models prefer multiples of 16; 1080 -> 1088.
SR_INTERNAL_MULTIPLE = int(os.getenv("HYVIDEO_SR_INTERNAL_MULTIPLE", "16"))

PIPE: Optional[HunyuanVideo15Pipeline] = None
PIPE_MODEL_ID: Optional[str] = None

# SR components (loaded once)
SR_TRANSFORMER: Optional[torch.nn.Module] = None
SR_UPSAMPLER: Optional[torch.nn.Module] = None
SR_SCHEDULER: Optional[FlowMatchEulerDiscreteScheduler] = None

_PIPE_LOCK = Lock()


def _configure_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    level = (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger(__name__).info("logging configured level=%s log_dir=%s", level, LOG_DIR)


def _dtype_from_str(s: str):
    v = (s or "").strip().lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_DIR, "videos"), exist_ok=True)


def _pick_seed(seed: Optional[int]):
    if isinstance(seed, int) and seed >= 0:
        return int(seed)
    return int(time.time() * 1000) ^ (uuid.uuid4().int & 0x7FFFFFFF)


def _device():
    if torch.cuda.is_available():
        return DEFAULT_DEVICE
    return "cpu"


def _ceil_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return x
    return ((x + m - 1) // m) * m


def _maybe_set_attention_backend(pipe: HunyuanVideo15Pipeline, backend: str) -> None:
    if not backend:
        return
    try:
        if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "set_attention_backend"):
            pipe.transformer.set_attention_backend(backend)
            logging.getLogger(__name__).info("attention backend set: %s", backend)
        else:
            logging.getLogger(__name__).warning("attention backend requested but transformer has no set_attention_backend()")
    except Exception:
        logging.getLogger(__name__).warning("failed to set attention backend=%s\n%s", backend, traceback.format_exc())


def _maybe_enable_tiling_and_offload(pipe: HunyuanVideo15Pipeline, dev: str) -> None:
    # VAE tiling
    if DEFAULT_VAE_TILING and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        try:
            pipe.vae.enable_tiling()
            logging.getLogger(__name__).info("vae tiling enabled")
        except Exception:
            logging.getLogger(__name__).warning("failed to enable vae tiling\n%s", traceback.format_exc())

    # CPU offload
    if DEFAULT_CPU_OFFLOAD and hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
            logging.getLogger(__name__).info("model cpu offload enabled")
            return
        except Exception:
            logging.getLogger(__name__).warning("failed to enable model cpu offload; will .to(device)\n%s", traceback.format_exc())

    try:
        pipe.to(dev)
        logging.getLogger(__name__).info("pipeline moved to device=%s", dev)
    except Exception:
        logging.getLogger(__name__).warning("failed moving pipeline to device=%s\n%s", dev, traceback.format_exc())


def _validate_local_dir(path: str, label: str) -> bool:
    if not isinstance(path, str) or not path.strip():
        logging.getLogger(__name__).error("%s: invalid path=%r", label, path)
        return False
    if not path.startswith("/opt/models/"):
        logging.getLogger(__name__).error("%s: must be local /opt/models path; got=%r", label, path)
        return False
    if not os.path.isdir(path) or not os.listdir(path):
        logging.getLogger(__name__).error("%s: missing or empty dir: %r", label, path)
        return False
    return True


def _load_base_pipe(model_id: str) -> Optional[HunyuanVideo15Pipeline]:
    dev = _device()
    t_dtype = _dtype_from_str(DEFAULT_TRANSFORMER_DTYPE)
    p_dtype = _dtype_from_str(DEFAULT_PIPE_DTYPE)

    logging.getLogger(__name__).info(
        "load_base_pipe start model_id=%s device=%s transformer_dtype=%s pipe_dtype=%s cpu_offload=%s vae_tiling=%s attn_backend=%s",
        model_id, dev, t_dtype, p_dtype, DEFAULT_CPU_OFFLOAD, DEFAULT_VAE_TILING, DEFAULT_ATTENTION_BACKEND or "(default)"
    )

    if not _validate_local_dir(model_id, "base_model"):
        return None

    t0 = time.monotonic()
    try:
        # Load pipeline normally - it will load all components including guider
        # We can optionally override transformer dtype after loading
        pipe = HunyuanVideo15Pipeline.from_pretrained(
            model_id,
            torch_dtype=p_dtype,
            local_files_only=True,
        )
        
        # Override transformer dtype if different from pipe dtype
        if t_dtype != p_dtype and hasattr(pipe, "transformer"):
            pipe.transformer = pipe.transformer.to(dtype=t_dtype)

        _maybe_set_attention_backend(pipe, DEFAULT_ATTENTION_BACKEND)
        _maybe_enable_tiling_and_offload(pipe, dev)

        # Log guider info (important: no guidance_scale arg at runtime)
        try:
            logging.getLogger(__name__).info("base guider: %s", getattr(pipe, "guider", None))
        except Exception:
            pass

        logging.getLogger(__name__).info("load_base_pipe done dur_ms=%d", int((time.monotonic() - t0) * 1000))
        return pipe
    except Exception:
        logging.getLogger(__name__).error("load_base_pipe failed\n%s", traceback.format_exc())
        return None


def _load_sr_components(sr_root: str) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], Optional[FlowMatchEulerDiscreteScheduler]]:
    """
    Load:
    - SR transformer: transformer/1080p_sr_distilled
    - SR upsampler : upsampler/1080p_sr_distilled
    - SR scheduler : scheduler

    All from local sr_root (/opt/models/hunyuan).
    """
    if not SR_ENABLE:
        logging.getLogger(__name__).info("SR disabled by HYVIDEO_SR_ENABLE=0")
        return None, None, None

    if not _validate_local_dir(sr_root, "sr_root"):
        return None, None, None

    dev = _device()
    t_dtype = _dtype_from_str(DEFAULT_TRANSFORMER_DTYPE)
    p_dtype = _dtype_from_str(DEFAULT_PIPE_DTYPE)

    logging.getLogger(__name__).info(
        "load_sr_components start sr_root=%s sr_transformer=%s sr_upsampler=%s sr_steps=%d sr_guidance=%s target=%dx%d internal_multiple=%d",
        sr_root, SR_TRANSFORMER_SUBFOLDER, SR_UPSAMPLER_SUBFOLDER, SR_NUM_INFERENCE_STEPS, SR_GUIDANCE_SCALE, SR_TARGET_W, SR_TARGET_H, SR_INTERNAL_MULTIPLE
    )

    try:
        # SR transformer - load from subfolder, but construct full path to avoid root config.json issues
        sr_transformer_path = os.path.join(sr_root, SR_TRANSFORMER_SUBFOLDER)
        if not os.path.isdir(sr_transformer_path):
            raise RuntimeError(f"SR transformer path does not exist: {sr_transformer_path}")
        
        # Load config and fix patch_size from list to tuple (JSON doesn't support tuples)
        # We need to fix this before the model is initialized
        try:
            # Load config.json directly and fix patch_size values
            config_path = os.path.join(sr_transformer_path, "config.json")
            if not os.path.isfile(config_path):
                raise RuntimeError(f"config.json not found at {config_path}")
            
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # Fix patch_size and patch_size_t if they are lists (JSON loads as lists, PyTorch needs tuples)
            fixed = False
            if "patch_size" in config_dict and isinstance(config_dict["patch_size"], list):
                config_dict["patch_size"] = tuple(config_dict["patch_size"])
                fixed = True
            if "patch_size_t" in config_dict and isinstance(config_dict["patch_size_t"], list):
                config_dict["patch_size_t"] = tuple(config_dict["patch_size_t"])
                fixed = True
            
            if fixed:
                logging.getLogger(__name__).info("SR transformer: fixed patch_size from list to tuple in config")
                # Create a persistent temp directory (not auto-deleted) to hold the fixed config
                import tempfile
                import shutil
                tmpdir = tempfile.mkdtemp(prefix="hunyuan_sr_")
                try:
                    # Copy all files to temp dir
                    for item in os.listdir(sr_transformer_path):
                        src = os.path.join(sr_transformer_path, item)
                        dst = os.path.join(tmpdir, item)
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
                        elif os.path.isdir(src):
                            shutil.copytree(src, dst)
                    # Write fixed config
                    tmp_config_path = os.path.join(tmpdir, "config.json")
                    with open(tmp_config_path, "w", encoding="utf-8") as f:
                        json.dump(config_dict, f, indent=2)
                    # Load model from temp dir (from_pretrained loads weights automatically)
                    sr_transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
                        tmpdir,
                        torch_dtype=t_dtype,
                        local_files_only=True,
                    )
                    # Note: tmpdir is not deleted - it will persist for the lifetime of the process
                    # This is acceptable as it's only created once at startup
                except Exception:
                    # Clean up temp dir on error
                    try:
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass
                    raise
            else:
                # No fix needed, load normally
                sr_transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
                    sr_transformer_path,
                    torch_dtype=t_dtype,
                    local_files_only=True,
                )
        except Exception as load_ex:
            logging.getLogger(__name__).error("SR transformer: failed to load with config fix: %s", load_ex, exc_info=True)
            raise

        # SR scheduler - load from subfolder
        sr_scheduler_path = os.path.join(sr_root, SR_SCHEDULER_SUBFOLDER)
        if not os.path.isdir(sr_scheduler_path):
            raise RuntimeError(f"SR scheduler path does not exist: {sr_scheduler_path}")
        sr_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            sr_scheduler_path,
            local_files_only=True,
        )

        # SR upsampler - load from subfolder
        sr_upsampler_path = os.path.join(sr_root, SR_UPSAMPLER_SUBFOLDER)
        if not os.path.isdir(sr_upsampler_path):
            raise RuntimeError(f"SR upsampler path does not exist: {sr_upsampler_path}")
        sr_upsampler = AutoModel.from_pretrained(
            sr_upsampler_path,
            torch_dtype=p_dtype,
            local_files_only=True,
        )

        # Attention backend on SR transformer too
        try:
            if DEFAULT_ATTENTION_BACKEND and hasattr(sr_transformer, "set_attention_backend"):
                sr_transformer.set_attention_backend(DEFAULT_ATTENTION_BACKEND)
                logging.getLogger(__name__).info("SR transformer attention backend set=%s", DEFAULT_ATTENTION_BACKEND)
        except Exception:
            logging.getLogger(__name__).warning("failed setting SR attention backend\n%s", traceback.format_exc())

        logging.getLogger(__name__).info("load_sr_components done (models loaded; device=%s)", dev)
        return sr_transformer, sr_upsampler, sr_scheduler

    except Exception:
        logging.getLogger(__name__).error("load_sr_components failed\n%s", traceback.format_exc())
        return None, None, None


def _resolve_local_image(image_ref: str):
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
    return Image.open(p).convert("RGB")


def _coerce_int(v: Any, default: int):
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    return int(default)


def _coerce_float(v: Any, default: float):
    if isinstance(v, bool):
        return float(default)
    if isinstance(v, (int, float)):
        return float(v)
    return float(default)


def _decode_latents_to_np_video(pipe: HunyuanVideo15Pipeline, latents: torch.Tensor) -> np.ndarray:
    """
    Decode video latents -> numpy video [F,H,W,C] in uint8-ish float range [0..255] or [0..1]?
    We'll output float [0..255] uint8 for export_to_video compatibility.
    """
    vae = getattr(pipe, "vae", None)
    if vae is None:
        raise RuntimeError("pipeline has no VAE")

    # Best-effort scaling factor
    scaling = None
    try:
        scaling = getattr(getattr(vae, "config", None), "scaling_factor", None)
    except Exception:
        scaling = None

    lat = latents
    if scaling and isinstance(scaling, (float, int)) and float(scaling) != 0.0:
        lat = lat / float(scaling)

    with torch.no_grad():
        decoded = vae.decode(lat, return_dict=True)
        sample = getattr(decoded, "sample", None)
        if sample is None:
            # fallback for older return types
            sample = decoded[0] if isinstance(decoded, (tuple, list)) else decoded

    # sample expected in [-1, 1]
    video = (sample / 2 + 0.5).clamp(0, 1)

    # [B, C, F, H, W] -> [F, H, W, C]
    if video.ndim != 5:
        raise RuntimeError(f"unexpected decoded video shape: {tuple(video.shape)}")

    video0 = video[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()  # F,H,W,C float 0..1
    video0_u8 = (video0 * 255.0).round().clip(0, 255).astype(np.uint8)
    return video0_u8


def _center_crop_video_np(video: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if not isinstance(video, np.ndarray) or video.ndim != 4:
        return video
    f, h, w, c = video.shape
    if h == target_h and w == target_w:
        return video
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    out = video[:, top:top + target_h, left:left + target_w, :]
    return out


def _call_upsampler(upsampler: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """
    Best-effort call into SR upsampler.
    We inspect signature and try common arg names.
    """
    sig = None
    try:
        sig = inspect.signature(upsampler.forward)
    except Exception:
        sig = None

    if sig is not None:
        params = list(sig.parameters.keys())
        # common single-input names
        for name in ("latents", "hidden_states", "x", "sample", "inputs"):
            if name in params:
                return upsampler(**{name: latents})
        # if it takes exactly one positional (besides self)
        if len(params) == 1:
            return upsampler(latents)

    # fallback
    try:
        return upsampler(latents)
    except Exception:
        # last resort: try keyword latents
        return upsampler(latents=latents)


def _run_sr_720_to_1080(
    pipe: HunyuanVideo15Pipeline,
    prompt: str,
    negative_prompt: Optional[str],
    lowres_latents: torch.Tensor,
    num_frames: int,
    seed: int,
) -> np.ndarray:
    """
    Run Tencent SR (720->1080) using:
      - SR_UPSAMPLER to produce conditional latents at high-res
      - SR_TRANSFORMER + SR_SCHEDULER inside the same pipeline by temporarily swapping modules
      - A patched prepare_cond_latents_and_mask to inject cond latents + mask
    Returns a numpy video [F,H,W,C] uint8 at internal resolution (cropping is applied by caller).
    """
    global SR_TRANSFORMER, SR_UPSAMPLER, SR_SCHEDULER

    if not (SR_TRANSFORMER and SR_UPSAMPLER and SR_SCHEDULER):
        raise RuntimeError("SR components not loaded")

    dev = _device()
    gen_sr = torch.Generator(device=dev).manual_seed(seed ^ 0xA5A5A5A5) if dev != "cpu" else torch.Generator().manual_seed(seed ^ 0xA5A5A5A5)

    # Compute internal SR size (crop later to exact 1080p)
    h_int = _ceil_to_multiple(SR_TARGET_H, SR_INTERNAL_MULTIPLE)
    w_int = _ceil_to_multiple(SR_TARGET_W, SR_INTERNAL_MULTIPLE)

    logger = logging.getLogger(__name__)
    logger.info(
        "SR start target=%dx%d internal=%dx%d steps=%d guidance=%s (recommended: steps=8, cfg=1, flow_shift=2 per Tencent)",
        SR_TARGET_W, SR_TARGET_H, w_int, h_int, SR_NUM_INFERENCE_STEPS, SR_GUIDANCE_SCALE
    )

    # Move lowres latents to a sensible dtype/device for upsampler
    lr = lowres_latents
    if lr.device.type != ("cuda" if torch.cuda.is_available() else "cpu"):
        lr = lr.to(dev)

    # Upsample latents using Tencent upsampler model
    t0_up = time.monotonic()
    with torch.no_grad():
        cond_latents_hr = _call_upsampler(SR_UPSAMPLER, lr)
    logger.info(
        "SR upsampler done dur_ms=%d lr_shape=%s hr_shape=%s",
        int((time.monotonic() - t0_up) * 1000),
        tuple(lr.shape),
        tuple(cond_latents_hr.shape) if hasattr(cond_latents_hr, "shape") else type(cond_latents_hr),
    )

    if not isinstance(cond_latents_hr, torch.Tensor):
        raise RuntimeError(f"upsampler returned non-tensor: {type(cond_latents_hr)}")

    # Build a mask (1 channel) in the same spatial/temporal shape
    if cond_latents_hr.ndim != 5:
        raise RuntimeError(f"unexpected cond_latents_hr shape: {tuple(cond_latents_hr.shape)} (expected B,C,F,H,W)")

    mask = torch.ones(
        (cond_latents_hr.shape[0], 1, cond_latents_hr.shape[2], cond_latents_hr.shape[3], cond_latents_hr.shape[4]),
        dtype=cond_latents_hr.dtype,
        device=cond_latents_hr.device,
    )

    # Patch prepare_cond_latents_and_mask so the pipeline injects our cond latents
    orig_prepare = getattr(pipe, "prepare_cond_latents_and_mask", None)
    orig_transformer = getattr(pipe, "transformer", None)
    orig_scheduler = getattr(pipe, "scheduler", None)
    orig_guider = getattr(pipe, "guider", None)

    def _sr_prepare_cond_latents_and_mask(latents: torch.Tensor, dtype=None, device=None):
        # latents is the *main* latent batch the pipeline is working with, possibly expanded for CFG
        B = latents.shape[0]
        c = cond_latents_hr.to(device=latents.device, dtype=latents.dtype)
        m = mask.to(device=latents.device, dtype=latents.dtype)

        # Expand across batch if pipeline expanded for guidance
        if c.shape[0] != B:
            if B % c.shape[0] == 0:
                rep = B // c.shape[0]
                c = c.repeat_interleave(rep, dim=0)
                m = m.repeat_interleave(rep, dim=0)
            else:
                c = c.expand(B, -1, -1, -1, -1)
                m = m.expand(B, -1, -1, -1, -1)

        return c, m

    try:
        # Swap in SR transformer + scheduler
        pipe.register_modules(transformer=SR_TRANSFORMER, scheduler=SR_SCHEDULER)

        # SR wants CFG ~ 1; disable guider if possible (fastest) else set guidance_scale=1
        try:
            if orig_guider is not None and hasattr(orig_guider, "new"):
                pipe.guider = orig_guider.new(enabled=False, guidance_scale=float(SR_GUIDANCE_SCALE))
                logger.info("SR guider configured: enabled=False guidance_scale=%s", SR_GUIDANCE_SCALE)
        except Exception:
            logger.warning("SR guider config failed; continuing\n%s", traceback.format_exc())

        # Patch method
        if orig_prepare is None:
            raise RuntimeError("pipe has no prepare_cond_latents_and_mask to patch")
        pipe.prepare_cond_latents_and_mask = _sr_prepare_cond_latents_and_mask  # type: ignore

        # Run SR generation (returns frames at high-res)
        t0 = time.monotonic()
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=h_int,
            width=w_int,
            num_frames=int(num_frames),
            num_inference_steps=int(SR_NUM_INFERENCE_STEPS),
            generator=gen_sr,
            output_type="np",
            return_dict=True,
        )
        dur_ms = int((time.monotonic() - t0) * 1000)

        frames0 = out.frames[0]
        # frames0 should be np array [F,H,W,C] with output_type="np"
        if isinstance(frames0, list):
            # convert list of PIL to np
            frames0 = np.stack([np.array(im) for im in frames0], axis=0)

        if not isinstance(frames0, np.ndarray):
            raise RuntimeError(f"SR output unexpected type: {type(frames0)}")

        logger.info("SR pipe done dur_ms=%d out_shape=%s", dur_ms, frames0.shape)
        return frames0

    finally:
        # Restore base components no matter what
        try:
            if orig_transformer is not None and orig_scheduler is not None:
                pipe.register_modules(transformer=orig_transformer, scheduler=orig_scheduler)
        except Exception:
            logger.warning("failed restoring base transformer/scheduler\n%s", traceback.format_exc())

        try:
            if orig_prepare is not None:
                pipe.prepare_cond_latents_and_mask = orig_prepare  # type: ignore
        except Exception:
            logger.warning("failed restoring prepare_cond_latents_and_mask\n%s", traceback.format_exc())

        try:
            if orig_guider is not None:
                pipe.guider = orig_guider
        except Exception:
            logger.warning("failed restoring guider\n%s", traceback.format_exc())


def _run_generate_dict(data: Dict[str, Any]):
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse(status_code=422, content={"ok": False, "error": {"code": "missing_prompt", "message": "prompt (string) is required"}})

    # Back-compat alias
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

        # IMPORTANT: HunyuanVideo15Pipeline does not accept guidance_scale at runtime;
        # update via pipe.guider.new(...)
        if isinstance(guidance_scale, (int, float)):
            try:
                if hasattr(pipe, "guider") and hasattr(pipe.guider, "new"):
                    pipe.guider = pipe.guider.new(guidance_scale=float(guidance_scale))
                    logging.getLogger(__name__).info("base guider updated guidance_scale=%s", guidance_scale)
            except Exception:
                logging.getLogger(__name__).warning("failed to update guider guidance_scale=%s\n%s", guidance_scale, traceback.format_exc())

        kwargs: Dict[str, Any] = {
            "prompt": prompt.strip(),
            "generator": gen,
            "output_type": "latent",   # we need latents for SR conditioning
            "return_dict": True,
        }
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
        if isinstance(num_videos_per_prompt, int) and num_videos_per_prompt > 0:
            kwargs["num_videos_per_prompt"] = int(num_videos_per_prompt)

        # init_image is NOT supported by HunyuanVideo15Pipeline (T2V).
        # We keep behavior explicit & safe.
        if isinstance(init_image_ref, str) and init_image_ref.strip():
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": {"code": "init_image_unsupported", "message": "HunyuanVideo-1.5 T2V pipeline does not support init_image in this service build"}},
            )

        # Merge extra args
        for k, v in extra.items():
            if k not in kwargs:
                kwargs[k] = v

        log = logging.getLogger(__name__)
        log.info(
            "generate start job_id=%s model_id=%s device=%s seed=%d prompt_len=%d frames=%s steps=%s size=%sx%s fps=%s sr_enable=%s",
            job,
            PIPE_MODEL_ID,
            dev,
            used_seed,
            len(prompt.strip()),
            kwargs.get("num_frames"),
            kwargs.get("num_inference_steps"),
            kwargs.get("width"),
            kwargs.get("height"),
            fps,
            SR_ENABLE,
        )
        if request_meta:
            log.info("generate request_meta keys=%s", sorted(list(request_meta.keys()))[:64])

        # --- Stage 1: 720p generation (latents) ---
        t0 = time.monotonic()
        try:
            out = pipe(**kwargs)
            dur_base_ms = int((time.monotonic() - t0) * 1000)
        except Exception as gen_err:
            dur_base_ms = int((time.monotonic() - t0) * 1000)
            log.error("base generation failed dur_ms=%d\n%s", dur_base_ms, traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "generation_failed",
                        "message": f"Failed to generate video: {str(gen_err)[:200]}",
                    },
                },
            )

        lat0 = out.frames[0]
        if not isinstance(lat0, torch.Tensor):
            log.error("expected latent tensor; got %s", type(lat0))
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "invalid_output",
                        "message": f"Pipeline returned unexpected output type: {type(lat0)}",
                    },
                },
            )
        log.info("base done dur_ms=%d latents_shape=%s dtype=%s device=%s", dur_base_ms, tuple(lat0.shape), lat0.dtype, lat0.device)

        # Decode 720p for fallback
        t0d = time.monotonic()
        try:
            video_720 = _decode_latents_to_np_video(pipe, lat0)
            dur_dec_ms = int((time.monotonic() - t0d) * 1000)
            log.info("base decode done dur_ms=%d video720_shape=%s", dur_dec_ms, video_720.shape)
        except Exception as decode_err:
            dur_dec_ms = int((time.monotonic() - t0d) * 1000)
            log.error("base decode failed dur_ms=%d\n%s", dur_dec_ms, traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "decode_failed",
                        "message": f"Failed to decode latents: {str(decode_err)[:200]}",
                    },
                },
            )

        # --- Stage 2: SR 720->1080 (Tencent SR) ---
        sr_ok = False
        sr_err = None
        final_video = video_720
        final_kind = "720p"

        if SR_ENABLE and SR_TRANSFORMER and SR_UPSAMPLER and SR_SCHEDULER:
            try:
                video_hr = _run_sr_720_to_1080(
                    pipe=pipe,
                    prompt=prompt.strip(),
                    negative_prompt=negative_prompt.strip() if isinstance(negative_prompt, str) and negative_prompt.strip() else None,
                    lowres_latents=lat0,
                    num_frames=int(kwargs.get("num_frames") or 121),
                    seed=used_seed,
                )
                # Crop to exact 1080p if we generated 1088 (or any larger)
                video_hr = _center_crop_video_np(video_hr, SR_TARGET_H, SR_TARGET_W)
                final_video = video_hr
                final_kind = "1080p"
                sr_ok = True
                log.info("SR success final_shape=%s", final_video.shape)
            except Exception:
                sr_err = traceback.format_exc()
                sr_ok = False
                log.error("SR failed; falling back to 720p\n%s", sr_err)

        # Export (encoding only; no ffmpeg scaling)
        t0e = time.monotonic()
        try:
            export_to_video(final_video, out_path, fps=int(fps) if isinstance(fps, int) and fps > 0 else 15)
            dur_export_ms = int((time.monotonic() - t0e) * 1000)
        except Exception as export_err:
            dur_export_ms = int((time.monotonic() - t0e) * 1000)
            log.error("export_to_video failed dur_ms=%d\n%s", dur_export_ms, traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "export_failed",
                        "message": f"Failed to export video: {str(export_err)[:200]}",
                    },
                },
            )

        total_ms = int((time.monotonic() - t0) * 1000.0)
        log.info(
            "generate done job_id=%s total_ms=%d base_ms=%d decode720_ms=%d export_ms=%d final=%s out=%s",
            job,
            total_ms,
            dur_base_ms,
            dur_dec_ms,
            dur_export_ms,
            final_kind,
            out_path,
        )

    url = (PUBLIC_BASE_URL.rstrip("/") + f"/uploads/videos/{job}/output.mp4") if PUBLIC_BASE_URL else None
    return {
        "ok": True,
        "result": {
            "job_id": job,
            "model_id": PIPE_MODEL_ID,
            "device": dev,
            "seed": used_seed,
            "output_kind": final_kind,
            "sr_ok": sr_ok,
            "sr_error": (sr_err[-4000:] if isinstance(sr_err, str) else None),
            "output_path": out_path,
            "output_url": url,
            "request_meta": request_meta,
            # Include all result fields in meta for orchestrator compatibility
            "meta": {
                "job_id": job,
                "model_id": PIPE_MODEL_ID,
                "device": dev,
                "seed": used_seed,
                "output_kind": final_kind,
                "sr_ok": sr_ok,
                "sr_error": (sr_err[-4000:] if isinstance(sr_err, str) else None),
                "request_meta": request_meta,
            },
        },
    }




@APP.get("/healthz")
def healthz():
    return {
        "ok": True,
        "base_model_id": PIPE_MODEL_ID,
        "sr_enable": SR_ENABLE,
        "sr_loaded": bool(SR_TRANSFORMER and SR_UPSAMPLER and SR_SCHEDULER),
        "cuda": bool(torch.cuda.is_available()),
        "device": _device(),
        "attention_backend": DEFAULT_ATTENTION_BACKEND or None,
        "sr_target": {"w": SR_TARGET_W, "h": SR_TARGET_H},
        "sr_steps": SR_NUM_INFERENCE_STEPS,
        "sr_guidance_scale": SR_GUIDANCE_SCALE,
    }


@APP.post("/v1/video/generate")
async def generate(req: Request):
    body_bytes = await req.body()
    body_txt = body_bytes.decode("utf-8", errors="replace")
    if not body_txt.strip():
        data: Dict[str, Any] = {}
    else:
        parser = JSONParser()
        data = parser.parse(body_txt, {})
        if not isinstance(data, dict):
            return JSONResponse(status_code=400, content={"ok": False, "error": {"code": "invalid_json", "message": "request body must be JSON"}})

    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"ok": False, "error": {"code": "invalid_json", "message": "request body must be a JSON object"}})

    try:
        out = _run_generate_dict(data)
        return out
    except Exception:
        logging.getLogger(__name__).error("unhandled error in generate\n%s", traceback.format_exc())
        return JSONResponse(status_code=500, content={"ok": False, "error": {"code": "internal_error", "message": "unhandled exception"}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=DEFAULT_PORT, log_level="info")
