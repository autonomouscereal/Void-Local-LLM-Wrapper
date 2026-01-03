import io
import logging
import os
import time
from typing import List, Optional

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore


logger = logging.getLogger(__name__)

_MUSIC_MODEL: Optional[MusicgenForConditionalGeneration] = None
_MUSIC_PROCESSOR = None
_MUSIC_DEVICE = None  # Will be determined dynamically
_MUSIC_DEVICE_MAP = None  # For CPU offloading / multi-GPU
_ENGINE_LOADED = False

# Minimum free GPU memory required (in GB) to use a GPU
_MIN_GPU_MEMORY_GB = float(os.getenv("MUSIC_MIN_GPU_MEMORY_GB", "2.0"))
# Enable CPU offloading (offload parts of model to CPU when GPU memory is tight)
_ENABLE_CPU_OFFLOAD = os.getenv("MUSIC_CPU_OFFLOAD", "1").strip().lower() in ("1", "true", "yes", "on")


def _log_music_generate_state(
    prompt: str,
    seconds: int,
    seed: Optional[int],
    model: MusicgenForConditionalGeneration,
    device: str,
    inputs: dict,
    max_new_tokens: int,
):
    """
    Log detailed information about the MusicGen generation call without dumping
    full tensor contents. Intended for debugging around model.generate().
    """
    try:
        logger.info("========== [MUSICGEN DEBUG] ==========")
        logger.info("prompt=%r", prompt)
        logger.info("seconds=%s seed=%s", seconds, seed)
        logger.info("device=%s", device)

        logger.info("model type=%s", type(model))
        model_id = getattr(getattr(model, "config", None), "name_or_path", None)
        logger.info("model id=%r", model_id)

        generate_attr = getattr(model, "generate", None)
        # Don't use %r here - it would print the entire model structure via repr() of the bound method
        generate_attr_str = (
            f"<bound method {type(model).__name__}.generate>"
            if generate_attr is not None
            else "None"
        )
        logger.info("model.generate attr=%s", generate_attr_str)
        logger.info("model.generate type=%s", type(generate_attr))
        logger.info("model.generate callable=%s", callable(generate_attr))

        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            logger.info(
                "generation_config.return_dict_in_generate=%r",
                getattr(gen_cfg, "return_dict_in_generate", None),
            )
            logger.info(
                "generation_config.max_new_tokens=%r",
                getattr(gen_cfg, "max_new_tokens", None),
            )
            logger.info(
                "generation_config.do_sample=%r",
                getattr(gen_cfg, "do_sample", None),
            )
            logger.info(
                "generation_config.guidance_scale=%r",
                getattr(gen_cfg, "guidance_scale", None),
            )

        input_keys = list(inputs.keys()) if isinstance(inputs, dict) else []
        logger.info("inputs keys=%r", input_keys)
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                value_type = type(value)
                shape = getattr(value, "shape", None)
                tensor_device = getattr(value, "device", None)
                logger.info(
                    "  - %s type=%s shape=%r device=%r",
                    key,
                    value_type,
                    shape,
                    tensor_device,
                )

        logger.info("extra kwargs: max_new_tokens=%s do_sample=%s guidance_scale=%s",
                    max_new_tokens, True, 3.0)
        logger.info("========== [END MUSICGEN DEBUG] ==========")
    except Exception:
        # This should never break generation; log and continue.
        logger.exception("Failed to log MusicGen generate state")


def _find_best_gpu() -> Optional[int]:
    """
    Find the GPU with the most free memory across all available GPUs.
    Returns the GPU index (0-based) or None if no suitable GPU found.
    """
    if not torch.cuda.is_available():
        return None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    
    best_gpu = None
    best_free_memory = 0.0
    
    for gpu_idx in range(num_gpus):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_idx)
            free_mem_gb = free_mem / (1024 ** 3)
            total_mem_gb = total_mem / (1024 ** 3)
            
            logger.info(
                "musicgen.gpu_check gpu=%d free_gb=%.2f total_gb=%.2f",
                gpu_idx, free_mem_gb, total_mem_gb
            )
            
            if free_mem_gb >= _MIN_GPU_MEMORY_GB and free_mem_gb > best_free_memory:
                best_gpu = gpu_idx
                best_free_memory = free_mem_gb
        except Exception as e:
            logger.warning("musicgen.gpu_check failed gpu=%d error=%s", gpu_idx, e)
            continue
    
    if best_gpu is not None:
        logger.info(
            "musicgen.gpu_selected gpu=%d free_gb=%.2f",
            best_gpu, best_free_memory
        )
    
    return best_gpu


def _get_device_and_map():
    """
    Determine the best device configuration: prefer GPU with CPU offloading fallback.
    Returns (device_string, device_map_dict) where device_map can be None for simple single-device.
    """
    best_gpu = _find_best_gpu()
    
    if best_gpu is None:
        logger.info("musicgen.device_selection using CPU (no suitable GPU found)")
        return "cpu", None
    
    device_str = f"cuda:{best_gpu}"
    
    # If CPU offloading is enabled, use device_map for automatic offloading
    if _ENABLE_CPU_OFFLOAD:
        try:
            # Try to get memory info to decide on offloading strategy
            free_mem, total_mem = torch.cuda.mem_get_info(best_gpu)
            free_mem_gb = free_mem / (1024 ** 3)
            
            # If free memory is less than 8GB, use CPU offloading
            if free_mem_gb < 8.0:
                logger.info(
                    "musicgen.device_selection using GPU %d with CPU offloading (free_gb=%.2f)",
                    best_gpu, free_mem_gb
                )
                # Use device_map="auto" to let transformers handle CPU offloading
                return device_str, "auto"
            else:
                logger.info(
                    "musicgen.device_selection using GPU %d without CPU offloading (free_gb=%.2f)",
                    best_gpu, free_mem_gb
                )
                return device_str, None
        except Exception as e:
            logger.warning("musicgen.device_selection error checking GPU memory: %s", e)
            return device_str, None
    else:
        logger.info("musicgen.device_selection using GPU %d (CPU offloading disabled)", best_gpu)
        return device_str, None


def _resolve_model_path(model_dir: str, model_id_env: str):
    """
    Resolve a local model path for MusicGen.

    If model_dir contains files, use it as the local directory. Otherwise fall
    back to the HF repo id in MUSIC_MODEL_ID and let transformers handle it.
    """
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        return model_dir
    model_id = os.getenv("MUSIC_MODEL_ID", "").strip()
    return model_id or model_id_env


def ensure_music_engine_loaded():
    """
    Load the primary music generation engine (MusicGen or compatible) exactly
    once into module-level globals. The model object never leaves this module.
    """
    global _MUSIC_MODEL, _MUSIC_PROCESSOR, _MUSIC_DEVICE, _MUSIC_DEVICE_MAP, _ENGINE_LOADED
    if _ENGINE_LOADED and _MUSIC_MODEL is not None and _MUSIC_PROCESSOR is not None:
        return
    
    # Determine device configuration
    _MUSIC_DEVICE, _MUSIC_DEVICE_MAP = _get_device_and_map()
    
    # Resolve either a local snapshot path or an HF repo id.
    model_dir = os.getenv("MUSIC_MODEL_DIR", "/opt/models/music")
    model_path = _resolve_model_path(model_dir, os.getenv("MUSIC_MODEL_ID", "facebook/musicgen-large"))
    if not model_path:
        # stack trace at the service layer when generation is attempted.
        logger.error("musicgen.init.error missing MUSIC_MODEL_ID/MUSIC_MODEL_DIR (model_path empty)")
        _ENGINE_LOADED = False
        _MUSIC_MODEL = None
        _MUSIC_PROCESSOR = None
        return
    
    dtype = torch.float16 if _MUSIC_DEVICE.startswith("cuda") else torch.float32
    
    # Load model with device_map if CPU offloading is enabled
    if _MUSIC_DEVICE_MAP == "auto":
        logger.info("musicgen.init loading with device_map=auto for CPU offloading")
        _MUSIC_MODEL = MusicgenForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        _MUSIC_MODEL = MusicgenForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype
        )
        _MUSIC_MODEL = _MUSIC_MODEL.to(_MUSIC_DEVICE)
    
    _MUSIC_PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _ENGINE_LOADED = True
    logger.info(
        "musicgen.init model=%s device=%s device_map=%s dtype=%s",
        model_path, _MUSIC_DEVICE, _MUSIC_DEVICE_MAP, dtype
    )


def generate_music(
    prompt: str,
    seconds: int,
    seed: Optional[int],
    refs: Optional[List[str]],
):
    """
    Generate music audio using the globally loaded MusicGen-compatible engine.

    The engine (MusicgenForConditionalGeneration + AutoProcessor) is owned by
    this module and never passed around as an argument.

    Returns WAV bytes at 32 kHz.
    """
    # Caller (/generate) is responsible for validating prompt; avoid raising here.
    # Ensure engine is initialized; rely on ensure_music_engine_loaded to log any
    # configuration problems, which will surface as errors at the service layer.
    ensure_music_engine_loaded()
    model = _MUSIC_MODEL
    proc = _MUSIC_PROCESSOR
    device = _MUSIC_DEVICE
    # Basic seeding for determinism
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    # Build conditioning text; refs can be wired in later if desired.
    text = prompt.strip()
    inputs = proc(text=[text], padding=True, return_tensors="pt")
    
    # Move inputs to the appropriate device
    # With device_map="auto", inputs should go to the primary device (the GPU we selected)
    # The model will handle moving tensors between devices as needed
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    # Crude mapping: ~50 tokens/s for small/medium configs
    max_new_tokens = max(50, int(seconds) * 50)

    # Structured logging around the model.generate call.
    logger.info(
        "musicgen.request seconds=%s seed=%s device=%s max_new_tokens=%s",
        seconds,
        seed,
        device,
        max_new_tokens,
    )
    _log_music_generate_state(
        prompt=prompt,
        seconds=seconds,
        seed=seed,
        model=model,
        device=device,
        inputs=inputs,
        max_new_tokens=max_new_tokens,
    )

    t0 = time.time()
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                audio_values = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=max_new_tokens,
                )
        else:
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=max_new_tokens,
            )
    elapsed = time.time() - t0
    logger.info(
        "musicgen.generate.finished seconds=%s device=%s elapsed_s=%.3f max_new_tokens=%s",
        seconds,
        device,
        elapsed,
        max_new_tokens,
    )
    # Expect [batch, channels, samples]
    audio = audio_values[0, 0].cpu().numpy().astype(np.float32)
    sr = 32000
    # Encode to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()



