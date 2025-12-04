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
_MUSIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ENGINE_LOADED = False


def _log_music_generate_state(
    prompt: str,
    seconds: int,
    seed: Optional[int],
    model: MusicgenForConditionalGeneration,
    device: str,
    inputs: dict,
    max_new_tokens: int,
) -> None:
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
        logger.info("model.generate attr=%r", generate_attr)
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


def _resolve_model_path(model_dir: str, model_id_env: str) -> str:
    """
    Resolve a local model path for MusicGen.

    If model_dir contains files, use it as the local directory. Otherwise fall
    back to the HF repo id in MUSIC_MODEL_ID and let transformers handle it.
    """
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        return model_dir
    model_id = os.getenv("MUSIC_MODEL_ID", "").strip()
    return model_id or model_id_env


def ensure_music_engine_loaded() -> None:
    """
    Load the primary music generation engine (MusicGen or compatible) exactly
    once into module-level globals. The model object never leaves this module.
    """
    global _MUSIC_MODEL, _MUSIC_PROCESSOR, _MUSIC_DEVICE, _ENGINE_LOADED
    if _ENGINE_LOADED and _MUSIC_MODEL is not None and _MUSIC_PROCESSOR is not None:
        return
    # Resolve either a local snapshot path or an HF repo id.
    model_dir = os.getenv("MUSIC_MODEL_DIR", "/opt/models/music")
    model_path = _resolve_model_path(model_dir, os.getenv("MUSIC_MODEL_ID", "facebook/musicgen-large"))
    if not model_path:
        # Do not raise here; log a clear error and let the caller surface a full
        # stack trace at the service layer when generation is attempted.
        logger.error("musicgen.init.error missing MUSIC_MODEL_ID/MUSIC_MODEL_DIR (model_path empty)")
        _ENGINE_LOADED = False
        _MUSIC_MODEL = None
        _MUSIC_PROCESSOR = None
        return
    dtype = torch.float16 if _MUSIC_DEVICE.startswith("cuda") else torch.float32
    _MUSIC_MODEL = MusicgenForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    _MUSIC_MODEL = _MUSIC_MODEL.to(_MUSIC_DEVICE)
    _MUSIC_PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _ENGINE_LOADED = True
    logger.info("musicgen.init model=%s device=%s dtype=%s", model_path, _MUSIC_DEVICE, dtype)


def generate_music(
    prompt: str,
    seconds: int,
    seed: Optional[int],
    refs: Optional[List[str]],
) -> bytes:
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



