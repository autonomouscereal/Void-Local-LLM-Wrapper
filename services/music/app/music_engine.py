import os
from typing import Optional, List

import io
import torch  # type: ignore
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore


_MUSIC_MODEL: Optional[MusicgenForConditionalGeneration] = None
_MUSIC_PROCESSOR = None
_MUSIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ENGINE_LOADED = False


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
        raise RuntimeError("MUSIC_MODEL_ID or MUSIC_MODEL_DIR must be set for music engine")
    dtype = torch.float16 if _MUSIC_DEVICE.startswith("cuda") else torch.float32
    _MUSIC_MODEL = MusicgenForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    _MUSIC_MODEL = _MUSIC_MODEL.to(_MUSIC_DEVICE)
    _MUSIC_PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _ENGINE_LOADED = True


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
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required for generate_music")
    # Ensure engine is initialized.
    ensure_music_engine_loaded()
    if _MUSIC_MODEL is None or _MUSIC_PROCESSOR is None:
        raise RuntimeError("Music engine not initialized")
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
    # Expect [batch, channels, samples]
    audio = audio_values[0, 0].cpu().numpy().astype(np.float32)
    sr = 32000
    # Encode to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()



