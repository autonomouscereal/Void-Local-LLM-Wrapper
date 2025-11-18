import os
from typing import Optional, List

import torch  # type: ignore
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore


_MODEL = None
_PROC = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_music_engine(model_dir: str, device: str = "cpu") -> MusicgenForConditionalGeneration:
    """
    Load the primary music generation engine (MusicGen or compatible) from a
    local directory populated by bootstrap, falling back to HF repo id if
    needed.
    """
    global _MODEL, _PROC, _DEVICE
    if _MODEL is not None and _PROC is not None:
        return _MODEL
    _DEVICE = device or _DEVICE
    # Resolve either a local snapshot path or a HF repo id.
    model_path = _resolve_model_path(model_dir, os.getenv("MUSIC_MODEL_ID", "facebook/musicgen-large"))
    if not model_path:
        raise RuntimeError("MUSIC_MODEL_ID or MUSIC_MODEL_DIR must be set for music engine")
    dtype = torch.float16 if _DEVICE.startswith("cuda") else torch.float32
    _MODEL = MusicgenForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    _MODEL = _MODEL.to(_DEVICE)
    _PROC = AutoProcessor.from_pretrained(model_path)
    return _MODEL


def generate_music(
    model,
    prompt: str,
    seconds: int,
    seed: Optional[int],
    refs: Optional[List[str]],
    device: str = "cpu",
) -> bytes:
    """
    Generate music audio using the loaded MusicGen-compatible model.

    The engine is a transformers MusicgenForConditionalGeneration loaded from
    facebook/musicgen-* (or equivalent). Some callers historically passed the
    (model, processor) tuple returned by load_music_engine; to be robust we
    defensively unwrap that here and always operate on the underlying model
    instance that exposes .generate(...).

    Returns WAV bytes at 32 kHz.
    """
    if not prompt:
        raise ValueError("prompt is required for generate_music")
    # Resolve the effective engine: prefer the global _MODEL if initialized,
    # otherwise fall back to the passed-in model. In all cases, unwrap tuples
    # so we never call .generate on (model, processor) or similar.
    global _MODEL
    engine = _MODEL if _MODEL is not None else model
    if isinstance(engine, (tuple, list)):
        try:
            engine = engine[0]
        except Exception:
            raise RuntimeError(f"generate_music: unexpected tuple engine type {type(engine)}")
    if not hasattr(engine, "generate"):
        raise RuntimeError(f"generate_music: music engine has no .generate attribute (type={type(engine)})")
    proc = _PROC
    if proc is None:
        raise RuntimeError("Music processor not initialized")
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
                audio_values = engine.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=max_new_tokens,
                )
        else:
            audio_values = engine.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=max_new_tokens,
            )
    # Expect [batch, channels, samples]
    audio = audio_values[0, 0].cpu().numpy().astype(np.float32)
    sr = 32000
    # Encode to WAV bytes
    import io

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()



