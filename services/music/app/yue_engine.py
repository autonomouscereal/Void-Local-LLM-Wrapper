import os
from typing import Optional, List


def load_yue(model_dir: str, device: str = "cpu"):
    # TODO: replace with real YuE loading
    if not (os.path.isdir(model_dir) and os.listdir(model_dir)):
        raise RuntimeError(f"YuE model directory missing or empty: {model_dir}")
    return object()


def generate_song(model, prompt: str, seconds: int, seed: Optional[int], refs: Optional[List[str]], device: str = "cpu") -> bytes:
    # TODO: replace with real YuE inference
    # Return 1 second of silence at 32kHz as placeholder WAV bytes
    import io
    import numpy as np
    import soundfile as sf
    sr = 32000
    dur = max(1, int(seconds))
    wav = np.zeros(sr * dur, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


