from __future__ import annotations

import io, os, sys, json
import numpy as np
import soundfile as sf
import torch
import pyloudnorm as pl
import librosa
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s:%(name)s:%(message)s")

try:
    import crepe  # type: ignore
except Exception:
    crepe = None


# --- Device selection (per script, no helpers, no gating) ---
try:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        print("[warn] vocalfix: CUDA not available; using CPU", file=sys.stderr, flush=True)
except Exception as _e:
    DEVICE = "cpu"
    print(f"[warn] vocalfix: torch check failed ({_e}); using CPU", file=sys.stderr, flush=True)
USE_FP16 = (os.environ.get("USE_FP16", "0") == "1") and DEVICE.startswith("cuda")
# --- end device selection ---


def _read_wav_payload(wav_url: str | None, wav_bytes_b64: str | None) -> tuple[np.ndarray, int]:
    if wav_bytes_b64:
        import base64
        data = base64.b64decode(wav_bytes_b64)
        y, sr = sf.read(io.BytesIO(data), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), int(sr)
    if wav_url:
        import requests
        y, sr = sf.read(io.BytesIO(requests.get(wav_url, timeout=30).content), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), int(sr)
    raise ValueError("missing wav_url or wav_bytes")


def lufs(y: np.ndarray, sr: int) -> float:
    return pl.Meter(sr).integrated_loudness(y)


def pitch_correct(y: np.ndarray, sr: int, key: str = "E minor", strength: float = 0.6) -> np.ndarray:
    if crepe is None:
        return y
    try:
        _, f0, _, _ = crepe.predict(y, sr, viterbi=True, verbose=0)
        cents = 1200.0 * np.log2(np.maximum(f0, 1e-6) / 440.0)
        cents_q = np.round(cents / 100.0) * 100.0
        cents_blend = (1.0 - strength) * cents + strength * cents_q
        _ = 440.0 * 2 ** (cents_blend / 1200.0)
        # Placeholder: return unchanged; integrate PSOLA/rubberband for actual shift
        return y
    except Exception:
        return y


def time_align(y: np.ndarray, sr: int, score_notes: list[dict]) -> np.ndarray:
    # Placeholder: no-op; integrate DTW-based stretch per-note
    return y


def de_ess(y: np.ndarray, sr: int, thresh_db: float = -16.0) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = (freqs >= 5000) & (freqs <= 9000)
    sibil = S[band].mean() if np.any(band) else 0.0
    if 20 * np.log10(float(sibil) + 1e-9) > thresh_db:
        S[band] *= 0.75
    y_fix = librosa.istft(S, hop_length=256)
    return y_fix.astype(np.float32)


app = FastAPI(title="Vocal Fixer", version="0.1.0")
_device: str | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/vocal_fix")
async def vocal_fix(body: dict):
    global _device
    _device = _device or DEVICE
    try:
        wav_url = body.get("wav_url")
        wav_bytes = body.get("wav_bytes")
        score_json = body.get("score_json") or {}
        key = body.get("key") or os.getenv("AUDIO_KEY", "E minor")
        ops = body.get("ops") or ["pitch", "align", "deess"]
        y, sr = _read_wav_payload(wav_url, wav_bytes)
        before = {"lufs": lufs(y, sr)}
        if "pitch" in ops:
            y = pitch_correct(y, sr, key=key, strength=0.6)
        if "align" in ops and isinstance(score_json, dict):
            y = time_align(y, sr, score_json.get("notes", []))
        if "deess" in ops:
            y = de_ess(y, sr, float(os.getenv("AUDIO_SIBILANCE_DB_MAX", "-16")))
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        import base64
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        after = {"lufs": lufs(y, sr)}
        return {"ok": True, "audio_wav_base64": b64, "metrics_before": before, "metrics_after": after}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


