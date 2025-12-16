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

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "vocalfix.log")
    _lvl = getattr(logging, (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_lvl,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(_log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("vocalfix.logging").info("vocalfix logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("vocalfix.logging").warning("vocalfix file logging disabled: %s", _ex, exc_info=True)

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
        y, sr = sf.read(io.BytesIO(requests.get(wav_url).content), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), int(sr)


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


def _process_vocal(y: np.ndarray, sr: int, key: str, ops: list[str], score_json: dict | None) -> dict:
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
    return {"ok": True, "audio_wav_base64": b64, "sample_rate": sr, "metrics_before": before, "metrics_after": after}


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
        result = _process_vocal(y, sr, key, ops, score_json)
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/v1/vocal/fix")
async def vocal_fix_v1(body: dict):
    """
    Canonical API for orchestrator: audio_wav_base64 + sample_rate + optional mode/metadata.
    """
    global _device
    _device = _device or DEVICE
    audio_b64 = body.get("audio_wav_base64")
    sr = body.get("sample_rate")
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error_code": "missing_audio_wav_base64", "error_message": "audio_wav_base64 is required"},
        )
    if not isinstance(sr, int) or sr <= 0:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error_code": "missing_sample_rate", "error_message": "sample_rate must be a positive integer"},
        )
    try:
        import base64
        data = base64.b64decode(audio_b64)
        y, sr2 = sf.read(io.BytesIO(data), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        sr_eff = int(sr2 or sr)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error_code": "decode_error", "error_message": str(e)},
        )
    key = body.get("key") or os.getenv("AUDIO_KEY", "E minor")
    ops = body.get("ops") or ["pitch", "align", "deess"]
    score_json = body.get("score_json") or {}
    result = _process_vocal(y, sr_eff, key, ops, score_json if isinstance(score_json, dict) else {})
    # Preserve the canonical contract: ok + audio_wav_base64 + sample_rate + metrics.
    return result


