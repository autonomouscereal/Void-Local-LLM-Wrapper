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

import uuid

from void_envelopes import ToolEnvelope
import traceback

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
    logging.getLogger(__name__).info("vocalfix logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger(__name__).warning("vocalfix file logging disabled: %s", _ex, exc_info=True)

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


def _read_wav_payload(wav_url: str | None, wav_bytes_b64: str | None):
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


def lufs(y: np.ndarray, sr: int):
    return pl.Meter(sr).integrated_loudness(y)


def pitch_correct(y: np.ndarray, sr: int, key: str = "E minor", strength: float = 0.6):
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


def time_align(y: np.ndarray, sr: int, score_notes: list[dict]):
    """
    Time-align audio to a target score note timeline.

    Best current implementation (no new deps): global time-stretch to match total
    score duration using librosa phase vocoder. This preserves pitch (approx) and
    improves downstream per-note alignment compared to a no-op.
    """
    aligned = y
    have_target = False
    bad_input = False
    stretch_exception = False
    # Log everything; never silently no-op.
    logging.getLogger(__name__).info(
        "vocalfix.time_align start sr=%s y_len=%s notes=%s",
        int(sr) if isinstance(sr, int) else sr,
        int(len(y)) if isinstance(y, np.ndarray) else None,
        int(len(score_notes)) if isinstance(score_notes, list) else None,
    )
    target_dur = None
    if isinstance(score_notes, list) and score_notes:
        starts = []
        ends = []
        dur_sum = 0.0
        has_any_time = False
        for n in score_notes:
            if not isinstance(n, dict):
                continue
            # Common fields: start_s/end_s or t_start/t_end or duration_s
            s0 = n.get("start_s")
            e0 = n.get("end_s")
            if s0 is None:
                s0 = n.get("t_start")
            if e0 is None:
                e0 = n.get("t_end")
            if isinstance(s0, (int, float)) and isinstance(e0, (int, float)):
                has_any_time = True
                starts.append(float(s0))
                ends.append(float(e0))
            d = n.get("duration_s") or n.get("dur_s")
            if isinstance(d, (int, float)) and float(d) > 0:
                dur_sum += float(d)
        if has_any_time and starts and ends:
            smin = min(starts)
            emax = max(ends)
            if isinstance(smin, (int, float)) and isinstance(emax, (int, float)) and float(emax) > float(smin):
                target_dur = float(emax) - float(smin)
        if target_dur is None and dur_sum > 0:
            target_dur = float(dur_sum)
    if not isinstance(target_dur, (int, float)) or float(target_dur) <= 0:
        logging.getLogger(__name__).warning(
            "vocalfix.time_align no_target_duration (notes missing timing/durations) notes=%s",
            score_notes,
        )
        bad_input = True
    if not isinstance(sr, int) or sr <= 0 or not isinstance(y, np.ndarray) or y.size <= 0:
        logging.getLogger(__name__).warning(
            "vocalfix.time_align invalid_audio sr=%r y_type=%s y_size=%s target_dur=%s",
            sr,
            type(y).__name__,
            int(y.size) if isinstance(y, np.ndarray) else None,
            target_dur,
        )
        bad_input = True
    cur_dur = float(len(y)) / float(sr) if sr > 0 else 0.0
    if cur_dur <= 0.0:
        logging.getLogger(__name__).warning("vocalfix.time_align zero_duration cur_dur=%s target_dur=%s", cur_dur, target_dur)
        bad_input = True
    # librosa time_stretch rate: >1 speeds up (shorter); <1 slows down (longer)
    have_target = (not bad_input) and isinstance(target_dur, (int, float)) and float(target_dur) > 0
    rate = (float(cur_dur) / float(target_dur)) if have_target else 1.0
    # Guard against absurd stretch ratios.
    if not (0.1 <= rate <= 10.0):
        logging.getLogger(__name__).warning(
            "vocalfix.time_align rate_out_of_range cur_dur=%s target_dur=%s rate=%s",
            cur_dur,
            target_dur,
            rate,
        )
        bad_input = True
    if (not bad_input) and have_target:
        try:
            aligned = librosa.effects.time_stretch(y.astype(np.float32), rate=rate).astype(np.float32)
            out_dur = float(len(aligned)) / float(sr) if sr > 0 else 0.0
            logging.getLogger(__name__).info(
                "vocalfix.time_align done cur_dur=%.4f target_dur=%.4f out_dur=%.4f rate=%.6f",
                cur_dur,
                float(target_dur),
                out_dur,
                rate,
            )
        except Exception:
            logging.getLogger(__name__).error("vocalfix.time_align exception", exc_info=True)
            stretch_exception = True
            aligned = y
    return aligned


def de_ess(y: np.ndarray, sr: int, thresh_db: float = -16.0):
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


def _process_vocal(y: np.ndarray, sr: int, key: str, ops: list[str], score_json: dict | None):
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
    trace_id = str(body.get("trace_id") or "").strip() if isinstance(body.get("trace_id"), (str, int)) else ""
    conversation_id = str(body.get("conversation_id") or "").strip() if isinstance(body.get("conversation_id"), (str, int)) else ""
    audio_b64 = body.get("audio_wav_base64")
    sr = body.get("sample_rate")
    resp = None
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        resp = ToolEnvelope.failure(
            code="missing_audio_wav_base64",
            message="audio_wav_base64 is required",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=400,
            details={"trace_id": trace_id, "conversation_id": conversation_id, "body_keys": sorted([str(k) for k in (body or {}).keys()])},
        )
    if resp is None and (not isinstance(sr, int) or sr <= 0):
        resp = ToolEnvelope.failure(
            code="missing_sample_rate",
            message="sample_rate must be a positive integer",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=400,
            details={"trace_id": trace_id, "conversation_id": conversation_id, "sample_rate": sr},
        )
    y = None
    sr_eff = None
    if resp is None:
        try:
            import base64
            data = base64.b64decode(audio_b64)
            y2, sr2 = sf.read(io.BytesIO(data), always_2d=False)
            if y2.ndim > 1:
                y2 = y2.mean(axis=1)
            y = y2.astype(np.float32)
            sr_eff = int(sr2 or sr)
        except Exception as e:
            resp = ToolEnvelope.failure(
                code="decode_error",
                message=str(e),
                trace_id=trace_id,
                conversation_id=conversation_id,
                status=400,
                details={"trace_id": trace_id, "conversation_id": conversation_id, "stack": traceback.format_exc()},
            )
    key = body.get("key") or os.getenv("AUDIO_KEY", "E minor")
    ops = body.get("ops") or ["pitch", "align", "deess"]
    score_json = body.get("score_json") or {}
    result = None
    if resp is None:
        try:
            result = _process_vocal(y, int(sr_eff or sr), key, ops, score_json if isinstance(score_json, dict) else {})
        except Exception as e:
            resp = ToolEnvelope.failure(
                code="process_error",
                message=str(e),
                trace_id=trace_id,
                conversation_id=conversation_id,
                status=500,
                details={"trace_id": trace_id, "conversation_id": conversation_id, "stack": traceback.format_exc()},
            )
    if resp is None and (not isinstance(result, dict) or not bool(result.get("ok"))):
        resp = ToolEnvelope.failure(
            code="vocalfix_failed",
            message="vocalfix returned a non-ok result",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=500,
            details={"trace_id": trace_id, "conversation_id": conversation_id, "raw": result},
        )
    if resp is None:
        # Return canonical ToolEnvelope: ok/result/error (HTTP 200 always).
        result_payload = {
            "audio_wav_base64": result.get("audio_wav_base64") if isinstance(result, dict) else None,
            "sample_rate": result.get("sample_rate") if isinstance(result, dict) else (int(sr_eff or sr) if isinstance(sr, int) else None),
            "metrics_before": result.get("metrics_before") if isinstance(result, dict) else None,
            "metrics_after": result.get("metrics_after") if isinstance(result, dict) else None,
            "trace_id": trace_id,
            "conversation_id": conversation_id,
        }
        resp = ToolEnvelope.success(result=result_payload, trace_id=trace_id, conversation_id=conversation_id)
    return resp


