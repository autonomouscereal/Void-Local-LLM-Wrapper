from __future__ import annotations

import os
import sys
import json
import base64
from typing import Dict, Any, Tuple

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI(title="RVC Voice Conversion Service", version="1.0.0")

REGISTRY_PATH = os.getenv("RVC_REGISTRY_PATH", "/rvc/assets/registry.json")
os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

# Model root populated by bootstrap (blaise-tk/TITAN under /opt/models/rvc_titan)
RVC_MODEL_ROOT = os.getenv("RVC_MODEL_ROOT", "/opt/models/rvc_titan")
if not (os.path.isdir(RVC_MODEL_ROOT) and os.listdir(RVC_MODEL_ROOT)):
    raise RuntimeError(f"RVC model directory missing or empty: {RVC_MODEL_ROOT}")

# Titan HF model id for metadata/logging.
RVC_MODEL_NAME = "blaise-tk/TITAN"

# Load the Titan RVC implementation directly from the bootstrapped snapshot.
# We do NOT rely on any external 'hf-rvc' wheel from PyPI; instead, we import
# the hf_rvc package from the local RVC_MODEL_ROOT where bootstrap downloaded
# the blaise-tk/TITAN repository.
if RVC_MODEL_ROOT not in sys.path:
    sys.path.insert(0, RVC_MODEL_ROOT)
try:
    from hf_rvc import RVCFeatureExtractor, RVCModel  # type: ignore
except ImportError as ex:  # pragma: no cover - hard startup failure
    raise RuntimeError(
        "hf_rvc package not found under RVC_MODEL_ROOT; ensure blaise-tk/TITAN "
        "has been bootstrapped into /opt/models/rvc_titan and that the repo "
        "contains the hf_rvc package."
    ) from ex

# Load feature extractor and model once at import time. Any failure here should
# cause the container to fail fast so orchestrator startup will also fail.
FEATURE_EXTRACTOR = RVCFeatureExtractor.from_pretrained(RVC_MODEL_ROOT)
RVC_MODEL = RVCModel.from_pretrained(RVC_MODEL_ROOT)
FEATURE_EXTRACTOR.set_f0_method("rmvpe")
RVC_MODEL.eval()


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_registry(reg: Dict[str, Any]) -> None:
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, REGISTRY_PATH)


@app.get("/healthz")
async def healthz():
    reg = _load_registry()
    return {"status": "ok", "voices": list(reg.keys())}


@app.post("/v1/voice/register")
async def register_voice(body: Dict[str, Any]):
    voice_lock_id = (body.get("voice_lock_id") or "").strip()
    if not voice_lock_id:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": {"code": "missing_voice_lock_id", "message": "voice_lock_id is required"},
            },
        )
    reg = _load_registry()
    ref_b64 = body.get("reference_wav_b64")
    ref_path = body.get("reference_wav_path")
    if ref_b64:
        # Persist reference audio under a deterministic path.
        try:
            data = base64.b64decode(ref_b64)
        except Exception as ex:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": {"code": "invalid_b64", "message": str(ex)},
                },
            )
        voices_root = os.getenv("RVC_VOICES_ROOT", "/rvc/assets/voices")
        os.makedirs(voices_root, exist_ok=True)
        out_path = os.path.join(voices_root, f"{voice_lock_id}.wav")
        with open(out_path, "wb") as f:
            f.write(data)
        ref_path = out_path
    if not isinstance(ref_path, str) or not ref_path.strip():
        # Allow placeholder registration without an explicit reference path.
        ref_path = ""
    reg[voice_lock_id] = {
        "model_type": "rvc_titan",
        "model_name": body.get("model_name") or "blaise-tk/TITAN",
        "reference_wav_path": ref_path,
        "f0_method": body.get("f0_method") or "rmvpe",
        "transpose": int(body.get("transpose") or 0),
        "sample_rate": int(body.get("sample_rate") or 32000),
    }
    _save_registry(reg)
    return {"ok": True, "voice_lock_id": voice_lock_id}


def _decode_wav_from_b64(b64: str) -> bytes:
    try:
        return base64.b64decode(b64)
    except Exception as ex:
        raise ValueError(f"invalid base64 audio: {ex}") from ex


def _decode_wav_to_mono_float32(data: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode WAV bytes into mono float32 PCM and return (audio, sample_rate).
    """
    audio, sr = sf.read(io.BytesIO(data), always_2d=False)  # type: ignore[name-defined]
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    return audio, int(sr)


def _encode_mono_float32_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """
    Encode mono float32 PCM into WAV bytes.
    """
    buf = io.BytesIO()  # type: ignore[name-defined]
    sf.write(buf, y, int(sr), format="WAV")
    return buf.getvalue()


@app.post("/v1/audio/convert")
async def convert_v1(body: Dict[str, Any]):
    voice_lock_id = (body.get("voice_lock_id") or "").strip()
    if not voice_lock_id:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": {"code": "missing_voice_lock_id", "message": "voice_lock_id is required"},
            },
        )
    # Accept both source_wav_base64 (preferred) and legacy source_wav_b64.
    src_b64 = body.get("source_wav_base64") or body.get("source_wav_b64")
    if not isinstance(src_b64, str) or not src_b64.strip():
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": {"code": "missing_source_wav_b64", "message": "source_wav_b64 is required"},
            },
        )
    reg = _load_registry()
    if voice_lock_id not in reg:
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "error": {"code": "unknown_voice_lock_id", "message": f"no registry entry for {voice_lock_id}"},
            },
        )
    try:
        wav_bytes = _decode_wav_from_b64(src_b64)
    except ValueError as ex:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"code": "invalid_b64", "message": str(ex)}},
        )
    # Decode source audio to mono float32.
    try:
        src_y, src_sr = _decode_wav_to_mono_float32(wav_bytes)
    except Exception as ex:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"code": "decode_error", "message": str(ex)}},
        )
    # Load reference voice from registry when available.
    ref_cfg = reg.get(voice_lock_id) or {}
    ref_path = ref_cfg.get("reference_wav_path") if isinstance(ref_cfg, dict) else None
    ref_y: np.ndarray
    ref_sr: int
    if isinstance(ref_path, str) and ref_path.strip() and os.path.exists(ref_path):
        try:
            ref_pcm, ref_sr = sf.read(ref_path, always_2d=False)
            if getattr(ref_pcm, "ndim", 1) > 1:
                ref_pcm = ref_pcm.mean(axis=1)
            ref_y = np.asarray(ref_pcm, dtype=np.float32)
        except Exception as ex:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": {"code": "ref_decode_error", "message": str(ex)},
                },
            )
    else:
        # Fallback: use source as its own reference when no explicit reference path is set.
        ref_y, ref_sr = src_y, src_sr
    # Build features for source and reference using hf-rvc feature extractor.
    try:
        src_feats = FEATURE_EXTRACTOR(audio=src_y, sampling_rate=src_sr, return_tensors="pt")
        ref_feats = FEATURE_EXTRACTOR(audio=ref_y, sampling_rate=ref_sr, return_tensors="pt")
        # Apply optional transpose/f0_method from registry if present.
        transpose = int(ref_cfg.get("transpose") or 0) if isinstance(ref_cfg, dict) else 0
        f0_method = ref_cfg.get("f0_method") if isinstance(ref_cfg, dict) else "rmvpe"
        # hf-rvc API: use the generic convert interface; details may need adjustment
        # based on the installed hf-rvc version.
        converted = RVC_MODEL.convert(
            source_features=src_feats,
            reference_features=ref_feats,
            transpose=transpose,
            f0_method=f0_method or "rmvpe",
        )
        # converted is expected to be a 1D float tensor; convert to numpy.
        if hasattr(converted, "detach"):
            converted_np = converted.detach().cpu().numpy().astype("float32")
        else:
            converted_np = np.asarray(converted, dtype=np.float32)
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "rvc_inference_error",
                    "message": str(ex),
                },
            },
        )
    # Encode back to WAV bytes at the desired sample rate.
    out_sr = int(body.get("sample_rate") or ref_cfg.get("sample_rate") or src_sr or 32000)
    try:
        out_wav_bytes = _encode_mono_float32_to_wav_bytes(converted_np, out_sr)
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {"code": "encode_error", "message": str(ex)},
            },
        )
    out_b64 = base64.b64encode(out_wav_bytes).decode("ascii")
    return {
        "ok": True,
        "audio_wav_base64": out_b64,
        "sample_rate": out_sr,
    }


@app.post("/convert_path")
async def convert_path(body: Dict[str, Any]):
    """
    Backwards-compatible path-based API used by older callers.
    Reads wav_path, converts it internally using the same engine as /v1/audio/convert,
    and returns a new wav path on disk.
    """
    wav_path = body.get("wav_path")
    voice_lock_id = (body.get("voice_lock_id") or "").strip()
    if not isinstance(wav_path, str) or not wav_path.strip():
        return JSONResponse(status_code=400, content={"error": "missing wav_path"})
    if not voice_lock_id:
        return JSONResponse(status_code=400, content={"error": "missing voice_lock_id"})
    if not os.path.exists(wav_path):
        return JSONResponse(status_code=404, content={"error": f"wav_path not found: {wav_path}"})
    with open(wav_path, "rb") as f:
        src = f.read()
    b64 = base64.b64encode(src).decode("ascii")
    res = await convert_v1({"source_wav_b64": b64, "voice_lock_id": voice_lock_id})
    # If convert_v1 returned a JSONResponse error, surface it directly.
    if isinstance(res, JSONResponse):
        return res
    if not isinstance(res, dict) or not res.get("ok"):
        return JSONResponse(status_code=500, content={"error": "conversion_failed", "detail": res})
    out_bytes = base64.b64decode(res.get("audio_wav_base64") or "")
    v_dir, v_name = os.path.dirname(wav_path), os.path.basename(wav_path)
    v_base, v_ext = os.path.splitext(v_name)
    out_path = os.path.join(v_dir, f"{v_base}.rvc{v_ext or '.wav'}")
    with open(out_path, "wb") as f:
        f.write(out_bytes)
    return {"ok": True, "path": out_path}
