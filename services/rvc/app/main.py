from __future__ import annotations

import os
import json
import base64
import glob
import io
import logging
import traceback
from typing import Dict, Any, Tuple

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI(title="RVC Voice Conversion Service", version="1.1.0")

REGISTRY_PATH = os.getenv("RVC_REGISTRY_PATH", "/rvc/assets/registry.json")
os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

# Model root populated by bootstrap (blaise-tk/TITAN under /opt/models/rvc_titan)
RVC_MODEL_ROOT = os.getenv("RVC_MODEL_ROOT", "/opt/models/rvc_titan")


def ensure_rvc_weights_present() -> None:
    """
    Verify that Titan RVC weights exist under RVC_MODEL_ROOT.

    This is independent of the hf_rvc Python package; failures here indicate a
    bootstrap/volume problem, not a missing library.
    """
    if not os.path.isdir(RVC_MODEL_ROOT):
        raise RuntimeError(f"RVC_MODEL_ROOT does not exist: {RVC_MODEL_ROOT}")
    entries = os.listdir(RVC_MODEL_ROOT)
    if not entries:
        raise RuntimeError(f"RVC model directory is empty: {RVC_MODEL_ROOT}")
    logging.info("RVC weights present in %s: %s", RVC_MODEL_ROOT, entries)


# Titan HF model id for metadata/logging only; model weights live under RVC_MODEL_ROOT.
RVC_MODEL_NAME = os.getenv("RVC_MODEL_ID", "blaise-tk/TITAN")

# hf-rvc is a Python package installed into the rvc_service image. If it is
# missing, this is an image/build problem, not a Titan bootstrap problem.
try:
    from hf_rvc import RVCFeatureExtractor, RVCModel  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover - hard startup failure
    raise RuntimeError(
        "hf_rvc Python package is not installed in the rvc_service container. "
        "Ensure the rvc_service image includes:\n"
        "    pip install git+https://github.com/esnya/hf-rvc.git#egg=hf-rvc"
    ) from e

# Ensure Titan weights are present before attempting to load the engine.
ensure_rvc_weights_present()

# Prefer loading Titan from the local snapshot under RVC_MODEL_ROOT so we do not
# depend on network access at runtime. Fall back to the HF model id if needed.
_model_target = RVC_MODEL_ROOT if os.path.isdir(RVC_MODEL_ROOT) else RVC_MODEL_NAME
try:
    FEATURE_EXTRACTOR = RVCFeatureExtractor.from_pretrained(_model_target)
    RVC_MODEL = RVCModel.from_pretrained(_model_target)
    FEATURE_EXTRACTOR.set_f0_method("rmvpe")
    RVC_MODEL.eval()
except Exception as ex:  # pragma: no cover - hard startup failure
    raise RuntimeError(
        f"Failed to load RVC Titan model from '{_model_target}'. "
        f"Check that Titan has been bootstrapped into {RVC_MODEL_ROOT} and that hf_rvc is compatible. "
        f"Underlying error: {ex}"
    ) from ex


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
        return {
            "ok": False,
            "error": {
                "code": "missing_voice_lock_id",
                "message": "voice_lock_id is required",
                "status": 400,
            },
        }
    reg = _load_registry()
    ref_b64 = body.get("reference_wav_b64")
    ref_path = body.get("reference_wav_path")
    if ref_b64:
        # Persist reference audio under a deterministic path.
        try:
            data = base64.b64decode(ref_b64)
        except Exception as ex:
            return {
                "ok": False,
                "error": {
                    "code": "invalid_b64",
                    "message": str(ex),
                    "status": 400,
                    "type": ex.__class__.__name__,
                    "traceback": traceback.format_exc(),
                },
            }
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
    audio, sr = sf.read(io.BytesIO(data), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    return audio, int(sr)


def _encode_mono_float32_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """
    Encode mono float32 PCM into WAV bytes.
    """
    buf = io.BytesIO()
    sf.write(buf, y, int(sr), format="WAV")
    return buf.getvalue()


def _run_rvc_convert(src_y: np.ndarray, src_sr: int, ref_y: np.ndarray, ref_sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Titan-backed RVC conversion using hf_rvc's feature extractor and model.

    This loads content features for both source and reference and calls the
    generic convert() interface on the Titan RVC model.
    """
    if not isinstance(src_y, np.ndarray) or src_y.size == 0:
        raise ValueError("rvc_convert: empty source audio")
    if not isinstance(ref_y, np.ndarray) or ref_y.size == 0:
        raise ValueError("rvc_convert: empty reference audio")
    # Build features for source and reference using Titan's feature extractor.
    src_feats = FEATURE_EXTRACTOR(audio=src_y, sampling_rate=src_sr, return_tensors="pt")
    ref_feats = FEATURE_EXTRACTOR(audio=ref_y, sampling_rate=ref_sr, return_tensors="pt")
    # Apply optional transpose/f0_method from registry if present.
    transpose = int(cfg.get("transpose") or 0) if isinstance(cfg, dict) else 0
    f0_method = cfg.get("f0_method") if isinstance(cfg, dict) else "rmvpe"
    converted = RVC_MODEL.convert(
        source_features=src_feats,
        reference_features=ref_feats,
        transpose=transpose,
        f0_method=f0_method or "rmvpe",
    )
    # converted is expected to be a 1D float tensor; convert to numpy.
    if hasattr(converted, "detach"):
        return converted.detach().cpu().numpy().astype("float32")
    return np.asarray(converted, dtype=np.float32)


@app.post("/v1/audio/convert")
async def convert_v1(body: Dict[str, Any]):
    voice_lock_id = (body.get("voice_lock_id") or "").strip()
    if not voice_lock_id:
        return {
            "ok": False,
            "error": {
                "code": "missing_voice_lock_id",
                "message": "voice_lock_id is required",
                "status": 400,
            },
        }
    # Accept both source_wav_base64 (preferred) and legacy source_wav_b64.
    src_b64 = body.get("source_wav_base64") or body.get("source_wav_b64")
    if not isinstance(src_b64, str) or not src_b64.strip():
        return {
            "ok": False,
            "error": {
                "code": "missing_source_wav_b64",
                "message": "source_wav_b64 is required",
                "status": 400,
            },
        }
    reg = _load_registry()
    if voice_lock_id not in reg:
        return {
            "ok": False,
            "error": {
                "code": "unknown_voice_lock_id",
                "message": f"no registry entry for {voice_lock_id}",
                "status": 404,
            },
        }
    try:
        wav_bytes = _decode_wav_from_b64(src_b64)
    except ValueError as ex:
        return {
            "ok": False,
            "error": {
                "code": "invalid_b64",
                "message": str(ex),
                "status": 400,
                "type": ex.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        }
    # Decode source audio to mono float32.
    try:
        src_y, src_sr = _decode_wav_to_mono_float32(wav_bytes)
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": "decode_error",
                "message": str(ex),
                "status": 400,
                "type": ex.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        }
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
            return {
                "ok": False,
                "error": {
                    "code": "ref_decode_error",
                    "message": str(ex),
                    "status": 400,
                    "type": ex.__class__.__name__,
                    "traceback": traceback.format_exc(),
                },
            }
    else:
        # For RVC cloning to be meaningful, a reference voice must be present.
        # Treat missing or non-existent reference paths as a hard, explicit error
        # instead of silently falling back to using the source as its own reference.
        return {
            "ok": False,
            "error": {
                "code": "rvc_reference_missing",
                "message": f"No reference_wav_path configured or file not found for voice_lock_id '{voice_lock_id}'.",
                "status": 400,
            },
        }
    try:
        converted_np = _run_rvc_convert(src_y, src_sr, ref_y, ref_sr, ref_cfg if isinstance(ref_cfg, dict) else {})
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": "rvc_inference_error",
                "message": str(ex),
                "status": 500,
                "type": ex.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        }
    # Encode back to WAV bytes at the desired sample rate.
    out_sr = int(body.get("sample_rate") or ref_cfg.get("sample_rate") or src_sr or 32000)
    try:
        out_wav_bytes = _encode_mono_float32_to_wav_bytes(converted_np, out_sr)
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": "encode_error",
                "message": str(ex),
                "status": 500,
                "type": ex.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        }
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
        return {
            "ok": False,
            "error": {
                "code": "missing_wav_path",
                "message": "wav_path is required",
                "status": 400,
            },
        }
    if not voice_lock_id:
        return {
            "ok": False,
            "error": {
                "code": "missing_voice_lock_id",
                "message": "voice_lock_id is required",
                "status": 400,
            },
        }
    if not os.path.exists(wav_path):
        return {
            "ok": False,
            "error": {
                "code": "wav_path_not_found",
                "message": f"wav_path not found: {wav_path}",
                "status": 404,
            },
        }
    with open(wav_path, "rb") as f:
        src = f.read()
    b64 = base64.b64encode(src).decode("ascii")
    res = await convert_v1({"source_wav_b64": b64, "voice_lock_id": voice_lock_id})
    if not isinstance(res, dict) or not res.get("ok"):
        return {
            "ok": False,
            "error": {
                "code": "conversion_failed",
                "detail": res,
                "status": 500,
            },
        }
    out_bytes = base64.b64decode(res.get("audio_wav_base64") or "")
    v_dir, v_name = os.path.dirname(wav_path), os.path.basename(wav_path)
    v_base, v_ext = os.path.splitext(v_name)
    out_path = os.path.join(v_dir, f"{v_base}.rvc{v_ext or '.wav'}")
    with open(out_path, "wb") as f:
        f.write(out_bytes)
    return {"ok": True, "path": out_path}
