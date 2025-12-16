from __future__ import annotations

# Import marker: if you don't see this in container logs, you're not running this file.
print("RVC_SERVICE_MARKER: workspace main.py 2025-11-24-b")

import os
import json
import base64
import glob
import io
import logging
import traceback
import sys
from typing import Dict, Any, Tuple, List

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore
import httpx  # type: ignore
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Import voice miner
from . import voice_miner


app = FastAPI(title="RVC Voice Conversion Service", version="1.2.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "rvc.log")
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
    logging.getLogger("rvc.logging").info("rvc logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("rvc.logging").warning("rvc file logging disabled: %s", _ex, exc_info=True)

REGISTRY_PATH = os.getenv("RVC_REGISTRY_PATH", "/rvc/assets/registry.json")
os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

# Model root populated by bootstrap (blaise-tk/TITAN under /opt/models/rvc_titan)
RVC_MODEL_ROOT = os.getenv("RVC_MODEL_ROOT", "/opt/models/rvc_titan")

# RVC models root (shared with rvc_python and rvc_trainer)
RVC_MODELS_ROOT = os.getenv("RVC_MODELS_ROOT", "/srv/rvc_models")

# External rvc-python engine base URL
RVC_ENGINE_BASE = os.getenv("RVC_ENGINE_BASE", "http://127.0.0.1:5050")

# External rvc trainer base URL
RVC_TRAINER_BASE = os.getenv("RVC_TRAINER_BASE", "http://127.0.0.1:7070")

# Global state for rvc-python engine
_ENGINE_DEVICE_SET = False
_LOADED_MODELS: Dict[str, bool] = {}


def ensure_rvc_weights_present() -> None:
    """
    Verify that Titan RVC weights exist under RVC_MODEL_ROOT.

    Failures here indicate a bootstrap/volume problem (missing Titan weights),
    not a missing Python library.
    """
    if not os.path.isdir(RVC_MODEL_ROOT):
        logging.error("RVC weights missing: RVC_MODEL_ROOT does not exist: %s", RVC_MODEL_ROOT)
        return
    entries = os.listdir(RVC_MODEL_ROOT)
    if not entries:
        logging.error("RVC weights missing: model directory is empty: %s", RVC_MODEL_ROOT)
        return
    logging.info("RVC weights present in %s: %s", RVC_MODEL_ROOT, entries)


@app.on_event("startup")
async def _check_rvc_weights_on_startup() -> None:
    """
    On service startup, verify that Titan RVC weights are present under RVC_MODEL_ROOT.
    Any failure is logged with a full stack trace; callers should rely on runtime
    error envelopes from endpoints rather than process-level crashes.
    """
    try:
        ensure_rvc_weights_present()
    except Exception:
        logging.error("RVC weights check failed:\n%s", traceback.format_exc())


def ensure_rvc_engine_loaded(cfg: Dict[str, Any]) -> None:
    """
    Ensure the external rvc-python engine is ready for the requested voice/model.

    - Sets the device once (cuda:0 or cpu) via /set_device.
    - Loads the requested model via /models/{model_name} once per model.
    """
    global _ENGINE_DEVICE_SET, _LOADED_MODELS

    model_name = (cfg.get("model_name") or "").strip()
    if not model_name:
        # Let callers surface this as a structured ValidationError instead of
        # raising from the helper.
        logging.error("rvc_engine: cfg['model_name'] missing or empty in ensure_rvc_engine_loaded")
        return

    base = RVC_ENGINE_BASE.rstrip("/")

    if not _ENGINE_DEVICE_SET:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        resp = httpx.post(f"{base}/set_device", json={"device": device}, timeout=None)
        if resp.status_code < 200 or resp.status_code >= 300:
            logging.error("rvc_engine: /set_device failed: %s %s", resp.status_code, resp.text)
            return
        _ENGINE_DEVICE_SET = True

    if model_name in _LOADED_MODELS and _LOADED_MODELS[model_name]:
        return

    resp = httpx.post(f"{base}/models/{model_name}", timeout=None)
    if resp.status_code < 200 or resp.status_code >= 300:
        logging.error("rvc_engine: /models/%s failed: %s %s", model_name, resp.status_code, resp.text)
        return

    _LOADED_MODELS[model_name] = True


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
                "code": "ValidationError",
                "message": "voice_lock_id is required",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    model_name = (body.get("model_name") or "").strip()
    if not model_name:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "model_name is required for rvc-python",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
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
                    "stack": traceback.format_exc(),
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
    
    # Support additional refs (for accumulating more training data over time)
    additional_refs = body.get("additional_refs") or []
    if not isinstance(additional_refs, list):
        additional_refs = []
    
    # If this voice already exists, merge additional refs
    existing_cfg = reg.get(voice_lock_id, {})
    if isinstance(existing_cfg, dict):
        existing_additional = existing_cfg.get("additional_refs") or []
        if isinstance(existing_additional, list):
            # Merge, avoiding duplicates
            for ref in additional_refs:
                if isinstance(ref, str) and ref not in existing_additional:
                    existing_additional.append(ref)
            additional_refs = existing_additional
    
    reg[voice_lock_id] = {
        "model_type": "rvc_titan",
        "model_name": model_name,
        "reference_wav_path": ref_path,
        "additional_refs": additional_refs,
        "f0_method": body.get("f0_method") or "rmvpe",
        "transpose": int(body.get("transpose") or 0),
        "sample_rate": int(body.get("sample_rate") or 32000),
        "index_rate": float(body.get("index_rate") or 0.0),
        "filter_radius": int(body.get("filter_radius") or 0),
        "rms_mix_rate": float(body.get("rms_mix_rate") or 0.0),
        "protect": float(body.get("protect") or 0.33),
    }
    _save_registry(reg)
    return {"ok": True, "voice_lock_id": voice_lock_id}


def _decode_wav_from_b64(b64: str) -> bytes:
    return base64.b64decode(b64)


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


def _run_rvc_convert(
    src_y: np.ndarray,
    src_sr: int,
    ref_y: np.ndarray,
    ref_sr: int,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    RVC conversion using external rvc-python API.

    - src_y/src_sr: source mono float32 PCM and sample rate.
    - ref_y/ref_sr: reference voice audio (currently used only as a policy check; the
      external rvc-python engine operates on a pre-trained model for this voice).
    - cfg: registry entry for this voice_lock_id; must contain model_name and may
      include f0_method, transpose, index_rate, etc.
    """
    if not isinstance(src_y, np.ndarray) or src_y.size == 0:
        # Leave semantic enforcement (empty source) to the caller to surface as
        # a ValidationError in the JSON envelope.
        logging.error("rvc_convert: empty source audio")
        return src_y
    if not isinstance(ref_y, np.ndarray) or ref_y.size == 0:
        # keep RVC mandatory semantics: require a reference clip at registration time
        logging.error("rvc_convert: empty reference audio")
        return src_y

    # Ensure external engine and model are ready
    ensure_rvc_engine_loaded(cfg)

    # Optional: push per-voice parameters into rvc-python
    params: Dict[str, Any] = {
        "f0method": cfg.get("f0_method") or "rmvpe",
        "f0up_key": int(cfg.get("transpose") or 0),
        "index_rate": float(cfg.get("index_rate") or 0.0),
        "filter_radius": int(cfg.get("filter_radius") or 0),
        "resample_sr": int(cfg.get("sample_rate") or 0),
        "rms_mix_rate": float(cfg.get("rms_mix_rate") or 0.0),
        "protect": float(cfg.get("protect") or 0.33),
    }
    base = RVC_ENGINE_BASE.rstrip("/")
    resp_params = httpx.post(
        f"{base}/params",
        json={"params": params},
        timeout=None,
    )
    if resp_params.status_code < 200 or resp_params.status_code >= 300:
        logging.error("rvc_engine: /params failed: %s %s", resp_params.status_code, resp_params.text)
        return src_y

    # Encode source audio to WAV bytes and base64 for /convert
    wav_bytes = _encode_mono_float32_to_wav_bytes(src_y, src_sr)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    resp_conv = httpx.post(
        f"{base}/convert",
        json={"audio_data": audio_b64},
        timeout=None,
    )
    if resp_conv.status_code < 200 or resp_conv.status_code >= 300:
        logging.error("rvc_engine: /convert failed: %s %s", resp_conv.status_code, resp_conv.text)
        return src_y

    # rvc-python returns raw WAV bytes in the body, not JSON
    out_bytes = resp_conv.content
    if not out_bytes:
        logging.error("rvc_engine: /convert returned empty body")
        return src_y

    out_y, out_sr = _decode_wav_to_mono_float32(out_bytes)
    # You can resample to cfg["sample_rate"] later if needed; for now we return what engine produced
    return out_y


# Note: train_steps computation is now done in trainer_api.py (source of truth)
# This function is kept for potential future use but trainer handles train_steps
def compute_total_duration(ref_paths: List[str]) -> float:
    """
    Compute total duration of all reference audio files.
    
    Returns total duration in seconds, or 0.0 if unable to compute.
    """
    total_sec = 0.0
    for ref_path in ref_paths:
        if not os.path.exists(ref_path):
            continue
        try:
            audio, sr = sf.read(ref_path, always_2d=False)
            duration = len(audio) / float(sr) if sr > 0 else 0.0
            total_sec += duration
        except Exception:
            # Skip files that can't be read
            continue
    return total_sec


def train_rvc_model_blocking(speaker_id: str, model_name: str, ref_paths: List[str], train_steps: int = 4000) -> Dict[str, Any]:
    """
    Blocking call to train an RVC model via rvc_trainer.
    
    - speaker_id: identifier for the speaker/voice
    - model_name: name for the trained model (will be available as model_name in registry)
    - ref_paths: list of file paths to training audio files
    - train_steps: number of training steps (default 4000)
    
    Returns the response from the trainer service.
    """
    payload = {
        "speaker_id": speaker_id,
        "rvc_model_name": model_name,
        "refs": ref_paths,
        "train_steps": train_steps,
    }
    resp = httpx.post(
        RVC_TRAINER_BASE.rstrip("/") + "/train_now",
        json=payload,
        timeout=None,  # no timeout: block until training finishes
    )
    return resp.json()


@app.post("/v1/voice/train")
async def train_voice(body: Dict[str, Any]):
    """
    Train an RVC model for a voice.
    
    Expected body:
    - voice_lock_id: str (required) - voice identifier
    - train_steps: int (optional, default 4000)
    
    Uses reference audio from the voice registry.
    """
    voice_lock_id = (body.get("voice_lock_id") or "").strip()
    if not voice_lock_id:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "voice_lock_id is required",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    
    reg = _load_registry()
    voice_cfg = reg.get(voice_lock_id)
    if not isinstance(voice_cfg, dict):
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": f"no registry entry for {voice_lock_id}",
                "status": 404,
            },
        }
    
    model_name = (voice_cfg.get("model_name") or "").strip()
    if not model_name:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "voice registry entry missing model_name",
                "status": 400,
            },
        }
    
    # Get reference audio paths - collect ALL refs for this voice
    ref_paths = []
    
    # Primary reference from registry
    ref_path = voice_cfg.get("reference_wav_path")
    if isinstance(ref_path, str) and ref_path.strip() and os.path.exists(ref_path):
        ref_paths.append(ref_path)
    
    # Additional refs from registry (if stored as list)
    additional_refs = voice_cfg.get("additional_refs") or []
    if isinstance(additional_refs, list):
        for ref in additional_refs:
            if isinstance(ref, str) and ref.strip() and os.path.exists(ref):
                if ref not in ref_paths:
                    ref_paths.append(ref)
    
    # Also check voices directory for additional refs
    voices_root = os.getenv("RVC_VOICES_ROOT", "/rvc/assets/voices")
    if os.path.isdir(voices_root):
        # Look for additional reference files for this voice
        voice_files = glob.glob(os.path.join(voices_root, f"{voice_lock_id}*.wav"))
        for f in voice_files:
            if f not in ref_paths:
                ref_paths.append(f)
    
    if not ref_paths:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": f"No reference audio files found for voice {voice_lock_id}",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    
    # Prevent training on default voices
    model_name = (voice_cfg.get("model_name") or "").strip()
    if model_name in (RVC_DEFAULT_MALE_VOICE_ID, RVC_DEFAULT_FEMALE_VOICE_ID):
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": f"Voice '{model_name}' is a reserved default and cannot be trained.",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    
    # train_steps: pass through if provided, otherwise let trainer compute
    train_steps = body.get("train_steps")
    if train_steps is not None:
        train_steps = int(train_steps)
    # If not provided, trainer will auto-compute based on duration
    
    try:
        result = train_rvc_model_blocking(
            speaker_id=voice_lock_id,
            model_name=model_name,
            ref_paths=ref_paths,
            train_steps=train_steps,
        )
        
        # Extract trainer result and update registry
        if isinstance(result, dict) and result.get("ok"):
            trainer_result = result.get("result") or {}
            returned_speaker_id = trainer_result.get("speaker_id")
            returned_model_name = trainer_result.get("model_name")
            is_new_speaker = bool(trainer_result.get("is_new_speaker"))
            match_info = trainer_result.get("match") or {}
            
            # Update registry entry with trainer results
            if isinstance(voice_cfg, dict):
                if returned_model_name:
                    voice_cfg["model_name"] = returned_model_name
                if returned_speaker_id:
                    voice_cfg["speaker_id"] = returned_speaker_id
                voice_cfg["last_train_steps"] = trainer_result.get("train_steps")
                voice_cfg["last_train_duration_s"] = trainer_result.get("duration_s")
                voice_cfg["is_new_speaker"] = is_new_speaker
                voice_cfg["last_speaker_match"] = match_info
                reg[voice_lock_id] = voice_cfg
                _save_registry(reg)
            
            # Return response that bubbles up to orchestrator
            return {
                "ok": True,
                "voice_lock_id": voice_lock_id,
                "speaker_id": returned_speaker_id,
                "model_name": returned_model_name,
                "is_new_speaker": is_new_speaker,
                "match": match_info,
                "train_steps": trainer_result.get("train_steps"),
                "duration_s": trainer_result.get("duration_s"),
            }
        
        # If trainer returned error, pass it through
        return result
        
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": ex.__class__.__name__ or "InternalError",
                "message": str(ex),
                "status": 500,
                "type": ex.__class__.__name__,
                "stack": traceback.format_exc(),
            },
        }


@app.post("/v1/voice/rename")
async def voice_rename(body: Dict[str, Any]):
    """
    Rename or merge voice IDs.
    
    Expected body:
    - old_voice_id: str (required)
    - new_voice_id: str (required)
    - merge_into_existing: bool (optional, defaults to true - always merge if match exists)
    """
    old_voice_id = (body.get("old_voice_id") or "").strip()
    new_voice_id = (body.get("new_voice_id") or "").strip()
    merge_into_existing_flag = bool(body.get("merge_into_existing", True))  # Default to True
    
    if not old_voice_id:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "old_voice_id is required for rename.",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    
    if not new_voice_id:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "new_voice_id is required for rename.",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    
    if old_voice_id == new_voice_id:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "old_voice_id and new_voice_id cannot be the same.",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    
    # Load registry
    reg = _load_registry()
    
    # Collect all voice_lock_ids that reference the old_voice_id
    affected = []
    for vlk, cfg in reg.items():
        if not isinstance(cfg, dict):
            continue
        existing_model = (cfg.get("model_name") or "").strip()
        if existing_model == old_voice_id:
            affected.append(vlk)
    
    # Determine if new_voice_id is already in use as a model_name
    new_used = False
    new_voice_cfg = None
    for vlk, cfg in reg.items():
        if isinstance(cfg, dict):
            if (cfg.get("model_name") or "").strip() == new_voice_id:
                new_used = True
                new_voice_cfg = cfg
                break
    
    # Always merge if target exists (always true - disregard flag)
    if new_used:
        # Merge: move refs from old to new, update registry
        old_refs = []
        old_additional_refs = []
        
        # Collect refs from all affected voice_lock_ids
        for vlk in affected:
            cfg = reg.get(vlk, {})
            if isinstance(cfg, dict):
                ref_path = cfg.get("reference_wav_path")
                if isinstance(ref_path, str) and ref_path.strip():
                    old_refs.append(ref_path)
                additional = cfg.get("additional_refs") or []
                if isinstance(additional, list):
                    old_additional_refs.extend(additional)
        
        # Merge into new_voice_cfg
        if isinstance(new_voice_cfg, dict):
            new_refs = new_voice_cfg.get("additional_refs") or []
            if not isinstance(new_refs, list):
                new_refs = []
            # Add old refs to new (avoid duplicates)
            for ref in old_refs + old_additional_refs:
                if isinstance(ref, str) and ref not in new_refs:
                    new_refs.append(ref)
            new_voice_cfg["additional_refs"] = new_refs
        
        # Update all affected registry entries to point to new_voice_id
        for vlk in affected:
            reg[vlk]["model_name"] = new_voice_id
        
        # Remove old model directory if it exists
        old_dir = os.path.join(RVC_MODELS_ROOT, old_voice_id)
        if os.path.isdir(old_dir):
            try:
                shutil.rmtree(old_dir)
            except Exception as ex:
                logging.warning("Failed to remove old model directory %s: %s", old_dir, str(ex))
        
        _save_registry(reg)
        
        # Update speaker index on merge
        spk_index_path = os.path.join(RVC_MODELS_ROOT, "speaker_index.json")
        try:
            if os.path.exists(spk_index_path):
                with open(spk_index_path, "r", encoding="utf-8") as f:
                    spk_index = json.load(f)
            else:
                spk_index = {}
            
            old_entry = spk_index.get(old_voice_id)
            new_entry = spk_index.get(new_voice_id)
            
            if new_entry and old_entry:
                # Merge embeddings: use same logic as update_speaker_embedding
                try:
                    old_emb = np.array(old_entry.get("embedding", []), dtype=np.float32)
                    old_count = int(old_entry.get("count", 0))
                    new_emb = np.array(new_entry.get("embedding", []), dtype=np.float32)
                    new_count = int(new_entry.get("count", 0))
                    if old_emb.shape == new_emb.shape and old_count > 0 and new_count > 0:
                        total = old_count + new_count
                        merged = (old_emb * old_count + new_emb * new_count) / float(total)
                        spk_index[new_voice_id] = {"embedding": merged.tolist(), "count": total}
                except Exception as ex:
                    # If merge fails, just keep new_entry, but do not be silent.
                    logging.warning("speaker_index merge failed old=%s new=%s: %s", old_voice_id, new_voice_id, str(ex), exc_info=True)
            elif old_entry and not new_entry:
                # Move old entry to new key
                spk_index[new_voice_id] = old_entry
            
            # Remove old entry
            if old_voice_id in spk_index:
                del spk_index[old_voice_id]
            
            # Save back
            with open(spk_index_path, "w", encoding="utf-8") as f:
                json.dump(spk_index, f, indent=2)
        except Exception as ex:
            logging.warning("failed to update speaker_index on rename: %s", str(ex))
        
        return {
            "ok": True,
            "old_voice_id": old_voice_id,
            "new_voice_id": new_voice_id,
            "merged": True,
            "affected_voice_lock_ids": affected,
        }
    
    else:
        # Simple rename: new_voice_id doesn't exist
        # Rename model directory
        old_dir = os.path.join(RVC_MODELS_ROOT, old_voice_id)
        new_dir = os.path.join(RVC_MODELS_ROOT, new_voice_id)
        
        if os.path.isdir(old_dir):
            try:
                if os.path.exists(new_dir):
                    # If new_dir exists, merge contents (shouldn't happen if new_used is False, but handle it)
                    for item in os.listdir(old_dir):
                        src = os.path.join(old_dir, item)
                        dst = os.path.join(new_dir, item)
                        if os.path.exists(dst):
                            continue
                        shutil.move(src, dst)
                    shutil.rmtree(old_dir)
                else:
                    os.rename(old_dir, new_dir)
            except Exception as ex:
                return {
                    "ok": False,
                    "error": {
                        "code": ex.__class__.__name__ or "InternalError",
                        "message": f"Failed to rename model directory: {str(ex)}",
                        "status": 500,
                        "stack": traceback.format_exc(),
                    },
                }
        
        # Update all affected registry entries
        for vlk in affected:
            cfg = reg.get(vlk, {})
            if isinstance(cfg, dict):
                cfg["model_name"] = new_voice_id
                # Optionally update speaker_id if stored
                if cfg.get("speaker_id") == old_voice_id:
                    cfg["speaker_id"] = new_voice_id
                reg[vlk] = cfg
        
        _save_registry(reg)
        
        # Update speaker index on rename
        spk_index_path = os.path.join(RVC_MODELS_ROOT, "speaker_index.json")
        try:
            if os.path.exists(spk_index_path):
                with open(spk_index_path, "r", encoding="utf-8") as f:
                    spk_index = json.load(f)
            else:
                spk_index = {}
            
            # Pure rename: move old entry to new key
            if old_voice_id in spk_index:
                spk_index[new_voice_id] = spk_index[old_voice_id]
                del spk_index[old_voice_id]
            
            # Save back
            with open(spk_index_path, "w", encoding="utf-8") as f:
                json.dump(spk_index, f, indent=2)
        except Exception as ex:
            logging.warning("failed to update speaker_index on rename: %s", str(ex))
        
        return {
            "ok": True,
            "old_voice_id": old_voice_id,
            "new_voice_id": new_voice_id,
            "merged": False,
            "affected_voice_lock_ids": affected,
        }


@app.post("/v1/voice/mine/audio")
async def mine_voices_audio(body: Dict[str, Any]):
    """
    Mine voices from audio reference (song, podcast, raw WAV).
    
    Expected body:
    - audio_path: str (required) - path to audio file
    - project_id: str (optional) - project identifier
    - expected_speakers: List[Dict] (optional) - speaker hints
      [{"id": "speaker:cyril.main", "hint": "Cyril"}, ...]
    - auto_train: bool (optional, default false) - whether to train immediately
    - language: str (optional) - language hint for transcription
    """
    result = voice_miner.mine_voices_from_audio(body)
    if result.get("ok"):
        return result
    else:
        return JSONResponse(
            status_code=400,
            content=result,
        )


@app.post("/v1/voice/mine/video")
async def mine_voices_video(body: Dict[str, Any]):
    """
    Mine voices from video reference (film, character alignment).
    
    Expected body:
    - video_path: str (required) - path to video file
    - project_id: str (optional) - project identifier
    - character_hints: List[Dict] (optional) - character hints
      [{"character_id": "char:shadow", "hint": "hedgehog voice"}, ...]
    - auto_train: bool (optional, default false) - whether to train immediately
    - language: str (optional) - language hint for transcription
    """
    result = voice_miner.mine_voices_from_video(body)
    if result.get("ok"):
        return result
    else:
        return JSONResponse(
            status_code=400,
            content=result,
        )


@app.post("/v1/audio/convert")
async def convert_v1(body: Dict[str, Any]):
    voice_lock_id = (body.get("voice_lock_id") or "").strip()
    
    # Accept both source_wav_base64 (preferred) and legacy source_wav_b64.
    src_b64 = body.get("source_wav_base64") or body.get("source_wav_b64")
    if not isinstance(src_b64, str) or not src_b64.strip():
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "source_wav_b64 is required",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    
    reg = _load_registry()
    voice_cfg = reg.get(voice_lock_id) if voice_lock_id else None
    
    # If no voice_lock_id or no registry entry, use default fallback
    model_name = ""
    if not voice_lock_id or not isinstance(voice_cfg, dict):
        # Choose default voice (simplistic: prefer female, fallback to male)
        # You can refine this later based on voice_role or other hints
        fallback_model = RVC_DEFAULT_FEMALE_VOICE_ID or RVC_DEFAULT_MALE_VOICE_ID
        voice_cfg = {
            "model_name": fallback_model,
            "speaker_id": fallback_model,
            "f0_method": "rmvpe",
            "transpose": 0,
            "sample_rate": 32000,
            "index_rate": 0.0,
            "filter_radius": 0,
            "rms_mix_rate": 0.0,
            "protect": 0.33,
        }
        model_name = fallback_model
        logging.info("Using default fallback voice: %s", fallback_model)
    else:
        # Check if model actually exists
        model_name = (voice_cfg.get("model_name") or "").strip()
        if model_name:
            model_path = os.path.join(RVC_MODELS_ROOT, model_name, f"{model_name}.pth")
            if not os.path.exists(model_path):
                # Model missing, fallback to default
                logging.warning("Model %s not found, falling back to default", model_name)
                fallback_model = RVC_DEFAULT_FEMALE_VOICE_ID or RVC_DEFAULT_MALE_VOICE_ID
                voice_cfg = {
                    "model_name": fallback_model,
                    "speaker_id": fallback_model,
                    "f0_method": voice_cfg.get("f0_method") or "rmvpe",
                    "transpose": int(voice_cfg.get("transpose") or 0),
                    "sample_rate": int(voice_cfg.get("sample_rate") or 32000),
                    "index_rate": float(voice_cfg.get("index_rate") or 0.0),
                    "filter_radius": int(voice_cfg.get("filter_radius") or 0),
                    "rms_mix_rate": float(voice_cfg.get("rms_mix_rate") or 0.0),
                    "protect": float(voice_cfg.get("protect") or 0.33),
                }
                model_name = fallback_model
    try:
        wav_bytes = _decode_wav_from_b64(src_b64)
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": ex.__class__.__name__ or "InternalError",
                "message": str(ex),
                "status": 400,
                "type": ex.__class__.__name__,
                "stack": traceback.format_exc(),
            },
        }
    # Decode source audio to mono float32.
    try:
        src_y, src_sr = _decode_wav_to_mono_float32(wav_bytes)
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": ex.__class__.__name__ or "InternalError",
                "message": str(ex),
                "status": 400,
                "type": ex.__class__.__name__,
                "stack": traceback.format_exc(),
            },
        }
    # Use voice_cfg we determined above (with fallback if needed)
    ref_cfg = voice_cfg if isinstance(voice_cfg, dict) else {}
    ref_path = ""
    if isinstance(ref_cfg, dict):
        ref_path = (ref_cfg.get("reference_wav_path") or "").strip()
    
    ref_y: np.ndarray
    ref_sr: int
    
    # For default voices, we may not have a reference_wav_path, but we still need to proceed
    # The model itself is the reference, so we can use source as ref or skip ref validation
    is_default_voice = False
    if model_name in (RVC_DEFAULT_MALE_VOICE_ID, RVC_DEFAULT_FEMALE_VOICE_ID):
        is_default_voice = True
    
    if isinstance(ref_path, str) and ref_path.strip() and os.path.exists(ref_path):
        try:
            ref_pcm, ref_sr = sf.read(ref_path, always_2d=False)
            if getattr(ref_pcm, "ndim", 1) > 1:
                ref_pcm = ref_pcm.mean(axis=1)
            ref_y = np.asarray(ref_pcm, dtype=np.float32)
        except Exception as ex:
            if is_default_voice:
                # For default voices, use source as ref if ref file is missing
                ref_y = src_y.copy()
                ref_sr = src_sr
            else:
                return {
                    "ok": False,
                    "error": {
                        "code": "ref_decode_error",
                        "message": str(ex),
                        "status": 400,
                        "type": ex.__class__.__name__,
                        "stack": traceback.format_exc(),
                    },
                }
    elif is_default_voice:
        # Default voices: use source as reference (model is pre-trained)
        ref_y = src_y.copy()
        ref_sr = src_sr
    else:
        # For non-default voices, require a reference
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": f"No reference_wav_path configured or file not found for voice_lock_id '{voice_lock_id or 'N/A'}'.",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    try:
        converted_np = _run_rvc_convert(src_y, src_sr, ref_y, ref_sr, ref_cfg)
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": ex.__class__.__name__ or "InternalError",
                "message": str(ex),
                "status": 500,
                "type": ex.__class__.__name__,
                "stack": traceback.format_exc(),
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
                "code": ex.__class__.__name__ or "InternalError",
                "message": str(ex),
                "status": 500,
                "type": ex.__class__.__name__,
                "stack": traceback.format_exc(),
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
                "code": "ValidationError",
                "message": "wav_path is required",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    if not voice_lock_id:
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": "voice_lock_id is required",
                "status": 400,
                "stack": "".join(traceback.format_stack()),
            },
        }
    if not os.path.exists(wav_path):
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "message": f"wav_path not found: {wav_path}",
                "status": 404,
                "stack": "".join(traceback.format_stack()),
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
                "code": "InternalError",
                "detail": res,
                "status": 500,
                "stack": "".join(traceback.format_stack()),
            },
        }
    out_bytes = base64.b64decode(res.get("audio_wav_base64") or "")
    v_dir, v_name = os.path.dirname(wav_path), os.path.basename(wav_path)
    v_base, v_ext = os.path.splitext(v_name)
    out_path = os.path.join(v_dir, f"{v_base}.rvc{v_ext or '.wav'}")
    with open(out_path, "wb") as f:
        f.write(out_bytes)
    return {"ok": True, "path": out_path}
