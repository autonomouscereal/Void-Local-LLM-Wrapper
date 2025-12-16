from __future__ import annotations

import os
import json
import time
import subprocess
import shutil
import glob
import uuid
import logging
import sys
import soundfile as sf
import numpy as np
import torch
import torchaudio  # type: ignore
from typing import Dict, Any, List, Tuple
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse


# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "rvc_trainer.log")
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
    logging.getLogger(__name__).info(
        "rvc_trainer logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl)
    )
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger(__name__).warning("rvc_trainer file logging disabled: %s", _ex, exc_info=True)

log = logging.getLogger(__name__)

try:
    from speechbrain.pretrained import EncoderClassifier
    SPKREC_AVAILABLE = True
except ImportError:
    SPKREC_AVAILABLE = False
    EncoderClassifier = None  # type: ignore

app = FastAPI(title="RVC Trainer Service", version="1.0.0")

RVC_TRAIN_DATA_ROOT = os.getenv("RVC_TRAIN_DATA_ROOT", "/srv/data")
RVC_MODELS_ROOT = os.getenv("RVC_MODELS_ROOT", "/srv/rvc_models")
RVC_WEBUI_ROOT = os.getenv("RVC_WEBUI_ROOT", "/srv/rvc_webui")

os.makedirs(RVC_TRAIN_DATA_ROOT, exist_ok=True)
os.makedirs(RVC_MODELS_ROOT, exist_ok=True)

# Speaker recognition model
SPKREC_MODEL_ID = os.environ.get("SPKREC_MODEL_ID", "speechbrain/spkrec-ecapa-voxceleb")
_SPKREC_MODEL = None
SPK_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPK_INDEX_PATH = os.path.join(RVC_MODELS_ROOT, "speaker_index.json")


def get_speaker_encoder() -> Any:
    """
    Lazily load the speaker embedding model (ECAPA-TDNN).
    This runs only once per process.
    """
    global _SPKREC_MODEL
    
    if _SPKREC_MODEL is None:
        log.info("speaker_encoder loading model_id=%r device=%s", SPKREC_MODEL_ID, SPK_DEVICE)
        t0 = time.perf_counter()
        _SPKREC_MODEL = EncoderClassifier.from_hparams(
            source=SPKREC_MODEL_ID,
            run_opts={"device": SPK_DEVICE},
        )
        log.info("speaker_encoder loaded duration_ms=%s", int((time.perf_counter() - t0) * 1000))
    return _SPKREC_MODEL


def compute_speaker_embedding(wav_path: str) -> np.ndarray:
    """
    Compute a speaker embedding for a single WAV file using the ECAPA encoder.
    """
    encoder = get_speaker_encoder()
    t0 = time.perf_counter()
    waveform, sample_rate = torchaudio.load(wav_path)
    target_sr = 16000
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    with torch.no_grad():
        emb = encoder.encode_batch(waveform.to(SPK_DEVICE))  # [1, feat_dim]
    emb_np = emb.squeeze(0).cpu().numpy()
    # Normalize to unit length for cosine similarity
    norm = np.linalg.norm(emb_np)
    if norm > 0:
        emb_np = emb_np / float(norm)
    log.info(
        "speaker_embedding computed wav=%r in_sr=%s target_sr=%s frames=%s emb_dim=%s duration_ms=%s",
        wav_path,
        int(sample_rate),
        int(target_sr),
        int(waveform.shape[-1]) if hasattr(waveform, "shape") else -1,
        int(emb_np.shape[-1]) if hasattr(emb_np, "shape") else -1,
        int((time.perf_counter() - t0) * 1000),
    )
    return emb_np


def load_speaker_index() -> Dict[str, Any]:
    """
    Load the speaker index mapping speaker_id -> {"embedding": [...], "count": int}.
    """
    if not os.path.exists(SPK_INDEX_PATH):
        log.info("speaker_index missing path=%r", SPK_INDEX_PATH)
        return {}
    try:
        with open(SPK_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning("speaker_index invalid_type path=%r type=%s", SPK_INDEX_PATH, type(data).__name__)
            return {}
        log.info("speaker_index loaded path=%r speakers=%s", SPK_INDEX_PATH, len(data))
        return data
    except Exception as ex:
        log.warning("speaker_index load_failed path=%r err=%s", SPK_INDEX_PATH, ex, exc_info=True)
        return {}


def save_speaker_index(index: Dict[str, Any]) -> None:
    """Save the speaker index to disk."""
    os.makedirs(os.path.dirname(SPK_INDEX_PATH), exist_ok=True)
    tmp_path = SPK_INDEX_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp_path, SPK_INDEX_PATH)
    log.info("speaker_index saved path=%r speakers=%s", SPK_INDEX_PATH, len(index))


def update_speaker_embedding(index: Dict[str, Any], speaker_id: str, new_emb: np.ndarray) -> Dict[str, Any]:
    """
    Maintain a running average embedding per speaker_id.
    Always merges if speaker_id exists.
    """
    key = speaker_id
    if key in index:
        entry = index[key]
        prev_emb = np.array(entry.get("embedding", []), dtype=np.float32)
        prev_count = int(entry.get("count", 0))
        if prev_count > 0 and prev_emb.shape == new_emb.shape:
            total_count = prev_count + 1
            merged = (prev_emb * prev_count + new_emb) / float(total_count)
            index[key] = {"embedding": merged.tolist(), "count": total_count}
        else:
            index[key] = {"embedding": new_emb.tolist(), "count": 1}
    else:
        index[key] = {"embedding": new_emb.tolist(), "count": 1}
    return index


def find_best_matching_speaker(new_emb: np.ndarray, index: Dict[str, Any]) -> Tuple[str, float]:
    """
    Return (speaker_id, cosine_similarity) for the best match in the index.
    If index is empty, returns ("", 0.0).
    """
    best_id = ""
    best_score = 0.0
    for speaker_id, entry in index.items():
        vec = np.array(entry.get("embedding", []), dtype=np.float32)
        if vec.shape != new_emb.shape:
            continue
        score = float(np.dot(vec, new_emb))  # both normalized
        if score > best_score:
            best_score = score
            best_id = speaker_id
    return best_id, best_score


def setup_dataset(speaker_id: str, ref_paths: List[str]) -> str:
    """
    Set up training dataset structure for RVC WebUI.
    
    Creates:
    - /srv/data/speakers/<speaker_id>/wavs/ (audio files)
    - /srv/data/speakers/<speaker_id>/train.list
    - /srv/data/speakers/<speaker_id>/val.list
    
    RVC WebUI expects train.list/val.list format:
    <absolute_path_to_wav>\t<dummy_text_or_id>
    
    Returns the dataset root path.
    """
    dataset_root = os.path.join(RVC_TRAIN_DATA_ROOT, "speakers", speaker_id)
    wavs_dir = os.path.join(dataset_root, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    log.info("dataset setup start speaker_id=%s refs=%s dataset_root=%r", speaker_id, len(ref_paths or []), dataset_root)
    
    # Copy or symlink reference files to wavs directory
    wav_abs_paths = []
    for ref_path in ref_paths:
        if not os.path.exists(ref_path):
            log.warning("dataset ref missing ref_path=%r", ref_path)
            continue
        basename = os.path.basename(ref_path)
        target_path = os.path.join(wavs_dir, basename)
        if not os.path.exists(target_path):
            try:
                shutil.copy2(ref_path, target_path)
                log.info("dataset copied ref=%r -> %r", ref_path, target_path)
            except Exception:
                # If copy fails, try symlink
                try:
                    os.symlink(ref_path, target_path)
                    log.info("dataset symlinked ref=%r -> %r", ref_path, target_path)
                except Exception:
                    log.warning("dataset copy+symlink failed ref=%r target=%r", ref_path, target_path, exc_info=True)
                    continue
        # Store absolute path for train.list/val.list
        wav_abs_paths.append(os.path.abspath(target_path))
    
    # Create train.list and val.list (80/20 split)
    train_paths = wav_abs_paths[:int(len(wav_abs_paths) * 0.8)]
    val_paths = wav_abs_paths[int(len(wav_abs_paths) * 0.8):]
    
    # Ensure at least one file in each set
    if not val_paths:
        val_paths = [train_paths.pop()] if train_paths else []
    if not train_paths:
        train_paths = [val_paths.pop()] if val_paths else []
    
    train_list_path = os.path.join(dataset_root, "train.list")
    val_list_path = os.path.join(dataset_root, "val.list")
    
    # Write train.list in RVC WebUI format: <absolute_path>\t<dummy_text>
    with open(train_list_path, "w", encoding="utf-8") as f:
        for wav_path in train_paths:
            # Use basename as dummy text/ID
            dummy_id = os.path.splitext(os.path.basename(wav_path))[0]
            f.write(f"{wav_path}\t{dummy_id}\n")
    
    # Write val.list
    with open(val_list_path, "w", encoding="utf-8") as f:
        for wav_path in val_paths:
            dummy_id = os.path.splitext(os.path.basename(wav_path))[0]
            f.write(f"{wav_path}\t{dummy_id}\n")
    
    log.info(
        "dataset setup done speaker_id=%s wavs=%s train=%s val=%s train_list=%r val_list=%r",
        speaker_id,
        len(wav_abs_paths),
        len(train_paths),
        len(val_paths),
        train_list_path,
        val_list_path,
    )
    return dataset_root


def generate_training_config(dataset_root: str, model_name: str, train_steps: int, is_finetune: bool, pretrain_path: str | None) -> str:
    """
    Generate a per-speaker training config JSON.
    
    Reads the base 48k config, modifies epochs/steps, and writes to dataset_root/config.json.
    
    Returns the path to the generated config.
    """
    base_config_path = os.path.join(RVC_WEBUI_ROOT, "configs", "48k", "config.json")
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Modify training parameters
    # RVC WebUI typically uses epochs, but we can also set total_step
    # Estimate epochs from steps (rough heuristic: ~100 steps per epoch for small datasets)
    estimated_epochs = max(1, train_steps // 100)
    
    if "train" in config:
        config["train"]["epochs"] = estimated_epochs
        config["train"]["total_step"] = train_steps
        # If fine-tuning, set pretrain path
        if is_finetune and pretrain_path:
            config["train"]["pretrained_G"] = pretrain_path
            config["train"]["pretrained_D"] = pretrain_path.replace("G_", "D_").replace(".pth", ".pth")
    
    # Write config to dataset root
    config_path = os.path.join(dataset_root, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    return config_path


def find_latest_model(model_name: str) -> str | None:
    """
    Find the latest trained model checkpoint in WebUI logs.
    
    Looks for G_*.pth files in /srv/rvc_webui/logs/<model_name>/
    Returns the path to the latest checkpoint, or None if not found.
    """
    logs_dir = os.path.join(RVC_WEBUI_ROOT, "logs", model_name)
    if not os.path.isdir(logs_dir):
        return None
    
    # Find all G_*.pth files
    g_files = glob.glob(os.path.join(logs_dir, "G_*.pth"))
    if not g_files:
        return None
    
    # Sort by modification time, return latest
    g_files.sort(key=os.path.getmtime, reverse=True)
    return g_files[0]


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "rvc_trainer"}


@app.post("/train_now")
async def train_now(body: Dict[str, Any]):
    """
    Blocking training endpoint. Trains an RVC model and returns when complete.
    
    Expected body:
    - speaker_id: str (optional, will auto-detect if not provided)
    - rvc_model_name: str (optional, defaults to speaker_id)
    - refs: List[str] (required, list of file paths to training audio)
    - train_steps: int (optional, will be estimated if not provided)
    """
    requested_speaker_id = (body.get("speaker_id") or "").strip()
    requested_model_name = (body.get("rvc_model_name") or "").strip()
    refs = body.get("refs") or []
    req_id = uuid.uuid4().hex[:12]
    log.info(
        "train_now start req_id=%s speaker_id=%r model_name=%r refs=%s train_steps=%r spkrec=%s",
        req_id,
        requested_speaker_id,
        requested_model_name,
        len(refs) if isinstance(refs, list) else -1,
        body.get("train_steps"),
        bool(SPKREC_AVAILABLE),
    )
    
    # train_steps will be computed later if not provided
    train_steps = body.get("train_steps")
    if train_steps is not None:
        train_steps = int(train_steps)
    
    if not isinstance(refs, list) or not refs:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"code": "missing_refs", "message": "At least one reference WAV path is required for training."}},
        )
    
    # Choose primary ref for embedding (first one)
    primary_ref = refs[0]
    if not isinstance(primary_ref, str) or not os.path.exists(primary_ref):
        log.warning("train_now invalid_primary_ref req_id=%s primary_ref=%r", req_id, primary_ref)
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"code": "invalid_primary_ref", "message": f"Primary reference not found: {primary_ref}"}},
        )
    
    # Speaker/model resolution
    is_new_speaker = False
    matched_speaker_id = ""
    match_score = 0.0
    
    if isinstance(requested_speaker_id, str) and requested_speaker_id.strip():
        # Caller provided explicit speaker_id; use as-is
        speaker_id = requested_speaker_id.strip()
        model_name = requested_model_name.strip() if requested_model_name.strip() else speaker_id
        log.info("train_now explicit_speaker req_id=%s speaker_id=%s model_name=%s", req_id, speaker_id, model_name)
    else:
        # No speaker_id provided: run speaker detection
        if not SPKREC_AVAILABLE:
            log.error("train_now speaker_detection_unavailable req_id=%s", req_id)
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": {"code": "speaker_detection_unavailable", "message": "speechbrain not available; speaker_id is required"}},
            )
        
        try:
            new_emb = compute_speaker_embedding(primary_ref)
            index = load_speaker_index()
            best_id, best_score = find_best_matching_speaker(new_emb, index)
            matched_speaker_id = best_id
            match_score = best_score
            log.info(
                "train_now speaker_match req_id=%s best_id=%r score=%.4f speakers_in_index=%s",
                req_id,
                best_id,
                float(best_score),
                len(index),
            )
            
            if best_id and best_score >= 0.80:
                # Reuse existing speaker/model (always merge)
                speaker_id = best_id
                model_name = best_id
            else:
                # Create a new speaker/model ID
                new_id = uuid.uuid4().hex
                speaker_id = f"spk-{new_id}"
                model_name = speaker_id
                is_new_speaker = True
            log.info(
                "train_now speaker_resolved req_id=%s speaker_id=%s model_name=%s is_new=%s matched=%r score=%.4f",
                req_id,
                speaker_id,
                model_name,
                bool(is_new_speaker),
                matched_speaker_id,
                float(match_score),
            )
        except Exception as ex:
            log.error("train_now speaker_detection_failed req_id=%s err=%s", req_id, ex, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": {"code": "speaker_detection_failed", "message": str(ex)}},
            )
    
    # Enforce that we never train on reserved default voices
    default_female = os.environ.get("RVC_DEFAULT_FEMALE_VOICE_ID", "").strip()
    default_male = os.environ.get("RVC_DEFAULT_MALE_VOICE_ID", "").strip()
    if speaker_id and (speaker_id == default_female or speaker_id == default_male):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"code": "forbidden_speaker", "message": f"Speaker '{speaker_id}' is reserved for default voices and cannot be trained."}},
        )
    
    start_time = time.time()
    log.info("train_now run_start req_id=%s speaker_id=%s model_name=%s primary_ref=%r", req_id, speaker_id, model_name, primary_ref)
    
    try:
        # Compute train_steps if not provided (estimate from total duration)
        if train_steps is None:
            total_sec = 0.0
            for ref_path in refs:
                if not os.path.exists(ref_path):
                    log.warning("train_now duration missing req_id=%s ref=%r", req_id, ref_path)
                    continue
                try:
                    audio, sr = sf.read(ref_path, always_2d=False)
                    duration = len(audio) / float(sr) if sr > 0 else 0.0
                    total_sec += duration
                    log.info(
                        "train_now duration req_id=%s ref=%r sr=%s samples=%s seconds=%.3f",
                        req_id,
                        ref_path,
                        int(sr),
                        int(len(audio)) if hasattr(audio, "__len__") else -1,
                        float(duration),
                    )
                except Exception:
                    log.warning("train_now duration read_failed req_id=%s ref=%r", req_id, ref_path, exc_info=True)
                    continue
            # Simple heuristic: estimate steps from duration
            minutes = total_sec / 60.0
            if minutes < 0.1:
                train_steps = 400
            elif minutes < 0.5:
                train_steps = 800
            elif minutes < 2.0:
                train_steps = 2000
            elif minutes < 5.0:
                train_steps = 4000
            elif minutes < 10.0:
                train_steps = 8000
            else:
                train_steps = 12000

            log.info(
                "train_now train_steps estimated req_id=%s total_sec=%.3f minutes=%.3f train_steps=%s",
                req_id,
                float(total_sec),
                float(minutes),
                int(train_steps),
            )
        
        # Set up dataset
        dataset_root = setup_dataset(speaker_id, refs)
        log.info("train_now dataset_ready req_id=%s dataset_root=%r", req_id, dataset_root)
        
        # Check if this is fine-tuning (model already exists) or first training
        model_dir = os.path.join(RVC_MODELS_ROOT, model_name)
        existing_model = os.path.join(model_dir, f"{model_name}.pth")
        is_finetune = os.path.exists(existing_model)
        log.info(
            "train_now finetune_check req_id=%s is_finetune=%s existing_model=%r",
            req_id,
            bool(is_finetune),
            existing_model,
        )
        
        # Determine pretrain path
        pretrain_path = None
        if is_finetune:
            # Fine-tune from existing model
            pretrain_path = existing_model
        else:
            # First training: use Titan pretrain
            titan_g = os.path.join(RVC_WEBUI_ROOT, "pretrained", "f0G48k.pth")
            if os.path.exists(titan_g):
                pretrain_path = titan_g
        
        # Generate per-speaker config with train_steps
        config_path = generate_training_config(
            dataset_root=dataset_root,
            model_name=model_name,
            train_steps=train_steps,
            is_finetune=is_finetune,
            pretrain_path=pretrain_path,
        )
        log.info(
            "train_now config_ready req_id=%s config_path=%r pretrain_path=%r",
            req_id,
            config_path,
            pretrain_path,
        )
        
        # Build training command
        # RVC WebUI train.py typically expects:
        # -c config_path
        # -m model_name
        # -n dataset_name (speaker_id)
        # -g gpu_id
        cmd = [
            "python3",
            "train.py",
            "-c",
            config_path,  # Use generated per-speaker config
            "-m",
            model_name,
            "-n",
            speaker_id,
            "-g",
            "0",
        ]
        
        # Set working directory and environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        log.info("train_now subprocess_start req_id=%s cwd=%r cmd=%r", req_id, RVC_WEBUI_ROOT, cmd)
        
        # Run training (blocking)
        proc = subprocess.run(
            cmd,
            cwd=RVC_WEBUI_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=None,  # No timeout - let it run as long as needed
        )
        
        duration = time.time() - start_time
        log.info(
            "train_now subprocess_done req_id=%s returncode=%s duration_s=%.3f stdout_len=%s stderr_len=%s",
            req_id,
            int(proc.returncode),
            float(duration),
            int(len(proc.stdout or "")),
            int(len(proc.stderr or "")),
        )
        
        if proc.returncode != 0:
            log.error(
                "train_now nonzero_exit req_id=%s returncode=%s stdout_tail=%r stderr_tail=%r",
                req_id,
                int(proc.returncode),
                (proc.stdout or "")[-2000:],
                (proc.stderr or "")[-2000:],
            )
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "train_nonzero_exit",
                        "message": "train.py returned non-zero exit code",
                        "stdout": proc.stdout[-5000:] if proc.stdout else "",  # Last 5k chars
                        "stderr": proc.stderr[-5000:] if proc.stderr else "",
                        "returncode": proc.returncode,
                    },
                    "duration_s": duration,
                },
            )
        
        # Find and copy the trained model
        latest_model = find_latest_model(model_name)
        if not latest_model:
            log.error("train_now model_not_found req_id=%s model_name=%s", req_id, model_name)
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "model_not_found",
                        "message": f"Trained model not found in logs for {model_name}",
                        "logs_dir": os.path.join(RVC_WEBUI_ROOT, "logs", model_name),
                    },
                    "duration_s": duration,
                },
            )
        
        # Copy to rvc_models directory
        os.makedirs(model_dir, exist_ok=True)
        target_model = os.path.join(model_dir, f"{model_name}.pth")
        shutil.copy2(latest_model, target_model)
        log.info("train_now model_copied req_id=%s src=%r dst=%r", req_id, latest_model, target_model)
        
        # Optional: Try to find and copy index file if it exists
        logs_dir = os.path.join(RVC_WEBUI_ROOT, "logs", model_name)
        index_files = glob.glob(os.path.join(logs_dir, "*.index"))
        if index_files:
            # Use the latest index file
            index_files.sort(key=os.path.getmtime, reverse=True)
            target_index = os.path.join(model_dir, f"{model_name}.index")
            shutil.copy2(index_files[0], target_index)
            log.info("train_now index_copied req_id=%s src=%r dst=%r", req_id, index_files[0], target_index)
        
        # Update speaker index after successful training
        try:
            if SPKREC_AVAILABLE:
                new_emb = compute_speaker_embedding(primary_ref)
                index = load_speaker_index()
                index = update_speaker_embedding(index, speaker_id, new_emb)
                save_speaker_index(index)
        except Exception as exc:
            log.warning("speaker_index_update_failed req_id=%s speaker_id=%s error=%s", req_id, speaker_id, str(exc), exc_info=True)

        log.info("train_now success req_id=%s speaker_id=%s model_name=%s duration_s=%.3f", req_id, speaker_id, model_name, float(duration))
        
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "result": {
                    "speaker_id": speaker_id,
                    "model_name": model_name,
                    "is_new_speaker": is_new_speaker,
                    "match": {
                        "matched_speaker_id": matched_speaker_id,
                        "score": match_score,
                    },
                    "train_steps": train_steps,
                    "duration_s": duration,
                    "model_path": target_model,
                },
                "error": None,
            },
        )
        
    except ValueError as ex:
        duration = time.time() - start_time
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": {
                    "code": "dataset_setup_failed",
                    "message": str(ex),
                },
                "duration_s": duration,
            },
        )
    except Exception as ex:
        duration = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "train_subprocess_failed",
                    "message": str(ex),
                    "type": ex.__class__.__name__,
                },
                "duration_s": duration,
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7070)

