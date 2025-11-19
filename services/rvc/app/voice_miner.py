"""
Voice Mining Module

Extracts and trains voice models from reference media (songs, videos, raw audio).

Pipeline:
1. Extract audio from video (if needed)
2. Isolate vocals using demucs_service
3. Run diarization (whisper_service)
4. Segment by speaker and write ref WAVs
5. Register voices in RVC registry
6. Optionally trigger training
"""

from __future__ import annotations

import os
import json
import uuid
import base64
import logging
import tempfile
import shutil
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import httpx
import soundfile as sf
import numpy as np

# Try to import torch/torchaudio for resampling (optional)
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    torchaudio = None  # type: ignore

# Try to import ECAPA for speaker embeddings (reuse trainer's model)
try:
    from speechbrain.pretrained import EncoderClassifier
    SPKREC_AVAILABLE = True
except ImportError:
    SPKREC_AVAILABLE = False
    EncoderClassifier = None  # type: ignore

# Service URLs (from environment)
# Defaults updated to match typical compose ports
DEMUCS_API_URL = os.getenv("DEMUCS_API_URL", "http://127.0.0.1:9101")
WHISPER_API_URL = os.getenv("WHISPER_API_URL", "http://127.0.0.1:7861")
VOCALFIX_API_URL = os.getenv("VOCALFIX_API_URL", "http://127.0.0.1:7866")
RVC_API_URL = os.getenv("RVC_API_URL", "http://127.0.0.1:7863")
RVC_VOICES_ROOT = os.getenv("RVC_VOICES_ROOT", "/rvc/assets/voices")
RVC_MODELS_ROOT = os.getenv("RVC_MODELS_ROOT", "/srv/rvc_models")
RVC_TRAIN_DATA_ROOT = os.getenv("RVC_TRAIN_DATA_ROOT", "/srv/data")

# Speaker index path (same as trainer uses)
SPK_INDEX_PATH = os.path.join(RVC_MODELS_ROOT, "speaker_index.json")
SPKREC_MODEL_ID = os.environ.get("SPKREC_MODEL_ID", "speechbrain/spkrec-ecapa-voxceleb")

# Global ECAPA model (lazy load)
_SPKREC_MODEL = None
SPK_DEVICE = "cuda" if TORCH_AVAILABLE and torch and torch.cuda.is_available() else "cpu"


def get_speaker_encoder() -> Any:
    """Lazily load ECAPA speaker encoder."""
    global _SPKREC_MODEL
    if not SPKREC_AVAILABLE:
        return None
    if _SPKREC_MODEL is None:
        _SPKREC_MODEL = EncoderClassifier.from_hparams(
            source=SPKREC_MODEL_ID,
            run_opts={"device": SPK_DEVICE},
        )
    return _SPKREC_MODEL


def compute_speaker_embedding(wav_path: str) -> Optional[np.ndarray]:
    """Compute ECAPA embedding for a WAV file."""
    encoder = get_speaker_encoder()
    if encoder is None:
        return None
    
    try:
        if TORCH_AVAILABLE and torchaudio:
            waveform, sample_rate = torchaudio.load(wav_path)
            target_sr = 16000
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            with torch.no_grad():
                emb = encoder.encode_batch(waveform.to(SPK_DEVICE))
            emb_np = emb.squeeze(0).cpu().numpy()
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / float(norm)
            return emb_np
    except Exception as ex:
        logging.warning(f"Failed to compute ECAPA embedding for {wav_path}: {ex}")
    return None


def load_speaker_index() -> Dict[str, Any]:
    """Load speaker index for matching."""
    if not os.path.exists(SPK_INDEX_PATH):
        return {}
    try:
        with open(SPK_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def find_best_matching_speaker(embedding: np.ndarray, index: Dict[str, Any]) -> Tuple[str, float]:
    """Find best matching speaker in index. Returns (speaker_id, score)."""
    best_id = ""
    best_score = 0.0
    for speaker_id, entry in index.items():
        vec = np.array(entry.get("embedding", []), dtype=np.float32)
        if vec.shape != embedding.shape:
            continue
        score = float(np.dot(vec, embedding))
        if score > best_score:
            best_score = score
            best_id = speaker_id
    return best_id, best_score


def normalize_audio_to_wav(input_path: str, output_path: str, target_sr: int = 44100) -> None:
    """Convert audio to canonical WAV format using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(target_sr), "-ac", "2",
                "-y", output_path
            ],
            check=True,
            capture_output=True,
        )
    except Exception as ex:
        raise RuntimeError(f"Failed to normalize audio: {ex}")


def extract_audio_from_video(video_path: str, output_path: str) -> None:
    """Extract audio from video using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                "-y", output_path
            ],
            check=True,
            capture_output=True,
        )
    except Exception as ex:
        raise RuntimeError(f"Failed to extract audio from video: {ex}")


def slice_audio_segment(audio_path: str, start: float, end: float, output_path: str) -> None:
    """Slice audio segment from start to end."""
    try:
        with sf.SoundFile(audio_path) as f:
            sr = f.samplerate
            start_frame = int(start * sr)
            frames_to_read = int((end - start) * sr)
            
            if frames_to_read <= 0:
                raise ValueError(f"Invalid duration for segment {start}-{end}")
                
            f.seek(start_frame)
            audio = f.read(frames_to_read)
            
            if len(audio) == 0:
                raise ValueError(f"Empty segment read {start}-{end}")
                
            sf.write(output_path, audio, sr)
    except Exception as ex:
        raise RuntimeError(f"Failed to slice segment {start}-{end}: {ex}")


def group_by_speaker(segments: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Group segments by speaker_label."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for seg in segments:
        label = seg.get("speaker_label", "unknown")
        groups.setdefault(label, []).append(seg)
    return list(groups.items())


def derive_voice_lock_id(speaker_id: str, project_id: str = "default") -> str:
    """
    Derive voice_lock_id from speaker_id and project_id.
    Example: voice:music:shadow_vs_charizard:spk-abc123
    """
    # Clean up speaker_id base
    if speaker_id.startswith("speaker:"):
        base = speaker_id.split("speaker:", 1)[1]
    elif speaker_id.startswith("char:"):
        base = speaker_id.split("char:", 1)[1]
    else:
        base = speaker_id
    
    # Sanitize base
    base = base.replace(":", "_")
    
    if project_id and project_id != "default":
        return f"voice:{project_id}:{base}"
    return f"voice:{base}"


def isolate_vocals_with_demucs(audio_path: str, project_dir: str) -> Tuple[str, str]:
    """
    Use demucs_service to extract vocals.
    Returns (vocals_path, instrumental_path).
    """
    os.makedirs(os.path.join(project_dir, "demucs"), exist_ok=True)
    
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Use timeout=None for long running operations
    with httpx.Client(timeout=None) as client:
        resp = client.post(
            DEMUCS_API_URL.rstrip("/") + "/v1/audio/stems",
            json={"b64": audio_b64, "stems": ["vocals", "other"]},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"demucs_service failed: {resp.status_code} {resp.text}")
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"demucs_service error: {data.get('error')}")
        
        stems = data.get("stems_b64", {})
        vocals_b64 = stems.get("vocals")
        other_b64 = stems.get("other")
        
        vocals_path = os.path.join(project_dir, "demucs", "vocals.wav")
        instrumental_path = os.path.join(project_dir, "demucs", "instrumental.wav")
        
        if vocals_b64:
            with open(vocals_path, "wb") as f:
                f.write(base64.b64decode(vocals_b64))
        else:
            # Fallback: use original as vocals
            logging.warning("No vocals stem, using original audio")
            shutil.copy2(audio_path, vocals_path)
        
        if other_b64:
            with open(instrumental_path, "wb") as f:
                f.write(base64.b64decode(other_b64))
        else:
            instrumental_path = vocals_path  # Fallback
        
        return vocals_path, instrumental_path


def transcribe_with_whisper(audio_path: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Use whisper_service to transcribe and get segments.
    Returns list of segments with {start, end, text, speaker_label}.
    """
    with httpx.Client(timeout=None) as client:
        resp = client.post(
            WHISPER_API_URL.rstrip("/") + "/transcribe",
            json={"audio_url": f"file://{audio_path}", "language": language},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"whisper_service failed: {resp.status_code} {resp.text}")
        data = resp.json()
        
        # If whisper service returns segments (future proofing)
        if "segments" in data and isinstance(data["segments"], list):
            return data["segments"]
            
        # Fallback: current whisper service returns just text
        text = data.get("text", "")
        
        # Get audio duration
        try:
            with sf.SoundFile(audio_path) as f:
                duration = float(len(f) / f.samplerate)
        except Exception:
            duration = 0.0
        
        # Return single segment covering whole file
        return [
            {
                "start": 0.0,
                "end": duration,
                "text": text,
                "speaker_label": "SPEAKER_00",
            }
        ]


def register_voice_with_rvc(
    voice_lock_id: str,
    primary_ref_path: str,
    additional_refs: List[str],
    model_name: Optional[str] = None,
) -> bool:
    """Register voice with rvc_service."""
    try:
        with open(primary_ref_path, "rb") as f:
            ref_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "voice_lock_id": voice_lock_id,
            "reference_wav_b64": ref_b64,
            "additional_refs": additional_refs,
        }
        if model_name:
            payload["model_name"] = model_name
        
        with httpx.Client(timeout=None) as client:
            resp = client.post(
                RVC_API_URL.rstrip("/") + "/v1/voice/register",
                json=payload,
            )
            if resp.status_code == 200:
                return True
            logging.warning(f"Failed to register voice {voice_lock_id}: {resp.text}")
            return False
    except Exception as ex:
        logging.warning(f"Failed to register voice {voice_lock_id}: {ex}")
        return False


def train_voice_with_rvc(voice_lock_id: str) -> Optional[Dict[str, Any]]:
    """
    Train voice with rvc_service.
    Returns a dictionary with training results (wrapped in 'result' key if successful).
    """
    try:
        with httpx.Client(timeout=None) as client:  # No timeout for training
            resp = client.post(
                RVC_API_URL.rstrip("/") + "/v1/voice/train",
                json={"voice_lock_id": voice_lock_id},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    # Wrap flat response into a structure compatible with downstream expectations
                    return {"ok": True, "result": data}
            logging.warning(f"Failed to train voice {voice_lock_id}: {resp.text}")
            return None
    except Exception as ex:
        logging.warning(f"Failed to train voice {voice_lock_id}: {ex}")
        return None


def mine_voices_from_audio(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mine voices from audio reference (song, podcast, raw WAV).
    """
    audio_path = job.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        return {
            "ok": False,
            "error": {
                "code": "audio_not_found",
                "message": f"Audio file not found: {audio_path}",
            },
        }
    
    project_id = job.get("project_id", "default")
    expected_speakers = job.get("expected_speakers") or []
    auto_train = bool(job.get("auto_train", False))
    language = job.get("language")
    
    project_dir = os.path.join(RVC_TRAIN_DATA_ROOT, "voice_miner", project_id)
    os.makedirs(project_dir, exist_ok=True)
    
    try:
        # Step 1: Normalize input
        normalized_path = os.path.join(project_dir, "normalized.wav")
        normalize_audio_to_wav(audio_path, normalized_path)
        
        # Step 2: Vocal/instrument separation
        logging.info("Isolating vocals using demucs")
        vocals_path, instrumental_path = isolate_vocals_with_demucs(normalized_path, project_dir)
        
        # Step 3: Speaker diarization
        logging.info("Transcribing with whisper")
        segments = transcribe_with_whisper(vocals_path, language)
        
        # Slice segments into individual WAV files
        segments_dir = os.path.join(project_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        
        for i, seg in enumerate(segments):
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                speaker_label = seg.get("speaker_label", "SPEAKER_00")
                
                seg_wav_path = os.path.join(segments_dir, f"{speaker_label}_seg_{i:03d}.wav")
                slice_audio_segment(vocals_path, start, end, seg_wav_path)
                seg["wav_path"] = seg_wav_path
            except Exception as ex:
                logging.warning(f"Failed to slice segment {i}: {ex}")
                continue
        
        # Step 4: Map diarized speakers → speaker IDs
        speaker_clusters = group_by_speaker(segments)
        results = []
        
        # Step 5: Process each speaker cluster
        for speaker_label, segs in speaker_clusters:
            if not segs:
                continue
            
            # Collect valid segments
            valid_segs = [s for s in segs if s.get("wav_path") and os.path.exists(s["wav_path"])]
            if not valid_segs:
                continue
                
            # Determine best matching speaker
            longest_seg = max(valid_segs, key=lambda s: float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
            embedding = compute_speaker_embedding(longest_seg["wav_path"])
            
            best_id = None
            best_score = 0.0
            
            if embedding is not None:
                index = load_speaker_index()
                best_id, best_score = find_best_matching_speaker(embedding, index)
            
            # Decide speaker_id
            speaker_id = f"spk-{uuid.uuid4().hex[:8]}"
            
            if best_id and best_score >= 0.80:
                speaker_id = best_id
            else:
                # Check hints
                seg_text_full = " ".join([s.get("text", "") for s in valid_segs]).lower()
                for exp in expected_speakers:
                    hint = str(exp.get("hint", "")).lower()
                    if hint and hint in seg_text_full:
                        speaker_id = exp.get("id", speaker_id)
                        break
            
            # Derive voice_lock_id and model_name
            voice_lock_id = derive_voice_lock_id(speaker_id, project_id)
            model_name = speaker_id.split(":", 1)[-1] if ":" in speaker_id else speaker_id
            
            # Step 6: Write ref WAVs
            voice_dir = os.path.join(RVC_VOICES_ROOT, voice_lock_id)
            os.makedirs(voice_dir, exist_ok=True)
            
            ref_paths = []
            for i, seg in enumerate(valid_segs):
                seg_path = seg["wav_path"]
                ref_filename = f"{voice_lock_id}__seg_{i:03d}.wav"
                ref_path = os.path.join(voice_dir, ref_filename)
                shutil.copy2(seg_path, ref_path)
                ref_paths.append(ref_path)
            
            if not ref_paths:
                continue
            
            # Step 7: Register
            register_voice_with_rvc(
                voice_lock_id,
                ref_paths[0], # Longest/primary is usually best, or just first
                ref_paths[1:],
                model_name,
            )
            
            # Step 8: Optionally train
            train_result = None
            if auto_train:
                train_result = train_voice_with_rvc(voice_lock_id)
            
            # Assemble final result using trainer feedback if available
            final_speaker_id = speaker_id
            final_model_name = model_name
            is_new_speaker = True
            
            if train_result and isinstance(train_result, dict) and train_result.get("ok"):
                res_inner = train_result.get("result") or {}
                final_speaker_id = res_inner.get("speaker_id") or speaker_id
                final_model_name = res_inner.get("model_name") or model_name
                is_new_speaker = bool(res_inner.get("is_new_speaker", True))
            else:
                # If we matched an existing ID via index, it's not new
                if best_id and best_score >= 0.80 and speaker_id == best_id:
                    is_new_speaker = False
            
            results.append({
                "speaker_id": final_speaker_id,
                "voice_lock_id": voice_lock_id,
                "model_name": final_model_name,
                "is_new_speaker": is_new_speaker,
                "matched_existing": best_id if best_score >= 0.80 else None,
                "match_score": best_score,
                "ref_paths": ref_paths,
                "train_result": train_result,
            })
        
        return {
            "ok": True,
            "speakers": results,
        }
        
    except Exception as ex:
        logging.error(f"Voice mining failed: {ex}", exc_info=True)
        return {
            "ok": False,
            "error": {
                "code": "mining_failed",
                "message": str(ex),
            },
        }


def mine_voices_from_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mine voices from video reference (film, character alignment).
    """
    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return {
            "ok": False,
            "error": {
                "code": "video_not_found",
                "message": f"Video file not found: {video_path}",
            },
        }
    
    project_id = job.get("project_id", "default")
    character_hints = job.get("character_hints") or []
    auto_train = bool(job.get("auto_train", False))
    language = job.get("language")
    
    project_dir = os.path.join(RVC_TRAIN_DATA_ROOT, "voice_miner", project_id)
    os.makedirs(project_dir, exist_ok=True)
    
    try:
        # Step 1: Extract audio
        logging.info("Extracting audio from video")
        audio_path = os.path.join(project_dir, "extracted_audio.wav")
        extract_audio_from_video(video_path, audio_path)
        
        # Step 2: Demucs (optional but good for clean dialogue)
        vocals_path, _ = isolate_vocals_with_demucs(audio_path, project_dir)
        
        # Step 3: Transcribe
        logging.info("Transcribing with whisper")
        segments = transcribe_with_whisper(vocals_path, language)
        
        # Slice segments
        segments_dir = os.path.join(project_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        
        for i, seg in enumerate(segments):
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                label = seg.get("speaker_label", "SPEAKER_00")
                seg_wav_path = os.path.join(segments_dir, f"{label}_seg_{i:03d}.wav")
                slice_audio_segment(vocals_path, start, end, seg_wav_path)
                seg["wav_path"] = seg_wav_path
            except Exception as ex:
                logging.warning(f"Failed to slice segment {i}: {ex}")
                continue
        
        # Step 4: Map segments to character IDs
        # Group by character based on hints (or face tracking in future)
        character_clusters: Dict[str, List[Dict[str, Any]]] = {}
        
        for seg in segments:
            assigned_char = None
            seg_text = seg.get("text", "").lower()
            
            # Check textual hints
            for hint in character_hints:
                char_id = hint.get("character_id")
                hint_text = str(hint.get("hint", "")).lower()
                if char_id and hint_text and hint_text in seg_text:
                    assigned_char = char_id
                    break
            
            # Fallback to speaker label if no hint match
            if not assigned_char:
                label = seg.get("speaker_label", "SPEAKER_00")
                assigned_char = f"char:{label}"
            
            character_clusters.setdefault(assigned_char, []).append(seg)
        
        # Step 5: Process per character
        results = []
        for char_id, segs in character_clusters.items():
            if not segs:
                continue
            
            valid_segs = [s for s in segs if s.get("wav_path") and os.path.exists(s["wav_path"])]
            if not valid_segs:
                continue
            
            # Speaker ID is the character ID
            speaker_id = char_id
            
            # Check against existing index to see if this character is actually a known speaker
            longest_seg = max(valid_segs, key=lambda s: float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
            embedding = compute_speaker_embedding(longest_seg["wav_path"])
            
            best_id = None
            best_score = 0.0
            if embedding is not None:
                index = load_speaker_index()
                best_id, best_score = find_best_matching_speaker(embedding, index)
            
            # If very high match, we might want to alias/merge, but for video characters
            # we usually want to keep the character ID stable (e.g. char:shadow).
            # But we can record the match for metadata.
            
            voice_lock_id = derive_voice_lock_id(speaker_id, project_id)
            model_name = speaker_id.split(":", 1)[-1] if ":" in speaker_id else speaker_id
            
            # Write refs
            voice_dir = os.path.join(RVC_VOICES_ROOT, voice_lock_id)
            os.makedirs(voice_dir, exist_ok=True)
            
            ref_paths = []
            for i, seg in enumerate(valid_segs):
                ref_path = os.path.join(voice_dir, f"{voice_lock_id}__seg_{i:03d}.wav")
                shutil.copy2(seg["wav_path"], ref_path)
                ref_paths.append(ref_path)
            
            if not ref_paths:
                continue
            
            # Register
            register_voice_with_rvc(
                voice_lock_id,
                ref_paths[0],
                ref_paths[1:],
                model_name,
            )
            
            # Train
            train_result = None
            if auto_train:
                train_result = train_voice_with_rvc(voice_lock_id)
            
            # Determine final status
            final_speaker_id = speaker_id
            final_model_name = model_name
            is_new_speaker = True
            
            if train_result and isinstance(train_result, dict) and train_result.get("ok"):
                res_inner = train_result.get("result") or {}
                final_speaker_id = res_inner.get("speaker_id") or speaker_id
                final_model_name = res_inner.get("model_name") or model_name
                is_new_speaker = bool(res_inner.get("is_new_speaker", True))
            
            results.append({
                "speaker_id": final_speaker_id,
                "voice_lock_id": voice_lock_id,
                "model_name": final_model_name,
                "is_new_speaker": is_new_speaker,
                "matched_existing": best_id if best_score >= 0.80 else None,
                "match_score": best_score,
                "ref_paths": ref_paths,
                "train_result": train_result,
            })
            
        return {
            "ok": True,
            "speakers": results,
        }

    except Exception as ex:
        logging.error(f"Video voice mining failed: {ex}", exc_info=True)
        return {
            "ok": False,
            "error": {
                "code": "mining_failed",
                "message": str(ex),
            },
        }
