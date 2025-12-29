from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import json
import math
import struct
import logging
import time
import hashlib
from io import BytesIO
from typing import Any, Dict, List, Optional
from .common import now_ts, ensure_dir, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_envelope, bump_envelope, assert_envelope
from ..artifacts.index import add_artifact as _ctx_add
from ..tracing.runtime import trace_event
from void_artifacts import build_artifact, generate_artifact_id, artifact_id_to_safe_filename

log = logging.getLogger(__name__)


def _sine_wav_bytes(freq: float, length_s: float, sample_rate: int = 22050) -> bytes:
    frames = int(max(0.05, length_s) * sample_rate)
    buf = BytesIO()
    # Write minimal WAV header + audio (mono 16-bit)
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack('<I', 36 + frames * 2))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack('<I', frames * 2))
    for i in range(frames):
        t = i / sample_rate
        s = int(32767.0 * math.sin(2.0 * math.pi * freq * t))
        buf.write(struct.pack('<h', s))
    return buf.getvalue()


def run_sfx_compose(*, trace_id: str, conversation_id: str, manifest: dict, type: str = "beep", length_s: float = 1.0, pitch: float = 440.0, seed: Any = None, lock_bundle: Optional[Dict[str, Any]] = None, sfx_event_ids: Optional[List[str]] = None, artifact_id: Optional[str] = None, **kwargs) -> dict:
    if trace_id:
        trace_event("tool.audio.sfx.compose.start", {"trace_id": trace_id, "conversation_id": conversation_id, "type": type})
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "sfx", conversation_id)
    ensure_dir(outdir)
    args = {
        "type": type or "beep",
        "length_s": float(length_s or 1.0),
        "pitch": float(pitch or 440.0),
        "seed": seed,
        # Optional lock-related hints for future SFX engines
        "lock_bundle": lock_bundle if isinstance(lock_bundle, dict) else None,
        "sfx_event_ids": sfx_event_ids if isinstance(sfx_event_ids, list) else None,
    }
    args = stamp_tool_args("audio.sfx.compose", args)
    wav = _sine_wav_bytes(freq=float(args.get("pitch") or 440.0), length_s=float(args.get("length_s") or 1.0))
    model = "builtin:sfx"
    # Generate unique artifact_id BEFORE creating file, then use it for filename
    artifact_id = generate_artifact_id(
        trace_id=trace_id,
        tool_name="audio.sfx.compose",
        conversation_id=conversation_id,
        suffix_data=len(wav),
        existing_id=artifact_id,
    )
    # Create safe filename from artifact_id (artifact_id is already sanitized, but use helper for consistency)
    safe_filename = artifact_id_to_safe_filename(artifact_id, ".wav")
    path = os.path.join(outdir, safe_filename)
    with open(path, "wb") as f: f.write(wav)
    stem = os.path.splitext(safe_filename)[0]
    # Minimal lock metadata placeholder for distillation
    sfx_locks = {}
    if isinstance(args.get("lock_bundle"), dict):
        sfx_locks["bundle"] = args.get("lock_bundle")
    # Initial placeholder SFX lock metrics (to be refined with real analysis)
    if sfx_locks:
        sfx_locks.setdefault("sfx_timbre_lock", 0.5)
        sfx_locks.setdefault("sfx_timing_lock", 1.0)
    sidecar_payload = {"tool": "audio.sfx.compose", **args, "model": model}
    if sfx_locks:
        sidecar_payload["locks"] = sfx_locks
    sidecar(path, sidecar_payload)
    add_manifest_row(manifest, path, step_id="audio.sfx.compose")
    try:
        # Persist to artifacts index for context/review selection.
        url = f"/uploads/artifacts/audio/sfx/{conversation_id}/{os.path.basename(path)}" if conversation_id else None
        _ctx_add(conversation_id, "audio", path, url, None, ["sfx"], {"type": args.get("type"), "trace_id": trace_id, "tool": "audio.sfx.compose"})
    except Exception:
        # SFX is non-critical; never fail the tool because context indexing failed.
        log.debug(f"audio.sfx.compose: context add failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r}", exc_info=True)
    env = {
        "meta": {
            "model": model,
            "ts": now_ts(),
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "step": 0,
            "state": "halt",
            "cont": {"present": False, "state_hash": None, "reason": None},
        },
        "reasoning": {"goal": "sfx", "constraints": ["json-only"], "decisions": ["sfx done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "sfx generated"},
        "tool_calls": [{"tool_name": "audio.sfx.compose", "tool": "audio.sfx.compose", "args": args, "arguments": args, "status": "done", "artifact_id": artifact_id}],
        "artifacts": [
            build_artifact(
                artifact_id=artifact_id,
                kind="audio",
                path=path,
                trace_id=trace_id,
                conversation_id=conversation_id,
                tool_name="audio.sfx.compose",
                url=(f"/uploads/artifacts/audio/sfx/{conversation_id}/{os.path.basename(path)}" if conversation_id else None),
                summary=stem,
                bytes=len(wav),
                tags=[],
            )
        ],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    if sfx_locks:
        meta_block = env.setdefault("meta", {})
        if isinstance(meta_block, dict):
            meta_block["locks"] = sfx_locks
    env = normalize_envelope(env); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "audio.sfx.compose", model)
    from ..tracing.training import append_training_sample
    try:
        append_training_sample("tts", {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "tool": "audio.sfx.compose",
            "type": args.get("type"),
            "length_s": float(args.get("length_s") or 0),
            "pitch": float(args.get("pitch") or 0),
            "seed": int(args.get("seed") or 0),
            "model": model,
            "path": path,
        })
    except Exception as exc:
        log.debug(f"audio.sfx.compose: trace append failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r} exc={exc!r}", exc_info=True)
    if trace_id:
        trace_event("tool.audio.sfx.compose.complete", {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "tool": "audio.sfx.compose",
            "model": model,
            "path": path,
            "bytes": len(wav),
            "type": args.get("type"),
        })
    log.info(f"audio.sfx.compose: completed conversation_id={conversation_id!r} trace_id={trace_id!r} model={model!r} path={path!r} bytes={len(wav)} type={args.get('type')}")
    return env


