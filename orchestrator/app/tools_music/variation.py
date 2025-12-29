from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import json
import wave
import struct
import logging
import time
import hashlib
from typing import Any, List
from .common import now_ts, ensure_dir, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_envelope, bump_envelope, assert_envelope
from ..ref_library.registry import append_provenance
from ..artifacts.index import add_artifact as _ctx_add
from ..artifacts.index import resolve_reference as _ctx_resolve, resolve_global as _glob_resolve
from ..tracing.training import append_training_sample
from ..tracing.runtime import trace_event
from void_artifacts import build_artifact, generate_artifact_id, artifact_id_to_safe_filename

log = logging.getLogger(__name__)


def _read_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate(); ch = wf.getnchannels(); sw = wf.getsampwidth(); n = wf.getnframes()
        data = wf.readframes(n)
    return sr, ch, sw, n, data


def _write_wav(path: str, sr: int, ch: int, sw: int, frames: bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(sw); wf.setframerate(sr)
        wf.writeframes(frames)


def _apply_gain(data: bytes, sw: int, gain: float) -> bytes:
    if sw != 2:
        return data
    out = bytearray(len(data))
    for i in range(0, len(data), 2):
        s = struct.unpack('<h', data[i:i+2])[0]
        v = int(max(-32768, min(32767, s * gain)))
        out[i:i+2] = struct.pack('<h', v)
    return bytes(out)


def run_music_variation(*, manifest: dict, trace_id: str = "", conversation_id: str = "", desc: str = "", prompt: str = "", variation_of: str = "", n: int = 1, intensity: float = 0.4, seed: Any = None, music_refs: List[str] | None = None, artifact_id: str | None = None, **kwargs) -> dict:
    if trace_id:
        trace_event("tool.music.variation.start", {"trace_id": trace_id, "conversation_id": conversation_id})
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", conversation_id)
    ensure_dir(outdir)
    # Use music_refs directly if provided, otherwise use artifact_id to build lock
    lock = music_refs if isinstance(music_refs, dict) else {}
    if artifact_id and not lock:
        lock = {"artifact_id": artifact_id}
    base_hint = str(desc or prompt or "")
    base_path = variation_of
    if not base_path:
        try:
            rec = _ctx_resolve(conversation_id, base_hint, "audio")
            if rec and isinstance(rec.get("path"), str):
                base_path = rec.get("path")
            if not base_path:
                gre = _glob_resolve(base_hint, "audio")
                if gre and isinstance(gre.get("path"), str):
                    base_path = gre.get("path")
        except Exception as exc:
            log.debug(f"music.variation: failed to resolve base audio from context (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r} exc={exc!r}", exc_info=True)
            base_path = None
    args = {
        "variation_of": base_path,
        "n": max(1, min(int(n or 1), 4)),
        "intensity": float(intensity or 0.4),
        "music_lock": lock,
        "seed": seed,
    }
    args = stamp_tool_args("music.variation", args)
    base = args.get("variation_of")
    artifacts = []
    try:
        sr, ch, sw, n, pcm = _read_wav(base)
    except Exception:
        sr, ch, sw, n, pcm = 44100, 2, 2, 0, b""
    for i in range(1, args["n"] + 1):
        g = max(0.5, min(1.2, 1.0 - i * (args["intensity"] * 0.2)))
        frames = _apply_gain(pcm, sw, g) if pcm else b""
        # Generate unique artifact_id BEFORE creating file, then use it for filename
        variant_artifact_id = generate_artifact_id(
            trace_id=trace_id,
            tool_name="music.variation",
            conversation_id=conversation_id,
            suffix_data=f"{i}:{len(frames)}",
            existing_id=f"{artifact_id}:variant_{i}" if artifact_id else None,
        )
        # Create safe filename from artifact_id (artifact_id is already sanitized, but use helper for consistency)
        safe_filename = artifact_id_to_safe_filename(variant_artifact_id, ".wav")
        path = os.path.join(outdir, safe_filename)
        _write_wav(path, sr, ch, sw, frames)
        stem = os.path.splitext(safe_filename)[0]
        sidecar(path, {"tool": "music.variation", **args, "variant_index": i, "gain": g})
        add_manifest_row(manifest, path, step_id="music.variation")
        artifacts.append(
            build_artifact(
                artifact_id=variant_artifact_id,
                kind="audio",
                path=path,
                trace_id=trace_id,
                conversation_id=conversation_id,
                tool_name="music.variation",
                url=(f"/uploads/artifacts/music/{conversation_id}/{os.path.basename(path)}" if conversation_id else None),
                summary=stem,
                tags=[],
                variant_index=i,
            )
        )
        try:
            url = f"/uploads/artifacts/music/{conversation_id}/{os.path.basename(path)}" if conversation_id else None
            _ctx_add(conversation_id, "audio", path, url, args.get("variation_of"), ["music", "variant"], {"trace_id": trace_id, "tool": "music.variation", "variant_index": i})
        except Exception as exc:
            log.debug(f"music.variation: context add failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r} exc={exc!r}", exc_info=True)
        try:
            append_training_sample("music", {
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "tool": "music.variation",
                "variation_of": args.get("variation_of"),
                "index": i,
                "seed": int(args.get("seed") or 0),
                "path": path,
            })
        except Exception as exc:
            log.debug(f"music.variation: trace append failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r} exc={exc!r}", exc_info=True)
        try:
            if variant_artifact_id:
                append_provenance(variant_artifact_id, {"when": now_ts(), "tool": "music.variation", "artifact": path, "seed": int(args.get("seed") or 0)})
        except Exception as exc:
            log.debug(f"music.variation: append_provenance failed (non-fatal) conversation_id={conversation_id!r} exc={exc!r}", exc_info=True)
    env = {
        "meta": {"model": "variation-local", "ts": now_ts(), "conversation_id": conversation_id, "trace_id": trace_id, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "music variation", "constraints": ["json-only"], "decisions": [f"music.variation x{len(artifacts)} done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "music variations generated"},
        "tool_calls": [{"tool_name": "music.variation", "tool": "music.variation", "args": args, "arguments": args, "status": "done", "artifact_id": (artifacts[0].get("artifact_id") if artifacts and isinstance(artifacts[0], dict) else None)}],
        "artifacts": artifacts,
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_envelope(env); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "music.variation", "variation-local")
    if trace_id:
        trace_event("tool.music.variation.complete", {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "tool": "music.variation",
            "variants": len(artifacts),
            "paths": [a.get("path") for a in artifacts if isinstance(a.get("path"), str)],
        })
    log.info(f"music.variation: completed conversation_id={conversation_id!r} trace_id={trace_id!r} variants={len(artifacts)}")
    return env


