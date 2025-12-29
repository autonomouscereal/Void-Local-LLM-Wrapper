from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import json
import wave
import struct
import logging
import time
import hashlib
from typing import Any, Dict, List
from .common import now_ts, ensure_dir, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_envelope, bump_envelope, assert_envelope
from ..ref_library.registry import append_provenance
from ..artifacts.index import add_artifact as _ctx_add
from ..tracing.training import append_training_sample
from ..tracing.runtime import trace_event
from void_artifacts import build_artifact, generate_artifact_id, artifact_id_to_safe_filename

log = logging.getLogger(__name__)


def _read_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate(); ch = wf.getnchannels(); sw = wf.getsampwidth(); n = wf.getnframes()
        data = wf.readframes(n)
    return sr, ch, sw, n, data


def _mix(stems: list, sample_rate: int, channels: int) -> bytes:
    # stems: [{path, gain_db, pan(-1..+1)}]
    # Assumes 16-bit PCM; simple sum with gains and pan to target channels.
    frames_list = []
    min_len = None
    for st in stems:
        sr, ch, sw, n, data = _read_wav(st.get("path"))
        if sw != 2:
            continue
        if sr != sample_rate:
            # naive drop/keep: if mismatch, skip for safety
            continue
        frames_list.append((ch, data, float(st.get("gain_db") or 0.0), float(st.get("pan") or 0.0)))
        if min_len is None or len(data) < min_len:
            min_len = len(data)
    if not frames_list:
        return b""
    if min_len is None:
        min_len = 0
    out = bytearray(min_len if channels == 1 else min_len)
    for i in range(0, min_len, 2 if channels == 1 else 4):
        acc_l = 0.0; acc_r = 0.0
        for (ch, data, gain_db, pan) in frames_list:
            # sample(s) from source
            if ch == 1:
                s = struct.unpack('<h', data[i:i+2])[0]
                left = right = s
            else:
                s_l = struct.unpack('<h', data[i:i+2])[0]
                s_r = struct.unpack('<h', data[i+2:i+4])[0]
                left = s_l; right = s_r
            g = pow(10.0, gain_db / 20.0)
            # simple constant power pan
            pl = max(0.0, min(1.0, 0.5 * (1.0 - pan)))
            pr = max(0.0, min(1.0, 0.5 * (1.0 + pan)))
            acc_l += left * g * pl
            acc_r += right * g * pr
        l = int(max(-32768, min(32767, acc_l)))
        r = int(max(-32768, min(32767, acc_r)))
        if channels == 1:
            m = int(max(-32768, min(32767, (l + r) / 2)))
            out[i:i+2] = struct.pack('<h', m)
        else:
            out[i:i+2] = struct.pack('<h', l)
            out[i+2:i+4] = struct.pack('<h', r)
    return bytes(out)


def run_music_mixdown(
    *,
    conversation_id: str,
    trace_id: str,
    stems: List[Dict[str, Any]],
    sample_rate: int,
    channels: int,
    seed: Any,
    manifest: Dict[str, Any],
    artifact_id: str | None = None,
) -> Dict[str, Any]:
    if trace_id:
        trace_event("tool.music.mixdown.start", {"trace_id": trace_id, "conversation_id": conversation_id, "stems_count": len(stems or [])})
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", conversation_id)
    ensure_dir(outdir)
    args = {
        "stems": stems,
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "seed": seed,
    }
    args = stamp_tool_args("music.mixdown", args)
    pcm = _mix(args["stems"], args["sample_rate"], args["channels"])
    # Generate unique artifact_id BEFORE creating file, then use it for filename
    artifact_id = generate_artifact_id(
        trace_id=trace_id,
        tool_name="music.mixdown",
        conversation_id=conversation_id,
        suffix_data=len(pcm),
        existing_id=artifact_id,
    )
    # Create safe filename from artifact_id (artifact_id is already sanitized, but use helper for consistency)
    safe_filename = artifact_id_to_safe_filename(artifact_id, ".wav")
    path = os.path.join(outdir, safe_filename)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(args["channels"]); wf.setsampwidth(2); wf.setframerate(args["sample_rate"])
        wf.writeframes(pcm)
    stem = os.path.splitext(safe_filename)[0]
    sidecar(path, {"tool": "music.mixdown", **args})
    try:
        if isinstance(args.get("stems"), list) and artifact_id:
            append_provenance(artifact_id, {"when": now_ts(), "tool": "music.mixdown", "artifact": path, "seed": int(args.get("seed") or 0)})
    except Exception as e:
        log.debug(f"music.mixdown: append_provenance failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r} exc={e!r}", exc_info=True)
    add_manifest_row(manifest, path, step_id="music.mixdown")
    try:
        url = f"/uploads/artifacts/music/{conversation_id}/{os.path.basename(path)}"
        _ctx_add(conversation_id, "audio", path, url, None, ["music", "mixdown"], {"trace_id": trace_id, "tool": "music.mixdown"})
    except Exception as e:
        log.debug(f"music.mixdown: context add failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r} exc={e!r}", exc_info=True)
    try:
        append_training_sample("music", {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "tool": "music.mixdown",
            "stems": args.get("stems"),
            "seed": int(args.get("seed") or 0),
            "path": path,
        })
    except Exception as e:
        log.debug(f"music.mixdown: trace append failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r} exc={e!r}", exc_info=True)
    env = {
        "meta": {"model": "mix-local", "ts": now_ts(), "conversation_id": conversation_id, "trace_id": trace_id, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "music mixdown", "constraints": ["json-only"], "decisions": ["music.mixdown done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "music mixed"},
        "tool_calls": [{"tool_name": "music.mixdown", "tool": "music.mixdown", "args": args, "arguments": args, "status": "done", "artifact_id": artifact_id}],
        "artifacts": [
            build_artifact(
                artifact_id=artifact_id,
                kind="audio",
                path=path,
                trace_id=trace_id,
                conversation_id=conversation_id,
                tool_name="music.mixdown",
                url=f"/uploads/artifacts/music/{conversation_id}/{os.path.basename(path)}",
                summary=stem,
                bytes=len(pcm),
                tags=[],
            )
        ],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_envelope(env); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "music.mixdown", "mix-local")
    if trace_id:
        trace_event("tool.music.mixdown.complete", {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "tool": "music.mixdown",
            "path": path,
            "bytes": len(pcm),
            "stems": len(args.get("stems") or []),
        })
    log.info(f"music.mixdown: completed conversation_id={conversation_id!r} trace_id={trace_id!r} path={path!r} bytes={len(pcm)} stems={len(args.get('stems') or [])}")
    return env


