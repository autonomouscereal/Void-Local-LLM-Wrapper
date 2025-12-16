from __future__ import annotations

import os
import json
import math
import struct
from io import BytesIO
from .common import now_ts, ensure_dir, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_to_envelope, bump_envelope, assert_envelope


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


def run_sfx_compose(job: dict, manifest: dict) -> dict:
    cid = job.get("cid") or f"sfx-{now_ts()}"
    outdir = os.path.join("/workspace", "uploads", "artifacts", "audio", "sfx", cid)
    ensure_dir(outdir)
    args = {
        "type": job.get("type") or "beep",
        "length_s": float(job.get("length_s") or 1.0),
        "pitch": float(job.get("pitch") or 440.0),
        "seed": job.get("seed"),
        # Optional lock-related hints for future SFX engines
        "lock_bundle": job.get("lock_bundle") if isinstance(job.get("lock_bundle"), dict) else None,
        "sfx_event_ids": job.get("sfx_event_ids") if isinstance(job.get("sfx_event_ids"), list) else None,
    }
    args = stamp_tool_args("audio.sfx.compose", args)
    wav = _sine_wav_bytes(freq=float(args.get("pitch") or 440.0), length_s=float(args.get("length_s") or 1.0))
    model = "builtin:sfx"
    stem = f"sfx_{now_ts()}"; path = os.path.join(outdir, stem + ".wav")
    with open(path, "wb") as f: f.write(wav)
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
    env = {
        "meta": {
            "model": model,
            "ts": now_ts(),
            "cid": cid,
            "step": 0,
            "state": "halt",
            "cont": {"present": False, "state_hash": None, "reason": None},
        },
        "reasoning": {"goal": "sfx", "constraints": ["json-only"], "decisions": ["sfx done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "sfx generated"},
        "tool_calls": [{"tool": "audio.sfx.compose", "args": args, "status": "done", "result_ref": os.path.basename(path)}],
        "artifacts": [{"id": os.path.basename(path), "kind": "audio-ref", "summary": stem, "bytes": len(wav)}],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    if sfx_locks:
        meta_block = env.setdefault("meta", {})
        if isinstance(meta_block, dict):
            meta_block["locks"] = sfx_locks
    env = normalize_to_envelope(json.dumps(env)); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "audio.sfx.compose", model)
    return env


