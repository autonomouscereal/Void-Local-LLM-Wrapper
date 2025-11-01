from __future__ import annotations

import os
import json
from .common import now_ts, ensure_dir, tts_edge_defaults, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope
from ..refs.voice import resolve_voice_lock


def run_tts_speak(job: dict, provider, manifest: dict) -> dict:
    """
    job: {
      "text": str,
      "voice": str|None,
      "voice_id": str|None,
      "voice_refs": {"voice_samples":[...], "transcript": "..."}|None,
      "rate": float|None,
      "pitch": float|None,
      "sample_rate": int|None,
      "max_seconds": int|None,
      "seed": int|None,
      "cid": str|None,
      "edge": bool|None
    }
    provider: exposes .speak(args) -> {"wav_bytes": b"...", "duration_s": float, "model": "..."}
    """
    cid = job.get("cid") or f"tts-{now_ts()}"
    outdir = os.path.join("/workspace", "uploads", "artifacts", "audio", "tts", cid)
    ensure_dir(outdir)
    params = tts_edge_defaults({
        "sample_rate": job.get("sample_rate"),
        "max_seconds": job.get("max_seconds"),
        "voice": job.get("voice"),
    }, edge=bool(job.get("edge")))
    lock = resolve_voice_lock(job.get("voice_id"), job.get("voice_refs"))
    args = {
        "text": job.get("text") or "",
        "voice": params.get("voice"),
        "rate": float(job.get("rate") or 1.0),
        "pitch": float(job.get("pitch") or 0.0),
        "sample_rate": int(params.get("sample_rate") or 22050),
        "channels": 1,
        "max_seconds": int(params.get("max_seconds") or 20),
        "voice_lock": lock,
        "seed": job.get("seed"),
    }
    args = stamp_tool_args("tts.speak", args)
    res = provider.speak(args)
    wav_bytes = res.get("wav_bytes") or b""
    dur = float(res.get("duration_s") or 0.0)
    model = res.get("model", "unknown")
    stem = f"tts_{now_ts()}"; wav_path = os.path.join(outdir, stem + ".wav")
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)
    sidecar(wav_path, {
        "tool": "tts.speak", "text": args.get("text"), "voice": args.get("voice"),
        "rate": args.get("rate"), "pitch": args.get("pitch"), "sample_rate": args.get("sample_rate"),
        "max_seconds": args.get("max_seconds"), "seed": args.get("seed"), "voice_lock": lock,
        "model": model, "duration_s": dur,
    })
    add_manifest_row(manifest, wav_path, step_id="tts.speak")
    env = {
        "meta": {"model": model, "ts": now_ts(), "cid": cid, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "tts", "constraints": ["json-only", "edge-safe"], "decisions": ["tts.speak done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "tts generated"},
        "tool_calls": [{"tool": "tts.speak", "args": args, "status": "done", "result_ref": os.path.basename(wav_path)}],
        "artifacts": [{"id": os.path.basename(wav_path), "kind": "audio-ref", "summary": stem, "bytes": len(wav_bytes)}],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_to_envelope(json.dumps(env)); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "tts.speak", model)
    return env


