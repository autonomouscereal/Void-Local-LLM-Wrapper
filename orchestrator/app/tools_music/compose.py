from __future__ import annotations

import os
import json
import base64
from .common import now_ts, ensure_dir, music_edge_defaults, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope
from ..refs.music import resolve_music_lock
from ..refs.registry import append_provenance
from .export import append_music_sample


def run_music_compose(job: dict, provider, manifest: dict) -> dict:
    cid = job.get("cid") or f"music-{now_ts()}"
    outdir = os.path.join("/workspace", "uploads", "artifacts", "music", cid)
    ensure_dir(outdir)
    params = music_edge_defaults({
        "bpm": job.get("bpm"),
        "length_s": job.get("length_s"),
        "sample_rate": job.get("sample_rate"),
        "channels": job.get("channels"),
        "structure": job.get("structure"),
    }, edge=bool(job.get("edge")))
    lock = resolve_music_lock(job.get("music_id"), job.get("music_refs"))
    args = {
        "prompt": job.get("prompt") or "",
        "bpm": params.get("bpm"),
        "length_s": params.get("length_s"),
        "structure": params.get("structure"),
        "sample_rate": params.get("sample_rate"),
        "channels": params.get("channels"),
        "music_lock": lock,
        "seed": job.get("seed"),
    }
    args = stamp_tool_args("music.compose", args)
    res = provider.compose(args)
    wav = res.get("wav_bytes") or b""
    model = res.get("model", "music")
    stems = res.get("stems") or []
    stem = f"compose_{now_ts()}"
    track_path = os.path.join(outdir, stem + ".wav")
    with open(track_path, "wb") as f:
        f.write(wav)
    stems_meta = []
    if stems:
        stems_dir = os.path.join(outdir, "stems"); ensure_dir(stems_dir)
        for s in stems:
            sp = os.path.join(stems_dir, f"{s.get('name','stem')}.wav")
            with open(sp, "wb") as f:
                f.write(s.get("wav_bytes") or b"")
            stems_meta.append({"name": s.get("name"), "path": sp})
    sidecar(track_path, {"tool": "music.compose", **args, "model": model, "stems": stems_meta})
    try:
        if job.get("music_id"):
            append_provenance(job.get("music_id"), {"when": now_ts(), "tool": "music.compose", "artifact": track_path, "seed": int(args.get("seed") or 0)})
    except Exception:
        pass
    try:
        append_music_sample(outdir, {
            "prompt": job.get("prompt"),
            "bpm": args.get("bpm"),
            "length_s": args.get("length_s"),
            "structure": args.get("structure"),
            "seed": int(args.get("seed") or 0),
            "music_lock": bool(lock),
            "track_ref": track_path,
            "stems": [s.get("path") for s in stems_meta],
            "model": model,
            "created_at": now_ts(),
        })
    except Exception:
        pass
    add_manifest_row(manifest, track_path, step_id="music.compose")
    env = {
        "meta": {"model": model, "ts": now_ts(), "cid": cid, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "music composition", "constraints": ["json-only", "edge-safe"], "decisions": ["music.compose done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "music composed"},
        "tool_calls": [{"tool": "music.compose", "args": args, "status": "done", "result_ref": os.path.basename(track_path)}],
        "artifacts": [{"id": os.path.basename(track_path), "kind": "audio-ref", "summary": stem, "bytes": len(wav)}],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_to_envelope(json.dumps(env)); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "music.compose", model)
    return env


