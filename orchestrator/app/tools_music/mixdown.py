from __future__ import annotations

import os
import json
import wave
import struct
from .common import now_ts, ensure_dir, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope
from ..refs.registry import append_provenance
from ..context.index import add_artifact as _ctx_add
from ..datasets.trace import append_sample as _trace_append


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


def run_music_mixdown(job: dict, manifest: dict) -> dict:
    cid = job.get("cid") or f"music-{now_ts()}"
    outdir = os.path.join("/workspace", "uploads", "artifacts", "music", cid); ensure_dir(outdir)
    args = {
        "stems": job.get("stems") or [],
        "sample_rate": int(job.get("sample_rate") or 44100),
        "channels": int(job.get("channels") or 2),
        "seed": job.get("seed"),
    }
    args = stamp_tool_args("music.mixdown", args)
    pcm = _mix(args["stems"], args["sample_rate"], args["channels"])
    stem = f"mix_{now_ts()}"; path = os.path.join(outdir, stem + ".wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(args["channels"]); wf.setsampwidth(2); wf.setframerate(args["sample_rate"])
        wf.writeframes(pcm)
    sidecar(path, {"tool": "music.mixdown", **args})
    try:
        if isinstance(args.get("stems"), list):
            # If a higher-level music_id was provided in job, record it; otherwise record per-stem provenance is optional
            mid = (job.get("music_id") if isinstance(job, dict) else None)
            if mid:
                append_provenance(mid, {"when": now_ts(), "tool": "music.mixdown", "artifact": path, "seed": int(args.get("seed") or 0)})
    except Exception:
        pass
    add_manifest_row(manifest, path, step_id="music.mixdown")
    try:
        _ctx_add(cid, "audio", path, None, None, ["music", "mixdown"], {})
    except Exception:
        pass
    try:
        _trace_append("music", {
            "cid": cid,
            "tool": "music.mixdown",
            "stems": args.get("stems"),
            "seed": int(args.get("seed") or 0),
            "path": path,
        })
    except Exception:
        pass
    env = {
        "meta": {"model": "mix-local", "ts": now_ts(), "cid": cid, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "music mixdown", "constraints": ["json-only"], "decisions": ["music.mixdown done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "music mixed"},
        "tool_calls": [{"tool": "music.mixdown", "args": args, "status": "done", "result_ref": os.path.basename(path)}],
        "artifacts": [{"id": os.path.basename(path), "kind": "audio-ref", "summary": stem, "bytes": len(pcm)}],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_to_envelope(json.dumps(env)); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "music.mixdown", "mix-local")
    return env


