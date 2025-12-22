from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import json
import wave
import struct
import logging
from .common import now_ts, ensure_dir, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_to_envelope, bump_envelope, assert_envelope
from ..locks.music_lock import resolve_music_lock
from ..ref_library.registry import append_provenance
from ..artifacts.index import add_artifact as _ctx_add
from ..artifacts.index import resolve_reference as _ctx_resolve, resolve_global as _glob_resolve
from ..tracing.training import append_training_sample

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


def run_music_variation(job: dict, manifest: dict) -> dict:
    cid = job.get("cid") or f"music-{now_ts()}"; outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", cid); ensure_dir(outdir)
    lock = resolve_music_lock(job.get("music_id"), job.get("music_refs"))
    base_hint = str(job.get("desc") or job.get("prompt") or "")
    base_path = job.get("variation_of")
    if not base_path:
        try:
            rec = _ctx_resolve(cid, base_hint, "audio")
            if rec and isinstance(rec.get("path"), str):
                base_path = rec.get("path")
            if not base_path:
                gre = _glob_resolve(base_hint, "audio")
                if gre and isinstance(gre.get("path"), str):
                    base_path = gre.get("path")
        except Exception as exc:
            log.debug("music.variation: failed to resolve base audio from context (non-fatal) cid=%s: %s", cid, exc, exc_info=True)
            base_path = None
    args = {
        "variation_of": base_path,
        "n": max(1, min(int(job.get("n") or 1), 4)),
        "intensity": float(job.get("intensity") or 0.4),
        "music_lock": lock,
        "seed": job.get("seed"),
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
        stem = f"var_{i}_{now_ts()}"; path = os.path.join(outdir, stem + ".wav")
        _write_wav(path, sr, ch, sw, frames)
        sidecar(path, {"tool": "music.variation", **args, "variant_index": i, "gain": g})
        add_manifest_row(manifest, path, step_id="music.variation")
        artifacts.append({"id": os.path.basename(path), "kind": "audio-ref", "summary": stem})
        try:
            _ctx_add(cid, "audio", path, None, args.get("variation_of"), ["music", "variant"], {})
        except Exception as exc:
            log.debug("music.variation: context add failed (non-fatal) cid=%s: %s", cid, exc, exc_info=True)
        try:
            append_training_sample("music", {
                "cid": cid,
                "tool": "music.variation",
                "variation_of": args.get("variation_of"),
                "index": i,
                "seed": int(args.get("seed") or 0),
                "path": path,
            })
        except Exception as exc:
            log.debug("music.variation: trace append failed (non-fatal) cid=%s: %s", cid, exc, exc_info=True)
        try:
            if job.get("music_id"):
                append_provenance(job.get("music_id"), {"when": now_ts(), "tool": "music.variation", "artifact": path, "seed": int(args.get("seed") or 0)})
        except Exception as exc:
            log.debug("music.variation: append_provenance failed (non-fatal) cid=%s: %s", cid, exc, exc_info=True)
    env = {
        "meta": {"model": "variation-local", "ts": now_ts(), "cid": cid, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "music variation", "constraints": ["json-only"], "decisions": [f"music.variation x{len(artifacts)} done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "music variations generated"},
        "tool_calls": [{"tool": "music.variation", "args": args, "status": "done", "result_ref": artifacts[0]["id"] if artifacts else None}],
        "artifacts": artifacts,
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_to_envelope(json.dumps(env)); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "music.variation", "variation-local")
    return env


