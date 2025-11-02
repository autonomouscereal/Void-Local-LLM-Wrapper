from __future__ import annotations

import os
import json
from .common import now_ts, ensure_dir, tts_edge_defaults, sidecar, stamp_env
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope
from ..refs.voice import resolve_voice_lock
from ..refs.registry import append_provenance
from .export import append_tts_sample
from ..context.index import add_artifact as _ctx_add
from ..context.index import list_recent as _ctx_list
import wave
import struct
import math
from ..analysis.media import analyze_audio, normalize_lufs
from ..datasets.trace import append_sample as _trace_append


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
    # If no explicit voice provided, try to reuse last voice from context
    inferred_voice = None
    try:
        recents = _ctx_list(cid, limit=10, kind_hint="audio")
        for it in reversed(recents):
            for t in (it.get("tags") or []):
                if isinstance(t, str) and t.startswith("voice:"):
                    inferred_voice = t.split(":",1)[1]
                    break
            if inferred_voice:
                break
    except Exception:
        inferred_voice = None
    args = {
        "text": job.get("text") or "",
        "voice": params.get("voice") or inferred_voice,
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
    # Committee QA: peak-normalize to -1 dBFS and hard-trim to max_seconds
    try:
        def _peak_normalize(path: str, target_dbfs: float = -1.0) -> dict:
            with wave.open(path, "rb") as wf:
                ch = wf.getnchannels(); sr = wf.getframerate(); sw = wf.getsampwidth(); n = wf.getnframes()
                pcm = wf.readframes(n)
            if sw != 2 or not pcm:
                return {"normalized": False}
            # Find peak
            peak = 0
            for i in range(0, len(pcm), 2 * ch):
                for c in range(ch):
                    s = struct.unpack('<h', pcm[i + 2*c:i + 2*c + 2])[0]
                    peak = max(peak, abs(s))
            if peak <= 0:
                return {"normalized": False}
            peak_dbfs = 20.0 * math.log10(max(1.0, peak) / 32767.0)
            gain = 10.0 ** ((target_dbfs - peak_dbfs) / 20.0)
            if gain <= 0.0 or abs(1.0 - gain) < 1e-3:
                return {"normalized": False, "peak_dbfs": peak_dbfs}
            out = bytearray(len(pcm))
            for i in range(0, len(pcm), 2 * ch):
                for c in range(ch):
                    s = struct.unpack('<h', pcm[i + 2*c:i + 2*c + 2])[0]
                    v = int(max(-32768, min(32767, s * gain)))
                    out[i + 2*c:i + 2*c + 2] = struct.pack('<h', v)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(ch); wf.setsampwidth(sw); wf.setframerate(sr)
                wf.writeframes(bytes(out))
            return {"normalized": True, "peak_dbfs": peak_dbfs, "applied_db": (target_dbfs - peak_dbfs)}
        def _hard_trim(path: str, max_seconds: float) -> dict:
            if max_seconds is None:
                return {"trimmed": False}
            with wave.open(path, "rb") as wf:
                ch = wf.getnchannels(); sr = wf.getframerate(); sw = wf.getsampwidth(); n = wf.getnframes()
                max_frames = int(max(0, max_seconds) * sr)
                if n <= max_frames:
                    return {"trimmed": False}
                wf.rewind()
                data = wf.readframes(max_frames)
            with wave.open(path, "wb") as wf2:
                wf2.setnchannels(ch); wf2.setsampwidth(sw); wf2.setframerate(sr)
                wf2.writeframes(data)
            return {"trimmed": True, "old_frames": n, "new_frames": max_frames}
        qa_norm = {"normalized": False}
        # Optional peak normalize only if requested
        if bool(job.get("normalize")):
            qa_norm = _peak_normalize(wav_path, -1.0)
        qa_trim = _hard_trim(wav_path, float(params.get("max_seconds") or 0))
        # Analyze LUFS always, normalize only if requested
        ainfo = analyze_audio(wav_path)
        if isinstance(ainfo, dict):
            qa_norm["lufs_before"] = ainfo.get("lufs")
        if bool(job.get("normalize_lufs")):
            applied = normalize_lufs(wav_path, -16.0)
            if applied is not None:
                qa_norm["lufs_gain_db"] = float(applied)
    except Exception:
        qa_norm = {"normalized": False}; qa_trim = {"trimmed": False}
    sidecar(wav_path, {
        "tool": "tts.speak", "text": args.get("text"), "voice": args.get("voice"),
        "rate": args.get("rate"), "pitch": args.get("pitch"), "sample_rate": args.get("sample_rate"),
        "max_seconds": args.get("max_seconds"), "seed": args.get("seed"), "voice_lock": lock,
        "model": model, "duration_s": dur,
        "committee": {"peak_normalize": qa_norm, "hard_trim": qa_trim},
    })
    # Emotion/pace gate: single revision to match target emotion if provided
    try:
        target_emotion = None
        try:
            if isinstance(job.get("voice_refs"), dict):
                target_emotion = job.get("voice_refs", {}).get("emotion")
        except Exception:
            target_emotion = None
        ainfo2 = analyze_audio(wav_path)
        if target_emotion and isinstance(ainfo2, dict):
            cur = (ainfo2.get("emotion") or "").lower()
            # Heuristic adjust rate/pitch to hit emotion
            if target_emotion == "excited" and cur != "excited":
                adj = dict(args)
                adj["rate"] = max(0.5, float(args.get("rate") or 1.0) * 1.15)
                adj["pitch"] = float(args.get("pitch") or 0.0) + 1.0
                res2 = provider.speak(adj)
                wb2 = res2.get("wav_bytes") or b""
                if wb2:
                    with open(wav_path, "wb") as f:
                        f.write(wb2)
                    sidecar(wav_path, {"tool": "tts.speak.committee", "target_emotion": target_emotion, "adjusted": True})
            if target_emotion == "calm" and cur != "calm":
                adj = dict(args)
                adj["rate"] = max(0.5, float(args.get("rate") or 1.0) * 0.9)
                adj["pitch"] = float(args.get("pitch") or 0.0) - 1.0
                res2 = provider.speak(adj)
                wb2 = res2.get("wav_bytes") or b""
                if wb2:
                    with open(wav_path, "wb") as f:
                        f.write(wb2)
                    sidecar(wav_path, {"tool": "tts.speak.committee", "target_emotion": target_emotion, "adjusted": True})
    except Exception:
        pass
    try:
        append_tts_sample(outdir, {
            "text": args.get("text"),
            "voice": args.get("voice"),
            "rate": args.get("rate"),
            "pitch": args.get("pitch"),
            "sample_rate": args.get("sample_rate"),
            "seed": int(args.get("seed") or 0),
            "voice_lock": bool(lock),
            "audio_ref": wav_path,
            "duration_s": dur,
            "model": model,
            "created_at": now_ts(),
        })
    except Exception:
        pass
    try:
        if job.get("voice_id"):
            append_provenance(job.get("voice_id"), {"when": now_ts(), "tool": "tts.speak", "artifact": wav_path, "seed": int(args.get("seed") or 0)})
    except Exception:
        pass
    add_manifest_row(manifest, wav_path, step_id="tts.speak")
    try:
        _ctx_add(cid, "audio", wav_path, None, None, ["voice", f"voice:{args.get('voice')}"] if args.get("voice") else ["voice"], {"model": model})
    except Exception:
        pass
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
    # Trace row for distillation
    try:
        ainfo = analyze_audio(wav_path)
        _trace_append("tts", {
            "cid": cid,
            "tool": "tts.speak",
            "text": args.get("text"),
            "voice": args.get("voice"),
            "rate": args.get("rate"),
            "pitch": args.get("pitch"),
            "seed": int(args.get("seed") or 0),
            "model": model,
            "path": wav_path,
            "lufs": ainfo.get("lufs") if isinstance(ainfo, dict) else None,
            "tempo_bpm": ainfo.get("tempo_bpm") if isinstance(ainfo, dict) else None,
            "emotion": ainfo.get("emotion") if isinstance(ainfo, dict) else None,
        })
    except Exception:
        pass
    return env


