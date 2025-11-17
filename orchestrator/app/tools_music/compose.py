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
from ..context.index import add_artifact as _ctx_add
import wave
import struct
from ..analysis.media import analyze_audio, normalize_lufs
from ..music.style_pack import style_score_for_track  # type: ignore
from ..datasets.trace import append_sample as _trace_append


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
    # Committee QA: try tempo detection and optional single revision if bpm far off; normalize peak
    qa = {"tempo_bpm": None, "rev": False, "rev_reason": None, "peak_dbfs": None}
    try:
        # peak normalize to -1 dBFS
        def _peak_dbfs_and_normalize(path: str, target_dbfs: float = -1.0) -> float | None:
            with wave.open(path, "rb") as wf:
                ch = wf.getnchannels(); sr = wf.getframerate(); sw = wf.getsampwidth(); n = wf.getnframes()
                pcm = wf.readframes(n)
            if sw != 2 or not pcm:
                return None
            peak = 0
            for i in range(0, len(pcm), 2 * ch):
                for c in range(ch):
                    s = struct.unpack('<h', pcm[i + 2*c:i + 2*c + 2])[0]
                    peak = max(peak, abs(s))
            if peak <= 0:
                return None
            import math
            peak_dbfs = 20.0 * math.log10(max(1.0, peak) / 32767.0)
            gain = 10.0 ** ((target_dbfs - peak_dbfs) / 20.0)
            if abs(1.0 - gain) < 1e-3:
                return peak_dbfs
            out = bytearray(len(pcm))
            for i in range(0, len(pcm), 2 * ch):
                for c in range(ch):
                    s = struct.unpack('<h', pcm[i + 2*c:i + 2*c + 2])[0]
                    v = int(max(-32768, min(32767, s * gain)))
                    out[i + 2*c:i + 2*c + 2] = struct.pack('<h', v)
            with wave.open(path, "wb") as wf2:
                wf2.setnchannels(ch); wf2.setsampwidth(sw); wf2.setframerate(sr)
                wf2.writeframes(bytes(out))
            return peak_dbfs
        # Optional peak normalize only if requested
        if bool(job.get("normalize") or (isinstance(lock, dict) and lock.get("normalize"))):
            qa["peak_dbfs"] = _peak_dbfs_and_normalize(track_path)
        # Analyze LUFS always, but normalize only if requested
        ainfo = analyze_audio(track_path)
        qa["lufs_before"] = ainfo.get("lufs") if isinstance(ainfo, dict) else None
        if bool(job.get("normalize_lufs") or (isinstance(lock, dict) and lock.get("normalize_lufs")) or (isinstance(job.get("post"), dict) and job.get("post", {}).get("normalize_lufs"))):
            lufs_gain = normalize_lufs(track_path, -14.0)
            qa["lufs_gain_db"] = float(lufs_gain) if lufs_gain is not None else None
        # tempo detection
        target_bpm = float(args.get("bpm") or 0)
        detected = None
        if target_bpm > 0:
            try:
                import librosa  # type: ignore
                import numpy as np  # type: ignore
                import soundfile as sf  # type: ignore
                y, sr = sf.read(track_path)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                detected = float(tempo)
                qa["tempo_bpm"] = detected
            except Exception:
                detected = None
        # Emotion/genre enforcement single pass based on lock
        target_emotion = None
        try:
            if isinstance(lock, dict):
                target_emotion = lock.get("emotion")
        except Exception:
            target_emotion = None
        if target_emotion and isinstance(ainfo, dict):
            cur_em = str(ainfo.get("emotion") or "")
            if target_emotion != cur_em:
                rev_args0 = dict(args)
                rev_args0["prompt"] = f"{args.get('prompt')} {target_emotion} mood, matching timbre"
                revx = provider.compose(rev_args0)
                wb = revx.get("wav_bytes") or b""
                if wb:
                    with open(track_path, "wb") as f:
                        f.write(wb)
                    qa["rev_emotion"] = target_emotion
                    ainfo = analyze_audio(track_path)
                    qa["emotion_after"] = ainfo.get("emotion") if isinstance(ainfo, dict) else None
        # Tempo enforcement only if explicitly requested; allow swing/rubato profiles
        strict = bool(job.get("bpm_strict") or job.get("tempo_enforce") or (isinstance(lock, dict) and (lock.get("bpm_strict") or lock.get("tempo_enforce"))))
        tempo_profile = (job.get("tempo_profile") or "").strip().lower()
        if tempo_profile in ("swing", "rubato", "freeform"):
            strict = False
        if target_bpm > 0 and strict:
            passes = 0
            while detected and abs(detected - target_bpm) / target_bpm > 0.2 and passes < 2:
                rev_args = dict(args)
                rev_args["prompt"] = f"{args.get('prompt')} tempo {int(target_bpm)} BPM, steady groove, metronomic"
                rev = provider.compose(rev_args)
                rbytes = rev.get("wav_bytes") or b""
                if rbytes:
                    with open(track_path, "wb") as f:
                        f.write(rbytes)
                    qa["rev"] = True; qa["rev_reason"] = "tempo_mismatch"; qa["rev_pass"] = passes
                    # re-analyze
                    ainfo2 = analyze_audio(track_path)
                    detected = float(ainfo2.get("tempo_bpm") or detected)
                passes += 1
    except Exception:
        pass
    stems_meta = []
    if stems:
        stems_dir = os.path.join(outdir, "stems"); ensure_dir(stems_dir)
        for s in stems:
            sp = os.path.join(stems_dir, f"{s.get('name','stem')}.wav")
            with open(sp, "wb") as f:
                f.write(s.get("wav_bytes") or b"")
            stems_meta.append({"name": s.get("name"), "path": sp})
    # Optional style pack scoring if present in the lock.
    style_score = None
    if isinstance(lock, dict):
        sp = lock.get("style_pack")
        if isinstance(sp, dict):
            style_score = style_score_for_track(track_path, sp)
            qa["style_score"] = style_score
    sidecar(track_path, {"tool": "music.compose", **args, "model": model, "stems": stems_meta, "committee": qa})
    try:
        if job.get("music_id"):
            append_provenance(job.get("music_id"), {"when": now_ts(), "tool": "music.compose", "artifact": track_path, "seed": int(args.get("seed") or 0)})
    except Exception:
        pass
    try:
        _ctx_add(cid, "audio", track_path, None, None, ["music", "track"], {"model": model})
        for s in stems_meta:
            _ctx_add(cid, "audio", s.get("path"), None, track_path, ["music", f"stem:{s.get('name') or 'stem'}"], {"model": model})
    except Exception:
        pass
    try:
        append_music_sample(
            outdir,
            {
                "prompt": job.get("prompt"),
                "bpm": args.get("bpm"),
                "length_s": args.get("length_s"),
                "structure": args.get("structure"),
                "seed": int(args.get("seed") or 0),
                "music_lock": bool(lock),
                "track_ref": track_path,
                "stems": [s.get("path") for s in stems_meta],
                "model": model,
                "style_score": style_score,
                "created_at": now_ts(),
            },
        )
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
    # Trace row for distillation
    try:
        ainfo3 = analyze_audio(track_path)
        _trace_append(
            "music",
            {
                "cid": cid,
                "tool": "music.compose",
                "prompt": job.get("prompt"),
                "bpm": args.get("bpm"),
                "length_s": args.get("length_s"),
                "seed": int(args.get("seed") or 0),
                "music_lock": lock,
                "model": model,
                "path": track_path,
                "lufs": ainfo3.get("lufs") if isinstance(ainfo3, dict) else None,
                "tempo_bpm": ainfo3.get("tempo_bpm") if isinstance(ainfo3, dict) else None,
                "key": ainfo3.get("key") if isinstance(ainfo3, dict) else None,
                "genre": ainfo3.get("genre") if isinstance(ainfo3, dict) else None,
                "style_score": style_score,
            },
        )
    except Exception:
        pass
    return env


