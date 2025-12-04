from __future__ import annotations

import os
import json
from typing import Any, Dict
import logging
from .common import now_ts, ensure_dir, tts_edge_defaults, sidecar, stamp_env
from ..locks import voice_embedding_from_path, tts_get_global, tts_get_voices
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
import httpx  # type: ignore


log = logging.getLogger(__name__)
def _cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return None
    num = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
    den_a = math.sqrt(sum(float(x) * float(x) for x in vec_a))
    den_b = math.sqrt(sum(float(y) * float(y) for y in vec_b))
    if den_a == 0.0 or den_b == 0.0:
        return None
    return num / (den_a * den_b)


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
    params = tts_edge_defaults(
        {
            "sample_rate": job.get("sample_rate"),
            "max_seconds": job.get("max_seconds"),
            "voice": job.get("voice"),
        },
        edge=bool(job.get("edge")),
    )
    lock = resolve_voice_lock(job.get("voice_id"), job.get("voice_refs"))
    lock_bundle = job.get("lock_bundle") if isinstance(job.get("lock_bundle"), dict) else None
    quality_profile = job.get("quality_profile")
    # TTS-specific lock defaults from lock_bundle.tts when present
    tts_global: Dict[str, Any] = {}
    tts_voice_id: str | None = None
    if lock_bundle:
        tts_global = tts_get_global(lock_bundle)
        if isinstance(tts_global, dict):
            dv = tts_global.get("default_voice_id")
            if isinstance(dv, str) and dv.strip():
                tts_voice_id = dv.strip()
    # If no explicit voice provided, try to reuse last voice from context
    inferred_voice = None
    try:
        recents = _ctx_list(cid, limit=10, kind_hint="audio")
        for it in reversed(recents):
            for t in (it.get("tags") or []):
                if isinstance(t, str) and t.startswith("voice:"):
                    inferred_voice = t.split(":", 1)[1]
                    break
            if inferred_voice:
                break
    except Exception as ex:
        # Context lookup for inferred_voice is best-effort; log failures so they
        # are visible in traces instead of silently ignoring them.
        log.warning("tts.speak: failed to infer voice from context for cid=%s: %s", cid, ex, exc_info=True)
        inferred_voice = None
    # Determine language with a hard default to English if unspecified.
    lang = job.get("language")
    if not isinstance(lang, str) or not lang.strip():
        lang = "en"
    # Logical voice identifier used for XTTS base-speaker mapping and RVC locks.
    logical_voice_id = job.get("voice_id") or params.get("voice") or tts_voice_id or inferred_voice
    # Ensure each call is associated with a stable segment identifier for downstream
    # RVC conversion and Film2/audio mixing.
    seg_raw = job.get("segment_id")
    if isinstance(seg_raw, str) and seg_raw.strip():
        segment_id = seg_raw.strip()
    else:
        segment_id = f"seg_{cid}"
    args = {
        "text": job.get("text") or "",
        "voice": logical_voice_id,
        "voice_id": logical_voice_id,
        "segment_id": segment_id,
        "rate": float(job.get("rate") or tts_global.get("default_speaking_rate") or 1.0),
        "pitch": float(job.get("pitch") or tts_global.get("default_pitch_shift_semitones") or 0.0),
        "sample_rate": int(params.get("sample_rate") or 22050),
        "channels": 1,
        "max_seconds": int(params.get("max_seconds") or 20),
        "voice_lock": lock,
        "seed": job.get("seed"),
        "lock_bundle": lock_bundle,
        "quality_profile": quality_profile,
        "language": lang,
        "trace_id": cid,
    }
    args = stamp_tool_args("tts.speak", args)
    res = provider.speak(args)
    # Provider may return either a raw payload or an envelope; handle both safely.
    if isinstance(res, dict) and "ok" in res:
        if not bool(res.get("ok")):
            # Bubble up error envelope unchanged so callers can see failure details,
            # including any stack traces from downstream XTTS/RVC/VocalFix.
            return res
        inner = res.get("result") if isinstance(res.get("result"), dict) else {}
        wav_bytes = inner.get("wav_bytes") or b""
        dur = float(inner.get("duration_s") or 0.0)
        model = inner.get("model", "unknown")
    else:
        wav_bytes = res.get("wav_bytes") or b""
        dur = float(res.get("duration_s") or 0.0)
        model = res.get("model", "unknown")
    # Mandatory RVC voice conversion for all vocal segments; configuration and
    # reference locks must be present or this tool returns an error.
    voice_lock_id = job.get("voice_lock_id") or job.get("voice_id") or logical_voice_id
    rvc_url = os.getenv("RVC_API_URL")
    if not isinstance(rvc_url, str) or not rvc_url.strip():
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "status": 500,
                "message": "RVC_API_URL is not set; RVC is mandatory for tts.speak.",
                "stack": "".join(__import__("traceback").format_stack()),
            },
        }
    if not isinstance(voice_lock_id, str) or not voice_lock_id.strip():
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "status": 500,
                "message": f"No voice_lock_id / reference configured for voice_id '{logical_voice_id}'.",
                "stack": "".join(__import__("traceback").format_stack()),
            },
        }
    import base64 as _b64_rvc
    payload_rvc = {
        "source_wav_base64": _b64_rvc.b64encode(wav_bytes or b"").decode("ascii"),
        "voice_lock_id": voice_lock_id.strip(),
    }
    with httpx.Client(timeout=None, trust_env=False) as client:
        r_rvc = client.post(rvc_url.rstrip("/") + "/v1/audio/convert", json=payload_rvc)
    from ..json_parser import JSONParser  # type: ignore

    parser = JSONParser()
    js_rvc = parser.parse_superset(
        r_rvc.text or "",
        {"ok": bool, "audio_wav_base64": str, "sample_rate": int},
    )["coerced"]
    if not isinstance(js_rvc, dict) or not bool(js_rvc.get("ok")):
        import traceback as _tb
        return {
            "ok": False,
            "error": {
                "code": js_rvc.get("error", {}).get("code") if isinstance(js_rvc.get("error"), dict) else "InternalError",
                "status": int(getattr(r_rvc, "status_code", 500) or 500),
                "message": "RVC /v1/audio/convert failed for tts.speak",
                "raw": js_rvc,
                "stack": js_rvc.get("error", {}).get("stack") if isinstance(js_rvc.get("error"), dict) else _tb.format_exc(),
            },
        }
    b64_rvc = js_rvc.get("audio_wav_base64") if isinstance(js_rvc.get("audio_wav_base64"), str) else None
    if not b64_rvc:
        import traceback as _tb2
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "status": 500,
                "message": "RVC /v1/audio/convert returned no audio_wav_base64",
                "stack": _tb2.format_stack(),
            },
        }
    wav_bytes = _b64_rvc.b64decode(b64_rvc)
    # Mandatory VocalFix quality stage for all TTS audio.
    vf_url = os.getenv("VOCAL_FIXER_API_URL")
    if not isinstance(vf_url, str) or not vf_url.strip():
        return {
            "ok": False,
            "error": {
                "code": "ValidationError",
                "status": 500,
                "message": "VOCAL_FIXER_API_URL must be configured for tts.speak",
                "stack": "".join(__import__("traceback").format_stack()),
            },
        }
    try:
        import base64 as _b64_vf
        payload_vf = {
            "audio_wav_base64": _b64_vf.b64encode(wav_bytes or b"").decode("ascii"),
            "sample_rate": int(params.get("sample_rate") or 22050),
            "ops": ["pitch", "align", "deess"],
        }
        with httpx.Client(timeout=None, trust_env=False) as client:
            r_vf = client.post(vf_url.rstrip("/") + "/v1/vocal/fix", json=payload_vf)
        try:
            from ..json_parser import JSONParser  # type: ignore
            parser = JSONParser()
            js_vf = parser.parse_superset(
                r_vf.text or "",
                {
                    "ok": bool,
                    "audio_wav_base64": str,
                    "sample_rate": int,
                    "metrics_before": dict,
                    "metrics_after": dict,
                },
            )["coerced"]
        except Exception:
            js_vf = {}
        if not isinstance(js_vf, dict) or not bool(js_vf.get("ok")):
            import traceback as _tb3
            inner_err = js_vf.get("error") if isinstance(js_vf, dict) else None
            return {
                "ok": False,
                "error": {
                    "code": (inner_err or {}).get("code") if isinstance(inner_err, dict) else "InternalError",
                    "status": int(getattr(r_vf, "status_code", 500) or 500),
                    "message": "VocalFix /v1/vocal/fix failed for tts.speak",
                    "raw": js_vf,
                    "stack": (inner_err or {}).get("stack") if isinstance(inner_err, dict) else _tb3.format_exc(),
                },
            }
        b64_vf = js_vf.get("audio_wav_base64") if isinstance(js_vf.get("audio_wav_base64"), str) else None
        if not b64_vf:
            import traceback as _tb4
            return {
                "ok": False,
                "error": {
                    "code": "ValidationError",
                    "status": 500,
                    "message": "VocalFix /v1/vocal/fix returned no audio_wav_base64",
                    "stack": _tb4.format_stack(),
                },
            }
        wav_bytes = _b64_vf.b64decode(b64_vf)
        # Attach VocalFix metrics to manifest for teacher/distillation.
        manifest.setdefault("items", []).append(
            {
                "kind": "tts.vocalfix",
                "metrics_before": js_vf.get("metrics_before") if isinstance(js_vf.get("metrics_before"), dict) else {},
                "metrics_after": js_vf.get("metrics_after") if isinstance(js_vf.get("metrics_after"), dict) else {},
            }
        )
    except Exception as ex:
        return {
            "ok": False,
            "error": {
                "code": "vocalfix_exception",
                "status": 500,
                "message": str(ex),
            },
        }
    stem = f"tts_{now_ts()}"
    wav_path = os.path.join(outdir, stem + ".wav")
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)
    # Committee QA: peak-normalize to -1 dBFS and hard-trim to max_seconds
    try:
        def _peak_normalize(path: str, target_dbfs: float = -1.0) -> dict:
            with wave.open(path, "rb") as wf:
                ch = wf.getnchannels()
                sr = wf.getframerate()
                sw = wf.getsampwidth()
                n = wf.getnframes()
                pcm = wf.readframes(n)
            if sw != 2 or not pcm:
                return {"normalized": False}
            # Find peak
            peak = 0
            for i in range(0, len(pcm), 2 * ch):
                for c in range(ch):
                    s = struct.unpack("<h", pcm[i + 2 * c : i + 2 * c + 2])[0]
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
                    s = struct.unpack("<h", pcm[i + 2 * c : i + 2 * c + 2])[0]
                    v = int(max(-32768, min(32767, s * gain)))
                    out[i + 2 * c : i + 2 * c + 2] = struct.pack("<h", v)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(sw)
                wf.setframerate(sr)
                wf.writeframes(bytes(out))
            return {"normalized": True, "peak_dbfs": peak_dbfs, "applied_db": (target_dbfs - peak_dbfs)}

        def _hard_trim(path: str, max_seconds: float) -> dict:
            if max_seconds is None:
                return {"trimmed": False}
            with wave.open(path, "rb") as wf:
                ch = wf.getnchannels()
                sr = wf.getframerate()
                sw = wf.getsampwidth()
                n = wf.getnframes()
                max_frames = int(max(0, max_seconds) * sr)
                if n <= max_frames:
                    return {"trimmed": False}
                wf.rewind()
                data = wf.readframes(max_frames)
            with wave.open(path, "wb") as wf2:
                wf2.setnchannels(ch)
                wf2.setsampwidth(sw)
                wf2.setframerate(sr)
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
        qa_norm = {"normalized": False}
        qa_trim = {"trimmed": False}
    locks_meta: Dict[str, Any] = {}
    if lock_bundle:
        locks_meta["bundle"] = lock_bundle
    audio_section = lock_bundle.get("audio") if isinstance(lock_bundle.get("audio"), dict) else {}
    ref_voice = audio_section.get("voice_embedding")
    if isinstance(ref_voice, list):
        voice_embed = voice_embedding_from_path(wav_path)
        if isinstance(voice_embed, list):
            sim = _cosine_similarity(ref_voice, voice_embed)
            if sim is not None:
                locks_meta["voice_score"] = max(0.0, min((sim + 1.0) / 2.0, 1.0))
                locks_meta["voice_embedding"] = voice_embed
    lyrics_segments = (
        audio_section.get("lyrics_segments") if isinstance(audio_section.get("lyrics_segments"), list) else []
    )
    hard_segments = [
        seg
        for seg in lyrics_segments
        if isinstance(seg, dict) and (seg.get("lock_mode") or "hard").lower() == "hard"
    ]
    if hard_segments:
        text_lower = (args.get("text") or "").lower()
        matched = 0
        for seg in hard_segments:
            seg_text = (seg.get("text") or "").lower()
            if seg_text and seg_text in text_lower:
                matched += 1
        locks_meta["lyrics_score"] = matched / len(hard_segments) if hard_segments else None
    # TTS prosody/emotion/timing QA from lock_bundle.tts when available
    if lock_bundle:
        ainfo3 = analyze_audio(wav_path)
        if isinstance(ainfo3, dict):
            pitch_mean = ainfo3.get("pitch_mean_hz")
            # Voice-level baseline
            tts_section = lock_bundle.get("tts") if isinstance(lock_bundle.get("tts"), dict) else {}
            voices = tts_get_voices(lock_bundle)
            # Select voice entry by default_voice_id or first voice
            target_voice_id = tts_voice_id
            voice_entry: Dict[str, Any] | None = None
            if isinstance(target_voice_id, str):
                for v in voices:
                    if isinstance(v, dict) and v.get("voice_id") == target_voice_id:
                        voice_entry = v
                        break
            if voice_entry is None and voices:
                first = voices[0]
                if isinstance(first, dict):
                    voice_entry = first
            # Prosody pitch lock
            if isinstance(pitch_mean, (int, float)) and voice_entry:
                base_pros = voice_entry.get("baseline_prosody") if isinstance(voice_entry.get("baseline_prosody"), dict) else {}
                base_pitch = base_pros.get("mean_pitch_hz")
                if isinstance(base_pitch, (int, float)) and base_pitch > 0.0:
                    err = abs(float(pitch_mean) - float(base_pitch)) / float(base_pitch)
                    prosody_pitch_lock = max(0.0, min(1.0 - err, 1.0))
                    locks_meta["prosody_pitch_lock"] = prosody_pitch_lock
            # Segment-level timing lock (match by exact text when possible)
            segments = tts_section.get("segments") if isinstance(tts_section.get("segments"), list) else []
            seg_match: Dict[str, Any] | None = None
            text_full = (args.get("text") or "").strip()
            if text_full and segments:
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    tref = seg.get("text_ref") if isinstance(seg.get("text_ref"), dict) else {}
                    txt = (tref.get("text") or "").strip()
                    if txt and txt == text_full:
                        seg_match = seg
                        break
            if seg_match:
                timing = seg_match.get("timing_targets") if isinstance(seg_match.get("timing_targets"), dict) else {}
                expected_dur = timing.get("expected_duration_s")
                max_dev = timing.get("max_deviation_s")
                if isinstance(expected_dur, (int, float)) and expected_dur > 0.0 and isinstance(max_dev, (int, float)):
                    err_t = abs(dur - float(expected_dur))
                    if err_t <= float(max_dev):
                        timing_lock = 1.0
                    else:
                        timing_lock = max(0.0, min(1.0 - (err_t / float(expected_dur)), 1.0))
                    locks_meta["timing_lock"] = timing_lock
    # Composite TTS lock score for distillation
    components: list[float] = []
    for key in ("voice_score", "lyrics_score", "prosody_pitch_lock", "timing_lock"):
        v = locks_meta.get(key)
        if isinstance(v, (int, float)):
            components.append(float(v))
    if components:
        locks_meta["lock_score_tts"] = min(components)
    if quality_profile:
        locks_meta.setdefault("quality_profile", quality_profile)
    sidecar_payload = {
        "tool": "tts.speak",
        "text": args.get("text"),
        "voice": args.get("voice"),
        "rate": args.get("rate"),
        "pitch": args.get("pitch"),
        "sample_rate": args.get("sample_rate"),
        "max_seconds": args.get("max_seconds"),
        "seed": args.get("seed"),
        "voice_lock": lock,
        "model": model,
        "duration_s": dur,
        "committee": {"peak_normalize": qa_norm, "hard_trim": qa_trim},
    }
    if locks_meta:
        sidecar_payload["locks"] = locks_meta
    sidecar(wav_path, sidecar_payload)
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
                    sidecar(
                        wav_path,
                        {"tool": "tts.speak.committee", "target_emotion": target_emotion, "adjusted": True},
                    )
            if target_emotion == "calm" and cur != "calm":
                adj = dict(args)
                adj["rate"] = max(0.5, float(args.get("rate") or 1.0) * 0.9)
                adj["pitch"] = float(args.get("pitch") or 0.0) - 1.0
                res2 = provider.speak(adj)
                wb2 = res2.get("wav_bytes") or b""
                if wb2:
                    with open(wav_path, "wb") as f:
                        f.write(wb2)
                    sidecar(
                        wav_path,
                        {"tool": "tts.speak.committee", "target_emotion": target_emotion, "adjusted": True},
                    )
    except Exception:
        pass
    try:
        append_tts_sample(
            outdir,
            {
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
            },
        )
    except Exception:
        pass
    try:
        if job.get("voice_id"):
            append_provenance(
                job.get("voice_id"),
                {"when": now_ts(), "tool": "tts.speak", "artifact": wav_path, "seed": int(args.get("seed") or 0)},
            )
    except Exception:
        pass
    add_manifest_row(manifest, wav_path, step_id="tts.speak")
    try:
        _ctx_add(
            cid,
            "audio",
            wav_path,
            None,
            None,
            ["voice", f"voice:{args.get('voice')}"] if args.get("voice") else ["voice"],
            {"model": model},
        )
    except Exception:
        pass
    env = {
        "meta": {
            "model": model,
            "ts": now_ts(),
            "cid": cid,
            "step": 0,
            "state": "halt",
            "cont": {"present": False, "state_hash": None, "reason": None},
            # Expose segment/voice identifiers at the envelope level so downstream
            # planners and mixers can stitch multi-singer timelines.
            "segment_id": segment_id,
            "voice_id": logical_voice_id,
            "voice_lock_id": voice_lock_id,
        },
        "reasoning": {"goal": "tts", "constraints": ["json-only", "edge-safe"], "decisions": ["tts.speak done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "tts generated"},
        "tool_calls": [{"tool": "tts.speak", "args": args, "status": "done", "result_ref": os.path.basename(wav_path)}],
        "artifacts": [
            {"id": os.path.basename(wav_path), "kind": "audio-ref", "summary": stem, "bytes": len(wav_bytes)}
        ],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env_meta = env.setdefault("meta", {})
    if quality_profile:
        env_meta.setdefault("quality_profile", quality_profile)
    if locks_meta:
        env_meta["locks"] = locks_meta
    env = normalize_to_envelope(json.dumps(env))
    env = bump_envelope(env)
    assert_envelope(env)
    env = stamp_env(env, "tts.speak", model)
    # Trace row for distillation
    try:
        ainfo = analyze_audio(wav_path)
        trace_row = {
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
        }
        if locks_meta:
            trace_row["locks"] = locks_meta
        _trace_append("tts", trace_row)
    except Exception:
        pass
    return env


