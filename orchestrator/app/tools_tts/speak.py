from __future__ import annotations

import os
import json
import base64
import traceback
import time
import hashlib
from typing import Any, Dict
import logging
import wave
import struct
import math

from ..json_parser import JSONParser
from .common import now_ts, ensure_dir, tts_edge_defaults, sidecar, stamp_env
from ..locks import voice_embedding_from_path, tts_get_global, tts_get_voices
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_envelope, bump_envelope, assert_envelope, _build_success_envelope, _build_error_envelope
from ..locks.voice_identity import resolve_voice_lock, resolve_voice_identity
from ..ref_library.registry import append_provenance
from ..artifacts.index import add_artifact as _ctx_add
from ..artifacts.index import list_recent as _ctx_list
from ..analysis.media import analyze_audio, normalize_lufs, cosine_similarity
from ..tracing.training import append_training_sample
from ..tracing.runtime import trace_event
from void_artifacts import build_artifact, generate_artifact_id, artifact_id_to_safe_filename
import httpx  # type: ignore


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")

log = logging.getLogger(__name__)


def _tts_peak_normalize_wav(path: str, target_dbfs: float = -1.0, *, trace_id: str = "", conversation_id: str = ""):
    """
    Peak-normalize an existing PCM16 WAV in-place.
    Never raises; returns {ok,result,error}.
    """
    if not isinstance(path, str) or not path.strip() or not os.path.exists(path.strip()):
        return _build_error_envelope(
            code="missing_wav",
            message="wav path missing",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=404,
            details={"path": path, "stack": "".join(traceback.format_stack())},
        )
    p = path.strip()
    try:
        with wave.open(p, "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            pcm = wf.readframes(n)
        if sw != 2 or not pcm:
            return _build_success_envelope(
                result={"normalized": False, "reason": "unsupported_sample_width_or_empty", "sampwidth": sw},
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
        # Find peak
        peak = 0
        for i in range(0, len(pcm), 2 * ch):
            for c in range(ch):
                s = struct.unpack("<h", pcm[i + 2 * c : i + 2 * c + 2])[0]
                peak = max(peak, abs(s))
        if peak <= 0:
            return _build_success_envelope(
                result={"normalized": False, "reason": "silent_audio"},
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
        peak_dbfs = 20.0 * math.log10(max(1.0, peak) / 32767.0)
        gain = 10.0 ** ((float(target_dbfs) - float(peak_dbfs)) / 20.0)
        if gain <= 0.0 or abs(1.0 - gain) < 1e-3:
            return _build_success_envelope(
                result={"normalized": False, "peak_dbfs": peak_dbfs, "reason": "already_close"},
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
        out = bytearray(len(pcm))
        for i in range(0, len(pcm), 2 * ch):
            for c in range(ch):
                s = struct.unpack("<h", pcm[i + 2 * c : i + 2 * c + 2])[0]
                v = int(max(-32768, min(32767, float(s) * float(gain))))
                out[i + 2 * c : i + 2 * c + 2] = struct.pack("<h", v)
        with wave.open(p, "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(sw)
            wf.setframerate(sr)
            wf.writeframes(bytes(out))
        return _build_success_envelope(
            result={
                "normalized": True,
                "peak_dbfs": peak_dbfs,
                "applied_db": (float(target_dbfs) - float(peak_dbfs)),
                "channels": ch,
                "sample_rate": sr,
            },
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        return _build_error_envelope(
            code="peak_normalize_exception",
            message=f"Exception during peak normalize {str(e)}",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=500,
            details={"path": p, "stack": traceback.format_exc()},
        )


def _tts_hard_trim_wav(path: str, max_seconds: Any, *, trace_id: str = "", conversation_id: str = ""):
    """
    Hard-trim an existing WAV in-place.
    Never raises; returns {ok,result,error}.
    """
    if not isinstance(path, str) or not path.strip() or not os.path.exists(path.strip()):
        return _build_error_envelope(
            code="missing_wav",
            message="wav path missing",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=404,
            details={"path": path, "stack": "".join(traceback.format_stack())},
        )
    p = path.strip()
    if max_seconds is None:
        return _build_success_envelope(
            result={"trimmed": False, "reason": "max_seconds_none"},
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    try:
        max_s = float(max_seconds)
    except Exception:
        max_s = 0.0
    if max_s <= 0.0:
        return _build_success_envelope(
            result={"trimmed": False, "reason": "max_seconds_nonpositive", "max_seconds": max_s},
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    try:
        with wave.open(p, "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            max_frames = int(max(0.0, float(max_s)) * int(sr))
            if n <= max_frames:
                return _build_success_envelope(
                    result={"trimmed": False, "old_frames": n, "new_frames": max_frames},
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                )
            wf.rewind()
            data = wf.readframes(max_frames)
        with wave.open(p, "wb") as wf2:
            wf2.setnchannels(ch)
            wf2.setsampwidth(sw)
            wf2.setframerate(sr)
            wf2.writeframes(data)
        return _build_success_envelope(
            result={"trimmed": True, "old_frames": n, "new_frames": max_frames, "channels": ch, "sample_rate": sr},
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        return _build_error_envelope(
            code="hard_trim_exception",
            message=f"Exception during hard trim {str(e)}",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=500,
            details={"path": p, "stack": traceback.format_exc()},
        )


def _augment_voice_refs_from_context(voice_id: str | None, voice_refs: dict | None, conversation_id: str) -> dict | None:
    """
    Best-effort helper: if no explicit voice_refs are provided but a voice_id
    is known, pull any recent audio artifacts tagged with that voice from the
    context index and use them as implicit voice_samples.
    Returns updated voice_refs dict or None if no augmentation needed.
    """
    if isinstance(voice_refs, dict):
        return voice_refs
    if not isinstance(voice_id, str) or not voice_id.strip():
        return voice_refs
    voice_tag = f"voice:{voice_id.strip()}"
    try:
        recents = _ctx_list(conversation_id, limit=50, kind_hint="audio")
    except Exception:
        return voice_refs
    samples: list[str] = []
    for it in recents or []:
        tags = it.get("tags") or []
        if not any(isinstance(t, str) and t == voice_tag for t in tags):
            continue
        pth = it.get("path")
        if isinstance(pth, str) and pth:
            samples.append(pth)
    if samples:
        return {"voice_samples": samples}
    return voice_refs


async def run_tts_speak(
    *,
    provider,
    manifest: dict,
    trace_id: str = "",
    conversation_id: str = "",
    text: str = "",
    voice: str | None = None,
    voice_id: str | None = None,
    voice_refs: dict | None = None,
    rate: float | None = None,
    pitch: float | None = None,
    sample_rate: int | None = None,
    max_seconds: int | None = None,
    seed: int | None = None,
    edge: bool = False,
    language: str | None = None,
    voice_gender: str | None = None,
    voice_lock_id: str | None = None,
    segment_id: str | None = None,
    normalize: bool = False,
    normalize_lufs: bool = False,
    lock_bundle: dict | None = None,
    quality_profile: str | None = None,
    artifact_id: str | None = None,
    **kwargs
):
    """
    TTS speak with explicit parameters.
    provider: exposes .speak(args) returns {"wav_bytes": b"...", "duration_s": float, "model": "..."}
    """
    if trace_id:
        trace_event("tool.tts.speak.start", {"trace_id": trace_id, "conversation_id": conversation_id, "text_length": len(str(text or ""))})
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "tts", conversation_id)
    ensure_dir(outdir)
    fatal_error: Dict[str, Any] | None = None
    wav_bytes: bytes = b""
    dur: float = 0.0
    model: str = "unknown"
    logical_voice_id: str | None = None
    voice_source: str = "unknown"
    segment_id_resolved: str = ""
    voice_lock_id_resolved: str | None = None
    locks_meta: Dict[str, Any] = {}
    ainfo_pre: Any = None
    # If no explicit refs were provided, try to infer reference samples for the
    # requested voice_id from recent context so that music/film pipelines and
    # prior TTS runs automatically feed into voice matching/training.
    voice_refs_augmented = _augment_voice_refs_from_context(voice_id, voice_refs, conversation_id)
    # Resolve canonical voice identity and lock from any provided voice_id / refs
    raw_voice_id = voice_id
    raw_voice_refs = voice_refs_augmented
    resolved_voice_id, lock, voice_meta = resolve_voice_identity(raw_voice_id, raw_voice_refs)
    if resolved_voice_id:
        voice_id = resolved_voice_id
    else:
        # Fall back to simple lock resolution for non-ref cases so existing behavior
        # (env defaults, lock-bundle defaults) still works.
        lock = resolve_voice_lock(raw_voice_id, raw_voice_refs)
    params = tts_edge_defaults(
        {
            "sample_rate": sample_rate,
            "max_seconds": max_seconds,
            "voice": voice,
        },
        edge=edge,
    )
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
        recents = _ctx_list(conversation_id, limit=10, kind_hint="audio")
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
        log.warning(f"tts.speak: failed to infer voice from context for conversation_id={conversation_id!r} ex={ex!r}", exc_info=True)
        inferred_voice = None
    # Determine language with a hard default to English if unspecified.
    lang = language
    if not isinstance(lang, str) or not lang.strip():
        lang = "en"
    else:
        # Normalize locale-style values like "en-US" to XTTS-compatible codes ("en").
        s = lang.strip().lower().replace("_", "-")
        if s in ("english", "eng"):
            s = "en"
        if s == "zh":
            s = "zh-cn"
        if "-" in s and s != "zh-cn":
            s = s.split("-", 1)[0].strip() or "en"
        # Force to English if we ended up with an empty value.
        lang = s or "en"
    # Logical voice identifier used for XTTS base-speaker mapping and RVC locks.
    logical_voice_id = resolved_voice_id or voice_id or params.get("voice") or tts_voice_id or inferred_voice
    # Hard policy: always resolve a concrete voice_id before calling downstream
    # XTTS/RVC/VocalFix. When missing, fall back to configured defaults instead
    # of letting microservices raise "no speaker" style errors.
    voice_source = "explicit"
    if not isinstance(logical_voice_id, str) or not logical_voice_id.strip():
        logical_voice_id = None
        voice_source = "unset"
    else:
        logical_voice_id = logical_voice_id.strip()
    if logical_voice_id is None:
        # Try lock-bundle-provided default voice first.
        dv = None
        if isinstance(tts_global, dict):
            dv = tts_global.get("default_voice_id")
            if isinstance(dv, str) and dv.strip():
                logical_voice_id = dv.strip()
                voice_source = "lock_bundle_default"
    if logical_voice_id is None:
        # Environment-level defaults for RVC/XTTS. Prefer gender-specific
        # defaults when a hint is available, otherwise fall back to a generic.
        gender_hint = str(voice_gender or "").lower()
        # NOTE: These env vars are optional; the project does not require users
        # to set them. We also treat the legacy compose placeholders
        # ("default_male"/"default_female") as unset so we don't route to
        # nonexistent models.
        default_female = os.getenv("RVC_DEFAULT_FEMALE_VOICE_ID")
        default_male = os.getenv("RVC_DEFAULT_MALE_VOICE_ID")
        default_generic = os.getenv("RVC_DEFAULT_VOICE_ID")
        if isinstance(default_female, str) and default_female.strip() in ("default_female", "default_male", "default_generic"):
            default_female = ""
        if isinstance(default_male, str) and default_male.strip() in ("default_female", "default_male", "default_generic"):
            default_male = ""
        if isinstance(default_generic, str) and default_generic.strip() in ("default_female", "default_male", "default_generic"):
            default_generic = ""
        # Hard fallback (no env required): use the base Titan checkpoint.
        # This matches services/rvc_python/entrypoint.sh which materializes:
        #   /srv/rvc_models/TITAN/TITAN.pth
        titan_default = "TITAN"
        chosen = None
        if "female" in gender_hint and isinstance(default_female, str) and default_female.strip():
            chosen = default_female.strip()
            voice_source = "env_default_female"
        elif "male" in gender_hint and isinstance(default_male, str) and default_male.strip():
            chosen = default_male.strip()
            voice_source = "env_default_male"
        elif isinstance(default_generic, str) and default_generic.strip():
            chosen = default_generic.strip()
            voice_source = "env_default_generic"
        elif isinstance(default_female, str) and default_female.strip():
            chosen = default_female.strip()
            voice_source = "env_default_female"
        elif isinstance(default_male, str) and default_male.strip():
            chosen = default_male.strip()
            voice_source = "env_default_male"
        else:
            chosen = titan_default
            voice_source = "builtin_default_titan"
        logical_voice_id = chosen
    if logical_voice_id is None:
        # Final guardrail: if we still cannot resolve a voice, fail BEFORE
        # calling XTTS so the error surfaces explicitly in the envelope instead
        # of as an opaque runtime from the microservice.
        fatal_error = {
            "code": "voice_resolution_failed",
            "status": 500,
            "message": "Unable to resolve TTS voice_id; no explicit voice, lock-bundle default, or env default configured.",
            "stack": "".join(traceback.format_stack()),
        }
    if fatal_error is None:
        # Log the resolved voice for traceability.
        log.info(f"[voice.resolve] conversation_id={conversation_id!r} voice_id={logical_voice_id!r} source={voice_source!r}")
    # If new reference samples were attached for this voice, register and train
    # the corresponding RVC model *before* conversion so this call benefits
    # from updated weights. Training is mandatory when refs are provided.
    new_samples = voice_meta.get("new_samples") if isinstance(voice_meta, dict) else []
    if fatal_error is None and logical_voice_id and new_samples:
        rvc_url_train = os.getenv("RVC_API_URL")
        if not isinstance(rvc_url_train, str) or not rvc_url_train.strip():
            fatal_error = {
                "code": "rvc_unconfigured",
                "status": 500,
                "message": "RVC_API_URL is not set; required for voice registration/training in tts.speak.",
                "stack": "".join(traceback.format_stack()),
            }
        payload_reg = {
            "voice_lock_id": str(logical_voice_id),
            "model_name": str(logical_voice_id),
            "reference_wav_path": new_samples[0],
            "additional_refs": new_samples[1:],
        }
        if fatal_error is None:
            with httpx.Client(timeout=None, trust_env=False) as client:
                # Voice registration
                r_reg = client.post(rvc_url_train.rstrip("/") + "/v1/voice/register", json=payload_reg)
                parser_reg = JSONParser()
                raw_reg = r_reg.text or ""
                reg_env = parser_reg.parse(raw_reg or "{}", {"schema_version": int, "trace_id": str, "ok": bool, "result": dict, "error": dict})
                if not isinstance(reg_env, dict) or not bool(reg_env.get("ok")):
                    err_obj = reg_env.get("error") if isinstance(reg_env, dict) and isinstance(reg_env.get("error"), dict) else {}
                    fatal_error = {
                        "code": (err_obj or {}).get("code") if isinstance(err_obj, dict) else "rvc_register_failed",
                        "status": int(getattr(r_reg, "status_code", 500) or 500),
                        "message": "RVC /v1/voice/register failed for tts.speak",
                        "raw": reg_env,
                        "stack": (err_obj or {}).get("stack") if isinstance(err_obj, dict) else traceback.format_exc(),
                    }
                # Blocking training: ensure updated model is ready before conversion.
                if fatal_error is None:
                    r_train = client.post(
                        rvc_url_train.rstrip("/") + "/v1/voice/train",
                        json={"voice_lock_id": str(logical_voice_id)},
                    )
                    parser_train = JSONParser()
                    raw_train = r_train.text or ""
                    train_env = parser_train.parse(raw_train or "{}", {"schema_version": int, "trace_id": str, "ok": bool, "result": dict, "error": dict})
                    if not isinstance(train_env, dict) or not bool(train_env.get("ok")):
                        err_obj = train_env.get("error") if isinstance(train_env, dict) and isinstance(train_env.get("error"), dict) else {}
                        fatal_error = {
                            "code": (err_obj or {}).get("code") if isinstance(err_obj, dict) else "rvc_train_failed",
                            "status": int(getattr(r_train, "status_code", 500) or 500),
                            "message": "RVC /v1/voice/train failed for tts.speak",
                            "raw": train_env,
                            "stack": (err_obj or {}).get("stack") if isinstance(err_obj, dict) else traceback.format_exc(),
                        }
    # Ensure each call is associated with a stable segment identifier for downstream
    # RVC conversion and Film2/audio mixing.
    if isinstance(segment_id, str) and segment_id.strip():
        segment_id_resolved = segment_id.strip()
    else:
        segment_id_resolved = f"seg_{conversation_id}"
    args = {
        "text": text or "",
        "voice": logical_voice_id,
        "voice_id": logical_voice_id,
        "segment_id": segment_id_resolved,
        # Defensive: user/planner supplied knobs must never raise.
        "rate": (tts_global.get("default_speaking_rate") or 1.0) if rate is None else rate,
        "pitch": (tts_global.get("default_pitch_shift_semitones") or 0.0) if pitch is None else pitch,
        "sample_rate": (params.get("sample_rate") or 22050),
        "channels": 1,
        "max_seconds": (params.get("max_seconds") or 20),
        "voice_lock": lock,
        "seed": seed,
        "lock_bundle": lock_bundle,
        "quality_profile": quality_profile,
        "language": lang,
        "trace_id": trace_id,
    }
    # Coerce numeric fields safely (in-line, no helper).
    try:
        args["rate"] = float(args.get("rate") or 1.0)
    except Exception as exc:
        log.warning(f"tts.speak: bad rate={args.get('rate')!r}; defaulting to 1.0 conversation_id={conversation_id!r}", exc_info=True)
        args["rate"] = 1.0
    try:
        args["pitch"] = float(args.get("pitch") or 0.0)
    except Exception as exc:
        log.warning(f"tts.speak: bad pitch={args.get('pitch')!r}; defaulting to 0.0 conversation_id={conversation_id!r}", exc_info=True)
        args["pitch"] = 0.0
    try:
        args["sample_rate"] = int(args.get("sample_rate") or 22050)
    except Exception as exc:
        log.warning(f"tts.speak: bad sample_rate={args.get('sample_rate')!r}; defaulting to 22050 conversation_id={conversation_id!r}", exc_info=True)
        args["sample_rate"] = 22050
    try:
        args["max_seconds"] = int(args.get("max_seconds") or 20)
    except Exception as exc:
        log.warning(f"tts.speak: bad max_seconds={args.get('max_seconds')!r}; defaulting to 20 conversation_id={conversation_id!r}", exc_info=True)
        args["max_seconds"] = 20
    args = stamp_tool_args("tts.speak", args)
    if fatal_error is None:
        res = provider.speak(args)
    else:
        res = {}
    # Provider may return either a raw payload or an envelope; handle both safely.
    if fatal_error is None:
        if isinstance(res, dict) and "ok" in res:
            if not bool(res.get("ok")):
                err0 = res.get("error")
                fatal_error = err0 if isinstance(err0, dict) else {"code": "tts_error", "status": 500, "message": str(err0 or "tts failed"), "raw": res, "stack": "".join(traceback.format_stack())}
            else:
                inner = res.get("result") if isinstance(res.get("result"), dict) else {}
                wb = inner.get("wav_bytes")
                wav_bytes = wb if isinstance(wb, (bytes, bytearray)) else b""
                d0 = inner.get("duration_s")
                dur = float(d0) if isinstance(d0, (int, float)) else 0.0
                model = inner.get("model", "unknown") if isinstance(inner.get("model"), str) else "unknown"
        else:
            wb = res.get("wav_bytes") if isinstance(res, dict) else None
            wav_bytes = wb if isinstance(wb, (bytes, bytearray)) else b""
            d0 = res.get("duration_s") if isinstance(res, dict) else None
            dur = float(d0) if isinstance(d0, (int, float)) else 0.0
            model = res.get("model", "unknown") if isinstance(res, dict) and isinstance(res.get("model"), str) else "unknown"
    # Mandatory RVC voice conversion for all vocal segments; configuration and
    # reference locks must be present or this tool returns an error.
    if fatal_error is None:
        voice_lock_id_resolved = voice_lock_id or voice_id or logical_voice_id
        rvc_url = os.getenv("RVC_API_URL")
        if not isinstance(rvc_url, str) or not rvc_url.strip():
            fatal_error = {
                "code": "rvc_unconfigured",
                "status": 500,
                "message": "RVC_API_URL is not set; RVC is mandatory for tts.speak.",
                "stack": "".join(traceback.format_stack()),
            }
        if fatal_error is None and (not isinstance(voice_lock_id_resolved, str) or not voice_lock_id_resolved.strip()):
            fatal_error = {
                "code": "missing_voice_lock_id",
                "status": 500,
                "message": f"No voice_lock_id / reference configured for voice_id '{logical_voice_id}'.",
                "stack": "".join(traceback.format_stack()),
            }
    payload_rvc: Dict[str, Any] = {}
    if fatal_error is None:
        payload_rvc = {
            "source_wav_base64": base64.b64encode(wav_bytes or b"").decode("ascii"),
            "voice_lock_id": str(voice_lock_id_resolved).strip(),
            "voice_id": logical_voice_id,
            "trace_id": trace_id,
            "segment_id": segment_id_resolved,
            "conversation_id": conversation_id,
        }
        r_rvc = None
        try:
            with httpx.Client(timeout=None, trust_env=False) as client:
                r_rvc = client.post(rvc_url.rstrip("/") + "/v1/audio/convert", json=payload_rvc)
        except Exception as ex:
            fatal_error = {
                "code": "rvc_http_exception",
                "status": 502,
                "message": str(ex),
                "payload": payload_rvc,
                "stack": traceback.format_exc(),
            }
        if fatal_error is None:
            parser = JSONParser()
            js_rvc = parser.parse(
                (r_rvc.text or "") if r_rvc is not None else "",
                {"schema_version": int, "trace_id": str, "ok": bool, "result": dict, "error": dict},
            )
            if not isinstance(js_rvc, dict) or not bool(js_rvc.get("ok")):
                err_block = js_rvc.get("error") if isinstance(js_rvc, dict) and isinstance(js_rvc.get("error"), dict) else {}
                fatal_error = {
                    "code": err_block.get("code") if isinstance(err_block, dict) else "rvc_error",
                    "status": int(getattr(r_rvc, "status_code", 500) or 500) if r_rvc is not None else 500,
                    "message": "RVC /v1/audio/convert failed for tts.speak",
                    "raw": js_rvc,
                    "stack": err_block.get("stack") if isinstance(err_block, dict) else "".join(traceback.format_stack()),
                }
            if fatal_error is None:
                inner_rvc = js_rvc.get("result") if isinstance(js_rvc.get("result"), dict) else {}
                b64_rvc = inner_rvc.get("audio_wav_base64") if isinstance(inner_rvc.get("audio_wav_base64"), str) else None
                if not b64_rvc:
                    fatal_error = {
                        "code": "rvc_missing_audio",
                        "status": 500,
                        "message": "RVC /v1/audio/convert returned no audio_wav_base64",
                        "raw": js_rvc,
                        "stack": "".join(traceback.format_stack()),
                    }
                else:
                    wav_bytes = base64.b64decode(b64_rvc)
    # Mandatory VocalFix quality stage for all TTS audio.
    if fatal_error is None:
        vf_url = os.getenv("VOCAL_FIXER_API_URL")
        if not isinstance(vf_url, str) or not vf_url.strip():
            fatal_error = {
                "code": "vocalfix_unconfigured",
                "status": 500,
                "message": "VOCAL_FIXER_API_URL must be configured for tts.speak",
                "stack": "".join(traceback.format_stack()),
            }
    if fatal_error is None:
        try:
            payload_vf = {
                "audio_wav_base64": base64.b64encode(wav_bytes or b"").decode("ascii"),
                "sample_rate": int(params.get("sample_rate") or 22050),
                "ops": ["pitch", "align", "deess"],
                "trace_id": trace_id,
            }
            with httpx.Client(timeout=None, trust_env=False) as client:
                r_vf = client.post(vf_url.rstrip("/") + "/v1/vocal/fix", json=payload_vf)
            parser = JSONParser()
            js_vf = parser.parse(
                r_vf.text or "",
                {"schema_version": int, "ok": bool, "result": dict, "error": dict},
            )
            if not isinstance(js_vf, dict) or not bool(js_vf.get("ok")):
                inner_err = js_vf.get("error") if isinstance(js_vf, dict) and isinstance(js_vf.get("error"), dict) else {}
                fatal_error = {
                    "code": (inner_err or {}).get("code") if isinstance(inner_err, dict) else "vocalfix_failed",
                    "status": int(getattr(r_vf, "status_code", 500) or 500),
                    "message": "VocalFix /v1/vocal/fix failed for tts.speak",
                    "raw": js_vf,
                    "stack": (inner_err or {}).get("stack") if isinstance(inner_err, dict) else "".join(traceback.format_stack()),
                }
            if fatal_error is None:
                inner_vf = js_vf.get("result") if isinstance(js_vf.get("result"), dict) else {}
                b64_vf = inner_vf.get("audio_wav_base64") if isinstance(inner_vf.get("audio_wav_base64"), str) else None
                if not b64_vf:
                    fatal_error = {
                        "code": "vocalfix_missing_audio",
                        "status": 500,
                        "message": "VocalFix /v1/vocal/fix returned no audio_wav_base64",
                        "raw": js_vf,
                        "stack": "".join(traceback.format_stack()),
                    }
                else:
                    wav_bytes = base64.b64decode(b64_vf)
                    # Attach VocalFix metrics to manifest for teacher/distillation.
                    manifest.setdefault("items", []).append(
                        {
                            "kind": "tts.vocalfix",
                            "metrics_before": inner_vf.get("metrics_before") if isinstance(inner_vf.get("metrics_before"), dict) else {},
                            "metrics_after": inner_vf.get("metrics_after") if isinstance(inner_vf.get("metrics_after"), dict) else {},
                        }
                    )
        except Exception as ex:
            fatal_error = {
                "code": "vocalfix_exception",
                "status": 500,
                "message": str(ex),
                "stack": traceback.format_exc(),
            }
    wav_path = ""
    stem = ""
    qa_norm: Dict[str, Any] = {}
    qa_trim: Dict[str, Any] = {}
    if fatal_error is None:
        # Generate unique artifact_id BEFORE creating file, then use it for filename
        artifact_id_generated = generate_artifact_id(
            trace_id=trace_id,
            tool_name="tts.speak",
            conversation_id=conversation_id,
            suffix_data=len(wav_bytes),
            existing_id=artifact_id,
        )
        # Create safe filename from artifact_id (artifact_id is already sanitized, but use helper for consistency)
        safe_filename = artifact_id_to_safe_filename(artifact_id_generated, ".wav")
        wav_path = os.path.join(outdir, safe_filename)
        stem = os.path.splitext(safe_filename)[0]
        try:
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
        except Exception:
            fatal_error = {
                "code": "tts_write_failed",
                "status": 500,
                "message": "Failed to write TTS wav file",
                "path": wav_path,
                "stack": traceback.format_exc(),
            }
    if fatal_error is None:
        # Post-processing (never fatal): peak-normalize / hard-trim / LUFS normalize.
        ainfo_pre = None
        qa_norm_env = _build_success_envelope(result={"normalized": False}, trace_id=trace_id, conversation_id=conversation_id)
        qa_trim_env = _build_success_envelope(result={"trimmed": False}, trace_id=trace_id, conversation_id=conversation_id)
        if normalize:
            qa_norm_env = _tts_peak_normalize_wav(wav_path, -1.0, trace_id=trace_id, conversation_id=conversation_id)
        qa_trim_env = _tts_hard_trim_wav(
            wav_path,
            (params.get("max_seconds") if isinstance(params, dict) else None),
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
        # Analyze audio once after preprocessing; reuse for downstream QA/locks.
        try:
            ainfo_pre = analyze_audio(wav_path)
        except Exception:
            ainfo_pre = {"error": {"code": "analyze_audio_exception", "message": "analyze_audio failed", "stack": traceback.format_exc()}}
        if isinstance(ainfo_pre, dict) and isinstance(qa_norm_env, dict) and isinstance(qa_norm_env.get("result"), dict):
            qa_norm_env["result"]["lufs_before"] = ainfo_pre.get("lufs")
        if normalize_lufs:
            try:
                applied = normalize_lufs(wav_path, -16.0)
                if applied is not None and isinstance(qa_norm_env, dict) and isinstance(qa_norm_env.get("result"), dict):
                    qa_norm_env["result"]["lufs_gain_db"] = float(applied)
            except Exception:
                if isinstance(qa_norm_env, dict) and isinstance(qa_norm_env.get("result"), dict):
                    qa_norm_env["result"]["lufs_error"] = {"code": "normalize_lufs_exception", "stack": traceback.format_exc()}
        manifest.setdefault("items", []).append({"kind": "tts.postproc", "peak_normalize": qa_norm_env, "hard_trim": qa_trim_env, "analyze_post": ainfo_pre})
        # Back-compat locals for sidecar payload below (keep full detail, not just booleans).
        qa_norm = qa_norm_env.get("result") if isinstance(qa_norm_env, dict) and isinstance(qa_norm_env.get("result"), dict) else {}
        qa_trim = qa_trim_env.get("result") if isinstance(qa_trim_env, dict) and isinstance(qa_trim_env.get("result"), dict) else {}
        locks_meta = {}
        if lock_bundle:
            locks_meta["bundle"] = lock_bundle
    audio_section = {}
    if isinstance(lock_bundle, dict):
        audio_section = lock_bundle.get("audio") if isinstance(lock_bundle.get("audio"), dict) else {}
    ref_voice = audio_section.get("voice_embedding")
    if isinstance(ref_voice, list):
        voice_embed = await voice_embedding_from_path(wav_path)
        if isinstance(voice_embed, list):
            sim = cosine_similarity(ref_voice, voice_embed)
            if sim is not None:
                locks_meta["voice_score"] = max(0.0, min((sim + 1.0) / 2.0, 1.0))
                locks_meta["voice_embedding"] = voice_embed
    lyrics_segments = (
        audio_section.get("lyrics_segments") if isinstance(audio_section.get("lyrics_segments"), list) else []
    )
    hard_segments = [
        segment
        for segment in lyrics_segments
        if isinstance(segment, dict) and (segment.get("lock_mode") or "hard").lower() == "hard"
    ]
    if hard_segments:
        text_lower = (args.get("text") or "").lower()
        matched = 0
        for segment in hard_segments:
            seg_text = (segment.get("text") or "").lower()
            if seg_text and seg_text in text_lower:
                matched += 1
        locks_meta["lyrics_score"] = matched / len(hard_segments) if hard_segments else None
    # TTS prosody/emotion/timing QA from lock_bundle.tts when available
    tts_section = {}
    if lock_bundle:
        if isinstance(lock_bundle, dict):
            tts_section = lock_bundle.get("tts") if isinstance(lock_bundle.get("tts"), dict) else {}
        if isinstance(ainfo_pre, dict):
            pitch_mean = ainfo_pre.get("pitch_mean_hz")
            # Voice-level baseline
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
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    tref = segment.get("text_ref") if isinstance(segment.get("text_ref"), dict) else {}
                    txt = (tref.get("text") or "").strip()
                    if txt and txt == text_full:
                        seg_match = segment
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
    audio_modified = False
    try:
        target_emotion = None
        try:
            if isinstance(voice_refs_augmented, dict):
                target_emotion = voice_refs_augmented.get("emotion")
        except Exception:
            target_emotion = None
        if target_emotion and isinstance(ainfo_pre, dict):
            cur = (ainfo_pre.get("emotion") or "").lower()
            # Heuristic adjust rate/pitch to hit emotion
            if target_emotion == "excited" and cur != "excited":
                adj = dict(args)
                try:
                    adj["rate"] = max(0.5, float(args.get("rate") or 1.0) * 1.15)
                except Exception as exc:
                    log.warning(f"tts.speak: bad rate={args.get('rate')!r} during emotion adjust conversation_id={conversation_id!r}", exc_info=True)
                    adj["rate"] = max(0.5, 1.0 * 1.15)
                try:
                    adj["pitch"] = float(args.get("pitch") or 0.0) + 1.0
                except Exception as exc:
                    log.warning(f"tts.speak: bad pitch={args.get('pitch')!r} during emotion adjust conversation_id={conversation_id!r}", exc_info=True)
                    adj["pitch"] = 1.0
                res2 = provider.speak(adj)
                wb2 = res2.get("wav_bytes") or b""
                if wb2:
                    with open(wav_path, "wb") as f:
                        f.write(wb2)
                    audio_modified = True
                    sidecar(
                        wav_path,
                        {"tool": "tts.speak.committee", "target_emotion": target_emotion, "adjusted": True},
                    )
            if target_emotion == "calm" and cur != "calm":
                adj = dict(args)
                try:
                    adj["rate"] = max(0.5, float(args.get("rate") or 1.0) * 0.9)
                except Exception as exc:
                    log.warning(f"tts.speak: bad rate={args.get('rate')!r} during emotion adjust conversation_id={conversation_id!r}", exc_info=True)
                    adj["rate"] = max(0.5, 1.0 * 0.9)
                try:
                    adj["pitch"] = float(args.get("pitch") or 0.0) - 1.0
                except Exception as exc:
                    log.warning(f"tts.speak: bad pitch={args.get('pitch')!r} during emotion adjust conversation_id={conversation_id!r}", exc_info=True)
                    adj["pitch"] = -1.0
                res2 = provider.speak(adj)
                wb2 = res2.get("wav_bytes") or b""
                if wb2:
                    with open(wav_path, "wb") as f:
                        f.write(wb2)
                    audio_modified = True
                    sidecar(
                        wav_path,
                        {"tool": "tts.speak.committee", "target_emotion": target_emotion, "adjusted": True},
                    )
    except Exception:
        log.debug(f"tts.speak: committee emotion adjustment failed (non-fatal) conversation_id={conversation_id!r}", exc_info=True)
    # (Removed) per-artifact tts_samples.jsonl writer. Canonical dataset stream is `datasets/stream.py`.
    _seed_raw = args.get("seed")
    seed_int = 0
    try:
        if _seed_raw is not None and not isinstance(_seed_raw, (dict, list)):
            seed_int = int(_seed_raw)
    except Exception as exc:
        log.warning(f"tts.speak: bad seed={_seed_raw!r}; defaulting to 0 conversation_id={conversation_id!r}", exc_info=True)
        seed_int = 0
    try:
        if voice_id:
            append_provenance(
                voice_id,
                {"when": now_ts(), "tool": "tts.speak", "artifact": wav_path, "seed": seed_int},
            )
    except Exception:
        log.debug(f"tts.speak: append_provenance failed (non-fatal) conversation_id={conversation_id!r}", exc_info=True)
    add_manifest_row(manifest, wav_path, step_id="tts.speak")
    try:
        _ctx_add(
            conversation_id,
            "audio",
            wav_path,
            None,
            None,
            ["voice", f"voice:{args.get('voice')}"] if args.get("voice") else ["voice"],
            {"model": model, "trace_id": trace_id, "tool": "tts.speak", "voice_id": logical_voice_id, "voice_lock_id": voice_lock_id_resolved, "segment_id": segment_id_resolved},
        )
    except Exception:
        log.debug(f"tts.speak: failed to add context artifact (non-fatal) conversation_id={conversation_id!r}", exc_info=True)
    rel = None
    view_url = None
    try:
        rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
        view_url = f"/uploads/{rel}"
    except Exception:
        rel = None
        view_url = None
    env = {
        "meta": {
            "model": model,
            "ts": now_ts(),
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "step": 0,
            "state": "halt",
            "cont": {"present": False, "state_hash": None, "reason": None},
            # Expose segment/voice identifiers at the envelope level so downstream
            # planners and mixers can stitch multi-singer timelines.
            "segment_id": segment_id_resolved,
            "voice_id": logical_voice_id,
            "voice_lock_id": voice_lock_id_resolved,
            "path": wav_path,
            "rel_path": rel,
            "url": view_url,
        },
        "reasoning": {"goal": "tts", "constraints": ["json-only", "edge-safe"], "decisions": ["tts.speak done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "tts generated"},
        "tool_calls": [{"tool_name": "tts.speak", "tool": "tts.speak", "args": args, "arguments": args, "status": "done", "artifact_id": artifact_id_generated}],
    }
    # artifact_id was already generated before file creation
    env["artifacts"] = [
            build_artifact(
                artifact_id=artifact_id_generated,
                kind="audio",
                path=wav_path,
                trace_id=trace_id,
                conversation_id=conversation_id,
                tool_name="tts.speak",
                summary=stem,
                bytes=len(wav_bytes),
                view_url=view_url,
                url=view_url,
                tags=[],
                segment_id=segment_id_resolved,
                voice_id=logical_voice_id,
                voice_lock_id=voice_lock_id_resolved,
            ),
            # Canonical: downstream URL collectors treat kind starting with "tts" as TTS audio.
            build_artifact(
                artifact_id=f"{artifact_id_generated}:tts",
                kind="tts",
                path=wav_path,
                trace_id=trace_id,
                conversation_id=conversation_id,
                tool_name="tts.speak",
                summary=stem,
                bytes=len(wav_bytes),
                view_url=view_url,
                url=view_url,
                tags=[],
                segment_id=segment_id_resolved,
                voice_id=logical_voice_id,
                voice_lock_id=voice_lock_id_resolved,
            ),
    ]
    env["telemetry"] = {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []}
    env_meta = env.setdefault("meta", {})
    if quality_profile:
        env_meta.setdefault("quality_profile", quality_profile)
    if locks_meta:
        env_meta["locks"] = locks_meta
    env = normalize_envelope(env)
    env = bump_envelope(env)
    assert_envelope(env)
    env = stamp_env(env, "tts.speak", model)
    # Trace row for distillation
    try:
        ainfo = analyze_audio(wav_path) if audio_modified else ainfo_pre
        trace_row = {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
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
        append_training_sample("tts", trace_row)
    except Exception:
        log.debug(f"tts.speak: trace append failed (non-fatal) conversation_id={conversation_id!r} trace_id={trace_id!r}", exc_info=True)
    if trace_id and not fatal_error:
        trace_event("tool.tts.speak.complete", {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "tool": "tts.speak",
            "model": model,
            "path": wav_path,
            "bytes": len(wav_bytes),
            "duration_s": dur,
            "voice_id": logical_voice_id,
            "voice_lock_id": voice_lock_id_resolved,
            "segment_id": segment_id_resolved,
        })
    if not fatal_error:
        log.info(f"tts.speak: completed conversation_id={conversation_id!r} trace_id={trace_id!r} model={model!r} path={wav_path!r} bytes={len(wav_bytes)} duration_s={dur}")
    # Single return point (no early returns): either tool-error dict or success envelope.
    final: Dict[str, Any] = env
    if fatal_error is not None:
        # Tool-style failure (not assistant envelope): orchestrator main forwards error dict to UI/AI.
        if "stack" not in fatal_error:
            fatal_error["stack"] = "".join(traceback.format_stack())
        fatal_error.setdefault("conversation_id", conversation_id)
        if trace_id:
            fatal_error.setdefault("trace_id", trace_id)
        if logical_voice_id:
            fatal_error.setdefault("voice_id", logical_voice_id)
        if voice_lock_id:
            fatal_error.setdefault("voice_lock_id", voice_lock_id)
        if segment_id:
            fatal_error.setdefault("segment_id", segment_id)
        final = {"ok": False, "error": fatal_error}
    return final


