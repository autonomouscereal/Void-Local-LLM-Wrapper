from __future__ import annotations

from typing import Dict, Any
import os
import traceback

import httpx  # type: ignore

from ..json_parser import JSONParser
from ..locks.voice_identity import resolve_voice_identity


async def run_voice_register(args: Dict[str, Any]):
    """
    Register or update a voice in the RVC registry from reference samples.
    Returns either {"result": {...}} or {"error": {...}} for the dispatcher.
    """
    rvc_url = os.getenv("RVC_API_URL")
    if not isinstance(rvc_url, str) or not rvc_url.strip():
        return {
            "error": {
                "code": "rvc_unconfigured",
                "message": "RVC_API_URL not configured for voice.register",
                "status": 500,
            }
        }
    a = args if isinstance(args, dict) else {}
    # Accept either a full inline voice_refs or a simple voice_samples list.
    inline = a.get("voice_refs") if isinstance(a.get("voice_refs"), dict) else None
    if inline is None and isinstance(a.get("voice_samples"), list):
        samples = [p for p in a.get("voice_samples") if isinstance(p, str) and p]
        if samples:
            inline = {"voice_samples": samples}
    vid_raw = a.get("voice_id")
    canonical_id, lock, meta = resolve_voice_identity(vid_raw, inline)
    if not canonical_id:
        return {
            "error": {
                "code": "voice_resolution_failed",
                "message": "Unable to resolve voice_id for voice.register; provide voice_id or voice_refs with samples.",
                "status": 400,
            }
        }
    samples = lock.get("voice_samples") if isinstance(lock.get("voice_samples"), list) else []
    if not samples:
        # Nothing to register yet; return canonical voice identity only.
        return {
            "result": {
                "voice_id": canonical_id,
                "registered": False,
                "reason": "no_samples",
                "meta": meta if isinstance(meta, dict) else {},
            }
        }
    payload_reg = {
        "voice_lock_id": str(canonical_id),
        "model_name": str(canonical_id),
        "reference_wav_path": samples[0],
        "additional_refs": samples[1:],
    }
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.post(rvc_url.rstrip("/") + "/v1/voice/register", json=payload_reg)
        raw = r.text or ""
        parser = JSONParser()
        env = parser.parse(raw or "{}", {"schema_version": int, "trace_id": str, "ok": bool, "result": dict, "error": dict})
    if isinstance(env, dict) and env.get("ok"):
        return {
            "result": {
                "voice_id": canonical_id,
                "registered": True,
                "rvc": env,
                "meta": meta if isinstance(meta, dict) else {},
            }
        }
    err_obj = env.get("error") if isinstance(env, dict) else {}
    return {
        "error": {
            "code": (err_obj or {}).get("code") or "rvc_register_failed",
            "message": (err_obj or {}).get("message") or "RVC /v1/voice/register failed",
            "status": int((err_obj or {}).get("status") or getattr(r, "status_code", 500) or 500),
            "raw": env,
            "stack_local": "".join(traceback.format_stack()),
        }
    }


async def run_voice_train(args: Dict[str, Any]):
    """
    Train or retrain the RVC model for a given voice_id using all registered refs.
    Returns either {"result": {...}} or {"error": {...}} for the dispatcher.
    """
    rvc_url = os.getenv("RVC_API_URL")
    if not isinstance(rvc_url, str) or not rvc_url.strip():
        return {
            "error": {
                "code": "rvc_unconfigured",
                "message": "RVC_API_URL not configured for voice.train",
                "status": 500,
            }
        }
    a = args if isinstance(args, dict) else {}
    vid = a.get("voice_id") or a.get("voice_lock_id")
    if not isinstance(vid, str) or not vid.strip():
        return {
            "error": {
                "code": "ValidationError",
                "message": "voice_id is required for voice.train",
                "status": 400,
            }
        }
    payload_train = {"voice_lock_id": vid.strip()}
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.post(rvc_url.rstrip("/") + "/v1/voice/train", json=payload_train)
        raw = r.text or ""
        parser = JSONParser()
        env = parser.parse(raw or "{}", {"schema_version": int, "trace_id": str, "ok": bool, "result": dict, "error": dict})
    if isinstance(env, dict) and env.get("ok"):
        return {"result": env}
    err_obj = env.get("error") if isinstance(env, dict) else {}
    return {
        "error": {
            "code": (err_obj or {}).get("code") or "rvc_train_failed",
            "message": (err_obj or {}).get("message") or "RVC /v1/voice/train failed",
            "status": int((err_obj or {}).get("status") or getattr(r, "status_code", 500) or 500),
            "raw": env,
            "stack_local": "".join(traceback.format_stack()),
        }
    }


