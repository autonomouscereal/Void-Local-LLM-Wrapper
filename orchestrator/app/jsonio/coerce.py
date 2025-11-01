from __future__ import annotations

import datetime as dt


def _iso_now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


DEFAULT_ENV = {
    "meta": {
        "schema_version": 1,
        "model": "unknown",
        "ts": _iso_now(),
        "cid": "",
        "step": 0,
        "state": "running",
        "cont": {"present": False, "state_hash": None, "reason": None},
        "device_profile": {"ram_mb": 0, "max_context_bytes": 0, "offline": False},
    },
    "reasoning": {"goal": "", "constraints": [], "decisions": []},
    "evidence": [],
    "message": {"role": "assistant", "type": "text", "content": ""},
    "tool_calls": [],
    "artifacts": [],
    "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
}


def coerce_to_envelope(obj: dict) -> dict:
    env = {
        k: (obj.get(k) if isinstance(obj.get(k), (dict, list, str, int, float, bool)) else None)
        for k in DEFAULT_ENV.keys()
    }
    # deep-merge defaults without clobbering present keys
    out = dict(DEFAULT_ENV)
    for k, v in env.items():
        if v is None:
            continue
        if isinstance(DEFAULT_ENV[k], dict) and isinstance(v, dict):
            merged = dict(DEFAULT_ENV[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    # minimal safety: ensure message/content exists
    msg = out.get("message") or {}
    if not isinstance(msg, dict):
        msg = {"role": "assistant", "type": "text", "content": str(obj)[:2000]}
    else:
        msg.setdefault("role", "assistant")
        msg.setdefault("type", "text")
        msg.setdefault("content", "")
    out["message"] = msg
    return out


