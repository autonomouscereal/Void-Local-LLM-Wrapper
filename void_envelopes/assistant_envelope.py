from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict

from void_json.json_parser import JSONParser


def _iso_now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


# High-level envelope shape for JSONParser to coerce into. This matches the
# canonical sections used throughout the orchestrator and keeps the parser
# schema-driven instead of treating everything as an untyped dict.
EXPECTED_ENVELOPE_SHAPE: Dict[str, Any] = {
    "meta": dict,
    "reasoning": dict,
    "evidence": list,
    "message": dict,
    "tool_calls": list,
    "artifacts": list,
    "telemetry": dict,
}


DEFAULT_ENV: Dict[str, Any] = {
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
    """
    Coerce a loosely-shaped dict into the canonical "assistant envelope".

    This envelope is used internally by the orchestrator pipeline (meta/message/tool_calls/etc).
    It is distinct from the ToolEnvelope (`ok/result/error`) used for tool-ish HTTP routes.
    """
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


def normalize_envelope(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill defaults for the envelope structure described by EXPECTED_ENVELOPE_SHAPE.
    """
    parser = JSONParser()
    base = parser.parse(obj if isinstance(obj, dict) else {}, EXPECTED_ENVELOPE_SHAPE)
    meta = base.get("meta", {}) or {}
    meta.setdefault("schema_version", 1)
    meta.setdefault("model", "")
    meta.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta.setdefault("cid", "")
    meta.setdefault("step", 0)
    meta.setdefault("state", "running")
    meta.setdefault("cont", {"present": False, "state_hash": None, "reason": None})
    meta.setdefault("device_profile", {"ram_mb": 0, "max_context_bytes": 0, "offline": False})
    base["meta"] = meta
    msg = base.get("message", {}) or {}
    msg.setdefault("role", "assistant")
    msg.setdefault("type", "text")
    msg.setdefault("content", "")
    base["message"] = msg
    base.setdefault("reasoning", {"goal": "", "constraints": [], "decisions": []})
    base.setdefault("evidence", [])
    base.setdefault("tool_calls", [])
    base.setdefault("artifacts", [])
    base.setdefault(
        "telemetry",
        {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    )
    return base


def _parse_json(text: str) -> Dict[str, Any]:
    parser = JSONParser()
    return parser.parse(text or "{}", EXPECTED_ENVELOPE_SHAPE)


def normalize_to_envelope(raw_text: str) -> Dict[str, Any]:
    """
    Convert any provider/tool output into the canonical assistant envelope using JSONParser.
    Must never throw on mildly malformed JSON—coerce or fill defaults instead.
    """
    obj = _parse_json(raw_text)
    env = coerce_to_envelope(obj)
    return env


