from __future__ import annotations

import time
from typing import Any, Dict
from ..json_parser import JSONParser


def normalize_envelope(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Fill defaults for the envelope structure described by the user
    expected = {
        "meta": dict,
        "reasoning": dict,
        "evidence": list,
        "message": dict,
        "tool_calls": list,
        "artifacts": list,
        "telemetry": dict,
    }
    parser = JSONParser()
    base = parser.ensure_structure(obj if isinstance(obj, dict) else {}, expected)
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
    base.setdefault("telemetry", {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []})
    return base


def to_envelope(raw_text: str, model_name: str, cid: str, step: int) -> Dict[str, Any]:
    # Parse arbitrary text as JSON (using hardened parser), then ensure envelope defaults
    parser = JSONParser()
    try:
        obj = parser.parse(raw_text or "{}", {})
    except Exception:
        obj = {}
    env = normalize_envelope(obj)
    env["meta"]["model"] = model_name or env["meta"].get("model")
    env["meta"]["cid"] = cid or env["meta"].get("cid")
    env["meta"]["step"] = int(step)
    return env


