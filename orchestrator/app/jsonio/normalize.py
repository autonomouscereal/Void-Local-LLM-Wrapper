from __future__ import annotations

import time
from typing import Any, Dict
from ..json_parser import JSONParser
from .coerce import coerce_to_envelope
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


def normalize_envelope(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Fill defaults for the envelope structure described by the user
    parser = JSONParser()
    # IMPORTANT: `ensure_structure` is an internal coercion implementation detail.
    # Use parse so normalization stays on the public API surface.
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
    base.setdefault("telemetry", {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []})
    return base


def _parse_json(text: str) -> Dict[str, Any]:
    parser = JSONParser()
    # Parse provider/tool output into the high-level envelope shape. Fields not
    # represented in EXPECTED_ENVELOPE_SHAPE are ignored at this layer;
    # downstream helpers (coerce_to_envelope, limits) operate on the coerced envelope.
    return parser.parse(text or "{}", EXPECTED_ENVELOPE_SHAPE)



def normalize_to_envelope(raw_text: str) -> Dict[str, Any]:
    """
    Converts any provider/tool output into the canonical envelope using the hardened parser.
    Must never throw on mildly malformed JSON—coerce or fill defaults instead.
    """
    obj = _parse_json(raw_text)
    env = coerce_to_envelope(obj)
    return env


