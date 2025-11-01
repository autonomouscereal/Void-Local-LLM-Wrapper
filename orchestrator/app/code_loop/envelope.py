from __future__ import annotations

import time


def make_artifact(id: str, kind: str, summary: str, bytes_count: int = 0):
    return {"id": id, "kind": kind, "summary": summary, "bytes": int(bytes_count)}


def make_envelope(model: str, content: str, artifacts: list, tool_calls: list, decisions: list):
    return {
        "meta": {"schema_version": 1, "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ")},
        "reasoning": {"goal": "code change", "constraints": ["diff-only", "json-only"], "decisions": (decisions or [])[-20:]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": content or ""},
        "tool_calls": tool_calls or [],
        "artifacts": artifacts or [],
        "telemetry": {"window": {"input_bytes": len((content or "").encode()), "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }


