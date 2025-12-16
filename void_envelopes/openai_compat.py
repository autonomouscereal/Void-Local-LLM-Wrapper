from __future__ import annotations

import re
import time
from typing import Any, Dict, List


_TAG_RE = re.compile(r"</?(CONT|HALT)\b[^>]*>", re.I)


def strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text or "")


def stitch_openai_response(partials: List[str], model_name: str) -> Dict[str, Any]:
    content = "\n\n".join([strip_tags(p) for p in (partials or []) if isinstance(p, str)])
    return {
        "id": "orc-windows-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def stitch_openai(final_partials: List[str], model_name: str) -> Dict[str, Any]:
    text = "".join(strip_tags(p) for p in (final_partials or []))
    return {
        "id": "chatcmpl_" + str(abs(hash(text)))[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name or "unknown",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


def merge_envelopes(step_envs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not step_envs:
        return {}
    final = dict(step_envs[-1])
    # message
    content = "".join(strip_tags((e.get("message", {}) or {}).get("content", "")) for e in step_envs)
    final.setdefault("message", {})
    final["message"]["content"] = content
    # tool_calls
    tc: List[Dict[str, Any]] = []
    for e in step_envs:
        t = e.get("tool_calls") or []
        if isinstance(t, list):
            tc.extend(t)
    final["tool_calls"] = tc
    # artifacts (unique by id)
    seen, arr = set(), []
    for e in step_envs:
        for a in e.get("artifacts", []) or []:
            aid = a.get("id") if isinstance(a, dict) else None
            if not aid or aid in seen:
                continue
            seen.add(aid)
            arr.append(a)
    final["artifacts"] = arr
    # reasoning: concat decisions tail
    decs: List[str] = []
    for e in step_envs:
        r = e.get("reasoning") or {}
        ds = r.get("decisions") or []
        if isinstance(ds, list):
            decs.extend([d for d in ds if isinstance(d, str)])
    final.setdefault("reasoning", {})
    final["reasoning"]["decisions"] = decs[-50:]
    return final


def build_openai_envelope(
    *,
    ok: bool,
    text: str,
    error: Dict[str, Any] | None,
    usage: Dict[str, Any],
    model: str,
    seed: int,
    id_: str,
) -> Dict[str, Any]:
    """
    Canonical OpenAI-style chat.completion envelope builder.

    Used by orchestrator endpoints that expose OpenAI-compatible responses on
    top of the internal assistant envelope / committee pipeline.
    """
    env: Dict[str, Any] = {
        "id": id_ or "orc-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": (text or "(no content)")},
            }
        ],
        "usage": usage,
        "ok": bool(ok),
        "error": (error or None),
        "seed": seed,
    }
    # cid/trace_id are injected by orchestrator callers (e.g., finalize_response)
    # via an out-of-band _meta block to keep the OpenAI schema surface stable.
    return env


