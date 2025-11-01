from __future__ import annotations

import time
from typing import List, Dict, Any
import re


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


