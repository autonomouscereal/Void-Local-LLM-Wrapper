from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Dict, Any, List


ROOT = os.getenv("PROMPTS_ROOT", os.path.join("/workspace", "uploads", "artifacts", "prompts"))


def _id_of(content: str) -> str:
    h = hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:16]
    return f"p_{h}"


def save_prompt(name: str, content: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    os.makedirs(ROOT, exist_ok=True)
    pid = _id_of(content or "")
    p = os.path.join(ROOT, f"{pid}.json")
    body = {"prompt_id": pid, "name": name, "content": content, "meta": meta or {}, "created_at": int(time.time())}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False, indent=2))
    os.replace(tmp, p)
    return body


def list_prompts() -> List[Dict[str, Any]]:
    if not os.path.exists(ROOT):
        return []
    out: List[Dict[str, Any]] = []
    for f in os.listdir(ROOT):
        if f.endswith(".json"):
            try:
                out.append(json.load(open(os.path.join(ROOT, f), "r", encoding="utf-8")))
            except Exception:
                continue
    return sorted(out, key=lambda x: x.get("created_at", 0), reverse=True)


