from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional


def _dir(cid: str) -> str:
    d = os.path.join("/workspace", "uploads", "artifacts", "film", cid, "snapshots")
    os.makedirs(d, exist_ok=True)
    return d


def save_shot_snapshot(cid: str, shot_id: str, payload: Dict[str, Any]) -> None:
    p = os.path.join(_dir(cid), f"{shot_id}.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(tmp, p)


def load_shot_snapshot(cid: str, shot_id: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(_dir(cid), f"{shot_id}.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return None


