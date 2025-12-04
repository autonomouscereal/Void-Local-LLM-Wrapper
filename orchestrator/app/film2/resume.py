from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")


class Phase:
    CLARIFY = 0
    BIBLES = 1
    PLANNER = 2
    STORYBOARDS = 3
    ANIMATIC = 4
    FINAL = 5
    EXPORTS = 6


@dataclass
class Checkpoint:
    cid: str
    phase: int
    extra: Optional[Dict[str, Any]] = None


def _cp_path(cid: str) -> str:
    root = os.path.join(UPLOAD_DIR, "film", cid)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "checkpoint.json")


def load_checkpoint(cid: str) -> Checkpoint:
    try:
        from ..json_parser import JSONParser
        with open(_cp_path(cid), "r", encoding="utf-8") as f:
            parser = JSONParser()
            schema = {"phase": int, "extra": dict}
            sup = parser.parse_superset(f.read(), schema)
            js = sup["coerced"]
            return Checkpoint(
                cid=cid,
                phase=int(js.get("phase", 0)),
                extra=js.get("extra") if isinstance(js, dict) else None,
            )
    except Exception:
        return Checkpoint(cid=cid, phase=0, extra=None)


def save_checkpoint(cid: str, phase: int, extra: Optional[Dict[str, Any]] = None) -> None:
    data = {"phase": int(phase)}
    if extra:
        data["extra"] = extra
    path = _cp_path(cid)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)


def _shots_done_path(cid: str) -> str:
    root = os.path.join(UPLOAD_DIR, "film", cid)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "shots_done.json")


def mark_shot_done(cid: str, shot_id: str) -> None:
    try:
        p = _shots_done_path(cid)
        done = []
        if os.path.exists(p):
            from ..json_parser import JSONParser
            with open(p, "r", encoding="utf-8") as f:
                parser = JSONParser()
                schema = [str]
                sup = parser.parse_superset(f.read(), schema)
                parsed = sup["coerced"]
                done = parsed if isinstance(parsed, list) else []
        if shot_id not in done:
            done.append(shot_id)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(done))
        os.replace(tmp, p)
    except Exception:
        return


def is_shot_done(cid: str, shot_id: str) -> bool:
    try:
        p = _shots_done_path(cid)
        if not os.path.exists(p):
            return False
        from ..json_parser import JSONParser
        with open(p, "r", encoding="utf-8") as f:
            parser = JSONParser()
            schema = [str]
            sup = parser.parse_superset(f.read(), schema)
            parsed = sup["coerced"]
            done = parsed if isinstance(parsed, list) else []
        return shot_id in (done or [])
    except Exception:
        return False


