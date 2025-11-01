from __future__ import annotations

import os
import json
import time
from ..determinism.seeds import SEEDS, stamp_tool_args, stamp_envelope


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def sidecar(path: str, data: dict) -> None:
    with open(path + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def make_outpaths(root: str, stem: str):
    ensure_dir(root)
    return os.path.join(root, stem + ".png"), os.path.join(root, stem + ".json")


def normalize_size(size: str | None, edge_safe: bool = False) -> str:
    if not size:
        return "512x512" if edge_safe else "1024x1024"
    return size


def stamp_env(env: dict, tool: str, model: str | None = None) -> dict:
    return stamp_envelope(env, tool=tool, model=model)


def now_ts() -> int:
    return int(time.time())


