from __future__ import annotations

import os
import json
import time
from ..determinism.seeds import stamp_tool_args, stamp_envelope


def now_ts() -> int:
    return int(time.time())


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def tts_edge_defaults(params: dict, edge: bool) -> dict:
    out = {
        "sample_rate": 22050 if edge else int(params.get("sample_rate") or 24000),
        "channels": 1,
        "max_seconds": min(int(params.get("max_seconds") or 20), 30),
        "voice": params.get("voice") or "narrator",
    }
    for k, v in (params or {}).items():
        if k not in out and v is not None:
            out[k] = v
    return out


def sidecar(base_path: str, data: dict) -> None:
    with open(base_path + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def stamp_env(env: dict, tool: str, model: str | None = None):
    return stamp_envelope(env, tool=tool, model=model)


