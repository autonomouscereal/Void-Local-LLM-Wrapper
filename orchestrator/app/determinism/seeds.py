from __future__ import annotations

import json
import time
from typing import Any, Dict

try:
    from ..determinism import seed_tool as _seed_tool
except Exception:
    def _seed_tool(name: str, trace_id: str, master: int) -> int:
        try:
            import xxhash  # type: ignore
            return xxhash.xxh64(f"tool|{name}|{trace_id}|{master}").intdigest()
        except Exception:
            import hashlib
            return int(hashlib.sha256(f"tool|{name}|{trace_id}|{master}".encode("utf-8")).hexdigest()[:16], 16)


SEEDS: Dict[str, int] = {}


def stamp_tool_args(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return args
    if "seed" in args and isinstance(args["seed"], int):
        return args
    trace_id = args.get("cid") or args.get("trace_id") or str(int(time.time()))
    master = int(args.get("master_seed", 0) or 0)
    args["seed"] = int(_seed_tool(tool or "tool", str(trace_id), master)) & ((1 << 31) - 1)
    return args


def stamp_envelope(env: Dict[str, Any], tool: str | None = None, model: str | None = None) -> Dict[str, Any]:
    if not isinstance(env, dict):
        return env
    meta = env.setdefault("meta", {})
    if tool:
        meta.setdefault("tool", tool)
    if model:
        meta.setdefault("model", model)
    if "seed" not in meta:
        trace_id = str(meta.get("cid") or meta.get("trace_id") or int(time.time()))
        meta["seed"] = int(_seed_tool(tool or "env", trace_id, int(meta.get("master_seed", 0) or 0))) & ((1 << 31) - 1)
    return env

from __future__ import annotations

import os
import time
import hashlib
from typing import Dict, Any


def _seed_from_entropy(tag: str = "") -> int:
    # Deterministic-ish 32-bit seed from time + tag + process entropy
    try:
        ent = os.urandom(8).hex()
    except Exception:
        ent = ""
    h = hashlib.sha256((str(time.time_ns()) + tag + ent).encode()).hexdigest()
    return int(h[:8], 16)


class SeedRegistry:
    def __init__(self):
        self._seeds: Dict[str, int] = {}

    def get(self, key: str, default: int | None = None) -> int:
        if key not in self._seeds:
            self._seeds[key] = (int(default) if default is not None else _seed_from_entropy(key))
        return int(self._seeds[key])

    def set(self, key: str, seed: int) -> None:
        self._seeds[key] = int(seed)

    def snapshot(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self._seeds.items()}


SEEDS = SeedRegistry()


def stamp_tool_args(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return args
    if ("seed" not in args) or (args.get("seed") is None):
        args["seed"] = SEEDS.get(f"tool:{tool}")
    return args


def stamp_envelope(env: dict, tool: str | None = None, model: str | None = None) -> dict:
    meta = env.setdefault("meta", {})
    det = meta.setdefault("determinism", {})
    if model:
        det.setdefault("model", {})
        det["model"].setdefault("seed", SEEDS.get(f"model:{model}"))
    if tool:
        det.setdefault("tools", {})
        det["tools"].setdefault(tool, {"seed": SEEDS.get(f"tool:{tool}")})
    return env


