from __future__ import annotations

import hashlib
from typing import Dict, Any

try:
    from xxhash import xxh64  # type: ignore
except Exception:
    xxh64 = None  # type: ignore


def _seed_from_tag(tag: str) -> int:
    """Deterministic 31-bit seed from a tag string."""
    if xxh64 is not None:
        try:
            return int(xxh64(tag).intdigest()) & ((1 << 31) - 1)
        except Exception:
            # Fall back to sha256 below.
            return int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16) & ((1 << 31) - 1)
    # Fallback: sha256 first 8 hex chars
    return int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16) & ((1 << 31) - 1)


class SeedRegistry:
    def __init__(self):
        self._seeds: Dict[str, int] = {}

    def get(self, key: str, default: int | None = None) -> int:
        if key not in self._seeds:
            self._seeds[key] = int(default) if default is not None else _seed_from_tag(key)
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
    if not isinstance(env, dict):
        return env
    meta = env.setdefault("meta", {})
    # Provide a top-level seed for convenience
    meta.setdefault("seed", SEEDS.get("env"))
    if model:
        meta.setdefault("model", model)
    det = meta.setdefault("determinism", {})
    if model:
        det.setdefault("model", {})
        det["model"].setdefault("seed", SEEDS.get(f"model:{model}"))
    if tool:
        det.setdefault("tools", {})
        det["tools"].setdefault(tool, {"seed": SEEDS.get(f"tool:{tool}")})
    return env


