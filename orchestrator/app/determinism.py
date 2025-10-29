from __future__ import annotations

from xxhash import xxh64


def seed_router(trace_id: str, master: int) -> int:
    return xxh64(f"router|{trace_id}|{master}").intdigest()


def seed_tool(name: str, trace_id: str, master: int) -> int:
    return xxh64(f"tool|{name}|{trace_id}|{master}").intdigest()


def round6(x: float) -> float:
    try:
        return float(f"{float(x):.6f}")
    except Exception:
        return 0.0


