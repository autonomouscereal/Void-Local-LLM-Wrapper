from __future__ import annotations

try:
    from xxhash import xxh64  # type: ignore
except Exception:  # fallback to hashlib
    import hashlib
    class _XX:
        @staticmethod
        def xxh64(s: str):
            class _H:
                def __init__(self, s: str):
                    self._d = int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)
                def intdigest(self) -> int:
                    return self._d
            return _H(s)
    xxh64 = _XX.xxh64  # type: ignore


def seed_router(trace_id: str, master: int) -> int:
    return int(xxh64(f"router|{trace_id}|{master}").intdigest())


def seed_tool(name: str, trace_id: str, master: int) -> int:
    return int(xxh64(f"tool|{name}|{trace_id}|{master}").intdigest())


def round6(x: float) -> float:
    try:
        return float(f"{float(x):.6f}")
    except Exception:
        return 0.0


