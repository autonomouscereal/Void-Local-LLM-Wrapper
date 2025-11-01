from __future__ import annotations

import os
import time
import hashlib
import random
import base64


def trace_id(seed: str = "") -> str:
    """26-char slug: ts + hash(seed+rand)"""
    t = int(time.time() * 1000)
    r = os.urandom(8)
    h = hashlib.sha1((seed + str(t)).encode() + r).digest()[:8]
    raw = t.to_bytes(6, "big") + h
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def step_id(prev: str | None = None) -> str:
    """Monotonic-ish; include ms time and 12 bits of rand"""
    t = int(time.time() * 1000)
    r = random.getrandbits(12)
    return f"{t:x}-{r:x}"


def short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


