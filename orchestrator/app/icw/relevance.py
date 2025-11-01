from __future__ import annotations

import math
import re


def score_chunk(chunk_text: str, query_text: str) -> float:
    # BM25-lite-ish: term overlap scaled by chunk length
    try:
        qt = set(re.findall(r"\w+", (query_text or "").lower()))
        ct = set(re.findall(r"\w+", (chunk_text or "").lower()))
        denom = 1.0 + math.log(1.0 + len(ct))
        return float(len(qt & ct)) / denom
    except Exception:
        return 0.0


