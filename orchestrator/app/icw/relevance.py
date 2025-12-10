from __future__ import annotations

import logging
import math
import re

log = logging.getLogger("orchestrator.icw")


def score_chunk(chunk_text: str, query_text: str) -> float:
    """Score a chunk against a query using a simple BM25-like heuristic.

    Any unexpected parsing error is logged and treated as zero relevance
    instead of raising, so ICW packing can continue while still surfacing
    the underlying problem in logs.
    """
    try:
        qt = set(re.findall(r"\w+", (query_text or "").lower()))
        ct = set(re.findall(r"\w+", (chunk_text or "").lower()))
        denom = 1.0 + math.log(1.0 + len(ct))
        return float(len(qt & ct)) / denom if denom > 0 else 0.0
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error(
            "icw.score_chunk failed for query=%r, chunk_prefix=%r: %s",
            (query_text or "")[:128],
            (chunk_text or "")[:128],
            exc,
            exc_info=True,
        )
        return 0.0
