from __future__ import annotations

import logging
import math
import re

log = logging.getLogger(__name__)


def score_chunk(chunk_text: str, query_text: str) -> float:
    """Score a chunk against a query using a simple BM25-like heuristic.

    Any unexpected parsing error is logged and treated as zero relevance
    instead of raising, so ICW packing can continue while still surfacing
    the underlying problem in logs.
    """
    try:
        q = query_text if isinstance(query_text, str) else str(query_text or "")
        c = chunk_text if isinstance(chunk_text, str) else str(chunk_text or "")
        qt = set(re.findall(r"\w+", (q or "").lower()))
        ct = set(re.findall(r"\w+", (c or "").lower()))
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
