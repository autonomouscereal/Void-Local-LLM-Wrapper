from __future__ import annotations

from typing import Dict, Any, List


def rrf_fuse(results_by_engine: Dict[str, List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    # Reciprocal Rank Fusion + CombMNZ with stable tie-break
    scores: Dict[str, float] = {}
    freq: Dict[str, int] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for eng, items in results_by_engine.items():
        for rank, it in enumerate(items, start=1):
            key = it.get("link") or it.get("url") or it.get("id") or f"{eng}:{rank}"
            rr = 1.0 / float(k + rank)
            scores[key] = scores.get(key, 0.0) + rr
            freq[key] = freq.get(key, 0) + 1
            if key not in meta:
                meta[key] = {"title": it.get("title"), "link": it.get("link") or it.get("url"), "snippet": it.get("snippet")}
    # Combine: CombMNZ = (sum of scores) * freq; round to 1e-6 and tie-break deterministically
    def _auth(x: Dict[str, Any]) -> float:
        return 0.0
    def _rec(x: Dict[str, Any]) -> float:
        return 0.0
    def _sha(x: str) -> str:
        import hashlib as _h
        return _h.sha256(x.encode("utf-8")).hexdigest()
    items: List[Dict[str, Any]] = []
    for key, sc in scores.items():
        m = meta.get(key) or {}
        comb = float(f"{(sc * float(freq.get(key, 1))):.6f}")
        items.append({"key": key, "score": comb, "rrf": float(f"{sc:.6f}"), "freq": int(freq.get(key, 1)), **m})
    items.sort(key=lambda r: (-r["score"], -_auth(r), -_rec(r), _sha(r.get("key") or "")))
    return items


