from __future__ import annotations

from typing import List, Dict


def build_timeline(ledger_rows: List[Dict]) -> Dict:
    out: List[Dict] = []
    for r in (ledger_rows or []):
        dt = r.get("ts")
        out.append({
            "date": dt,
            "title": (r.get("title", "") or "")[:160],
            "id": r.get("id"),
            "url": r.get("url", ""),
        })
    out.sort(key=lambda x: (x.get("date") or 0))
    return {"events": out}


