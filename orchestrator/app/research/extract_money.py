from __future__ import annotations

import re
from typing import List, Dict, Optional


AMT_RE = re.compile(r'(?i)\$?\b(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?)\b')
DATE_RE = re.compile(r'\b(20\d{2}|19\d{2})(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b')
VEHICLE = ["grant", "contract", "equity", "award", "donation", "loan", "purchase", "licensing"]


def find_parties(text: str) -> List[str]:
    return re.findall(r'\b([A-Z][A-Za-z0-9&\.\-]+(?:\s+[A-Z][A-Za-z0-9&\.\-]+){0,3})\b', text or "")[:6]


def infer_vehicle(text: str) -> Optional[str]:
    t = (text or "").lower()
    for v in VEHICLE:
        if v in t:
            return v
    return None


def extract_edges(ledger_rows: List[Dict], query: str = "") -> List[Dict]:
    edges: List[Dict] = []
    for r in (ledger_rows or []):
        t = f"{r.get('title','')} {r.get('excerpt','')}"
        amts = AMT_RE.findall(t)
        parties = find_parties(t)
        veh = infer_vehicle(t)
        dt = DATE_RE.search(t)
        if len(parties) >= 2 and amts:
            edges.append({
                "src": parties[0],
                "dst": parties[1],
                "amount": amts[0],
                "ccy": "USD",
                "vehicle": veh or "unspecified",
                "date": (dt.group(0) if dt else None),
                "evidence": [r.get("ledger_id") or r.get("id")],
            })
    return edges


