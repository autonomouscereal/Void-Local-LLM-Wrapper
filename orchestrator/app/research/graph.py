from __future__ import annotations

from typing import Dict, List, Set


def score_edge(e: Dict) -> float:
    s = 0.5
    if e.get("amount"):
        s += 0.2
    if e.get("vehicle") and e.get("vehicle") != "unspecified":
        s += 0.1
    if e.get("date"):
        s += 0.1
    s += 0.1 * min(3, len(set(e.get("evidence", []) or [])))
    return float(min(0.99, s))


def build_money_map(edges: List[Dict]) -> Dict:
    nodes: Set[str] = set()
    for e in (edges or []):
        nodes.add(str(e.get("src")))
        nodes.add(str(e.get("dst")))
        e["score"] = score_edge(e)
    return {"nodes": sorted(nodes), "edges": edges or []}


