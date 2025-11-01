from __future__ import annotations

from typing import Dict, List


def make_report(query: str, ledger_rows: List[Dict], money_map: Dict, timeline: Dict, judge_o: Dict) -> Dict:
    edges = list(money_map.get("edges", []) or [])
    edges.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
    findings: List[Dict] = []
    for e in edges[:20]:
        findings.append({
            "summary": f"{e.get('src')} → {e.get('dst')} ${e.get('amount')} via {e.get('vehicle')}",
            "evidence": list(e.get("evidence", []) or [])[:5],
            "score": float(e.get("score", 0.0)),
            "date": e.get("date"),
        })
    return {
        "query": query,
        "findings": findings,
        "money_map": money_map,
        "timeline": timeline,
        "uncertainties": list(judge_o.get("uncertainties", []) or []),
        "falsification": list(judge_o.get("falsification", []) or []),
    }


