from __future__ import annotations

from typing import Dict, List


def judge_findings(money_map: Dict, timeline: Dict) -> Dict:
    issues: List[Dict] = []
    nodes = set(money_map.get("nodes", []) or [])
    for e in (money_map.get("edges") or []):
        if (e.get("src") not in nodes) or (e.get("dst") not in nodes):
            issues.append({"issue": "dangling_edge", "edge": e})
    uncertainties: List[str] = []
    if issues:
        uncertainties.append("Graph contains dangling edges.")
    uncertainties.append("Amounts/vehicles inferred heuristically; verify primary docs.")
    falsification = [
        "Obtain primary contract/grant filings and compare dates/amounts.",
        "Seek independent sources for each edge with corroborating quotes.",
        "Confirm party identities (disambiguate similarly named orgs).",
    ]
    return {"uncertainties": uncertainties, "falsification": falsification, "issues": issues}


