from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class QaReport:
    fail_rate: float
    issues: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"fail_rate": float(self.fail_rate), "issues": list(self.issues or [])}


def qa_storyboards(sboards_dir, char_bible) -> QaReport:
    issues: List[Dict[str, Any]] = []
    total = 0
    fails = 0
    for sb in (sboards_dir or []):
        if not isinstance(sb, dict):
            continue
        total += 1
        if not sb.get("path"):
            fails += 1
            issues.append({"shot": sb.get("id"), "issue": "missing_image"})
        if not sb.get("style"):
            issues.append({"shot": sb.get("id"), "issue": "style_missing"})
    rate = (fails / float(total)) if total else 0.0
    return QaReport(fail_rate=rate, issues=issues)


def qa_animatic(animatic_path, shots, char_bible) -> QaReport:
    total = sum(int(s.get("duration_ms", 0)) for s in (shots or []))
    dur = int((animatic_path or {}).get("duration", 0)) * 1000
    dur_ok = abs(dur - total) <= int(total * 0.03) if total else True
    issues: List[Dict[str, Any]] = []
    if not dur_ok:
        issues.append({"issue": "duration_mismatch"})
    return QaReport(fail_rate=(0.0 if dur_ok else 0.2), issues=issues)


def apply_autofix(asset, report: QaReport):
    # Deterministic no-op placeholder; extend with rerender logic if needed.
    return asset


