from __future__ import annotations

from typing import List, Dict


def event(state: str, phase: str, progress: float, artifacts: List[Dict] | None = None, **extra) -> Dict:
    e = {"state": state, "phase": phase, "progress": round(float(progress or 0.0), 3), "artifacts": artifacts or []}
    e.update(extra or {})
    return e


