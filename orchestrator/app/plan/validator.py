from __future__ import annotations

from typing import Dict, Any, List

from .catalog import REQUIRED_INPUTS


def validate_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    steps = plan.get("plan") or []
    if not isinstance(steps, list):
        return [{"id": None, "error": "invalid_plan_type"}]
    for step in steps:
        sid = (step or {}).get("id")
        tool = (step or {}).get("tool")
        inputs = (step or {}).get("inputs") or {}
        if tool not in REQUIRED_INPUTS:
            errors.append({"id": sid, "error": f"unknown_tool:{tool}"})
            continue
        for key in REQUIRED_INPUTS[tool]:
            if key not in inputs:
                errors.append({"id": sid, "error": f"missing_input:{key}"})
    return errors


