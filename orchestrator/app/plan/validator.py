from __future__ import annotations

import logging
from typing import Dict, Any, List

from .catalog import REQUIRED_INPUTS

log = logging.getLogger("orchestrator.plan.validator")

def validate_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    if not isinstance(plan, dict):
        log.error("plan.validator validate_plan non-dict plan type=%s", type(plan).__name__)
        return [{"id": None, "error": "invalid_plan_root"}]
    steps = plan.get("plan") or plan.get("steps") or []
    if not isinstance(steps, list):
        log.error("plan.validator validate_plan invalid steps type=%s", type(steps).__name__)
        return [{"id": None, "error": "invalid_plan_type"}]
    for step in steps:
        if not isinstance(step, dict):
            errors.append({"id": None, "error": "invalid_step_type"})
            continue
        sid = step.get("id")
        tool = step.get("tool")
        inputs = step.get("inputs") or step.get("args") or {}
        if tool not in REQUIRED_INPUTS:
            errors.append({"id": sid, "error": f"unknown_tool:{tool}"})
            continue
        if not isinstance(inputs, dict):
            errors.append({"id": sid, "error": "invalid_inputs_type"})
            continue
        for key in REQUIRED_INPUTS[tool]:
            if key not in inputs:
                errors.append({"id": sid, "error": f"missing_input:{key}"})
    if errors:
        log.warning("plan.validator validate_plan errors=%s", errors[:10])
    return errors


