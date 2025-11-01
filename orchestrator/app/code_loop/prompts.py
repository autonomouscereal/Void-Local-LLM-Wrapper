from __future__ import annotations

import json


ARCH_PREAMBLE = """You are the Architect. Return a single JSON object with:{"plan":[{"file":"path","intent":"add|edit|delete","reason":"..."}], "notes":["constraints..."]}"""
IMPL_PREAMBLE = """You are the Implementer. Given the plan and file excerpts, output a single JSON object:{"patch":"<unified_diff>", "notes":["..."]}"""
REV_PREAMBLE = """You are the Reviewer. Return a single JSON object:{"patch":"<unified_diff possibly adjusted>", "findings":["risk, style, small fixes"]}"""


def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def build_architect_input(task: str, idx: dict) -> str:
    head = {"task": task, "files": idx.get("working", [])[:30], "symbols": {k: v.get("symbols") for k, v in list(idx.get("symbols", {}).items())[:20]}}
    return ARCH_PREAMBLE + "\n" + json_dump(head)


def build_implementer_input(plan: dict, excerpts: list[dict]) -> str:
    body = {"plan": plan.get("plan", []), "excerpts": excerpts}
    return IMPL_PREAMBLE + "\n" + json_dump(body)


def build_reviewer_input(task: str, plan: dict, patch: str) -> str:
    body = {"task": task, "plan": plan.get("plan", []), "patch": patch}
    return REV_PREAMBLE + "\n" + json_dump(body)


