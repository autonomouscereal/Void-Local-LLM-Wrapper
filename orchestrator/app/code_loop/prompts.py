from __future__ import annotations

import json


POLICY = (
    "Policy: Do NOT use try/except unless explicitly requested or strictly required by an external API; let errors surface. "
    "Do NOT set client timeouts unless explicitly requested. Do NOT use Pydantic. Do NOT use SQLAlchemy/ORM; use asyncpg + raw SQL. "
    "Do NOT design or implement HTTP proxies, reverse proxies, pass-through/relay endpoints, or WS relays under ANY circumstances unless the user explicitly grants permission in the current prompt. Always call direct service endpoints. "
    "Do NOT introduce or use any new library (including uvicorn/fastapi/etc.) without first performing a LIVE web search to the OFFICIAL docs for the latest stable version (no memory/RAG), reviewing constraints/APIs, and obtaining explicit user permission. Include the doc URL."
)
ARCH_PREAMBLE = """You are the Architect.
{policy}
Return a single JSON object with:{{"plan":[{{"file":"path","intent":"add|edit|delete","reason":"..."}}], "notes":["constraints..."]}}""".format(policy=POLICY)
IMPL_PREAMBLE = """You are the Implementer.
{policy}
Given the plan and file excerpts, output a single JSON object:{{"patch":"<unified_diff>", "notes":["..."]}}""".format(policy=POLICY)
REV_PREAMBLE = """You are the Reviewer.
{policy}
Return a single JSON object:{{"patch":"<unified_diff possibly adjusted>", "findings":["risk, style, small fixes"]}}""".format(policy=POLICY)


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


