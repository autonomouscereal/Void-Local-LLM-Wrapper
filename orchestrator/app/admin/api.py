from __future__ import annotations

import time
from typing import Dict, Any
from .gc import plan_gc, run_gc
from ..jobs.state import _jobs
from ..state.checkpoints import read_all


def get_jobs_list():
    out = []
    for jid, j in list(_jobs.items()):
        out.append({"id": jid, "tool": j.tool, "state": j.state, "phase": j.phase, "progress": j.progress, "updated_at": j.updated_at})
    return {"jobs": out}


def get_jobs_replay(cid: str):
    events = read_all("/workspace/uploads/state/traces", cid)
    return {"cid": cid, "events": events[-500:]}


def post_artifacts_gc(body: Dict[str, Any]):
    ttl = body.get("ttl_seconds")
    maxb = body.get("max_bytes_total")
    root = body.get("root") or "/workspace/uploads/artifacts"
    plan = plan_gc(root, ttl_seconds=ttl, max_bytes_total=maxb)
    res = run_gc(plan, dryrun=bool(body.get("dryrun", True)))
    return {"plan": {"summary": plan["summary"], "ttl": len(plan.get("ttl_candidates") or []), "budget": len(plan.get("budget_candidates") or [])}, "result": res}


