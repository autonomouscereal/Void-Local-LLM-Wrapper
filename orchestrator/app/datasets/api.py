from __future__ import annotations

import time
from typing import Dict, Any
from .registry import list_datasets, list_versions, _idx_path
from .jobs import run_export_job
from ..jobs.state import create_job, set_state, get_job
from ..jobs.progress import event


def post_datasets_start(body: Dict[str, Any], emit):
    jid = body.get("id") or f"ds-{int(time.time())}"
    j = create_job(jid=jid, tool="datasets.export", args=body)
    # Run synchronously for simplicity; emit a couple events
    set_state(j.id, "running", phase="start", progress=0.0)
    emit(event("running", "start", 0.0))
    result = run_export_job(j.id, body, emit)
    return {"id": j.id, **(result or {})}


def get_datasets_list():
    return {"datasets": list_datasets()}


def get_datasets_versions(name: str):
    return {"name": name, "versions": list_versions(name)}


def get_datasets_index(name: str, version: str):
    p = _idx_path(name, version)
    with open(p, "rb") as f:
        return f.read(), 200, {"Content-Type": "application/json"}


