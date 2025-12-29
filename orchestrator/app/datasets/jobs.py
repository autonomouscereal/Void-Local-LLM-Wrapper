from __future__ import annotations

import os
import time
from typing import Dict, Any
from .registry import new_version, write_index, _ver_dir
from .exporters import export_dataset_stream, export_research
from ..jobs.state import set_state
from ..jobs.progress import event


def run_export_job(jid: str, args: Dict[str, Any], emit):
    name = args.get("name", "training")
    include = set(args.get("include") or ["dataset", "research"])
    version = new_version()
    dst = _ver_dir(name, version)
    os.makedirs(dst, exist_ok=True)
    items = []

    def add_item(label: str, info: Dict[str, Any]):
        if info:
            items.append({"label": label, **info})

    set_state(jid, "running", phase="dataset")
    if "dataset" in include:
        add_item("dataset_stream", export_dataset_stream(dst))
        emit(event("running", "dataset", 0.6))

    set_state(jid, "running", phase="research")
    if "research" in include:
        add_item("research", export_research(dst))
        emit(event("running", "research", 0.9))

    index = {
        "dataset": name,
        "version": version,
        "created_at": int(time.time()),
        "items": items,
        "schema": {
            "dataset_stream": "dataset_stream/*",
            "research": "research/*",
        },
    }
    write_index(name, version, index)
    set_state(jid, "done", phase="done", progress=1.0)
    emit(event("done", "done", 1.0, artifacts=[{"artifact_id": "index.json", "kind": "json", "path": os.path.join(dst, "index.json")}]))
    return {"name": name, "version": version}


