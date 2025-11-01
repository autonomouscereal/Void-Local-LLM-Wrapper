from __future__ import annotations

import os
import time
from typing import Dict, Any
from .registry import new_version, write_index, _ver_dir
from .exporters import (
    export_facts,
    export_image_samples,
    export_tts_samples,
    export_music_samples,
    export_code_patches,
    export_research,
)
from ..jobs.state import set_state
from ..jobs.progress import event


def run_export_job(jid: str, args: Dict[str, Any], emit):
    name = args.get("name", "training")
    include = set(args.get("include") or ["facts", "images", "tts", "music", "research"])  # code optional
    version = new_version()
    dst = _ver_dir(name, version)
    os.makedirs(dst, exist_ok=True)
    items = []

    def add_item(label: str, info: Dict[str, Any]):
        if info:
            items.append({"label": label, **info})

    set_state(jid, "running", phase="facts")
    if "facts" in include:
        add_item("facts", export_facts(dst))
        emit(event("running", "facts", 0.2))

    set_state(jid, "running", phase="images")
    if "images" in include:
        add_item("image_samples", export_image_samples(dst))
        emit(event("running", "images", 0.4))

    set_state(jid, "running", phase="tts")
    if "tts" in include:
        add_item("tts_samples", export_tts_samples(dst))
        emit(event("running", "tts", 0.6))

    set_state(jid, "running", phase="music")
    if "music" in include:
        add_item("music_samples", export_music_samples(dst))
        emit(event("running", "music", 0.8))

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
            "facts": "facts.jsonl",
            "image_samples": "image_samples.jsonl",
            "tts_samples": "tts_samples.jsonl",
            "music_samples": "music_samples.jsonl",
            "code_patches": "code_patches.jsonl",
            "research": "research/*",
        },
    }
    write_index(name, version, index)
    set_state(jid, "done", phase="done", progress=1.0)
    emit(event("done", "done", 1.0, artifacts=[{"path": os.path.join(dst, "index.json")}]))
    return {"name": name, "version": version}


