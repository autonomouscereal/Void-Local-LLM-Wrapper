from __future__ import annotations

import os
import json
from typing import List, Dict, Any
import logging

log = logging.getLogger(__name__)


def write_music_samples(rows: List[Dict[str, Any]], outdir: str, run_id: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"music_samples_{run_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows or []:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def append_music_sample(outdir: str, row: Dict[str, Any]) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "music_samples.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Non-fatal: samples are optional diagnostics
        log.warning("tools_music.export.append_music_sample failed outdir=%s path=%s: %s", outdir, path, exc, exc_info=True)
    return path


