from __future__ import annotations

import os
import json
from typing import List, Dict, Any


def write_music_samples(rows: List[Dict[str, Any]], outdir: str, run_id: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"music_samples_{run_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows or []:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


