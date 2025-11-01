from __future__ import annotations

import os
import json
from typing import Dict, Any


def append_image_sample(outdir: str, row: Dict[str, Any]) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "image_samples.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return path


