from __future__ import annotations

import os
import json
from typing import Dict, Any
import logging

log = logging.getLogger(__name__)


def append_image_sample(outdir: str, row: Dict[str, Any]) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "image_samples.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Non-fatal: samples are optional diagnostics
        log.warning("tools_image.export.append_image_sample failed outdir=%s path=%s: %s", outdir, path, exc, exc_info=True)
    return path


