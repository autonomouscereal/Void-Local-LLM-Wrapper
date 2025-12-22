from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
import logging

log = logging.getLogger(__name__)


def append_ledger(row: Dict[str, Any], base_dir: str = "/workspace/distill") -> None:
    """
    Best-effort diagnostics ledger for QA/review loops.
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, "ledger.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}, ensure_ascii=False) + "\n")
    except Exception as ex:
        log.warning("append_ledger failed for base_dir=%s: %s", base_dir, ex, exc_info=True)
        return




