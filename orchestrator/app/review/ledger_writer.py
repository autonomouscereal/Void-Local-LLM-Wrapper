from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def append_ledger(row: Dict[str, Any], base_dir: str = "/workspace/distill") -> None:
    try:
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, "ledger.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}, ensure_ascii=False) + "\n")
    except Exception:
        pass


