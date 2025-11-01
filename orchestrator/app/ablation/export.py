from __future__ import annotations

import json
import os
import time
from typing import Dict, Any


def write_facts_jsonl(ablated: Dict[str, Any], run_id: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"facts_{run_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for fact in (ablated.get("facts") or []):
            rec = {"run_id": run_id, "t": int(time.time())}
            try:
                rec.update(fact)
            except Exception:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


