from __future__ import annotations

import json
import os
import time
import logging
from typing import Dict, Any

from ..datasets.stream import append_fact


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
            # Also append into the canonical dataset stream (single source of truth).
            try:
                append_fact(rec)
            except Exception:
                # Never fail ablation export due to dataset streaming.
                # But do log: silent failures break distillation.
                # Keep it lightweight: ablation export can run on large batches.
                logging.getLogger(__name__).warning("ablation.export.append_fact_failed run_id=%s", run_id, exc_info=True)
    return path


