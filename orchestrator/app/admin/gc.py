from __future__ import annotations

import os
import time
import shutil
from typing import Dict, Any


DEFAULTS = {
    "ttl_seconds": 30 * 24 * 3600,
    "max_bytes_total": 200 * 1024**3,
    "protect_suffixes": [".index.json", "manifest.json"],
}


def _size(p: str) -> int:
    try:
        if os.path.isdir(p):
            total = 0
            for root, _, files in os.walk(p):
                for f in files:
                    total += os.path.getsize(os.path.join(root, f))
            return total
        return os.path.getsize(p)
    except OSError:
        return 0


def _age_seconds(p: str, now: float | None = None) -> float:
    now = now or time.time()
    try:
        return now - os.path.getmtime(p)
    except OSError:
        return 0.0


def _is_protected(p: str, protect_suffixes: list[str]) -> bool:
    return any(p.endswith(sfx) for sfx in protect_suffixes)


def plan_gc(root: str = "/workspace/uploads/artifacts", ttl_seconds: int | None = None, max_bytes_total: int | None = None, protect_suffixes: list[str] | None = None) -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    if ttl_seconds is not None:
        cfg["ttl_seconds"] = int(ttl_seconds)
    if max_bytes_total is not None:
        cfg["max_bytes_total"] = int(max_bytes_total)
    if protect_suffixes:
        cfg["protect_suffixes"] = protect_suffixes
    now = time.time()
    all_files: list[Dict[str, Any]] = []
    for r, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(r, f)
            all_files.append({
                "path": p,
                "bytes": _size(p),
                "age_s": _age_seconds(p, now),
                "protected": _is_protected(p, cfg["protect_suffixes"]),
            })
    all_bytes = sum(f["bytes"] for f in all_files)
    ttl = sorted([f for f in all_files if (f["age_s"] > cfg["ttl_seconds"] and not f["protected"])], key=lambda x: x["age_s"], reverse=True)
    over = max(0, all_bytes - cfg["max_bytes_total"])
    budget: list[Dict[str, Any]] = []
    if over > 0:
        for f in sorted([x for x in all_files if not x["protected"]], key=lambda x: (x["age_s"], x["bytes"]), reverse=True):
            budget.append(f)
            over -= f["bytes"]
            if over <= 0:
                break
    return {"summary": {"bytes_total": all_bytes, "files_total": len(all_files)}, "cfg": cfg, "ttl_candidates": ttl, "budget_candidates": budget}


def run_gc(plan: Dict[str, Any], dryrun: bool = True) -> Dict[str, Any]:
    if dryrun:
        return {"deleted": [], "dryrun": True, "plan_counts": {"ttl": len(plan.get("ttl_candidates") or []), "budget": len(plan.get("budget_candidates") or [])}}
    deleted: list[Dict[str, Any]] = []
    for f in (plan.get("ttl_candidates") or []) + (plan.get("budget_candidates") or []):
        p = f.get("path")
        try:
            os.remove(p)
            deleted.append({"path": p, "bytes": f.get("bytes")})
        except IsADirectoryError:
            try:
                shutil.rmtree(p)
                deleted.append({"path": p, "bytes": 0})
            except Exception:
                pass
        except FileNotFoundError:
            pass
        except Exception:
            pass
    return {"deleted": deleted, "dryrun": False}


