from __future__ import annotations

from typing import List
import os


def _roe_dir(state_dir: str) -> str:
    return os.path.join(state_dir, "roe")


def load_roe_digest(state_dir: str, trace_id: str) -> List[str]:
    """
    Load previously persisted RoE digest lines for a given trace_id.
    Returns an empty list if none exists. No try/except — let errors surface if IO is misconfigured.
    """
    if not state_dir or not trace_id:
        return []
    path = os.path.join(_roe_dir(state_dir), f"{trace_id}.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    return [ln for ln in lines if isinstance(ln, str) and ln.strip()]


def save_roe_digest(state_dir: str, trace_id: str, lines: List[str]) -> None:
    """
    Save RoE digest lines for this trace_id. Overwrites previous content.
    """
    if not state_dir or not trace_id:
        return
    os.makedirs(_roe_dir(state_dir), exist_ok=True)
    path = os.path.join(_roe_dir(state_dir), f"{trace_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for ln in (lines or []):
            f.write(str(ln).rstrip("\n") + "\n")
