from __future__ import annotations

import re
import hashlib


def make_system_hint(target_out: int, state_hash: str) -> str:
    return (
        "You may be interrupted at any time. If incomplete, END with "
        f"<CONT state=\"{state_hash}\" reason=\"continue\"/> and repeat the last sentence fragment before the tag. "
        f"If solved, END with <HALT state=\"{state_hash}\" outcome=\"success\"/>. "
        f"Budget your output to ~{target_out} tokens for this step."
    )


HALT_RE = re.compile(r"<HALT state=\"([a-f0-9]{16})\".*?/>", re.I | re.S)
CONT_RE = re.compile(r"<CONT state=\"([a-f0-9]{16})\".*?/>", re.I | re.S)


def detect_cont_or_halt(text: str):
    m = HALT_RE.search(text or "")
    if m:
        return ("HALT", m.group(1))
    m = CONT_RE.search(text or "")
    if m:
        return ("CONT", m.group(1))
    return ("CONT", None)  # conservative fallback


def state_hash_from(state: dict) -> str:
    material = repr(
        (
            state.get("goals"),
            state.get("entities"),
            state.get("artifacts_meta"),
            state.get("turn"),
            state.get("planner_state"),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


