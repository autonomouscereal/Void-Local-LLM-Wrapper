from __future__ import annotations

import re
import hashlib


# Accept vendor variations and stray whitespace/newlines
HALT_RE = re.compile(r"<\s*HALT\b[^>]*state\s*=\s*\"([a-f0-9]{8,32})\"[^>]*\/\s*>", re.I | re.S)
CONT_RE = re.compile(r"<\s*CONT\b[^>]*state\s*=\s*\"([a-f0-9]{8,32})\"[^>]*\/\s*>", re.I | re.S)


def make_system_hint(target_out: int, state_hash: str) -> str:
    return (
        "You may be interrupted at any time. If incomplete, END with "
        f"<CONT state=\"{state_hash}\" reason=\"continue\"/> and repeat the last sentence fragment before the tag. "
        f"If solved, END with <HALT state=\"{state_hash}\" outcome=\"success\"/>. "
        f"Budget your output to ~{target_out} tokens for this step."
    )


def detect_cont_or_halt(text: str):
    # Prefer HALT if both appear (some models echo examples)
    m = HALT_RE.search(text or "")
    if m:
        return ("HALT", m.group(1))
    m = CONT_RE.search(text or "")
    if m:
        return ("CONT", m.group(1))
    # Fallback: treat as CONT to force multi-step safety
    return ("CONT", None)


def synthesize_cont(text: str, state_hash: str, last_fragment: str) -> str:
    """
    If model forgot tags, synthesize a CONT and repeat last fragment tail
    so stitching stays seamless.
    """
    frag = (last_fragment or "").rstrip()
    return (text.rstrip() + "\n" + frag[-120:] + f"\n<CONT state=\"{state_hash}\" reason=\"fallback\"/>")


SENT_EDGE = re.compile(r'(?<=[\.\?\!])\s+')


def last_sentence_fragment(s: str) -> str:
    parts = SENT_EDGE.split((s or "").strip())
    return parts[-1] if parts else (s or "")


def strip_tags(text: str) -> str:
    return re.sub(r"</?\s*(CONT|HALT)\b[^>]*>", "", text or "", flags=re.I)


def state_hash_from(state: dict) -> str:
    material = repr(
        (
            state.get("goals"),
            state.get("entities"),
            tuple(state.get("artifacts_meta", [])) if isinstance(state.get("artifacts_meta"), list) else state.get("artifacts_meta"),
            state.get("turn"),
            state.get("planner_state"),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


