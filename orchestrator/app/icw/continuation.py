from __future__ import annotations

import logging
import re
import hashlib

log = logging.getLogger(__name__)


# Accept vendor variations and stray whitespace/newlines
HALT_RE = re.compile(r"<\s*HALT\b[^>]*state\s*=\s*\"([a-f0-9]{8,32})\"[^>]*\/\s*>", re.I | re.S)
CONT_RE = re.compile(r"<\s*CONT\b[^>]*state\s*=\s*\"([a-f0-9]{8,32})\"[^>]*\/\s*>", re.I | re.S)


def make_system_hint(target_out: int, state_hash: str) -> str:
    try:
        to = int(target_out)
    except Exception as exc:  # pragma: no cover
        log.warning("icw.continuation make_system_hint bad target_out=%r: %s", target_out, exc, exc_info=True)
        to = 0
    sh = str(state_hash or "")
    return (
        "You may be interrupted at any time. If incomplete, END with "
        f"<CONT state=\"{sh}\" reason=\"continue\"/> and repeat the last sentence fragment before the tag. "
        f"If solved, END with <HALT state=\"{sh}\" outcome=\"success\"/>. "
        f"Budget your output to ~{to} tokens for this step."
    )


def detect_cont_or_halt(text: str):
    # Prefer HALT if both appear (some models echo examples)
    t = text if isinstance(text, str) else str(text or "")
    m = HALT_RE.search(t or "")
    if m:
        return ("HALT", m.group(1))
    m = CONT_RE.search(t or "")
    if m:
        return ("CONT", m.group(1))
    # Fallback: treat as CONT to force multi-step safety
    return ("CONT", None)


def synthesize_cont(text: str, state_hash: str, last_fragment: str) -> str:
    """
    If model forgot tags, synthesize a CONT and repeat last fragment tail
    so stitching stays seamless.
    """
    t = text if isinstance(text, str) else str(text or "")
    frag = (last_fragment or "").rstrip()
    sh = str(state_hash or "")
    return (t.rstrip() + "\n" + frag[-120:] + f"\n<CONT state=\"{sh}\" reason=\"fallback\"/>")


SENT_EDGE = re.compile(r'(?<=[\.\?\!])\s+')


def last_sentence_fragment(s: str) -> str:
    try:
        parts = SENT_EDGE.split((s or "").strip())
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.continuation last_sentence_fragment failed: %s", exc, exc_info=True)
        return str(s or "")
    return parts[-1] if parts else (s or "")


def strip_tags(text: str) -> str:
    try:
        t = text if isinstance(text, str) else str(text or "")
        return re.sub(r"</?\s*(CONT|HALT)\b[^>]*>", "", t or "", flags=re.I)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.continuation strip_tags failed: %s", exc, exc_info=True)
        return str(text or "")


def state_hash_from(state: dict) -> str:
    # Never raise from continuation helpers; invalid state should degrade safely.
    if not isinstance(state, dict):
        log.warning("icw.continuation state_hash_from expected dict, got %s", type(state).__name__)
        try:
            material = repr(state)
        except Exception:
            material = f"<unrepr:{type(state).__name__}>"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
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


