from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for x in value:
            if x is None:
                continue
            try:
                out.append(str(x))
            except Exception:  # pragma: no cover
                out.append(repr(x))
        return out
    # tolerate singletons (common JSON mistake)
    try:
        return [str(value)]
    except Exception:  # pragma: no cover
        return [repr(value)]


def need_reframe(history_hashes: list[str], window_steps: int, threshold: int = 3) -> bool:
    """If the same state hash repeats N times, we’re spinning."""
    try:
        n = int(threshold)
    except Exception:  # pragma: no cover
        n = 3
    if n <= 0:
        n = 3
    if window_steps < n:
        return False
    if not isinstance(history_hashes, list) or not history_hashes:
        return False
    tail = [str(x) for x in history_hashes[-n:]]
    spinning = len(set(tail)) == 1
    if spinning:
        log.warning("icw.reframe spinning detected window_steps=%s threshold=%s state=%s", window_steps, n, tail[-1][:8] if tail[-1] else "")
    return spinning


def build_reframe_prompt(goal: str, constraints: list[str], seen_blockers: list[str]) -> str:
    """One-shot reframe: restate goal, hard constraints, and the next atomic action."""
    g = str(goal or "")
    cs = _as_str_list(constraints)[:8]
    bs = _as_str_list(seen_blockers)[:5]
    lines = [
        "Reframe:",
        f"- Goal: {g}",
        f"- Constraints: {', '.join(cs)}",
        "- Observed blockers: " + (", ".join(bs) if bs else "none"),
        "- Next: produce 1 atomic step toward completion and stop with <CONT/>.",
    ]
    return "\n".join(lines)


