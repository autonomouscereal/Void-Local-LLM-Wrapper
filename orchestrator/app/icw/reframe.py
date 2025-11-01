from __future__ import annotations


def need_reframe(history_hashes: list[str], window_steps: int, threshold: int = 3) -> bool:
    """If the same state hash repeats N times, we’re spinning."""
    if window_steps < threshold:
        return False
    tail = history_hashes[-threshold:]
    return len(set(tail)) == 1


def build_reframe_prompt(goal: str, constraints: list[str], seen_blockers: list[str]) -> str:
    """One-shot reframe: restate goal, hard constraints, and the next atomic action."""
    lines = [
        "Reframe:",
        f"- Goal: {goal}",
        f"- Constraints: {', '.join(constraints[:8])}",
        "- Observed blockers: " + (", ".join(seen_blockers[:5]) if seen_blockers else "none"),
        "- Next: produce 1 atomic step toward completion and stop with <CONT/>.",
    ]
    return "\n".join(lines)


