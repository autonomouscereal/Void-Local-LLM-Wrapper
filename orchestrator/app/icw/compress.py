from __future__ import annotations

import logging
import re

log = logging.getLogger("orchestrator.icw.compress")


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("icw.compress bytes decode failed: %s", exc, exc_info=True)
            return repr(value)
    try:
        return str(value)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.compress str() failed for type=%s: %s", type(value), exc, exc_info=True)
        return repr(value)


def cleanup_lossless(text: str) -> str:
    t0 = _coerce_text(text)
    if not t0:
        return ""
    # strip duplicate headers, excessive whitespace, signatures, markdown fences (preserve code blocks but trim edges)
    try:
        t = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip(), t0)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.compress cleanup_lossless fence strip failed: %s", exc, exc_info=True)
        t = t0
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def compact_numbers_tables(text: str) -> str:
    # placeholder: could transform simple tables; keep as-is for now
    return _coerce_text(text)


def facts_pocket(text: str) -> list[str]:
    pat = r"(\b\d[\d\.\-x%]*\b|https?://\S+|[/~][\w\-/\.]+|\b[A-Z]{2,}\-\d+\b)"
    try:
        return sorted(set(re.findall(pat, _coerce_text(text) or "")))
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.compress facts_pocket failed: %s", exc, exc_info=True)
        return []


def attach_pocket(summary: str, pocket: list[str]) -> str:
    s = _coerce_text(summary)
    p = pocket if isinstance(pocket, list) else []
    try:
        return s + ("\n[Facts] " + " | ".join([str(x) for x in p[:64]]) if p else "")
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.compress attach_pocket failed: %s", exc, exc_info=True)
        return s


def summarize_block_extractive(text: str, target_ratio: float = 0.5) -> str:
    t = _coerce_text(text)
    try:
        sents = re.split(r"(?<=[\.\?\!])\s+", t or "")
        # Defensive: target_ratio can be str-ish.
        try:
            tr = float(target_ratio)
        except Exception:
            tr = 0.5
        keep = max(1, int(len(sents) * max(0.0, min(1.0, tr))))
        return " ".join(sents[:keep]).strip()
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.compress summarize_block_extractive failed: %s", exc, exc_info=True)
        return (t or "").strip()


