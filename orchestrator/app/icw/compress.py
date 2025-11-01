from __future__ import annotations

import re


def cleanup_lossless(text: str) -> str:
    if not text:
        return text
    # strip duplicate headers, excessive whitespace, signatures, markdown fences (preserve code blocks but trim edges)
    t = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip(), text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def compact_numbers_tables(text: str) -> str:
    # placeholder: could transform simple tables; keep as-is for now
    return text


def facts_pocket(text: str) -> list[str]:
    pat = r"(\b\d[\d\.\-x%]*\b|https?://\S+|[/~][\w\-/\.]+|\b[A-Z]{2,}\-\d+\b)"
    return sorted(set(re.findall(pat, text or "")))


def attach_pocket(summary: str, pocket: list[str]) -> str:
    return summary + ("\n[Facts] " + " | ".join(pocket[:64]) if pocket else "")


def summarize_block_extractive(text: str, target_ratio: float = 0.5) -> str:
    sents = re.split(r'(?<=[\.\?\!])\s+', text or "")
    keep = max(1, int(len(sents) * max(0.0, min(1.0, target_ratio))))
    return " ".join(sents[:keep])


