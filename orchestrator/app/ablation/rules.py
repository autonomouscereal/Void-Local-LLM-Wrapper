from __future__ import annotations

import re
import hashlib


_SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+')
_FILLER_PAT = re.compile(r"\b(as everyone knows|clearly|it should be noted|basically|literally|just|kind of|sort of)\b", re.I)
_INSTR_PAT = re.compile(r"^\s*(do|create|write|implement|refactor|return|export|generate)\b", re.I)
_CODE_FENCE = re.compile(r"```")


def split_sentences(text: str):
    return [s for s in _SENT_SPLIT.split((text or "").strip()) if (s or "").strip()]


def is_filler(s: str) -> bool:
    return bool(_FILLER_PAT.search(s or "")) and len(s or "") < 240


def is_instruction_or_code(s: str) -> bool:
    return bool(_INSTR_PAT.search(s or "")) or bool(_CODE_FENCE.search(s or ""))


def facts_pocket(s: str):
    pat = r"(\b\d[\d\.\-x%]*\b|https?://\S+|[/~][\w\-/\.]+|\b[A-Z]{2,}\-\d+\b)"
    return sorted(set(re.findall(pat, s or "")))


def _hash_quote(q: str) -> str:
    return "sha1:" + hashlib.sha1((q or "").encode("utf-8")).hexdigest()


def attach_support(sent: str, evidence: list) -> list:
    """
    naive grounding: if sentence shares 3+ tokens with any evidence quote/title,
    attach that evidence hash.
    """
    toks = set(re.findall(r"\w+", (sent or "").lower()))
    support = []
    for e in (evidence or []):
        blob = ((e.get("quote", "") + " " + e.get("src", "")).lower())
        if len(toks & set(re.findall(r"\w+", blob))) >= 3:
            q = (e.get("quote", "") or "")[:500]
            support.append(e.get("hash") or _hash_quote(q))
    out, seen = [], set()
    for h in support:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out[:4]


def resolve_contradictions(facts: list) -> list:
    """
    simple pattern: if two facts share an id/entity token and one negates the other,
    drop the lower-confidence one.
    """
    def neg(s: str) -> bool:
        return bool(re.search(r"\b(no|not|never|without|cannot|fails?)\b", (s or "").lower()))

    kept = []
    by_key = {}
    for f in (facts or []):
        key = "|".join(sorted(t for t in (f.get("tags", []) or []) if str(t).startswith(("id:", "entity:", "path:")))) or (f.get("claim", "")[:60])
        if key not in by_key:
            by_key[key] = f
            kept.append(f)
            continue
        other = by_key[key]
        if neg(f.get("claim", "")) != neg(other.get("claim", "")):
            if float(f.get("confidence", 0.0)) > float(other.get("confidence", 0.0)):
                by_key[key] = f
                kept[-1] = f
        else:
            kept.append(f)
    return kept


