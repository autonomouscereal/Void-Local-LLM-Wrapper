from __future__ import annotations

import re
from typing import Dict, Any
from .rules import split_sentences, is_filler, is_instruction_or_code, facts_pocket, attach_support, resolve_contradictions


def ablate(envelope: Dict[str, Any], scope_hint: str = "chat") -> Dict[str, Any]:
    """
    Deterministic ablation for a single envelope:
    - extract candidate sentences from message.content
    - keep grounded facts, compact instructions/code results
    - attach numbers/IDs/paths/URLs via facts-pocket
    - drop dup/filler/unverifiable
    - resolve simple contradictions
    """
    text = ((envelope.get("message", {}) or {}).get("content", "") or "")
    evidence = (envelope.get("evidence", []) or [])
    sents = split_sentences(text)
    seen = set()
    facts, drops = [], []
    for s in sents:
        s_norm = re.sub(r"\s+", " ", (s or "").strip())
        if not s_norm:
            continue
        if s_norm in seen:
            drops.append({"text": s_norm, "reason": "duplicate"})
            continue
        seen.add(s_norm)
        if is_filler(s_norm):
            drops.append({"text": s_norm, "reason": "filler"})
            continue
        grounded_support = attach_support(s_norm, evidence)
        if not grounded_support and not is_instruction_or_code(s_norm):
            drops.append({"text": s_norm, "reason": "unverifiable"})
            continue
        pocket = facts_pocket(s_norm)
        tags = []
        for p in pocket[:12]:
            if p.startswith("http"):
                tags.append(f"url:{p}")
            elif "/" in p:
                tags.append(f"path:{p}")
            elif re.match(r"^[A-Z]{2,}\-\d+\b", p):
                tags.append(f"id:{p}")
            elif re.match(r"^\d", p):
                tags.append(f"num:{p}")
        facts.append({
            "claim": s_norm,
            "support": grounded_support,
            "kind": "instruction" if is_instruction_or_code(s_norm) else "fact",
            "scope": scope_hint,
            "tags": tags[:16],
            "confidence": 0.8 if grounded_support else 0.6,
        })
    facts = resolve_contradictions(facts)
    return {
        "facts": facts,
        "drops": drops,
        "notes": [f"ablation: kept={len(facts)}, dropped={len(drops)}"],
    }


