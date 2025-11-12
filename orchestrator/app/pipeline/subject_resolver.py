from __future__ import annotations

from typing import Dict
import re


def resolve_subject_canon(text: str) -> Dict[str, object]:
	"""
	Lightweight subject resolver to populate subject_canon.
	- Extract a likely proper noun phrase (first 2-4 capitalized words)
	- Provide basic identity tokens heuristic from the text
	"""
	if not isinstance(text, str) or not text.strip():
		return {}
	# Proper noun heuristic: sequences of capitalized words
	caps = re.findall(r"(?:[A-Z][a-zA-Z]+(?:\\s+|$)){1,4}", text.strip())
	literal = (caps[0].strip() if caps else "")[:64]
	# Identity tokens: nouns/adjectives-like tokens from text (very light heuristic)
	tokens = []
	for w in re.findall(r"[A-Za-z]{3,}", text.lower()):
		if w in ("the", "and", "for", "with", "from", "into", "over", "under", "this", "that"):
			continue
		tokens.append(w)
	tokens = list(dict.fromkeys(tokens))[:24]
	out: Dict[str, object] = {}
	if literal:
		out["literal"] = literal
	if tokens:
		out["tokens"] = tokens
	return out


