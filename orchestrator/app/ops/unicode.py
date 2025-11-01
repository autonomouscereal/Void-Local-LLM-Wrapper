from __future__ import annotations

import unicodedata as _u
from typing import List, Dict, Any


def nfc(s: str) -> str:
    try:
        return _u.normalize("NFC", s)
    except Exception:
        return s


def nfc_msgs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        c = m.get("content")
        out.append({**m, "content": nfc(c) if isinstance(c, str) else c})
    return out


