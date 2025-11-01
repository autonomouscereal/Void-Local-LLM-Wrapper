from __future__ import annotations

def distill_line(row: dict) -> str:
    """
    Make a compact one-liner summary used when older shards are included in ICW.
    Keep IDs, URLs, numbers if present.
    """
    bits = []
    try:
        for k in ("id", "title", "url", "file", "amount", "date", "hash"):
            v = row.get(k)
            if v:
                s = str(v)
                bits.append(f"{k}:{s[:80]}")
        return " | ".join(bits) or (str(row)[:160])
    except Exception:
        return (str(row)[:160])

from __future__ import annotations


def distill_line(row: dict) -> str:
    """
    Make a compact one-liner summary used when older shards are included in ICW.
    Keep IDs, URLs, numbers if present.
    """
    bits = []
    for k in ("id", "title", "url", "file", "amount", "date", "hash"):
        try:
            v = row.get(k)
        except Exception:
            v = None
        if v:
            sval = str(v)
            bits.append(f"{k}:{sval[:80]}")
    return " | ".join(bits) or (str(row)[:160])


