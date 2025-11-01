from __future__ import annotations

import os
from .fsview import list_files, read_text


def recent_and_neighbors(root: str, n: int = 50, neighbor_radius: int = 2):
    files = list_files(root)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    recent = files[:n]
    neighbors = set()
    for p in recent:
        d = os.path.dirname(p)
        for f in files:
            if f.startswith(d) and len(neighbors) < n * neighbor_radius:
                neighbors.add(f)
    working = list(dict.fromkeys(recent + list(neighbors)))
    return working


def symbol_index(root: str, files: list[str]):
    idx = {}
    for p in files:
        text = read_text(p, max_bytes=200_000)
        syms = []
        for line in text.splitlines()[:400]:
            s = line.strip()
            if s.startswith(("def ", "class ", "interface ", "export function ", "const ", "type ")):
                syms.append(s[:160])
        idx[p] = {"size": len(text), "symbols": syms}
    return idx


def build_index(root: str):
    working = recent_and_neighbors(root)
    sym = symbol_index(root, working)
    return {"root": os.path.abspath(root), "working": working, "symbols": sym}


