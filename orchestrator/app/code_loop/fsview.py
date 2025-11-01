from __future__ import annotations

import os
import glob


def list_files(root: str, exts=(".py", ".ts", ".tsx", ".js", ".json", ".md", ".yaml", ".yml", ".toml")):
    root = os.path.abspath(root)
    out = []
    for ext in exts:
        out += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
    return [f for f in out if os.path.isfile(f)]


def read_text(path: str, max_bytes: int = 400_000) -> str:
    with open(path, "rb") as f:
        b = f.read(max_bytes)
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("utf-8", "ignore")


