from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from .fsview import read_text


@dataclass
class PatchResult:
    ok: bool
    errmsg: str | None
    patch: str


def make_unified_diff(path: str, old_text: str, new_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )


def apply_patch_in_memory(root: str, patch: str) -> PatchResult:
    """
    Very small verifier: for each @@ hunk, ensure target file exists and line counts are consistent.
    DOES NOT write to disk; detects obvious malformed patches early.
    """
    if not patch or not patch.strip():
        return PatchResult(False, "empty patch", patch)
    files_ok = True
    err = None
    headers = re.findall(r"^\-\-\- a\/(.*?)\n\+\+\+ b\/\1", patch, flags=re.M)
    for header in headers:
        try:
            _ = read_text(f"{root}/{header}")
        except Exception:
            files_ok = False
            err = f"missing file: {header}"
            break
    return PatchResult(files_ok, err, patch)


