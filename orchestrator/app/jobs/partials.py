from __future__ import annotations

from typing import Dict, Any
from .state import set_state, get_job
from .progress import event
from ..artifacts.manifest import write_manifest_atomic


def emit_partial(jid: str, phase: str, manifest: Dict[str, Any], emit) -> None:
    try:
        j = set_state(jid, "failed" if phase == "error" else "cancelled", phase=phase, progress=0.0)
    except Exception:
        j = None
    try:
        payload = event((j.state if j else "failed"), phase, 0.0, artifacts=manifest.get("items", []), notes=["partial-result"])  # type: ignore
        emit(payload)
    except Exception:
        pass
    try:
        root = manifest.get("root") or "."
        write_manifest_atomic(root, manifest)
    except Exception:
        pass


