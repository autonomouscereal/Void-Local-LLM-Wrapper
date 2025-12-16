from __future__ import annotations

from typing import Dict, Any
import logging
from .state import set_state, get_job
from .progress import event
from ..artifacts.manifest import write_manifest_atomic

log = logging.getLogger(__name__)


def emit_partial(jid: str, phase: str, manifest: Dict[str, Any], emit) -> None:
    try:
        j = set_state(jid, "failed" if phase == "error" else "cancelled", phase=phase, progress=0.0)
    except Exception as exc:
        log.warning("jobs.partials.emit_partial: set_state failed jid=%s phase=%s: %s", jid, phase, exc, exc_info=True)
        j = None
    try:
        payload = event((j.state if j else "failed"), phase, 0.0, artifacts=manifest.get("items", []), notes=["partial-result"])  # type: ignore
        emit(payload)
    except Exception as exc:
        log.warning("jobs.partials.emit_partial: emit failed jid=%s phase=%s: %s", jid, phase, exc, exc_info=True)
    try:
        root = manifest.get("root") or "."
        write_manifest_atomic(root, manifest)
    except Exception as exc:
        log.warning("jobs.partials.emit_partial: write_manifest_atomic failed jid=%s root=%s: %s", jid, root, exc, exc_info=True)


