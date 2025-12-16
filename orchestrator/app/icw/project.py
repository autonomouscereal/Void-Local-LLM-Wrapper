from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

log = logging.getLogger("orchestrator.icw.project")

FILM2_DATA_DIR = os.getenv("FILM2_DATA", "/srv/film2")


def _sanitize_job_id(job_id: str) -> str:
    jid = str(job_id or "").strip()
    if not jid:
        raise ValueError("job_id is empty")
    # Prevent path traversal / separator injection.
    if any(sep in jid for sep in ("/", "\\", os.sep)):
        raise ValueError(f"job_id contains a path separator: {jid!r}")
    if ".." in jid:
        raise ValueError(f"job_id contains '..': {jid!r}")
    # Keep filesystem-friendly IDs; allow common characters used in ids.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:.")
    if any(ch not in allowed for ch in jid):
        raise ValueError(f"job_id contains unsupported characters: {jid!r}")
    return jid


def project_dir(job_id: str) -> str:
    """
    Root directory for a film/ICW job, scoped under FILM2_DATA_DIR.
    """
    jid = _sanitize_job_id(job_id)
    base = os.path.join(FILM2_DATA_DIR, "jobs", jid)
    try:
        os.makedirs(base, exist_ok=True)
    except Exception as exc:
        log.error("icw.project project_dir makedirs failed base=%r: %s", base, exc, exc_info=True)
        raise
    return base


def capsules_dir(job_id: str) -> str:
    """
    Directory for OmniCapsule window JSON files for a given job.
    """
    d = os.path.join(project_dir(job_id), "capsules")
    try:
        os.makedirs(os.path.join(d, "windows"), exist_ok=True)
    except Exception as exc:
        log.error("icw.project capsules_dir makedirs failed d=%r: %s", d, exc, exc_info=True)
        raise
    return d


def project_capsule_path(job_id: str) -> str:
    """
    Path to the per-project OmniCapsule JSON file.
    """
    return os.path.join(capsules_dir(job_id), "OmniCapsule.json")


def windows_dir(job_id: str) -> str:
    """
    Directory where per-window OmniCapsule snapshots are stored.
    """
    return os.path.join(capsules_dir(job_id), "windows")


def read_json_safe(path: str) -> Dict[str, Any]:
    """
    Minimal JSON reader helper used by ICW/film project endpoints.
    """
    p = str(path or "")
    if not p:
        raise ValueError("read_json_safe path is empty")
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        log.warning("icw.project read_json_safe not found path=%r", p)
        raise
    except json.JSONDecodeError as exc:
        log.error("icw.project read_json_safe JSON decode failed path=%r: %s", p, exc, exc_info=True)
        raise
    except Exception as exc:
        log.error("icw.project read_json_safe failed path=%r: %s", p, exc, exc_info=True)
        raise
    if not isinstance(obj, dict):
        log.warning("icw.project read_json_safe non-dict root path=%r type=%s", p, type(obj).__name__)
        return {"_value": obj}
    return obj


def write_json_safe(path: str, obj: Dict[str, Any]) -> None:
    """
    Minimal JSON writer helper used by ICW/film project endpoints.
    """
    p = str(path or "")
    if not p:
        raise ValueError("write_json_safe path is empty")
    try:
        d = os.path.dirname(p) or "."
        os.makedirs(d, exist_ok=True)
    except Exception as exc:
        log.error("icw.project write_json_safe makedirs failed dir=%r: %s", os.path.dirname(p), exc, exc_info=True)
        raise
    tmp = p + ".tmp"
    payload: Dict[str, Any]
    if isinstance(obj, dict):
        payload = obj
    else:
        log.warning("icw.project write_json_safe non-dict obj type=%s path=%r", type(obj).__name__, p)
        payload = {"_value": obj}
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception as exc:
                # fsync not supported on some environments; best-effort only
                log.debug("icw.project write_json_safe: fsync failed (non-fatal) path=%r: %s", p, exc, exc_info=True)
        os.replace(tmp, p)
    except Exception as exc:
        log.error("icw.project write_json_safe failed path=%r tmp=%r: %s", p, tmp, exc, exc_info=True)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception as exc2:
            log.debug("icw.project write_json_safe cleanup failed tmp=%r: %s", tmp, exc2, exc_info=True)
        raise


