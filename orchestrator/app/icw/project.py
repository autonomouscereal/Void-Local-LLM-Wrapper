from __future__ import annotations

import json
import os
from typing import Any, Dict


FILM2_DATA_DIR = os.getenv("FILM2_DATA", "/srv/film2")


def project_dir(job_id: str) -> str:
    """
    Root directory for a film/ICW job, scoped under FILM2_DATA_DIR.
    """
    base = os.path.join(FILM2_DATA_DIR, "jobs", job_id)
    os.makedirs(base, exist_ok=True)
    return base


def capsules_dir(job_id: str) -> str:
    """
    Directory for OmniCapsule window JSON files for a given job.
    """
    d = os.path.join(project_dir(job_id), "capsules")
    os.makedirs(os.path.join(d, "windows"), exist_ok=True)
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_safe(path: str, obj: Dict[str, Any]) -> None:
    """
    Minimal JSON writer helper used by ICW/film project endpoints.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


