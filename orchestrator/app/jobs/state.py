from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional
import time


JobState = Literal["queued", "running", "cancelling", "cancelled", "done", "failed"]


@dataclass
class Job:
    id: str
    tool: str
    args: Dict[str, Any]
    state: JobState = "queued"
    phase: str = "init"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_flag: bool = False
    error: Optional[str] = None


_jobs: Dict[str, Job] = {}


def create_job(jid: str, tool: str, args: dict) -> Job:
    j = Job(id=jid, tool=tool, args=args)
    _jobs[jid] = j
    return j


def get_job(jid: str) -> Optional[Job]:
    return _jobs.get(jid)


def set_state(jid: str, state: JobState, phase: Optional[str] = None, progress: Optional[float] = None, error: Optional[str] = None) -> Optional[Job]:
    j = _jobs.get(jid)
    if not j:
        return None
    if state:
        j.state = state
    if phase is not None:
        j.phase = phase
    if progress is not None:
        j.progress = float(progress)
    if error is not None:
        j.error = error
    j.updated_at = time.time()
    return j


def request_cancel(jid: str) -> Optional[Job]:
    j = _jobs.get(jid)
    if not j:
        return None
    j.cancel_flag = True
    j.state = "cancelling"
    j.updated_at = time.time()
    return j


def clear_job(jid: str) -> None:
    _jobs.pop(jid, None)


