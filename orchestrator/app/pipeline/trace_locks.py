from __future__ import annotations

from typing import Optional

from app.state.lock import acquire_lock as _acquire_lock
from app.state.lock import release_lock as _release_lock
from app.state.checkpoints import append_event as checkpoints_append_event


def acquire_lock(state_dir: str, trace_id: str, timeout_s: int = 10) -> Optional[str]:
	"""
	Acquire a per-trace lock and emit a start checkpoint.
	Returns a token (path) if acquired, else None.
	"""
	token = None
	try:
		token = _acquire_lock(state_dir, trace_id, timeout_s=timeout_s)
		checkpoints_append_event(state_dir, trace_id, "start", {})
	except Exception:
		# Non-fatal; proceed without blocking the request
		token = None
	return token


def release_lock(state_dir: str, trace_id: str, token: Optional[str] = None) -> None:
	"""
	Release a per-trace lock and swallow any errors (non-fatal).
	"""
	try:
		_release_lock(state_dir, trace_id)
	except Exception:
		pass


