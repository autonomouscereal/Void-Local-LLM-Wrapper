from __future__ import annotations

from typing import Any, Dict, List
import os
import json
from datetime import datetime, timezone


def _dir(state_dir: str) -> str:
	return os.path.join(state_dir, "tool_evidence")


def _path(state_dir: str, trace_id: str) -> str:
	return os.path.join(_dir(state_dir), f"{trace_id}.ndjson")


def append_tool_evidence(state_dir: str, trace_id: str, entry: Dict[str, Any]) -> None:
	"""
	Append one evidence entry to the per-trace NDJSON file.
	Entry schema (recommended):
	- name: str (tool name)
	- ok: bool
	- label: str ('success' or 'failure')
	- raw: dict (RAW JSON per memo)
	"""
	if not state_dir or not trace_id:
		return
	os.makedirs(_dir(state_dir), exist_ok=True)
	rec = dict(entry or {})
	# Stamp ts if missing
	if not isinstance(rec.get("raw"), dict):
		rec["raw"] = {}
	if not rec.get("raw", {}).get("ts"):
		rec["raw"]["ts"] = datetime.now(timezone.utc).isoformat()
	with open(_path(state_dir, trace_id), "a", encoding="utf-8") as f:
		f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_recent_tool_evidence(state_dir: str, trace_id: str, limit: int = 4) -> List[Dict[str, Any]]:
	"""
	Load up to 'limit' most recent evidence entries for a trace.
	Returns list of {name, ok, label, raw}.
	"""
	if not state_dir or not trace_id:
		return []
	p = _path(state_dir, trace_id)
	if not os.path.exists(p):
		return []
	entries: List[Dict[str, Any]] = []
	with open(p, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if isinstance(obj, dict):
				entries.append(obj)
	# Return last 'limit' entries in reverse chronological order
	return entries[-limit:]


