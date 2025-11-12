from __future__ import annotations

import os
import json
import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.trace_utils import emit_trace as _emit_trace


router = APIRouter()
STATE_DIR = os.environ.get("STATE_DIR_LOCAL", "/workspace/state")


def ok_envelope(result, rid: str = "logs.tools.append") -> JSONResponse:
	return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


@router.post("/logs/tools.append")
async def tools_append(req: Request):
	body = await req.json()
	trace_id = str(body.get("trace_id") or body.get("tid") or "unknown")
	entry = {
		"t": int(time.time() * 1000),
		"trace_id": trace_id,
		"event": body.get("event") or "tool",
		"step_id": body.get("step_id"),
		"tool": body.get("tool"),
		"payload": body.get("payload"),
	}
	_emit_trace(STATE_DIR, trace_id, str(entry.get("event") or "tool"), entry)
	return ok_envelope({"appended": True})


