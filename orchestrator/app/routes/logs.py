from __future__ import annotations

import os
import json
import time
from fastapi import APIRouter, Request
from app.routes.toolrun import ToolEnvelope  # canonical envelope
from app.trace_utils import emit_trace as _emit_trace
from app.json_parser import JSONParser


router = APIRouter()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
STATE_DIR = os.path.join(UPLOAD_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)


@router.post("/logs/tools.append")
async def tools_append(req: Request):
    raw = await req.body()
    parser = JSONParser()
    schema = {
        "trace_id": str,
        "event": str,
        "step_id": str,
        "tool": str,
        "payload": dict,
    }
    body = parser.parse(
        raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw or ""),
        schema,
    )
    trace_id = body.get("trace_id") if isinstance(body, dict) else None
    if not isinstance(trace_id, str) or not trace_id:
        return ToolEnvelope.failure("missing_trace_id", "trace_id is required", status=422, request_id="logs.tools.append")
    entry = {
        "t": int(time.time() * 1000),
        "trace_id": trace_id,
        "event": body.get("event") or "tool",
        "step_id": body.get("step_id"),
        "tool": body.get("tool"),
        "payload": body.get("payload"),
    }
    _emit_trace(STATE_DIR, trace_id, str(entry.get("event") or "tool"), entry)
    return ToolEnvelope.success({"appended": True}, request_id="logs.tools.append")


