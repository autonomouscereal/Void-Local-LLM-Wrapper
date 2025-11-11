from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter()


def ok_envelope(result, rid: str) -> JSONResponse:
	return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


@router.post("/tool.validate")
async def tool_validate(req: Request):
	body = await req.json()
	name = body.get("name")
	args = body.get("args")
	return ok_envelope({"name": name, "valid": True, "args": args}, rid="tool.validate")


@router.post("/tool.run")
async def tool_run(req: Request):
	body = await req.json()
	name = body.get("name") or ""
	args = body.get("args") or {}
	return ok_envelope({"ids": {"job_id": f"mock-{name}"}, "meta": {"accepted": True, "args": args}}, rid="tool.run")


