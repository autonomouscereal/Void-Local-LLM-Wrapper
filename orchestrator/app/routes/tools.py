from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse


router = APIRouter()


def ok_envelope(result, rid: str = "tool.describe") -> JSONResponse:
	return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


_REGISTRY = {
	"image.dispatch": {
		"name": "image.dispatch",
		"description": "Dispatch an image generation job to ComfyUI backends.",
		"inputs": {
			"type": "object",
			"properties": {
				"prompt": {"type": "string"},
				"workflow": {"type": "string"},
				"seed": {"type": "integer"},
				"width": {"type": "integer"},
				"height": {"type": "integer"},
			},
			"required": ["prompt"],
		},
	},
}


@router.get("/tool.describe")
async def tool_describe(name: str = Query(..., alias="name")):
	sch = _REGISTRY.get(name)
	if not sch:
		return JSONResponse(
			{
				"schema_version": 1,
				"request_id": "tool.describe",
				"ok": False,
				"error": {"code": "tool_not_found", "message": f"unknown tool '{name}'", "details": {}},
			},
			status_code=404,
		)
	return ok_envelope({"schema": sch})


