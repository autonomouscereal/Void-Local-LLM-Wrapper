from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.main import _TOOL_SCHEMAS as _SCHEMAS


router = APIRouter()


def ok_envelope(result, rid: str = "tool.describe") -> JSONResponse:
	return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


_REGISTRY = {
	"image.dispatch": {
		"name": "image.dispatch",
		"description": "Dispatches an image generation job to ComfyUI using the stock image workflow.",
		"inputs": {
			"prompt": {"type": "string", "required": False},
			"negative_prompt": {"type": "string", "required": False},
			"seed": {"type": "integer", "required": False},
			"steps": {"type": "integer", "required": False},
			"cfg": {"type": "number", "required": False},
			"sampler_name": {"type": "string", "required": False},
			"scheduler": {"type": "string", "required": False},
			"width": {"type": "integer", "required": False},
			"height": {"type": "integer", "required": False},
			"model": {"type": "string", "required": False},
			"workflow_path": {"type": "string", "required": False},
			"timeout_sec": {"type": "integer", "required": False},
		},
		"result": {
			"ids": {"prompt_id": "string", "client_id": "string", "images": "array"},
			"meta": {"submitted": "boolean", "workflow_path": "string", "comfy_base": "string", "image_count": "integer", "timed_out": "boolean"},
		},
	},
}


@router.get("/tool.describe")
async def tool_describe(name: str = Query(..., alias="name")):
	key = (name or "").strip()
	# Prefer the full schema from app.main when available
	meta = _SCHEMAS.get(key) if isinstance(_SCHEMAS, dict) else None
	if meta and isinstance(meta.get("schema"), dict):
		return ok_envelope({
			"name": meta.get("name") or key,
			"version": meta.get("version"),
			"kind": meta.get("kind"),
			"schema": meta.get("schema"),
			"notes": meta.get("notes"),
			"examples": meta.get("examples", []),
		})
	# Fallback to local minimal registry entries
	sch = _REGISTRY.get(key)
	if not sch:
		return JSONResponse({"schema_version": 1, "request_id": "tool.describe", "ok": False, "error": {"code": "tool_not_found", "message": f"unknown tool '{name}'", "details": {}}}, status_code=404)
	return ok_envelope({"schema": sch})


