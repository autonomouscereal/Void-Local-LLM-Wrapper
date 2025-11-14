from __future__ import annotations

from fastapi import APIRouter, Query
from app.main import _TOOL_SCHEMAS as _SCHEMAS
from app.routes.toolrun import ToolEnvelope  # canonical envelope
from typing import Any, Dict


router = APIRouter()


_REGISTRY = {
	"image.dispatch": {
		"name": "image.dispatch",
		"description": "Dispatches an image generation job to ComfyUI using the stock image workflow.",
		"inputs": {
			"prompt": {"type": "string", "required": True},
			"negative": {"type": "string", "required": False},
			"seed": {"type": "integer", "required": False},
			"steps": {"type": "integer", "required": False},
			"cfg": {"type": "number", "required": False},
			"sampler": {"type": "string", "required": False},
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
		return ToolEnvelope.success(
			{
				"name": meta.get("name") or key,
				"version": meta.get("version"),
				"kind": meta.get("kind"),
				"schema": meta.get("schema"),
				"notes": meta.get("notes"),
				"examples": meta.get("examples", []),
			},
			request_id="tool.describe",
		)
	# Fallback to local minimal registry entries
	sch = _REGISTRY.get(key)
	if not sch:
		return ToolEnvelope.failure(
			"tool_not_found",
			f"unknown tool '{name}'",
			status=404,
			request_id="tool.describe",
			details={},
		)
	return ToolEnvelope.success({"schema": sch}, request_id="tool.describe")


@router.post("/tool.describe")
async def tool_describe_post(body: Dict[str, Any]):
	name = ((body or {}).get("name") or "").strip()
	meta = _SCHEMAS.get(name) if isinstance(_SCHEMAS, dict) else None
	# Preferred: expose input_schema for executor auto-fix use
	if meta and isinstance(meta.get("schema"), dict):
		return ToolEnvelope.success(
			{"input_schema": meta.get("schema")},
			request_id="tool.describe",
		)
	# Fallback: construct input_schema for image.dispatch locally
	if name == "image.dispatch":
		input_schema = {
			"type": "object",
			"additionalProperties": False,
			"properties": {
				"prompt": {"type": "string"},
				"negative": {"type": "string"},
				"seed": {"type": "integer"},
				"steps": {"type": "integer", "minimum": 1, "maximum": 150},
				"cfg": {"type": "number", "minimum": 0.0, "maximum": 30.0},
				"sampler": {"type": "string"},
				"scheduler": {"type": "string"},
				"width": {"type": "integer", "minimum": 64, "maximum": 2048, "multipleOf": 8},
				"height": {"type": "integer", "minimum": 64, "maximum": 2048, "multipleOf": 8},
				"model": {"type": "string"},
				"workflow_path": {"type": "string"},
				"workflow_graph": {"type": "object"},
				"autofix_422": {"type": "boolean", "default": True},
			},
			"required": ["prompt"],
		}
		return ToolEnvelope.success(
			{"input_schema": input_schema},
			request_id="tool.describe",
		)
	return ToolEnvelope.failure(
		"tool_not_found",
		f"unknown tool '{name}'",
		status=404,
		request_id="tool.describe",
		details={},
	)
