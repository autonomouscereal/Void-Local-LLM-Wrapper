from __future__ import annotations

import hashlib
import json
from fastapi import APIRouter, Query, Response
from typing import Any, Dict

from ..tools_schema import get_tool_introspection_registry
from .toolrun import ToolEnvelope  # canonical envelope


router = APIRouter()


@router.get("/tool.list")
async def tool_list():
    tools = []
    for n, meta in (get_tool_introspection_registry() or {}).items():
        tools.append(
            {
                "name": n,
                "version": meta.get("version"),
                "kind": meta.get("kind"),
                "describe_url": f"/tool.describe?name={n}",
            }
        )
    return ToolEnvelope.success({"tools": tools}, request_id="tool.list")


@router.get("/tool.describe")
async def tool_describe(name: str = Query(..., alias="name"), response: Response | None = None):
    key = (name or "").strip()
    # Prefer canonical schema derived from tools_schema.py (single source of truth).
    meta = get_tool_introspection_registry([key]).get(key)
    if isinstance(meta, dict) and isinstance(meta.get("schema"), dict):
        schema = meta.get("schema") or {}
        compact = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        shash = hashlib.sha256(compact).hexdigest()
        if response is not None:
            try:
                response.headers["ETag"] = f'W/"{shash}"'
            except Exception:
                pass
        return ToolEnvelope.success(
            {
                "name": meta.get("name") or key,
                "version": meta.get("version"),
                "kind": meta.get("kind"),
                "schema": schema,
                "schema_hash": shash,
                "notes": meta.get("notes"),
                "examples": meta.get("examples", []),
            },
            request_id="tool.describe",
        )
    return ToolEnvelope.failure(
        "tool_not_found",
        f"unknown tool '{name}'",
        status=404,
        request_id="tool.describe",
        details={},
    )


@router.post("/tool.describe")
async def tool_describe_post(body: Dict[str, Any]):
    name = ((body or {}).get("name") or "").strip()
    meta = get_tool_introspection_registry([name]).get(name)
    # Preferred: expose input_schema for executor auto-fix use.
    if isinstance(meta, dict) and isinstance(meta.get("schema"), dict):
        return ToolEnvelope.success(
            {"input_schema": meta.get("schema")},
            request_id="tool.describe",
        )
    return ToolEnvelope.failure(
        "tool_not_found",
        f"unknown tool '{name}'",
        status=404,
        request_id="tool.describe",
        details={},
    )


