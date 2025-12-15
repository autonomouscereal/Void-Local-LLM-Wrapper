from __future__ import annotations

from fastapi import FastAPI
from typing import Any, Dict

from ..tools_schema import get_tool_introspection_registry
from .toolrun import ToolEnvelope  # canonical envelope


def mount_tools_routes(app: FastAPI) -> None:
    """
    Mount tool schema introspection endpoints directly onto the FastAPI app.

    Policy: JSON-in only. No query params. No headers/ETag behaviors.
    """

    async def tool_list(body: Dict[str, Any]) -> Dict[str, Any]:
        # body is accepted for policy consistency; currently unused.
        tools = []
        for n, meta in (get_tool_introspection_registry() or {}).items():
            tools.append(
                {
                    "name": n,
                    "version": meta.get("version"),
                    "kind": meta.get("kind"),
                }
            )
        return ToolEnvelope.success({"tools": tools}, request_id="tool.list")

    async def tool_describe(body: Dict[str, Any]) -> Dict[str, Any]:
        name = ((body or {}).get("name") or "").strip()
        meta = get_tool_introspection_registry([name]).get(name)
        if isinstance(meta, dict) and isinstance(meta.get("schema"), dict):
            return ToolEnvelope.success(
                {
                    "name": meta.get("name") or name,
                    "version": meta.get("version"),
                    "kind": meta.get("kind"),
                    "schema": meta.get("schema") or {},
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

    # JSON-only endpoints
    app.add_api_route("/tool.list", tool_list, methods=["POST"])
    app.add_api_route("/tool.describe", tool_describe, methods=["POST"])


