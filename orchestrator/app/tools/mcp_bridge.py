from __future__ import annotations

from typing import Dict, Any
import traceback
import httpx  # type: ignore
from ..json_parser import JSONParser


async def call_mcp_tool(bridge_url: str | None, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not bridge_url:
        return {
            "ok": False,
            "result": None,
            "error": {"code": "mcp_bridge_unconfigured", "message": "MCP bridge URL not configured", "status": 500},
        }
    try:
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.post(
            bridge_url.rstrip("/") + "/call",
            json={"name": name, "arguments": arguments or {}},
        )
    except Exception as ex:
        return {
            "ok": False,
            "result": None,
            "error": {"code": "mcp_bridge_request_failed", "message": str(ex), "status": 502, "stack": traceback.format_exc()},
        }
    if int(getattr(r, "status_code", 0) or 0) < 200 or int(getattr(r, "status_code", 0) or 0) >= 300:
        return {
            "ok": False,
            "result": None,
            "error": {
                "code": "mcp_bridge_http_error",
                "message": f"MCP bridge returned HTTP {int(getattr(r, 'status_code', 0) or 0)}",
                "status": int(getattr(r, "status_code", 0) or 0) or 502,
                "body": (r.text or ""),
            },
        }
        parser = JSONParser()
        # Open schema: do not drop any keys returned by the MCP bridge.
        js = parser.parse(r.text or "", {})
    if not isinstance(js, dict):
        return {
            "ok": False,
            "result": None,
            "error": {"code": "mcp_bridge_bad_json", "message": "invalid mcp bridge response", "status": 502, "body": (r.text or "")},
        }
    return {"ok": True, "result": js, "error": None}


