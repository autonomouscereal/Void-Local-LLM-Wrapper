from __future__ import annotations

from typing import Dict, Any
import httpx  # type: ignore
from ..jsonio.helpers import resp_json as _resp_json


async def call_mcp_tool(bridge_url: str | None, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not bridge_url:
        return {"error": "MCP bridge URL not configured"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                bridge_url.rstrip("/") + "/call",
                json={"name": name, "arguments": arguments or {}},
            )
            r.raise_for_status()
            return _resp_json(r, {})
    except Exception as ex:
        return {"error": str(ex)}


