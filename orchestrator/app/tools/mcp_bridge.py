from __future__ import annotations

from typing import Dict, Any
import httpx  # type: ignore
from ..json_parser import JSONParser


async def call_mcp_tool(bridge_url: str | None, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not bridge_url:
        return {"error": "MCP bridge URL not configured"}
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.post(
            bridge_url.rstrip("/") + "/call",
            json={"name": name, "arguments": arguments or {}},
        )
        parser = JSONParser()
        js = parser.parse(r.text or "", {})
        return js if isinstance(js, dict) else {"error": "invalid mcp bridge response"}


