from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict, Tuple


class Preflight204Middleware:
    """
    Short-circuit all OPTIONS requests with a 204 No Content and permissive CORS headers.
    This must be registered as the outermost middleware to prevent downstream handling.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[..., Awaitable[Dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        if scope.get("type") == "http" and (scope.get("method") or "").upper() == "OPTIONS":
            headers: list[Tuple[bytes, bytes]] = [
                (b"access-control-allow-origin", b"*"),
                (b"access-control-allow-methods", b"GET, POST, PUT, PATCH, DELETE, OPTIONS"),
                (b"access-control-allow-headers", b"*"),
                (b"access-control-allow-private-network", b"true"),
                (b"vary", b"Origin"),
                (b"connection", b"close"),
            ]
            await send({
                "type": "http.response.start",
                "status": 204,
                "headers": headers,
            })
            await send({
                "type": "http.response.body",
                "body": b"",
            })
            return
        await self.app(scope, receive, send)


