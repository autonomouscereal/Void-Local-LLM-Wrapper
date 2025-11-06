from __future__ import annotations

from typing import Any, Callable, Awaitable


class PermissiveWebSocketMiddleware:
    """ASGI middleware that strips the Origin header from WebSocket handshakes
    to avoid upstream origin gate failures. HTTP requests are passed through unchanged.
    """

    def __init__(self, app: Callable[[dict, Callable, Callable], Awaitable[None]]):
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") == "websocket":
            headers = scope.get("headers") or []
            # remove Origin to prevent strict origin checks from rejecting WS
            filtered = [(k, v) for (k, v) in headers if (k or b"").lower() != b"origin"]
            scope = dict(scope)
            scope["headers"] = filtered
        await self.app(scope, receive, send)


