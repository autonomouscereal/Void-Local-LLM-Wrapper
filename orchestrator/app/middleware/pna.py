from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict, Tuple


class AllowPrivateNetworkMiddleware:
    """Append Access-Control-Allow-Private-Network on OPTIONS when requested."""

    def __init__(self, app):
        self.app = app

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[..., Awaitable[Dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = (scope.get("method") or "").upper()
        hdrs: Dict[bytes, bytes] = {}
        for k, v in (scope.get("headers") or []):
            k_l = (k or b"").lower()
            if k_l not in hdrs:
                hdrs[k_l] = v or b""
        want_pna = (hdrs.get(b"access-control-request-private-network", b"").lower() == b"true")

        async def send_wrapper(message: Dict[str, Any]) -> None:
            if message.get("type") == "http.response.start" and method == "OPTIONS":
                headers: list[Tuple[bytes, bytes]] = list(message.get("headers") or [])
                if want_pna:
                    headers.append((b"access-control-allow-private-network", b"true"))
                # Vary on relevant request headers for caches/proxies
                headers.append((b"vary", b"Origin, Access-Control-Request-Headers, Access-Control-Request-Private-Network"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


