from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict, Tuple, List


class AppendCommonHeadersMiddleware:
    """
    Ensure permissive cross-origin headers are present on all responses
    without overriding values set by upstream middleware/handlers.
    - Access-Control-Allow-Origin: *
    - Access-Control-Expose-Headers: *
    - Cross-Origin-Resource-Policy: cross-origin
    - Timing-Allow-Origin: *
    - Vary: Origin
    Additionally, for OPTIONS preflight, ensure standard fields exist (no caching/max-age):
    - Access-Control-Allow-Methods
    - Access-Control-Allow-Headers
    """
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

        async def send_wrapper(message: Dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers: List[Tuple[bytes, bytes]] = list(message.get("headers") or [])
                lower = {k.lower(): i for i, (k, _) in enumerate(headers)}

                def setdefault(name: bytes, value: bytes) -> None:
                    if name.lower() not in lower:
                        headers.append((name, value))

                # Always expose origin-related headers for browser consumption
                setdefault(b"access-control-allow-origin", b"*")
                setdefault(b"access-control-expose-headers", b"*")
                setdefault(b"cross-origin-resource-policy", b"cross-origin")
                setdefault(b"timing-allow-origin", b"*")
                # Cache/key vary on Origin
                # If Vary already present, avoid duplicating; else add
                vary_idx = lower.get(b"vary")
                if vary_idx is not None:
                    # no normalization; rely on upstream value
                    pass
                else:
                    headers.append((b"vary", b"Origin"))

                if method == "OPTIONS":
                    setdefault(b"access-control-allow-methods", b"GET, POST, PUT, PATCH, DELETE, OPTIONS")
                    setdefault(b"access-control-allow-headers", b"*")

                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


