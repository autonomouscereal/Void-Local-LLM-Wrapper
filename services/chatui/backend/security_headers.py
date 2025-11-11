from __future__ import annotations
import os
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


def _orch_base() -> str:
	# Prefer explicit ORCHESTRATOR_URL; fallback to common service name
	return (os.getenv("ORCHESTRATOR_URL") or "http://orchestrator:8000").rstrip("/")


class SecurityHeaders(BaseHTTPMiddleware):
	def __init__(self, app: ASGIApp):
		super().__init__(app)

	async def dispatch(self, request, call_next: Callable):
		resp = await call_next(request)
		orch = _orch_base()
		# Permit cross-origin fetch to orchestrator; do not block preflight or POST
		csp = (
			"default-src 'self'; "
			f"connect-src 'self' {orch} data: blob:; "
			"img-src 'self' data: blob:; "
			"style-src 'self' 'unsafe-inline'; "
			"script-src 'self'; "
			"frame-ancestors 'self'; "
			"base-uri 'self'"
		)
		resp.headers["Content-Security-Policy"] = csp
		# Optional hardening (does not affect CORS/CSP connect-src):
		resp.headers.setdefault("Referrer-Policy", "no-referrer")
		resp.headers.setdefault("X-Content-Type-Options", "nosniff")
		resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
		return resp


