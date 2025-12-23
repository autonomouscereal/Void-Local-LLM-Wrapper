from __future__ import annotations

import os
import socket
import ipaddress
from typing import Any, Dict, Iterable, Optional, Tuple, TypedDict
from urllib.parse import urlsplit

import httpx  # type: ignore

from ..json_parser import JSONParser


ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}
SAFE_RESPONSE_HEADERS = {
    "content-type",
    "content-length",
    "date",
    "server",
    "x-request-id",
    "x-correlation-id",
    "x-amzn-requestid",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "retry-after",
}
BODY_PREVIEW_BYTES = 65536


class HttpRequestConfig(TypedDict, total=False):
    url: str
    method: str
    headers: Dict[str, str]
    query: Dict[str, Any]
    body: Any
    expect_json: bool


def _coerce_header_map(headers: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in headers.items() if isinstance(k, str)}


def _coerce_query_map(query: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in query.items():
        if not isinstance(key, str):
            continue
        # IMPORTANT: bool is a subclass of int; handle it first.
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif value is None:
            out[key] = ""
        elif isinstance(value, (str, int, float)):
            out[key] = str(value)
        else:
            out[key] = str(value)
    return out


def _truncate_text(text: str, limit: int = BODY_PREVIEW_BYTES) -> str:
    # Full-fidelity: never truncate remote bodies (required for debugging/training).
    return text


def _subset_headers(headers: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    subset: Dict[str, str] = {}
    for key, value in headers:
        if len(subset) >= 12:
            break
        if key.lower() in SAFE_RESPONSE_HEADERS or key.lower().startswith("x-ratelimit"):
            subset[key] = value
    if not subset:
        for key, value in list(headers)[:6]:
            subset[key] = value
    return subset


def validate_remote_host(url: str) -> Optional[str]:
    if not url:
        return "missing_url"
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return "scheme_not_allowed"
    host = (parts.hostname or "").lower()
    if not host:
        return "missing_host"
    if host in ("localhost",) or host.startswith("127.") or host == "::1":
        return "host_not_allowed"

    def _host_of(env_var: str) -> str:
        candidate = os.getenv(env_var, "") or ""
        try:
            return urlsplit(candidate).hostname or ""
        except Exception:
            return ""

    forbidden_hosts = set(
        filter(
            None,
            [
                _host_of("EXECUTOR_BASE_URL"),
                _host_of("COMFYUI_API_URL"),
                _host_of("DRT_API_URL"),
                _host_of("ORCHESTRATOR_BASE_URL"),
            ],
        )
    )
    if host in forbidden_hosts:
        return "host_not_allowed"
    try:
        addrs = {
            info[-1][0]
            for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        }
        for addr in addrs:
            ip_obj = ipaddress.ip_address(addr)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                return "host_not_allowed"
    except Exception:
        # If we cannot resolve the host, we cannot safely SSRF-screen it.
        return "dns_lookup_failed"
    return None


def _prepare_body(body: Any) -> Tuple[Optional[Any], Optional[bytes], Optional[str]]:
    if body is None:
        return None, None, None
    if isinstance(body, (dict, list)):
        return body, None, "application/json"
    if isinstance(body, (bytes, bytearray)):
        return None, bytes(body), None
    if isinstance(body, str):
        return None, body.encode("utf-8"), "text/plain; charset=utf-8"
    return None, str(body).encode("utf-8"), "text/plain; charset=utf-8"


async def perform_http_request(config: HttpRequestConfig) -> Tuple[bool, Dict[str, Any]]:
    url = config.get("url", "")
    method = config.get("method", "GET").upper()
    headers = _coerce_header_map(config.get("headers") or {})
    query = _coerce_query_map(config.get("query") or {})
    body = config.get("body")
    expect_json = config.get("expect_json", True)

    method_allowed = method in ALLOWED_METHODS
    if not method_allowed:
        return False, {
            "code": "method_not_allowed",
            "message": f"HTTP method {method} is not permitted",
            "status": 422,
            "details": {"method": method},
        }

    host_error = validate_remote_host(url)
    if host_error:
        return False, {
            "code": host_error,
            "message": f"Host not allowed: {url}",
            "status": 422,
            "details": {"url": url},
        }

    json_payload, content_payload, explicit_content_type = _prepare_body(body)
    if explicit_content_type and "content-type" not in {k.lower() for k in headers}:
        headers["Content-Type"] = explicit_content_type

    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=False, trust_env=False) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=query,
                json=json_payload,
                content=content_payload,
            )
    except httpx.TimeoutException as ex:  # type: ignore
        return False, {
            "code": "remote_timeout",
            "message": "Remote request timed out",
            "status": 598,
            "details": {"error": str(ex), "url": url},
        }
    except Exception as ex:
        return False, {
            "code": "remote_network_error",
            "message": "Remote request failed",
            "status": 599,
            "details": {"error": str(ex), "url": url},
        }

    status = int(response.status_code)
    captured_headers = _subset_headers(response.headers.items())
    raw_body = response.content or b""
    body_preview = _truncate_text(
        raw_body.decode(response.encoding or "utf-8", errors="replace")
        if raw_body else ""
    )

    if 200 <= status < 300:
        if expect_json:
            txt = body_preview
            if not txt.strip():
                return True, {"status": status, "headers": captured_headers, "body": None}
            parser = JSONParser()
            try:
                # Use the first non-whitespace character as a loose hint for the
                # top-level container shape, but always delegate full repair and
                # parsing to JSONParser without any "looks like JSON" guards.
                first = txt.lstrip()[:1]
                expected: Any
                if first == "[":
                    expected = []
                else:
                    expected = {}
                parsed = parser.parse(txt, expected)
            except Exception as ex:
                return False, {
                    "code": "remote_invalid_json",
                    "message": "Remote response JSON parsing failed",
                    "status": status,
                    "details": {
                        "remote_status": status,
                        "remote_headers": captured_headers,
                        "remote_body": txt,
                        "parse_error": str(ex),
                    },
                }
            return True, {"status": status, "headers": captured_headers, "body": parsed}

        # expect_json is False → return text or base64
        if raw_body:
            try:
                decoded = response.text
            except Exception:
                decoded = body_preview
        else:
            decoded = ""
        return True, {"status": status, "headers": captured_headers, "body_text": _truncate_text(decoded)}

    # Non-2xx statuses => failure envelope
    details: Dict[str, Any] = {
        "remote_status": status,
        "remote_headers": captured_headers,
    }
    if expect_json:
        txt = body_preview
        parser = JSONParser()
        try:
            first = txt.lstrip()[:1]
            if first == "[":
                expected = []
            else:
                expected = {}
            details["remote_body"] = parser.parse(txt, expected)
        except Exception:
            details["remote_body"] = txt
            details["remote_body_truncated"] = len(txt) == BODY_PREVIEW_BYTES
    else:
        if raw_body:
            details["remote_body"] = _truncate_text(body_preview)
            if len(body_preview) == BODY_PREVIEW_BYTES:
                details["remote_body_truncated"] = True
        else:
            details["remote_body"] = ""

    return False, {
        "code": "remote_http_error",
        "message": f"Remote HTTP {status}",
        "status": status,
        "details": details,
    }


