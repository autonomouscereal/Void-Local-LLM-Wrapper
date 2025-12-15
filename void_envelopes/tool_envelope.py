from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse


def _build_success_envelope(result: Dict[str, Any] | None, rid: str) -> Dict[str, Any]:
    """
    Canonical success envelope for tool-ish routes.

    Shared between orchestrator and satellite services (e.g. music) so all
    tool-style responses use the same top-level schema.
    """
    res_obj: Dict[str, Any] = dict(result or {})
    cid_val: Optional[str] = None
    trace_val: Optional[str] = None

    if isinstance(res_obj, dict):
        cid_val = res_obj.get("cid")  # type: ignore[assignment]
        trace_val = res_obj.get("trace_id")  # type: ignore[assignment]
        meta_part = res_obj.get("meta") if isinstance(res_obj.get("meta"), dict) else {}
        if isinstance(meta_part, dict):
            if not cid_val:
                cid_val = meta_part.get("cid")
            if not trace_val:
                trace_val = meta_part.get("trace_id") or (meta_part.get("ids") or {}).get("trace_id")

    env: Dict[str, Any] = {
        "schema_version": 1,
        "request_id": rid,
        "ok": True,
        "result": res_obj,
        "error": None,
    }
    if isinstance(cid_val, (str, int)):
        env["cid"] = str(cid_val)
    if isinstance(trace_val, (str, int)) and str(trace_val).strip():
        env["trace_id"] = str(trace_val)
    return env


def _build_error_envelope(
    code: str,
    message: str,
    rid: str,
    *,
    status: int,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Canonical error envelope for tool-ish routes.

    HTTP status is always 200; semantic status lives on error.status.
    """
    err_details: Dict[str, Any] = dict(details or {})
    err_details.setdefault("status", int(status))
    env: Dict[str, Any] = {
        "schema_version": 1,
        "request_id": rid,
        "ok": False,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "status": int(status),
            "details": err_details,
        },
    }
    cid_val = err_details.get("cid")
    if isinstance(cid_val, (str, int)):
        env["cid"] = str(cid_val)
    trace_val = err_details.get("trace_id") or err_details.get("tid")
    if isinstance(trace_val, (str, int)) and str(trace_val).strip():
        env["trace_id"] = str(trace_val)
    return env


class ToolEnvelope:
    """
    Shared ToolEnvelope helpers.

    This is the single canonical implementation; orchestrator and all
    satellite services should import from `void_envelopes` instead of
    defining their own copies.
    """

    @staticmethod
    def success(result: Dict[str, Any], *, request_id: Optional[str] = None) -> JSONResponse:
        rid = request_id or uuid.uuid4().hex
        env = _build_success_envelope(result, rid)
        return JSONResponse(env, status_code=200)

    @staticmethod
    def failure(
        code: str,
        message: str,
        *,
        status: int,
        details: Dict[str, Any] | None = None,
        request_id: Optional[str] = None,
    ) -> JSONResponse:
        rid = request_id or uuid.uuid4().hex
        env = _build_error_envelope(code, message, rid, status=int(status), details=details)
        return JSONResponse(env, status_code=200)


