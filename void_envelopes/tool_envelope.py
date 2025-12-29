from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse


def _build_success_envelope(*, result: Dict[str, Any] | None, trace_id: str, conversation_id: str) -> Dict[str, Any]:
    """
    Canonical success envelope for tool-ish routes.

    Shared between orchestrator and satellite services (e.g. music) so all
    tool-style responses use the same top-level schema.
    """
    res_obj: Dict[str, Any] = dict(result or {})
    trace_id = trace_id if isinstance(trace_id, str) else ""
    conversation_id = conversation_id if isinstance(conversation_id, str) else ""
    env: Dict[str, Any] = {
        "schema_version": 1,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "ok": True,
        "result": res_obj,
        "error": None,
    }
    return env


def _build_error_envelope(
    *,
    code: str,
    message: str,
    trace_id: str,
    conversation_id: str,
    status: int,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Canonical error envelope for tool-ish routes.

    HTTP status is always 200; semantic status lives on error.status.
    """
    err_details: Dict[str, Any] = dict(details or {})
    err_details.setdefault("status", int(status))
    trace_id = trace_id if isinstance(trace_id, str) else ""
    conversation_id = conversation_id if isinstance(conversation_id, str) else ""
    env: Dict[str, Any] = {
        "schema_version": 1,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "ok": False,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "status": int(status),
            "details": err_details,
        },
    }
    return env


class ToolEnvelope:
    """
    Shared ToolEnvelope helpers.

    This is the single canonical implementation; orchestrator and all
    satellite services should import from `void_envelopes` instead of
    defining their own copies.
    """

    @staticmethod
    def success(*, result: Dict[str, Any], trace_id: str, conversation_id: str = "") -> JSONResponse:
        env = _build_success_envelope(result=result, trace_id=trace_id, conversation_id=conversation_id)
        return JSONResponse(env, status_code=200)

    @staticmethod
    def failure(
        *,
        code: str,
        message: str,
        trace_id: str,
        conversation_id: str = "",
        status: int,
        details: Dict[str, Any] | None = None,
    ) -> JSONResponse:
        env = _build_error_envelope(code=code, message=message, trace_id=trace_id, conversation_id=conversation_id, status=int(status), details=details)
        return JSONResponse(env, status_code=200)


