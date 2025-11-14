from __future__ import annotations

from typing import Any, Callable, Dict, List
import uuid as _uuid

from app.json_parser import JSONParser  # type: ignore
from .tool_evidence_store import append_tool_evidence as _tel_append  # type: ignore

import httpx as _hx  # type: ignore


async def validate_and_repair(
    tool_calls: List[Dict[str, Any]],
    *,
    base_url: str,
    trace_id: str,
    log_fn: Callable[..., None],
    state_dir: str,
) -> Dict[str, Any]:
    """
    Validate tool calls exactly once via /tool.validate.
    Validator is advisory-only: every planned tool call is still executed once.
    Returns the (optionally adjusted) calls alongside structured failures
    that downstream planners and committees can treat as evidence.
    """
    validated: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    base = (base_url or "").rstrip("/")
    if not base:
        # Advisory-only semantics: record a failure for evidence, but still execute all planned tool calls.
        for tc in tool_calls or []:
            name = str((tc.get("name") or "")).strip()
            args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
            envelope = {
                "schema_version": 1,
                "request_id": _uuid.uuid4().hex,
                "ok": False,
                "error": {
                    "code": "validator_base_url_missing",
                    "message": "tool.validate base_url missing",
                    "details": {},
                    "status": 0,
                },
            }
            log_fn("validate.result", trace_id=trace_id, tool=name, status=0, ok=False, error=envelope["error"])
            failures.append(
                {
                    "name": name,
                    "arguments": args,
                    "status": 0,
                    "envelope": envelope,
                }
            )
            _tel_append(
                state_dir,
                trace_id,
                {
                    "name": name,
                    "ok": False,
                    "label": "failure",
                    "raw": {
                        "ts": None,
                        "ok": False,
                        "args": args,
                        "error": envelope["error"],
                    },
                },
            )
            # Always execute the original tool call when the validator base is missing.
            validated.append(tc)
        return {
            "validated": validated,
            "pre_tool_failures": failures,
            "repairs_made": False,
            "_repair_success_any": False,
            "_patched_payload_emitted": False,
        }

    async with _hx.AsyncClient(trust_env=False, timeout=None) as client:
        for tc in tool_calls or []:
            name = str((tc.get("name") or "")).strip()
            args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
            payload = {"name": name, "args": args}
            parser = JSONParser()
            try:
                response = await client.post(base + "/tool.validate", json=payload)
                expected = {
                    "schema_version": int,
                    "request_id": str,
                    "ok": bool,
                    "result": dict,
                    "error": {"code": str, "message": str, "status": int, "details": dict},
                }
                env = parser.parse(response.text or "", expected)
            except Exception as ex:
                # Validation backend/network must never block execution; treat as soft failure.
                env = {
                    "schema_version": 1,
                    "request_id": _uuid.uuid4().hex,
                    "ok": False,
                    "result": None,
                    "error": {
                        "code": "validator_request_failed",
                        "message": str(ex),
                        "status": 0,
                        "details": {},
                    },
                }
                log_fn(
                    "validate.result",
                    trace_id=trace_id,
                    tool=name,
                    status=0,
                    ok=False,
                    error=env["error"],
                )
                failures.append(
                    {
                        "name": name,
                        "arguments": args,
                        "status": 0,
                        "envelope": env,
                    }
                )
                _tel_append(
                    state_dir,
                    trace_id,
                    {
                        "name": name,
                        "ok": False,
                        "label": "failure",
                        "raw": {
                            "ts": None,
                            "ok": False,
                            "args": args,
                            "error": env["error"],
                        },
                    },
                )
                # Soft-fail: still allow this tool call to be executed.
                validated.append(tc)
                continue

            ok_body = isinstance(env, dict) and env.get("ok") is True
            error_obj = (env.get("error") if isinstance(env, dict) else {}) or {}
            status_code = int(error_obj.get("status") or getattr(response, "status_code", 0) or 0)
            log_fn(
                "validate.result",
                trace_id=trace_id,
                tool=name,
                status=status_code,
                ok=bool(ok_body),
                error=error_obj,
            )
            if not isinstance(env, dict):
                env = {}
            if "schema_version" not in env:
                env["schema_version"] = 1
            if not isinstance(env.get("request_id"), str) or not env.get("request_id"):
                env["request_id"] = _uuid.uuid4().hex
            if not isinstance(env.get("error"), dict):
                env["error"] = {"code": "validation_failed", "message": "", "details": {}}
            if "status" not in env["error"]:
                env["error"]["status"] = status_code

            # Decide final arguments for execution (advisory-only):
            # default to original args, optionally adopt any repaired args the validator may provide.
            final_args = args
            result_obj = env.get("result") if isinstance(env.get("result"), dict) else {}
            if isinstance(result_obj, dict):
                maybe_fixed = result_obj.get("args") or result_obj.get("arguments")
                if isinstance(maybe_fixed, dict):
                    final_args = maybe_fixed

            # Record failures for evidence when the validator reports an error.
            has_error = (not bool(ok_body)) or bool(str(error_obj.get("code") or "").strip())
            if has_error:
                failures.append(
                    {
                        "name": name,
                        "arguments": args,
                        "status": status_code,
                        "envelope": env,
                    }
                )
                _tel_append(
                    state_dir,
                    trace_id,
                    {
                        "name": name,
                        "ok": False,
                        "label": "failure",
                        "raw": {
                            "ts": None,
                            "ok": False,
                            "args": args,
                            "error": env["error"],
                        },
                    },
                )

            # Always execute each planned tool exactly once with final_args, regardless of env.ok.
            tc_fixed = dict(tc)
            tc_fixed["arguments"] = final_args
            validated.append(tc_fixed)

    return {
        "validated": validated,
        "pre_tool_failures": failures,
        "repairs_made": False,
        "_repair_success_any": False,
        "_patched_payload_emitted": False,
    }

