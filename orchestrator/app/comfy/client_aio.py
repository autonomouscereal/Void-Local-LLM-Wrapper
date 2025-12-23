from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx  # type: ignore

from ..json_parser import JSONParser

BASE = (
    os.getenv("COMFYUI_BASE_URL")
    or os.getenv("COMFYUI_API_URL")
    or "http://comfyui:8188"
).rstrip("/")

log = logging.getLogger(__name__)


def comfy_submit(graph: Dict[str, Any], client_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Hard-blocking ComfyUI submit. No asyncio, no websockets, no retries/backoff.
    """
    comfy_client_id = client_id or str(uuid.uuid4())
    payload = {"prompt": graph.get("prompt") or graph, "client_id": comfy_client_id}
    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    logging.info("[comfy.submit] bytes=%s", len(body_bytes))
    try:
        Path("/tmp/last_comfy_payload.json").write_bytes(body_bytes)
    except Exception:
        log.debug("[comfy.submit] failed to persist /tmp/last_comfy_payload.json (non-fatal)", exc_info=True)
    try:
        with httpx.Client(timeout=None, trust_env=False) as client:
            r = client.post(f"{BASE}/prompt", content=body_bytes, headers={"Content-Type": "application/json"})
        text = r.text or ""
        if r.status_code < 200 or r.status_code >= 300:
            return {
                "ok": False,
                "error": {
                    "code": "comfy_prompt_http_error",
                    "message": f"/prompt {r.status_code}",
                    "status": int(r.status_code),
                    "details": {"body": text},
                },
            }
        parser = JSONParser()
        schema = {"prompt_id": str, "uuid": str, "id": str}
        return parser.parse(text, schema)
    except Exception as ex:
        log.warning("comfy_submit.error: %s", ex, exc_info=True)
        return {"ok": False, "error": {"code": "comfy_submit_exception", "message": str(ex), "status": 500}}


def comfy_history(prompt_id: str) -> Dict[str, Any]:
    """
    Hard-blocking ComfyUI history fetch. No retries/backoff.
    """
    try:
        with httpx.Client(timeout=None, trust_env=False) as client:
            r = client.get(f"{BASE}/history/{prompt_id}")
        text = r.text or ""
        if r.status_code < 200 or r.status_code >= 300:
            return {
                "ok": False,
                "error": {
                    "code": "comfy_history_http_error",
                    "message": f"/history {r.status_code}",
                    "status": int(r.status_code),
                    "details": {"body": text},
                },
            }
        parser = JSONParser()
        schema = {"history": dict}
        return parser.parse(text, schema)
    except Exception as ex:
        log.warning("comfy_history.error prompt_id=%s: %s", prompt_id, ex, exc_info=True)
        return {"ok": False, "error": {"code": "comfy_history_exception", "message": str(ex), "status": 500}}


def comfy_is_completed(detail: Dict[str, Any]) -> bool:
    """
    Best-effort completion detector for a single ComfyUI history entry.

    Considers explicit status.completed, common terminal status strings, or the
    presence of any outputs as signals of completion.
    """
    st = detail.get("status") or {}
    if st.get("completed") is True:
        return True
    s = (st.get("status") or "").lower()
    if s in ("completed", "success", "succeeded", "done", "finished"):
        return True
    outs = detail.get("outputs")
    if isinstance(outs, dict) and any(isinstance(v, list) and len(v) > 0 for v in outs.values()):
        return True
    return False


def comfy_upload_image(name_hint: str, b64_png: str) -> str:
    data = base64.b64decode(b64_png)
    filename = f"{name_hint or 'ref'}_{uuid.uuid4().hex[:8]}.png"
    try:
        with httpx.Client(timeout=None, trust_env=False) as client:
            files = {"image": (filename, data, "image/png")}
            r = client.post(f"{BASE}/upload/image", files=files)
        text = r.text or ""
        if r.status_code < 200 or r.status_code >= 300:
            return filename
        parser = JSONParser()
        schema = {"name": str}
        obj = parser.parse(text, schema)
        stored = obj.get("name") or filename if isinstance(obj, dict) else filename
        return stored
    except Exception as ex:
        log.warning("comfy_upload_image.error: %s", ex, exc_info=True)
        return filename


def comfy_upload_mask(name_hint: str, b64_png: str) -> str:
    data = base64.b64decode(b64_png)
    filename = f"{name_hint or 'mask'}_{uuid.uuid4().hex[:8]}.png"
    try:
        with httpx.Client(timeout=None, trust_env=False) as client:
            files = {"image": (filename, data, "image/png")}
            r = client.post(f"{BASE}/upload/mask", files=files)
        text = r.text or ""
        if r.status_code < 200 or r.status_code >= 300:
            return filename
        parser = JSONParser()
        schema = {"name": str}
        obj = parser.parse(text, schema)
        return obj.get("name") or filename if isinstance(obj, dict) else filename
    except Exception as ex:
        log.warning("comfy_upload_mask.error: %s", ex, exc_info=True)
        return filename


def comfy_view(filename: str) -> Tuple[bytes, str]:
    # Returns (bytes, content_type)
    try:
        with httpx.Client(timeout=None, trust_env=False) as client:
            r = client.get(f"{BASE}/view", params={"filename": filename})
        if r.status_code < 200 or r.status_code >= 300:
            return b"", r.headers.get("content-type", "application/octet-stream")
        return r.content or b"", r.headers.get("content-type", "application/octet-stream")
    except Exception as ex:
        log.warning("comfy_view.error filename=%s: %s", filename, ex, exc_info=True)
        return b"", "application/octet-stream"


def choose_sampler_name() -> str:
    # Legacy name retained; hard-blocking and deterministic (no object_info call).
    return "euler"


