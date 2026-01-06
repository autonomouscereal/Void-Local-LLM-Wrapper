from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx  # type: ignore

from ..json_parser import JSONParser
from .assets import normalize_history_entry, extract_comfy_asset_urls

BASE = (
    os.getenv("COMFYUI_BASE_URL")
    or os.getenv("COMFYUI_API_URL")
    or "http://comfyui:8188"
).rstrip("/")

log = logging.getLogger(__name__)


def comfy_submit(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hard-blocking ComfyUI submit. No asyncio, no websockets, no retries/backoff.
    """
    payload = {"prompt": graph.get("prompt") or graph}
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


def submit_to_comfyui_and_poll(
    prompt: Dict[str, Any], trace_id: str = "", conversation_id: str = ""
) -> Dict[str, Any]:
    """
    Submit a prompt to ComfyUI, poll until complete, and download images to the correct directory.
    
    Args:
        prompt: ComfyUI workflow/prompt dict
        trace_id: Optional trace ID for logging
        conversation_id: Optional conversation ID for logging
    
    Returns:
        Dict with:
            - ok: bool indicating success
            - prompt_id: str if successful
            - saved_paths: List[str] of downloaded image paths
            - save_dir: str path to directory where images were saved
            - error: Dict with error info if failed
    """
    upload_dir = os.getenv("UPLOAD_DIR", "/workspace/uploads")
    
    # Submit
    workflow_payload: Dict[str, Any] = {"prompt": prompt}
    submit_res = comfy_submit(workflow_payload)
    
    if not isinstance(submit_res, dict) or (isinstance(submit_res.get("error"), dict) and submit_res.get("error")):
        return {
            "ok": False,
            "error": submit_res.get("error") or {"code": "comfy_submit_failed", "message": "comfy submit failed"},
        }
    
    prompt_id = submit_res.get("prompt_id") or submit_res.get("uuid") or submit_res.get("id")
    if not isinstance(prompt_id, str) or not prompt_id:
        return {
            "ok": False,
            "error": {"code": "missing_prompt_id", "message": "missing prompt_id from comfy", "detail": submit_res},
        }
    
    # Poll until complete
    log.info("[comfy.poll] trace_id=%s prompt_id=%s starting poll", trace_id, prompt_id)
    while True:
        hist = comfy_history(prompt_id)
        if isinstance(hist, dict) and hist.get("error"):
            return {
                "ok": False,
                "error": hist.get("error"),
                "prompt_id": prompt_id,
            }
        
        detail = normalize_history_entry(hist if isinstance(hist, dict) else {}, prompt_id)
        if detail and comfy_is_completed(detail if isinstance(detail, dict) else {}):
            break
        time.sleep(0.75)
    
    log.info("[comfy.poll] trace_id=%s prompt_id=%s completed", trace_id, prompt_id)
    
    # Extract asset URLs
    base = BASE
    assets_list = extract_comfy_asset_urls(detail if isinstance(detail, dict) else {}, base)
    
    if not assets_list:
        return {
            "ok": True,
            "prompt_id": prompt_id,
            "saved_paths": [],
            "save_dir": None,
            "warning": "no assets found in history",
        }
    
    # Download outputs into uploads/artifacts
    artifact_group_id = str(prompt_id)
    save_dir = os.path.join(upload_dir, "artifacts", "image", artifact_group_id)
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths: List[str] = []
    
    with httpx.Client(timeout=None, trust_env=False) as client:
        for it in assets_list:
            if not isinstance(it, dict):
                continue
            fn = it.get("filename")
            url = it.get("url")
            if not isinstance(fn, str) or not fn:
                continue
            if not isinstance(url, str) or not url:
                continue
            
            # Fetch and persist
            try:
                resp = client.get(url)
                if int(getattr(resp, "status_code", 0) or 0) != 200:
                    log.warning("[comfy.download] trace_id=%s prompt_id=%s failed to fetch %s: status=%s", 
                              trace_id, prompt_id, url, resp.status_code)
                    continue
                
                safe_fn = os.path.basename(fn)
                dst = os.path.join(save_dir, safe_fn)
                with open(dst, "wb") as f:
                    f.write(resp.content)
                saved_paths.append(dst)
                log.debug("[comfy.download] trace_id=%s prompt_id=%s saved %s", trace_id, prompt_id, dst)
            except Exception as ex:
                log.warning("[comfy.download] trace_id=%s prompt_id=%s error downloading %s: %s", 
                           trace_id, prompt_id, url, ex, exc_info=True)
                continue
    
    return {
        "ok": True,
        "prompt_id": prompt_id,
        "saved_paths": saved_paths,
        "save_dir": save_dir,
    }

