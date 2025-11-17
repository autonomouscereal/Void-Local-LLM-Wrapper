from __future__ import annotations

import os
import json
import uuid
import base64
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

import aiohttp  # type: ignore
from ..json_parser import JSONParser


BASE = (
    os.getenv("COMFYUI_BASE_URL")
    or os.getenv("COMFYUI_API_URL")
    or "http://comfyui:8188"
).rstrip("/")


async def comfy_submit(graph: Dict[str, Any], client_id: Optional[str] = None, ws=None) -> Dict[str, Any]:
    cid = client_id or str(uuid.uuid4())
    payload = {"prompt": graph.get("prompt") or graph, "client_id": cid}
    # Validate/shape payload with project JSON parser
    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    logging.info(f"[comfy.submit] bytes={len(body_bytes)}")
    try:
        Path("/tmp/last_comfy_payload.json").write_bytes(body_bytes)
    except Exception:
        pass
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.post(f"{BASE}/prompt", data=body_bytes, headers={"Content-Type": "application/json"}) as r:
            text = await r.text()
            if r.status != 200:
                # Do not raise; return a structured error envelope for callers to surface.
                err = {
                    "ok": False,
                    "error": {
                        "code": "comfy_prompt_http_error",
                        "message": f"/prompt {r.status}",
                        "status": int(r.status),
                        "details": {"body": text[:500]},
                    },
                }
                if ws:
                    try:
                        await ws.send_json({"type": "error", "source": "comfy", "body": text[:500]})
                        await ws.close(code=1000)
                    except Exception:
                        pass
                return err
            parser = JSONParser()
            return parser.parse(text, {"prompt_id": str, "uuid": str, "id": str})


async def comfy_history(prompt_id: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.get(f"{BASE}/history/{prompt_id}") as r:
            text = await r.text()
            if r.status != 200:
                # Return a structured error instead of raising.
                return {
                    "ok": False,
                    "error": {
                        "code": "comfy_history_http_error",
                        "message": f"/history {r.status}",
                        "status": int(r.status),
                        "details": {"body": text[:500]},
                    },
                }
            parser = JSONParser()
            return parser.parse(text, {"history": dict})


async def comfy_upload_image(name_hint: str, b64_png: str) -> str:
    data = base64.b64decode(b64_png)
    filename = f"{name_hint or 'ref'}_{uuid.uuid4().hex[:8]}.png"
    form = aiohttp.FormData()
    form.add_field("image", data, filename=filename, content_type="image/png")
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.post(f"{BASE}/upload/image", data=form) as r:
            text = await r.text()
            if r.status != 200:
                # Return best-effort fallback name so callers can proceed.
                return filename
            parser = JSONParser()
            obj = parser.parse(text, {"name": str})
            stored = obj.get("name") or filename if isinstance(obj, dict) else filename
            return stored


async def comfy_upload_mask(name_hint: str, b64_png: str) -> str:
    data = base64.b64decode(b64_png)
    filename = f"{name_hint or 'mask'}_{uuid.uuid4().hex[:8]}.png"
    form = aiohttp.FormData()
    form.add_field("image", data, filename=filename, content_type="image/png")
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.post(f"{BASE}/upload/mask", data=form) as r:
            text = await r.text()
            if r.status != 200:
                # Return best-effort fallback name so callers can proceed.
                return filename
            parser = JSONParser()
            obj = parser.parse(text, {"name": str})
            return obj.get("name") or filename if isinstance(obj, dict) else filename


async def comfy_view(filename: str) -> Tuple[bytes, str]:
    # Returns (bytes, content_type)
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.get(f"{BASE}/view", params={"filename": filename}) as r:
            data = await r.read()
            if r.status != 200:
                # Return empty payload and generic content type instead of raising.
                return b"", r.headers.get("content-type", "application/octet-stream")
            return data, r.headers.get("content-type", "application/octet-stream")


async def comfy_object_info(session: aiohttp.ClientSession, node_class: str) -> Dict[str, Any]:
    async with session.get(f"{BASE}/object_info/{node_class}") as r:
        text = await r.text()
        if r.status != 200:
            return {}
        parser = JSONParser()
        obj = parser.parse(text, {"inputs": dict})
        return obj if isinstance(obj, dict) else {}


def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("+", "")


async def choose_sampler_name(session: aiohttp.ClientSession) -> str:
    info = await comfy_object_info(session, "KSampler")
    try:
        choices = ((info.get("inputs") or {}).get("sampler_name") or {}).get("choices") or []
        norm_map = {_norm(c): c for c in choices}
        for want in ("dpmpp2m", "dpmppsde", "euler", "eulera"):
            if want in norm_map:
                return norm_map[want]
        return choices[0] if choices else "euler"
    except Exception:
        return "euler"


