from __future__ import annotations

import os
import json
import uuid
import base64
from typing import Dict, Any, Optional, Tuple
import logging

import aiohttp  # type: ignore
from ..json_parser import JSONParser


BASE = os.getenv("COMFYUI_API_URL", "http://comfyui:8188").rstrip("/")


async def comfy_submit(graph: Dict[str, Any], client_id: Optional[str] = None) -> Dict[str, Any]:
    cid = client_id or str(uuid.uuid4())
    payload = {"prompt": graph.get("prompt") or graph, "client_id": cid}
    # Validate/shape payload with project JSON parser
    shaped = JSONParser().ensure_structure(payload, {"prompt": dict, "client_id": str})
    body_text = json.dumps(shaped, ensure_ascii=False, separators=(",", ":"))
    logging.info(f"[comfy.submit] bytes={len(body_text.encode('utf-8'))}")
    async with aiohttp.ClientSession() as s:
        async with s.post(
            f"{BASE}/prompt",
            data=body_text.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ) as r:
            text = await r.text()
            if r.status != 200:
                raise RuntimeError(f"/prompt {r.status}: {text[:500]}")
            # Parse response via JSONParser for safety
            return JSONParser().parse(text, {"prompt_id": str, "history": dict, "node_errors": dict})


async def comfy_history(prompt_id: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE}/history/{prompt_id}") as r:
            text = await r.text()
            if r.status != 200:
                raise RuntimeError(f"/history {r.status}: {text[:500]}")
            return JSONParser().parse(text, {prompt_id: dict})


async def comfy_upload_image(name_hint: str, b64_png: str) -> str:
    data = base64.b64decode(b64_png)
    filename = f"{name_hint or 'ref'}_{uuid.uuid4().hex[:8]}.png"
    form = aiohttp.FormData()
    form.add_field("image", data, filename=filename, content_type="image/png")
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE}/upload/image", data=form) as r:
            text = await r.text()
            if r.status != 200:
                raise RuntimeError(f"/upload/image {r.status}: {text[:500]}")
            obj = JSONParser().parse(text, {"name": str})
            stored = obj.get("name") or filename
            return stored


async def comfy_upload_mask(name_hint: str, b64_png: str) -> str:
    data = base64.b64decode(b64_png)
    filename = f"{name_hint or 'mask'}_{uuid.uuid4().hex[:8]}.png"
    form = aiohttp.FormData()
    form.add_field("image", data, filename=filename, content_type="image/png")
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE}/upload/mask", data=form) as r:
            text = await r.text()
            if r.status != 200:
                raise RuntimeError(f"/upload/mask {r.status}: {text[:500]}")
            obj = JSONParser().parse(text, {"name": str})
            return obj.get("name") or filename


async def comfy_view(filename: str) -> Tuple[bytes, str]:
    # Returns (bytes, content_type)
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE}/view", params={"filename": filename}) as r:
            data = await r.read()
            if r.status != 200:
                body = (await r.text()) if data else ""
                raise RuntimeError(f"/view {r.status}: {body[:500]}")
            return data, r.headers.get("content-type", "application/octet-stream")


