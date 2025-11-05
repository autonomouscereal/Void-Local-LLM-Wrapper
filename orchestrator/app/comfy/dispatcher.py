from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
import copy
from typing import Dict, Any, Optional

import httpx
from ..json_parser import JSONParser


COMFY_BASE = os.getenv("COMFYUI_API_URL", "http://comfyui:8188").rstrip("/")
COMFY_WS   = COMFY_BASE.replace("http", "ws") + "/ws?clientId=wrapper-001"
INPUT_DIR  = os.getenv("COMFY_INPUT_DIR", "/comfyui/input")
WF_PATH    = os.getenv("COMFY_WORKFLOW_PATH", "/workspace/services/image/workflows/pipeline_complex.json")


def load_workflow() -> Dict[str, Any]:
    text = Path(WF_PATH).read_text(encoding="utf-8")
    return JSONParser().parse(text, {})


def _patch_text_prompts(graph: Dict[str, Any], positive: Optional[str], negative: Optional[str]) -> None:
    if not positive and not negative:
        return
    for node in (graph.get("prompt") or {}).values():
        if node.get("class_type") == "CLIPTextEncode":
            if positive is not None:
                node.setdefault("inputs", {})["text"] = positive
                positive = None
            elif negative is not None:
                node.setdefault("inputs", {})["text"] = negative
                negative = None


def _patch_seeds(graph: Dict[str, Any], seed: Optional[int]) -> None:
    if seed is None:
        return
    for node in (graph.get("prompt") or {}).values():
        if node.get("class_type") in ("KSampler", "SamplerCustom", "KSamplerAdvanced"):
            if "seed" in (node.get("inputs") or {}):
                node["inputs"]["seed"] = int(seed)


def _patch_size(graph: Dict[str, Any], width: Optional[int], height: Optional[int]) -> None:
    for node in (graph.get("prompt") or {}).values():
        if node.get("class_type") in ("EmptyLatentImage", "ImageResize", "LatentUpscale"):
            inp = node.setdefault("inputs", {})
            if width is not None and "width" in inp:
                inp["width"] = int(width)
            if height is not None and "height" in inp:
                inp["height"] = int(height)


def _copy_into_input(path: str) -> Optional[str]:
    if not path:
        return None
    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        basename = os.path.basename(path)
        dst = os.path.join(INPUT_DIR, basename)
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copyfile(path, dst)
        return basename  # Comfy nodes expect just the filename
    except Exception:
        return None


def patch_assets(graph: Dict[str, Any], assets: Dict[str, Any]) -> None:
    for node_id, node in (graph.get("prompt") or {}).items():
        ct = node.get("class_type")
        if ct in ("LoadImage", "LoadImageMask"):
            if "image" in (node.get("inputs") or {}):
                ref = assets.get("reference") or assets.get("image")
                b = _copy_into_input(ref)
                if b:
                    node["inputs"]["image"] = b
        if ct in ("ControlNetApply", "ControlLoraApply", "ControlNetLoader"):
            img = assets.get("control_canny") or assets.get("control_depth") or assets.get("control_normal")
            if img and "image" in (node.get("inputs") or {}):
                b = _copy_into_input(img)
                if b:
                    node["inputs"]["image"] = b
        if ct and "InstantID" in ct:
            face = assets.get("instantid")
            if face and "image" in (node.get("inputs") or {}):
                b = _copy_into_input(face)
                if b:
                    node["inputs"]["image"] = b
        if ct and ct.startswith("IPAdapter"):
            w = assets.get("ip_adapter_weight")
            if w is not None and "weight" in (node.get("inputs") or {}):
                node["inputs"]["weight"] = float(w)
            ref = assets.get("reference")
            if ref and "image" in (node.get("inputs") or {}):
                b = _copy_into_input(ref)
                if b:
                    node["inputs"]["image"] = b


async def submit(graph: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"prompt": graph, "client_id": "wrapper-001"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(COMFY_BASE + "/api/prompt", json=payload)
        r.raise_for_status()
        j = JSONParser().parse(r.text, {"prompt_id": (str), "uuid": (str), "id": (str)})
        pid = j.get("prompt_id") or j.get("uuid") or j.get("id")
        if not isinstance(pid, str):
            return {"error": "missing prompt_id from comfy"}
    # WS wait
    import websockets  # type: ignore
    async with websockets.connect(COMFY_WS, ping_interval=None) as ws:
        while True:
            msg = await ws.recv()
            d = JSONParser().parse(msg, {"type": (str), "data": dict}) if isinstance(msg, (str, bytes)) else {}
            if isinstance(d, dict) and d.get("type") == "executed" and (d.get("data") or {}).get("prompt_id") == pid:
                break
    # History
    async with httpx.AsyncClient(timeout=60) as client:
        hr = await client.get(COMFY_BASE + f"/api/history/{pid}")
        if hr.status_code == 200:
            return {"prompt_id": pid, "history": JSONParser().parse(hr.text, {})}
        return {"prompt_id": pid, "history_error": hr.text}


def patch_workflow(base_graph: Dict[str, Any], *, prompt: Optional[str] = None, negative: Optional[str] = None, seed: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None, assets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    g = copy.deepcopy(base_graph)
    _patch_text_prompts(g, prompt, negative)
    _patch_seeds(g, seed)
    _patch_size(g, width, height)
    # Remap any RealESRGAN nodes in incoming workflows to ComfyUI core upscaler nodes
    for node in (g.get("prompt") or {}).values():
        ct = node.get("class_type")
        if ct == "RealESRGANModelLoader":
            node["class_type"] = "UpscaleModelLoader"
            inp = node.setdefault("inputs", {})
            if "model_name" not in inp:
                # Use configured default if not present
                inp["model_name"] = os.getenv("COMFY_UPSCALE_MODEL", "4x-UltraSharp.pth")
        if ct == "RealESRGAN":
            node["class_type"] = "ImageUpscaleWithModel"
            inp = node.setdefault("inputs", {})
            # Map 'model' -> 'upscale_model' if present
            if "model" in inp and "upscale_model" not in inp:
                inp["upscale_model"] = inp.pop("model")
            # 'image' key is already correct for core node
            # Remove obsolete 'scale' if present; core node doesn't need it
            if "scale" in inp:
                inp.pop("scale", None)
    if assets:
        patch_assets(g, assets)
    return g


