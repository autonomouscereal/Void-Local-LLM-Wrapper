from __future__ import annotations

import asyncio
import os
import json
import urllib.request
import urllib.error
import traceback
import logging
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Request, WebSocket
from app.routes.toolrun import ToolEnvelope  # canonical envelope

from ..json_parser import JSONParser
from ..plan.committee import make_full_plan
from ..plan.validator import validate_plan
from ..review.referee import build_delta_plan as _build_delta_plan
EXECUTOR_BASE_URL = os.getenv("EXECUTOR_BASE_URL", "http://127.0.0.1:8081")
EXECUTE_URL = EXECUTOR_BASE_URL.rstrip("/") + "/execute"

STATE_DIR_LOCAL = os.path.join(os.getenv("UPLOAD_DIR", "/workspace/uploads"), "state")
from ..state.checkpoints import append_ndjson as _append_jsonl
import time


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        # Strict parse with expected structure only (no std json.loads fallback)
        expected = {"produced": dict, "error": str, "detail": str, "traceback": str}
        return JSONParser().parse(raw, expected)



def _soften_plan(plan: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    """Make inputs more forgiving by filling common defaults so execution can proceed."""
    steps = plan.get("steps") or plan.get("plan") or []
    out = {"request_id": plan.get("request_id"), "steps": []}
    for s in steps:
        if not isinstance(s, dict):
            continue
        t = (s.get("tool") or "").strip()
        inputs = dict(s.get("inputs") or {})
        if t == "image.dispatch":
            if not isinstance(inputs.get("prompt"), str) or not inputs.get("prompt"):
                inputs["prompt"] = user_text
            if not isinstance(inputs.get("size"), str) or not inputs.get("size"):
                inputs["size"] = "1024x1024"
        if t == "audio.music.generate":
            if not isinstance(inputs.get("style"), str) or not inputs.get("style"):
                inputs["style"] = "auto"
            if not isinstance(inputs.get("duration_s"), (int, float)):
                inputs["duration_s"] = 15
        s2 = dict(s)
        s2["inputs"] = inputs
        out["steps"].append(s2)
    return out



router = APIRouter()


@router.get("/jobs")
async def jobs(req: Request):
    app = req.app
    jobs = getattr(app.state, "jobs", {})
    items: List[Dict[str, Any]] = []
    if isinstance(jobs, dict):
        for rid, info in jobs.items():
            if isinstance(info, dict):
                items.append({"request_id": rid, "status": info.get("status"), "trace_id": info.get("trace_id")})
    return ToolEnvelope.success({"jobs": items}, request_id="jobs")


def _strip_data_urls(obj: Any) -> None:
    if isinstance(obj, dict):
        if "data_url" in obj:
            obj.pop("data_url", None)
        if "poster_data_url" in obj:
            obj.pop("poster_data_url", None)
        for v in obj.values():
            _strip_data_urls(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_data_urls(v)


def _strip_stacks(obj: Any) -> None:
    if isinstance(obj, dict):
        if "traceback" in obj:
            obj.pop("traceback", None)
        for v in obj.values():
            _strip_stacks(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_stacks(v)


def _to_artifact_parts(step_result: Dict[str, Any]) -> list[Dict[str, Any]]:
    # Canonical reader: step_result must contain {"result": {"ids":{}, "meta":{}}}
    base = (step_result or {}).get("result") if isinstance(step_result, dict) else None
    ids = (base or {}).get("ids") or {}
    meta = (base or {}).get("meta") or {}
    out: list[Dict[str, Any]] = []
    # image
    if ids.get("image_id") and (meta.get("data_url") or meta.get("orch_view_url") or meta.get("view_url")):
        if isinstance(meta.get("data_url"), str):
            out.append({"kind": "image", "preview": {"data_url": meta["data_url"]}, "full": {"url": meta.get("orch_view_url") or meta.get("view_url"), "filename": meta.get("filename")}})
        else:
            out.append({"kind": "image", "preview": {"url": meta.get("orch_view_url") or meta.get("view_url")}, "full": {"url": meta.get("orch_view_url") or meta.get("view_url"), "filename": meta.get("filename")}})
    # audio
    if ids.get("audio_id") and (meta.get("data_url") or meta.get("url")):
        p: Dict[str, Any] = {"kind": "audio", "preview": {}, "full": {"url": meta.get("url"), "mime": meta.get("mime", "audio/wav")}}
        if isinstance(meta.get("data_url"), str):
            p["preview"]["data_url"] = meta["data_url"]
        if meta.get("duration_s") is not None:
            p["full"]["duration_sec"] = meta.get("duration_s")
        out.append(p)
    # video
    if ids.get("video_id") and meta.get("view_url"):
        p: Dict[str, Any] = {"kind": "video", "preview": {}, "full": {"url": meta.get("view_url"), "mime": meta.get("mime", "video/mp4")}}
        if isinstance(meta.get("poster_data_url"), str):
            p["preview"]["poster_data_url"] = meta.get("poster_data_url")
        out.append(p)
    return out


def _canonical_step_result(step_name: str, step_result: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce any legacy flat step outputs into canonical nested shape.
    Canonical: {"name": step_name, "result": {"ids": {...}, "meta": {...}}}
    """
    if isinstance(step_result, dict) and isinstance(step_result.get("result"), dict):
        res = step_result["result"]
        ids = res.get("ids") or {}
        meta = res.get("meta") or {}
        return {"name": step_name, "result": {"ids": ids, "meta": meta}}
    ids = step_result.get("ids") if isinstance(step_result, dict) else {}
    meta = step_result.get("meta") if isinstance(step_result, dict) else {}
    if not isinstance(ids, dict):
        ids = {}
    if not isinstance(meta, dict):
        meta = {}
    return {"name": step_name, "result": {"ids": ids, "meta": meta}}


def _canonicalize_produced(produced_map: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sid, val in (produced_map or {}).items():
        out[str(sid)] = _canonical_step_result(str((val or {}).get("name") or ""), val if isinstance(val, dict) else {})
    return out








