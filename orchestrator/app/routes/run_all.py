from __future__ import annotations

import asyncio
import os
import json
import urllib.request
import urllib.error
import traceback
import logging
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse

from ..json_parser import JSONParser
from ..plan.committee import make_full_plan
from ..plan.validator import validate_plan
from ..review.referee import build_delta_plan as _build_delta_plan
EXECUTOR_BASE_URL = os.getenv("EXECUTOR_BASE_URL", "http://executor:8081")
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
    steps = plan.get("plan") or []
    out = {"request_id": plan.get("request_id"), "plan": []}
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
        out["plan"].append(s2)
    return out



router = APIRouter()


def ok(result: Any, rid: str) -> JSONResponse:
    return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


def jerr(status: int, rid: str, code: str, msg: str, details: Any | None = None) -> JSONResponse:
    return JSONResponse({"schema_version": 1, "request_id": rid, "ok": False, "error": {"code": code, "message": msg, "details": details}}, status_code=status)


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
    ids = (step_result or {}).get("ids") or {}
    meta = (step_result or {}).get("meta") or {}
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


@router.post("/v1/run")
async def run_all(req: Request):
    rid = str(uuid.uuid4())
    raw = await req.body()
    try:
        body = JSONParser().parse(raw.decode("utf-8", errors="replace"), {})
    except Exception:
        return jerr(400, rid, "invalid_json", "Body must be valid JSON")
    if not isinstance(body, dict):
        return jerr(422, rid, "invalid_body", "Body must be JSON object")
    user_text = body.get("text") or ""
    if not isinstance(user_text, str):
        return jerr(422, rid, "invalid:text", "text must be string")
    # First plan
    plan = await make_full_plan(user_text)
    errs = validate_plan(plan)
    if errs:
        # Auto re-ask once, then proceed with softened plan even if still imperfect
        plan = await make_full_plan(user_text)
        errs = validate_plan(plan)
    if errs:
        plan = _soften_plan(plan, user_text)
    app = req.app
    if not hasattr(app.state, "jobs"):
        app.state.jobs = {}
    if not hasattr(app.state, "ws_clients"):
        app.state.ws_clients = {}
    app.state.jobs[rid] = {"status": "queued", "plan": plan, "result": None, "error": None}
    # Review config (defaults: enabled true, 1 iteration)
    review_cfg = (body.get("review") or {}) if isinstance(body.get("review"), dict) else {}
    review_enabled = bool(review_cfg.get("enabled", True))
    max_iters = int(review_cfg.get("max_iterations", 1) or 1)
    async def _runner():
        ws = app.state.ws_clients.get(rid)
        try:
            app.state.jobs[rid]["status"] = "running"
            if ws: await ws.send_json({"type": "status", "status": "running"})
            # Distilled job start
            _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", rid, "events.jsonl"), {"t": int(time.time()*1000), "event": "job_start", "rid": rid, "steps": len(plan.get("plan") or [])})
            # Trace: record plan submission
            _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", rid, "requests.jsonl"), {"t": int(time.time()*1000), "trace_id": rid, "plan": plan})
            url = EXECUTOR_BASE_URL.rstrip("/") + "/execute_plan"
            produced_map: Dict[str, Any] = {}
            for iter_idx in range(max(1, max_iters)):
                # Execute current plan
                res = await asyncio.get_event_loop().run_in_executor(None, lambda: _post_json(url, {"plan": plan}))
                # On schema/tool errors: re-plan once with softening and retry
                if (not isinstance(res, dict)) or (res.get("error") and not res.get("produced")):
                    if ws:
                        await ws.send_json({"type": "progress", "event": "repair_attempt", "note": "retrying with corrected plan"})
                    _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", rid, "events.jsonl"), {"t": int(time.time()*1000), "event": "repair_attempt", "rid": rid})
                    plan2 = _soften_plan(await make_full_plan(user_text), user_text)
                    res2 = await asyncio.get_event_loop().run_in_executor(None, lambda: _post_json(url, {"plan": plan2}))
                    if not isinstance(res2, dict) or (res2.get("error") and not res2.get("produced")):
                        raise RuntimeError(f"executor_error:{(res2 or res or {})}")
                    res = res2
                produced_map = (res or {}).get("produced") if isinstance(res, dict) else {}
                # Emit a chat.append preview for this iteration
                if ws:
                    parts = []
                    # Use normalizer to build chat parts
                    for step_name, step_result in (produced_map or {}).items():
                        if isinstance(step_result, dict):
                            parts.extend(_to_artifact_parts(step_result))
                    await ws.send_json({"type": "chat.append", "message": {"role": "assistant", "parts": parts}})
                # Record produced map for this iteration (strip data_url to keep traces clean)
                rec = {"produced": (produced_map or {})}
                _strip_data_urls(rec)
                _strip_stacks(rec)
                _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", rid, "responses.jsonl"), {"t": int(time.time()*1000), "trace_id": rid, **rec, "iter": iter_idx})
                # Decide whether to review and iterate
                if not review_enabled or (iter_idx + 1) >= max(1, max_iters):
                    break
                try:
                    if ws:
                        await ws.send_json({"type": "review.start", "iter": iter_idx + 1})
                    delta = _build_delta_plan({"produced": produced_map, "text": user_text})
                    if ws:
                        await ws.send_json({"type": "review.decision", "iter": iter_idx + 1, "accepted": bool(not delta), "delta": (delta or {})})
                    if not delta:
                        break
                    # Apply delta to produce a new plan
                    plan = {"request_id": plan.get("request_id") or rid, "plan": (delta.get("plan") or plan.get("plan") or [])}
                    _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", rid, "events.jsonl"), {"t": int(time.time()*1000), "event": "edit.plan", "rid": rid, "iter": iter_idx + 1})
                except Exception:
                    # On review failure, exit loop and finalize current results
                    break
            app.state.jobs[rid]["result"] = produced_map or {}
            _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", rid, "events.jsonl"), {"t": int(time.time()*1000), "event": "job_finish", "rid": rid, "produced_keys": sorted(list((produced_map or {}).keys()))})
            app.state.jobs[rid]["status"] = "done"
            if ws:
                # Final result to UI: strip any inline data to keep payload lean
                final_res = {"result": (produced_map or {})}
                _strip_data_urls(final_res)
                _strip_stacks(final_res)
                await ws.send_json({"type": "done", **final_res})
                try: await ws.close(code=1000)
                except Exception as _e:
                    _append_jsonl(os.path.join(STATE_DIR_LOCAL, "ws", "errors.jsonl"), {"t": int(time.time()*1000), "rid": rid, "where": "run_all.done", "error": str(_e)})
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(tb)
            # Route error with stack to errors corpus (keep primary traces clean)
            try:
                _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", rid, "errors.jsonl"), {
                    "t": int(time.time()*1000),
                    "error": {"message": str(e), "traceback": tb}
                })
            except Exception:
                logging.error(traceback.format_exc())
            app.state.jobs[rid]["error"] = str(e)
            app.state.jobs[rid]["status"] = "error"
            if ws:
                await ws.send_json({"type": "error", "message": str(e)[:500], "traceback": tb})
                try: await ws.close(code=1000)
                except Exception as _e:
                    _append_jsonl(os.path.join(STATE_DIR_LOCAL, "ws", "errors.jsonl"), {"t": int(time.time()*1000), "rid": rid, "where": "run_all.error", "error": str(_e)})
    asyncio.create_task(_runner())
    return ok({"request_id": rid, "plan": plan}, rid)


@router.websocket("/ws.run")
async def ws_run(websocket: WebSocket):
    await websocket.accept(subprotocol=websocket.headers.get("sec-websocket-protocol"))
    qs = websocket.query_params
    rid = qs.get("rid") or ""
    app = websocket.app
    if not hasattr(app.state, "ws_clients"):
        app.state.ws_clients = {}
    app.state.ws_clients[rid] = websocket
    try:
        await websocket.send_json({"type": "session", "rid": rid})
        # Keep open until server closes from runner
        while True:
            msg = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "echo": (msg[:100] if msg else "")})
    except Exception as e:
        tb = traceback.format_exc()
        try:
            await websocket.close(code=1000)
        except Exception as _e:
            _append_jsonl(os.path.join(STATE_DIR_LOCAL, "ws", "errors.jsonl"), {"t": int(time.time()*1000), "rid": rid, "where": "ws.run.close", "error": str(_e)})

