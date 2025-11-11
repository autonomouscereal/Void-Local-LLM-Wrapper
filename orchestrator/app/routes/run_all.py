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
from fastapi.responses import JSONResponse

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


def ok_envelope(result: Any, rid: str) -> JSONResponse:
    return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


def jerr(status: int, rid: str, code: str, msg: str, details: Any | None = None) -> JSONResponse:
    return JSONResponse({"schema_version": 1, "request_id": rid, "ok": False, "error": {"code": code, "message": msg, "details": details}}, status_code=status)


@router.get("/jobs")
async def jobs(req: Request):
    app = req.app
    jobs = getattr(app.state, "jobs", {})
    items: List[Dict[str, Any]] = []
    if isinstance(jobs, dict):
        for rid, info in jobs.items():
            if isinstance(info, dict):
                items.append({"request_id": rid, "status": info.get("status"), "trace_id": info.get("trace_id")})
    return ok_envelope({"jobs": items}, rid="jobs")


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


@router.post("/v1/run")
async def run_all(req: Request):
    rid = str(uuid.uuid4())
    tid = uuid.uuid4().hex
    raw = await req.body()
    parser = JSONParser()
    ok_strict, body = parser.parse_strict(raw.decode("utf-8", "replace"))
    if not ok_strict or not isinstance(body, dict):
        _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid, "errors.jsonl"), {
            "t": int(time.time()*1000), "event": "json_parse", "trace_id": tid, "rid": rid,
            "raw_len": len(raw or b""),
            "errors": (parser.errors[:5] if isinstance(parser.errors, list) else []),
            "repairs": (parser.repairs[:5] if isinstance(parser.repairs, list) else []),
        })
        return jerr(400, rid, "parse_error", "Invalid JSON")
    # Normalize request id if provided
    rid = str(body.get("request_id") or rid)
    user_text = body.get("text") or ""
    if not isinstance(user_text, str):
        return jerr(422, rid, "invalid_text", "text must be string")
    # Accept either explicit plan or steps; otherwise build via committee
    plan: Dict[str, Any] | None = None
    req_plan = body.get("plan")
    req_steps = body.get("steps")
    if isinstance(req_plan, dict):
        if isinstance(req_plan.get("steps"), list):
            plan = req_plan
        elif isinstance(req_plan.get("plan"), list):
            plan = {"request_id": req_plan.get("request_id") or rid, "steps": req_plan.get("plan")}
        else:
            plan = {"request_id": req_plan.get("request_id") or rid, "steps": []}
    elif isinstance(req_steps, list):
        plan = {"request_id": rid, "steps": req_steps}
    else:
        plan = await make_full_plan(user_text)
        # normalize to steps key if planner returned legacy shape
        if isinstance(plan, dict) and isinstance(plan.get("plan"), list) and not isinstance(plan.get("steps"), list):
            plan = {"request_id": plan.get("request_id") or rid, "steps": plan.get("plan")}
        errs = validate_plan(plan)
        if errs:
            # Auto re-ask once, then proceed with softened plan even if still imperfect
            plan = await make_full_plan(user_text)
            if isinstance(plan, dict) and isinstance(plan.get("plan"), list) and not isinstance(plan.get("steps"), list):
                plan = {"request_id": plan.get("request_id") or rid, "steps": plan.get("plan")}
            errs = validate_plan(plan)
        if errs:
            plan = _soften_plan(plan, user_text)
    # For provided plans, validate and soften if needed
    errs = validate_plan(plan)
    if errs:
        plan = _soften_plan(plan, user_text)
    app = req.app
    if not hasattr(app.state, "jobs"):
        app.state.jobs = {}
    if not hasattr(app.state, "ws_clients"):
        app.state.ws_clients = {}
    app.state.jobs[rid] = {"status": "queued", "plan": plan, "result": None, "error": None, "trace_id": tid}
    # Bind what the runner needs
    steps: List[Dict[str, Any]] = list(plan.get("steps") or [])
    steps_len: int = len(steps)
    # Emit job_start and request snapshot now
    _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid, "events.jsonl"), {"t": int(time.time()*1000), "event": "job_start", "rid": rid, "trace_id": tid, "steps": steps_len})
    _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid, "requests.jsonl"), {"t": int(time.time()*1000), "trace_id": tid, "plan": {"request_id": rid, "steps": steps}})
    # Review config (defaults: enabled true, 1 iteration)
    review_cfg = (body.get("review") or {}) if isinstance(body.get("review"), dict) else {}
    review_enabled = bool(review_cfg.get("enabled", True))
    max_iters = int(review_cfg.get("max_iterations", 1) or 1)
    async def _runner(rid_: str, tid_: str, steps0: List[Dict[str, Any]], user_text_: str, review_enabled_: bool, max_iters_: int):
        ws = app.state.ws_clients.get(rid_)
        app.state.jobs[rid_]["status"] = "running"
        if ws:
            await ws.send_json({"type": "status", "status": "running"})
        produced_map: Dict[str, Any] = {}
        steps_local: List[Dict[str, Any]] = list(steps0 or [])
        for iter_idx in range(max(1, max_iters_)):
            # Execute current steps
            # Log exec_post event for visibility
            _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid_, "events.jsonl"), {"t": int(time.time()*1000), "event": "exec_post", "url": EXECUTE_URL, "rid": rid_, "trace_id": tid_})
            try:
                res = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _post_json(EXECUTE_URL, {"schema_version": 1, "request_id": rid_, "trace_id": tid_, "steps": steps_local}),
                )
            except Exception as e:
                res = {"schema_version": 1, "ok": False, "code": "executor_http_error", "message": str(e), "result": []}
            # On schema/tool errors: re-plan once with softening and retry
            if (not isinstance(res, dict)) or (res.get("error") and not res.get("produced")):
                if ws:
                    await ws.send_json({"type": "progress", "event": "repair_attempt", "note": "retrying with corrected plan"})
                _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid_, "events.jsonl"), {"t": int(time.time()*1000), "event": "repair_attempt", "rid": rid_, "trace_id": tid_})
                plan2 = _soften_plan(await make_full_plan(user_text_), user_text_)
                if isinstance(plan2, dict) and isinstance(plan2.get("plan"), list) and not isinstance(plan2.get("steps"), list):
                    plan2 = {"request_id": plan2.get("request_id") or rid_, "steps": plan2.get("plan")}
                res2 = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _post_json(EXECUTE_URL, {"schema_version": 1, "request_id": rid_, "trace_id": tid_, "steps": (plan2.get("steps") or [])}),
                )
                if not isinstance(res2, dict) or (res2.get("error") and not res2.get("produced")):
                    return
                res = res2
                steps_local = list((plan2.get("steps") or []))
            produced_map = (res or {}).get("produced") if isinstance(res, dict) else {}
            # Always emit terminal result to WS/UI for this iteration
            if ws:
                try:
                    await ws.send_json({"type": "tool.result", "ok": bool((res or {}).get("ok", False)), "result": res})
                except Exception:
                    pass
            produced_map = _canonicalize_produced(produced_map)
            # Emit a chat.append preview for this iteration
            if ws:
                parts = []
                for _, step_result in (produced_map or {}).items():
                    if isinstance(step_result, dict):
                        parts.extend(_to_artifact_parts(step_result))
                await ws.send_json({"type": "chat.append", "message": {"role": "assistant", "parts": parts}})
            # Record produced map for this iteration (strip data_url to keep traces clean)
            rec = {"produced": (produced_map or {})}
            _strip_data_urls(rec)
            _strip_stacks(rec)
            _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid_, "responses.jsonl"), {"t": int(time.time()*1000), "trace_id": tid_, **rec, "iter": iter_idx})
            # Decide whether to review and iterate
            if not review_enabled_ or (iter_idx + 1) >= max(1, max_iters_):
                break
            if ws:
                await ws.send_json({"type": "review.start", "iter": iter_idx + 1})
            delta = _build_delta_plan({"produced": produced_map, "text": user_text_})
            if ws:
                await ws.send_json({"type": "review.decision", "iter": iter_idx + 1, "accepted": bool(not delta), "delta": (delta or {})})
            if not delta:
                break
            # Apply delta to produce new steps (no plan usage)
            next_steps = (delta.get("steps") or delta.get("plan") or steps_local or [])
            # remap needs to current identifiers (prefer explicit id else name)
            def _id_or_name(s: Dict[str, Any]) -> str:
                sid = s.get("id")
                return sid if isinstance(sid, str) and sid else str(s.get("name") or "")
            idx = { _id_or_name(s): _id_or_name(s) for s in next_steps if isinstance(s, dict) }
            for s in next_steps:
                if not isinstance(s, dict):
                    continue
                needs = s.get("needs") or []
                if isinstance(needs, list):
                    s["needs"] = [ idx.get(n, n) for n in needs ]
            steps_local = list(next_steps)
            _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid_, "events.jsonl"), {"t": int(time.time()*1000), "event": "edit.plan", "rid": rid_, "trace_id": tid_, "iter": iter_idx + 1})
        app.state.jobs[rid_]["result"] = produced_map or {}
        _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid_, "events.jsonl"), {"t": int(time.time()*1000), "event": "job_finish", "rid": rid_, "trace_id": tid_, "produced_keys": sorted(list((produced_map or {}).keys()))})
        app.state.jobs[rid_]["status"] = "done"
        if ws:
            final_res = {"result": (produced_map or {})}
            _strip_data_urls(final_res)
            _strip_stacks(final_res)
            ok_flag = True
            try:
                ok_flag = bool(final_res.get("result"))
            except Exception:
                ok_flag = False
            await ws.send_json({"type": "tool.result", "ok": ok_flag, "result": {"produced": (produced_map or {})}})
            # Compatibility terminal events for older UIs
            for tname in ("run.complete", "run.completed", "final", "done"):
                await ws.send_json({"type": tname, "ok": ok_flag})
            await ws.send_json({"type": "done", **final_res})
            await ws.close(code=1000)
    asyncio.create_task(_runner(rid, tid, steps, user_text, review_enabled, max_iters))
    return ok_envelope({"request_id": rid, "plan": plan}, rid)


@router.websocket("/ws.run")
async def ws_run(websocket: WebSocket):
    await websocket.accept(subprotocol=websocket.headers.get("sec-websocket-protocol"))
    qs = websocket.query_params
    rid_qs = (qs.get("rid") or "").strip()
    rid = rid_qs
    app = websocket.app
    if not hasattr(app.state, "ws_clients"):
        app.state.ws_clients = {}
    app.state.ws_clients[rid] = websocket
    # Compute or fetch the unique trace id for this run; never fall back to rid
    tid = None
    if hasattr(app.state, "jobs") and isinstance(app.state.jobs.get(rid), dict):
        tid = app.state.jobs[rid].get("trace_id")
    if not tid:
        tid = uuid.uuid4().hex
        if not hasattr(app.state, "jobs"):
            app.state.jobs = {}
        app.state.jobs[rid] = {"trace_id": tid}
    await websocket.send_json({"type": "session/ready", "trace_id": tid})
    # First message must be valid JSON (strict); mirror HTTP edge behavior
    try:
        msg0 = await websocket.receive_text()
    except Exception:
        await websocket.send_json({"type": "error", "error": {"code": "parse_error", "message": "Invalid JSON"}})
        await websocket.close(code=1000)
        return
    parser = JSONParser()
    ok0, body0 = parser.parse_strict(msg0 or "")
    if not ok0 or not isinstance(body0, dict):
        _append_jsonl(os.path.join(STATE_DIR_LOCAL, "traces", tid, "errors.jsonl"), {
            "t": int(time.time()*1000), "event": "json_parse", "trace_id": tid, "rid": (rid_qs or None),
            "raw_len": len(msg0 or ""),
            "errors": (parser.errors[:5] if isinstance(parser.errors, list) else []),
            "repairs": (parser.repairs[:5] if isinstance(parser.repairs, list) else []),
        })
        await websocket.send_json({"type": "error", "error": {"code": "parse_error", "message": "Invalid JSON"}})
        await websocket.close(code=1000)
        return
    # Normalize to unified plan/steps as in /v1/run
    rid_norm = str((body0.get("request_id") or rid_qs or uuid.uuid4().hex))
    plan0: Dict[str, Any] | None = None
    if isinstance(body0.get("plan"), dict):
        p = body0["plan"]
        if isinstance(p.get("steps"), list):
            plan0 = p
        else:
            plan0 = {"request_id": p.get("request_id") or rid_norm, "steps": (p.get("plan") or [])}
    elif isinstance(body0.get("steps"), list):
        plan0 = {"request_id": rid_norm, "steps": body0["steps"]}
    else:
        plan0 = {"request_id": rid_norm, "steps": []}
    if (not isinstance(plan0.get("steps"), list)) or (len(plan0.get("steps") or []) == 0):
        await websocket.send_json({"type": "error", "error": {"code": "invalid_plan", "message": "plan.steps required"}})
        await websocket.close(code=1000)
        return
    # Keep open until server closes from runner; echo subsequent messages for compatibility
    while True:
        msg = await websocket.receive_text()
        await websocket.send_json({"type": "ack", "echo": (msg[:100] if msg else "")})


