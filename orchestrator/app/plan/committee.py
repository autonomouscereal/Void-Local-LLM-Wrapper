from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from ..pipeline.compression_orchestrator import co_pack, frames_to_string
from ..committee_client import committee_ai_text, committee_jsonify
from .catalog import PLANNER_VISIBLE_TOOLS
from ..tools_schema import get_builtin_tools_schema

log = logging.getLogger("orchestrator.plan.committee")

SYSTEM_IMAGE = (
    "You are ImageOps. Output ONLY JSON per the schema. "
    "Always think in multi-step plans: start from the user goal, then decompose into small steps with clear purposes and dependencies. "
    "If an image is requested, include image.dispatch and sub-steps for pose/edge/depth/style locks when relevant. "
    "Respect existing lock bundles: call locks.get_bundle, locks.build_region_locks, and locks.update_region_modes to control which regions stay hard-locked versus flexible. "
    "Prefer using existing artifacts and locks over re-generating from scratch when possible."
)
SYSTEM_VIDEO = (
    "You are VideoOps. Output ONLY JSON per the schema. "
    "Always think in multi-step plans: include preparation/analysis steps (locks, references) before heavy generation steps. "
    "If any visual content is requested, include t2v/i2v plus stabilize and video.upscale where relevant. "
    "Use image.dispatch hero frames to seed Film-2 and keep character, region, and scene locks consistent across shots. "
    "Prefer patching or refining existing shots over re-running full generations when QA or committee metrics suggest minor issues."
)
SYSTEM_AUDIO = (
    "You are AudioOps. Output ONLY JSON per the schema. "
    "Always think in multi-step plans: separate composition, vocal generation, mastering, and lock adjustments into distinct steps with clear purposes. "
    "If any music is implied, you MUST use the single front-door music tool 'music.infinite.windowed' for composition and windowed scoring. "
    "Do not invent or call other music tools such as 'audio.music.generate' or 'search'; they are invalid. "
    "For vocals, use tts.speak, and for audio mastering/mixdown use audio.master or music.mixdown when available in the tool catalog. "
    "Manage lock bundles via locks.get_bundle, locks.update_audio_modes, and locks.update_region_modes (for visuals linked to audio). "
    "Keep tempo/key/stem/lyrics/voice locks aligned with the user directive and with any committee/QA metrics from prior runs."
)


def _tool_schema_by_name(name: str) -> Dict[str, Any]:
    """
    Extract the OpenAI-style JSON schema parameters block for a built-in tool.
    Returns {} when the tool isn't found or has no parameters block.
    """
    nm = str(name or "").strip()
    if not nm:
        return {}
    try:
        tools_schema = get_builtin_tools_schema() or []
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("planner _tool_schema_by_name get_builtin_tools_schema failed name=%r: %s", nm, exc, exc_info=True)
        return {}
    for t in tools_schema:
        try:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if isinstance(t.get("function"), dict) else {}
            tool_name = fn.get("name") if isinstance(fn.get("name"), str) else None
            if tool_name == nm:
                params = fn.get("parameters")
                return params if isinstance(params, dict) else {}
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("planner _tool_schema_by_name iterate failed name=%r: %s", nm, exc, exc_info=True)
            continue
    return {}


def _safe_message_list(messages: Any, *, trace_id: str) -> List[Dict[str, Any]]:
    if messages is None:
        return []
    if not isinstance(messages, list):
        log.warning("planner messages not list trace_id=%s type=%s", trace_id, type(messages).__name__)
        messages = [messages]
    out: List[Dict[str, Any]] = []
    dropped = 0
    for m in messages or []:
        if not isinstance(m, dict):
            dropped += 1
            continue
        role = m.get("role")
        content = m.get("content")
        if not isinstance(role, str) or not role.strip():
            dropped += 1
            continue
        # Content can be non-string in some adapter paths; coerce safely.
        if content is not None and not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                content = repr(content)
        out.append({"role": role.strip(), "content": (content or "")})
    if dropped:
        log.warning("planner messages dropped=%s kept=%s trace_id=%s", dropped, len(out), trace_id)
    return out


def _safe_tools_list(tools: Any, *, trace_id: str) -> Optional[List[Dict[str, Any]]]:
    if tools is None:
        return None
    if not isinstance(tools, list):
        log.warning("planner tools not list trace_id=%s type=%s", trace_id, type(tools).__name__)
        return None
    out: List[Dict[str, Any]] = []
    bad = 0
    for t in tools:
        if isinstance(t, dict):
            out.append(t)
        else:
            bad += 1
    if bad:
        log.warning("planner tools dropped_non_dict=%s kept=%s trace_id=%s", bad, len(out), trace_id)
    return out


async def produce_tool_plan(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    temperature: float,
    trace_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Central chat planner entry point.

    - Owns planning (no planner logic in /v1/chat/completions).
    - Uses the committee layer for model calls.
    - Returns (raw_planner_text, normalized_tool_calls, planner_env).
    """
    effective_mode = str(mode or "general").strip() or "general"
    t0 = time.perf_counter()
    tid = str(trace_id or "planner")
    msgs = _safe_message_list(messages or [], trace_id=tid)
    tools = _safe_tools_list(tools, trace_id=tid)
    try:
        temp = float(temperature)
    except Exception:
        temp = 0.2
    if temp < 0.0:
        temp = 0.0
    if temp > 2.0:
        temp = 2.0

    # Latest user text for goal anchoring.
    last_user = ""
    for m in reversed(msgs):
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m.get("content").strip()
        ):
            last_user = m.get("content").strip()
            break
    if not last_user:
        log.warning("planner no_last_user trace_id=%s msgs=%s", tid, len(msgs))

    # Planner-visible tool palette (fixed) — keep consistent with plan.catalog.
    allowed_tools = sorted([t for t in (PLANNER_VISIBLE_TOOLS or set()) if isinstance(t, str) and t.strip()])
    log.info(
        "planner.start trace_id=%s mode=%s msgs=%s tools_param=%s allowed_tools=%s temp=%s last_user_len=%s",
        tid,
        effective_mode,
        len(msgs),
        (len(tools) if isinstance(tools, list) else 0),
        len(allowed_tools),
        temp,
        len(last_user),
    )

    # Mode steering (kept inside the planner module, not in /v1/chat/completions).
    mode_system = "You are PlannerOps. Output ONLY JSON per the schema."
    if effective_mode == "film":
        mode_system = SYSTEM_VIDEO
    elif effective_mode == "image":
        mode_system = SYSTEM_IMAGE
    elif effective_mode == "audio":
        mode_system = SYSTEM_AUDIO

    # CO frames (prompt-only) to provide compact guidance + tail-safe RoE.
    co_env = {
        "schema_version": 1,
        "trace_id": str(trace_id or ""),
        "call_kind": "planner",
        "model_caps": {"num_ctx": 8192},
        "user_turn": {"role": "user", "content": last_user},
        "history": msgs,
        "attachments": [],
        "tool_memory": [],
        "rag_hints": [],
        "roe_incoming_instructions": [],
        "subject_canon": {},
        "percent_budget": {"icw_pct": [65, 70], "tools_pct": [18, 20], "roe_pct": [5, 10], "misc_pct": [3, 5], "buffer_pct": 5},
        "sweep_plan": ["0-90", "30-120", "60-150+wrap"],
    }
    try:
        co_out = co_pack(co_env)
        frames_text = frames_to_string(co_out.get("frames") or [])
    except Exception as exc:
        log.error("planner co_pack failed trace_id=%s: %s", tid, exc, exc_info=True)
        return "", [], {"ok": False, "error": {"code": "planner_co_pack_failed", "message": str(exc)}}

    # Tool catalog: strict, explicit, planner-visible tools only.
    catalog_lines: List[str] = [
        "### [TOOL CATALOG / SYSTEM]",
        "You must plan using ONLY these front-door tools. For each, the JSON args must follow the given schema.",
    ]
    for name in allowed_tools:
        try:
            schema = _tool_schema_by_name(name)
            catalog_lines.append(f"- tool: {name}")
            catalog_lines.append("  json_schema: " + json.dumps(schema or {}, ensure_ascii=False, sort_keys=True))
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("planner tool catalog build failed trace_id=%s tool=%r: %s", tid, name, exc, exc_info=True)
            catalog_lines.append(f"- tool: {name}")
            catalog_lines.append("  json_schema: {}")
    tool_catalog_frame = {"role": "system", "content": "\n".join(catalog_lines)}

    planner_rules = (
        "### [PLANNER / SYSTEM]\n"
        "Return ONLY strict JSON: {\"steps\":[{\"tool\":\"<name>\",\"args\":{...}}]} — no extra keys.\n"
        "For pure chat answers with no tool use, return {\"steps\":[]}.\n"
        "Rules:\n"
        f"- mode: {effective_mode}\n"
        f"- allowed_tools: {', '.join(allowed_tools) if allowed_tools else '(none)'}\n"
        "- Do NOT invent tool names.\n"
        "- If the user asks for any images/video/music/audio/TTS, you MUST include at least one tool step.\n"
    )

    plan_messages = [
        {"role": "system", "content": mode_system},
        {"role": "system", "content": frames_text},
        tool_catalog_frame,
        {"role": "system", "content": planner_rules},
    ] + msgs

    t_call = time.perf_counter()
    try:
        env = await committee_ai_text(
            plan_messages,
            trace_id=tid,
            temperature=float(temp),
        )
    except Exception as ex:
        log.error("planner committee_ai_text exception trace_id=%s: %s", tid, ex, exc_info=True)
        return "", [], {"ok": False, "error": {"code": "planner_committee_exception", "message": str(ex)}}
    call_ms = int((time.perf_counter() - t_call) * 1000)

    planner_env: Dict[str, Any] = env if isinstance(env, dict) else {"ok": False, "error": {"code": "planner_env_invalid", "message": str(env)}}
    if not planner_env.get("ok"):
        log.warning("planner committee not ok trace_id=%s call_ms=%s env=%r", tid, call_ms, planner_env.get("error") or planner_env)
        return "", [], planner_env

    res = planner_env.get("result") if isinstance(planner_env.get("result"), dict) else {}
    raw_text = res.get("text") if isinstance(res.get("text"), str) else ""
    log.info("planner committee ok trace_id=%s call_ms=%s raw_len=%s", tid, call_ms, len(raw_text))

    schema_steps = {
        "steps": [
            {
                "tool": str,
                "name": str,
                "id": str,
                "args": object,
                "arguments": object,
                "needs": list,
                "meta": dict,
            }
        ]
    }
    t_parse = time.perf_counter()
    try:
        parsed = await committee_jsonify(
            raw_text or "{}",
            expected_schema=schema_steps,
            trace_id=tid,
            temperature=0.0,
        )
    except Exception as ex:
        log.error("planner committee_jsonify exception trace_id=%s: %s", tid, ex, exc_info=True)
        parsed = {"steps": [], "error": {"code": "planner_jsonify_error", "message": str(ex)}}
    parse_ms = int((time.perf_counter() - t_parse) * 1000)

    steps_raw = parsed.get("steps") if isinstance(parsed, dict) else []
    tool_calls: List[Dict[str, Any]] = []
    allowed_set = set(allowed_tools)
    rejected = 0
    coerced_args = 0
    for st in steps_raw or []:
        if not isinstance(st, dict):
            rejected += 1
            continue
        tool_name = str(st.get("tool") or st.get("name") or "").strip()
        if not tool_name or tool_name not in allowed_set:
            rejected += 1
            continue
        args_val = st.get("args") if ("args" in st) else st.get("arguments")
        # Keep args always JSON-object-ish; downstream hardeners expect an object.
        if args_val is None:
            args_val = {}
            coerced_args += 1
        elif not isinstance(args_val, dict):
            if isinstance(args_val, str):
                try:
                    j = json.loads(args_val)
                    args_val = j if isinstance(j, dict) else {"_value": j}
                except Exception:
                    args_val = {"_raw": args_val}
                coerced_args += 1
            else:
                args_val = {"_value": args_val}
                coerced_args += 1
        tool_calls.append({"name": tool_name, "arguments": args_val})

    dt_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "planner.done trace_id=%s mode=%s steps_in=%s steps_out=%s rejected=%s coerced_args=%s parse_ms=%s total_ms=%s",
        tid,
        effective_mode,
        (len(steps_raw) if isinstance(steps_raw, list) else -1),
        len(tool_calls),
        rejected,
        coerced_args,
        parse_ms,
        dt_ms,
    )
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        planner_env = dict(planner_env)
        planner_env["parse_error"] = parsed.get("error")
    return raw_text, tool_calls, planner_env


