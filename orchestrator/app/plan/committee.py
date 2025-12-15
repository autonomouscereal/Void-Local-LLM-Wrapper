from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, Tuple

from ..pipeline.compression_orchestrator import co_pack, frames_to_string
from ..committee_client import committee_ai_text, committee_jsonify
from .catalog import PLANNER_VISIBLE_TOOLS
from ..tools_schema import get_builtin_tools_schema

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
    for t in get_builtin_tools_schema() or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        tool_name = fn.get("name") if isinstance(fn.get("name"), str) else None
        if tool_name == nm:
            params = fn.get("parameters")
            return params if isinstance(params, dict) else {}
    return {}


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
    msgs = messages or []

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

    # Planner-visible tool palette (fixed) — keep consistent with plan.catalog.
    allowed_tools = sorted([t for t in (PLANNER_VISIBLE_TOOLS or set()) if isinstance(t, str) and t.strip()])

    # Mode steering (kept inside the planner module, not in /v1/chat/completions).
    mode_system = "You are PlannerOps. Output ONLY JSON per the schema."
    if effective_mode == "film":
        mode_system = SYSTEM_VIDEO

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
    co_out = co_pack(co_env)
    frames_text = frames_to_string(co_out.get("frames") or [])

    # Tool catalog: strict, explicit, planner-visible tools only.
    catalog_lines: List[str] = [
        "### [TOOL CATALOG / SYSTEM]",
        "You must plan using ONLY these front-door tools. For each, the JSON args must follow the given schema.",
    ]
    for name in allowed_tools:
        schema = _tool_schema_by_name(name)
        catalog_lines.append(f"- tool: {name}")
        catalog_lines.append("  json_schema: " + json.dumps(schema or {}, ensure_ascii=False, sort_keys=True))
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

    try:
        env = await committee_ai_text(
            plan_messages,
            trace_id=str(trace_id or "planner"),
            temperature=float(temperature),
        )
    except Exception as ex:
        return "", [], {"ok": False, "error": {"code": "planner_committee_exception", "message": str(ex)}}

    planner_env: Dict[str, Any] = env if isinstance(env, dict) else {"ok": False, "error": {"code": "planner_env_invalid", "message": str(env)}}
    if not planner_env.get("ok"):
        return "", [], planner_env

    res = planner_env.get("result") if isinstance(planner_env.get("result"), dict) else {}
    raw_text = res.get("text") if isinstance(res.get("text"), str) else ""

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
    try:
        parsed = await committee_jsonify(
            raw_text or "{}",
            expected_schema=schema_steps,
            trace_id=str(trace_id or "planner"),
            temperature=0.0,
        )
    except Exception as ex:
        parsed = {"steps": [], "error": {"code": "planner_jsonify_error", "message": str(ex)}}

    steps_raw = parsed.get("steps") if isinstance(parsed, dict) else []
    tool_calls: List[Dict[str, Any]] = []
    allowed_set = set(allowed_tools)
    for st in steps_raw or []:
        if not isinstance(st, dict):
            continue
        tool_name = str(st.get("tool") or st.get("name") or "").strip()
        if not tool_name or tool_name not in allowed_set:
            continue
        args_val = st.get("args") if ("args" in st) else st.get("arguments")
        tool_calls.append({"name": tool_name, "arguments": args_val})

    return raw_text, tool_calls, planner_env


