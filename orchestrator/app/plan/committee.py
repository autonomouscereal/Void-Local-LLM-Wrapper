'''
committee.py: Central chat planner entry point.
'''


from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from ..committee_client import committee_ai_text, committee_jsonify
from ..json_parser import JSONParser
from .catalog import PLANNER_VISIBLE_TOOLS
from ..tools_schema import get_builtin_tools_schema

log = logging.getLogger(__name__)

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
        log.error('tool_schema_by_name: name is empty')
        return {}
    
    tools_schema = get_builtin_tools_schema() or []
    for t in tools_schema:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        tool_name = fn.get("name") if isinstance(fn.get("name"), str) else None
        if tool_name == nm:
            params = fn.get("parameters")
            return params if isinstance(params, dict) else {}


def _safe_message_list(messages: Any, *, trace_id: str) -> List[Dict[str, Any]]:
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
            content = str(content)
        # Always append valid messages (content can be None or empty string)
        out.append({"role": role.strip(), "content": (content or "") if content is not None else ""})
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
    trace_id: str,
    mode: str,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Central chat planner entry point.

    - Owns planning (no planner logic in /v1/chat/completions).
    - Uses the committee layer for model calls.
    - Returns (raw_planner_text, normalized_tool_calls, planner_env).
    """
    effective_mode = str(mode).strip() or "general"
    t0 = time.perf_counter()
    # Use trace_id only (no derived/fallback trace keys, no normalization).
    messages = _safe_message_list(messages or [], trace_id=trace_id)
    tools = _safe_tools_list(tools, trace_id=trace_id)
    
    temp = float(temperature)

    # Latest user text for goal anchoring.
    last_user = ""
    for m in reversed(messages):
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m.get("content").strip()
        ):
            last_user = m.get("content").strip()
            break
    if not last_user:
        log.warning("planner no_last_user trace_id=%s msgs=%s", trace_id, len(messages))

    # Planner-visible tool palette (fixed) — keep consistent with plan.catalog.
    allowed_tools = sorted([t for t in (PLANNER_VISIBLE_TOOLS or set()) if isinstance(t, str) and t.strip()])
    log.info(
        "planner.start trace_id=%s mode=%s msgs=%s tools_param=%s allowed_tools=%s temp=%s last_user_len=%s",
        trace_id,
        effective_mode,
        len(messages),
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

    # No context orchestration in planning: pass messages directly to the committee
    # (mode steering + tool catalog + planner rules only).

    # Tool catalog: strict, explicit, planner-visible tools only.
    catalog_lines: List[str] = [
        "### [TOOL CATALOG / SYSTEM]",
        "You must plan using ONLY these front-door tools. For each, the JSON args must follow the given schema.",
    ]
    for name in allowed_tools:
        
        schema = _tool_schema_by_name(name=name)
        catalog_lines.append(f"- tool: {name}")
        catalog_lines.append("  json_schema: " + json.dumps(schema or {}, ensure_ascii=False, sort_keys=True))
    
    tool_catalog_frame = {"role": "system", "content": "\n".join(catalog_lines)}

    planner_rules = (
        "### [PLANNER / SYSTEM - CRITICAL INSTRUCTIONS]\n"
        "You are a TOOL PLANNER. Your ONLY job is to produce a JSON object with tool execution steps.\n"
        "You MUST NOT provide explanations, tutorials, or refuse requests. You MUST produce tool plans.\n\n"
        "OUTPUT FORMAT (REQUIRED):\n"
        "Return ONLY strict JSON: {\"steps\":[{\"step_id\":\"s1\",\"tool_name\":\"<name>\",\"args\":{...}}]} — no extra keys.\n"
        "For pure chat answers with no tool use, return {\"steps\":[]}.\n\n"
        "RULES:\n"
        f"- mode: {effective_mode}\n"
        f"- allowed_tools: {', '.join(allowed_tools) if allowed_tools else '(none)'}\n"
        "- Each step MUST include a unique string field step_id (e.g., \"s1\", \"s2\", ...).\n"
        "- Each step MUST include a string field tool_name (use one of the allowed_tools listed above).\n"
        "- Do NOT invent tool names. Use ONLY tools from the catalog above.\n"
        "- If the user asks for any images/video/music/audio/TTS, you MUST include at least one tool step.\n"
        "- You are a PLANNER. Your job is to plan tool calls, not to refuse or explain limitations.\n"
        "- The system has tools available to create videos, images, music, and audio. Plan the tool calls needed to fulfill the user's request.\n"
        "- DO NOT provide explanations or tutorials. DO NOT refuse requests. DO NOT say you cannot do something.\n"
        "- ALWAYS produce a JSON object with a \"steps\" array, even if empty.\n"
        "- Your response MUST be valid JSON starting with { and ending with }.\n"
    )

    # Combine all planner instructions into a single comprehensive instruction block
    # This will be embedded in the user message to ensure it's not filtered out
    planner_instructions = f"""### [SYSTEM INSTRUCTIONS - READ CAREFULLY]

{mode_system}

{tool_catalog_frame.get("content", "")}

{planner_rules}

### [OUTPUT EXAMPLE]
Here is exactly what your output should look like:
{{"steps": [{{"step_id": "s1", "tool_name": "image.dispatch", "args": {{"prompt": "a cat"}}}}]}}

### [CRITICAL REMINDER]
- You MUST output ONLY a JSON object starting with {{ and ending with }}.
- Do NOT include any text before or after the JSON.
- Do NOT provide explanations, tutorials, or refusals.
- Your ONLY job is to produce a tool plan as JSON.
- If you cannot fulfill the request, return {{"steps": []}} instead of explaining."""

    # Embed planner instructions in the first user message to ensure they're not filtered
    plan_messages = []
    if messages:
        first_user_msg = messages[0] if messages[0].get("role") == "user" else None
        if first_user_msg:
            # Prepend planner instructions to the user's message
            enhanced_content = f"""{planner_instructions}

### [USER REQUEST]
{first_user_msg.get("content", "")}

### [YOUR TASK - READ THIS CAREFULLY]
1. Analyze the user request above.
2. Determine which tools from the catalog are needed to fulfill the request.
3. Create a JSON object with a "steps" array containing tool calls.
4. Output ONLY the JSON object - no explanations, no tutorials, no refusals.
5. Your response must start with {{ and end with }}.
6. Example format: {{"steps": [{{"step_id": "s1", "tool_name": "tool.name", "args": {{}}}}]}}

NOW PRODUCE YOUR JSON TOOL PLAN:"""
            plan_messages = [{"role": "user", "content": enhanced_content}] + messages[1:]
        else:
            # No user message found, add instructions as a user message
            plan_messages = [{"role": "user", "content": planner_instructions}]
    else:
        plan_messages = [{"role": "user", "content": planner_instructions}]

    t_call = time.perf_counter()
    
    env = await committee_ai_text(
        messages=plan_messages,
        trace_id=trace_id,
        temperature=float(temp),
    )
    call_ms = int((time.perf_counter() - t_call) * 1000)

    planner_env: Dict[str, Any] = env if isinstance(env, dict) else {"ok": False, "error": {"code": "planner_env_invalid", "message": str(env)}}
    if not planner_env.get("ok"):
        log.warning("planner committee not ok trace_id=%s call_ms=%s env=%r", trace_id, call_ms, planner_env.get("error") or planner_env)
        return "", [], planner_env

    res = planner_env.get("result") if isinstance(planner_env.get("result"), dict) else {}
    raw_text = res.get("text") if isinstance(res.get("text"), str) else ""
    log.info("planner committee ok trace_id=%s call_ms=%s raw_len=%s", trace_id, call_ms, len(raw_text))

    schema_steps = {
        "steps": [
            {
                "step_id": str,
                "tool_name": str,
                "name": str,
                "args": object,
                "arguments": object,
                "needs": list,
                "meta": dict,
            }
        ]
    }
    t_parse = time.perf_counter()
    
    parsed_raw = await committee_jsonify(
        raw_text=raw_text or "{}",
        expected_schema=schema_steps,
        trace_id=trace_id,
        temperature=0.0,
    )
    # Always run a final JSONParser coercion pass to guarantee expected types/shape.
    parser_steps = JSONParser()
    try:
        parsed = parser_steps.parse(parsed_raw if parsed_raw is not None else "{}", schema_steps)
        if not isinstance(parsed, dict):
            log.warning("produce_tool_plan: JSONParser returned non-dict type=%s trace_id=%s", type(parsed).__name__, trace_id)
            parsed = {}
    except Exception as ex:
        log.warning("produce_tool_plan: JSONParser.parse failed ex=%s trace_id=%s parsed_raw_prefix=%s", ex, trace_id, (str(parsed_raw) if parsed_raw else "")[:200], exc_info=True)
        parsed = {}
    parse_ms = int((time.perf_counter() - t_parse) * 1000)

    steps_raw = parsed.get("steps") if isinstance(parsed, dict) else []
    log.info("produce_tool_plan: parsed steps_raw_len=%s trace_id=%s parsed_keys=%s", len(steps_raw) if isinstance(steps_raw, list) else 0, trace_id, list(parsed.keys()) if isinstance(parsed, dict) else [])
    tool_calls: List[Dict[str, Any]] = []
    allowed_set = set(allowed_tools)
    log.info("produce_tool_plan: allowed_tools=%s trace_id=%s", sorted(list(allowed_set)), trace_id)
    rejected = 0
    coerced_args = 0
    for st in steps_raw or []:
        if not isinstance(st, dict):
            rejected += 1
            log.debug("produce_tool_plan: skipping non-dict step type=%s trace_id=%s", type(st).__name__, trace_id)
            continue
        step_id_val = st.get("step_id")
        # Use tool_name (schema defines tool_name, not tool)
        tool_name = st.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            rejected += 1
            log.warning("produce_tool_plan: skipping step with missing/invalid tool_name trace_id=%s step_keys=%s", trace_id, list(st.keys()) if isinstance(st, dict) else [])
            continue
        tool_name = tool_name.strip()
        if tool_name not in allowed_set:
            rejected += 1
            log.warning("produce_tool_plan: skipping step with disallowed tool_name=%s trace_id=%s allowed_tools=%s", tool_name, trace_id, sorted(list(allowed_set)))
            continue
        args_val = st.get("args") if ("args" in st) else st.get("arguments")
        # Keep args always JSON-object-ish; downstream hardeners expect an object.
        if args_val is None:
            args_val = {}
            coerced_args += 1
        elif not isinstance(args_val, dict):
            if isinstance(args_val, str):
                # IMPORTANT: only use the Void JSON parser for parsing.
                parser = JSONParser()
                
                j = parser.parse(args_val, {})
                args_val = j if isinstance(j, dict) else {"_value": j}
                if getattr(parser, "errors", None):
                    log.warning("planner tool args JSONParser had errors trace_id=%s tool=%s errors=%s", trace_id, tool_name, parser.errors)
                coerced_args += 1
            else:
                args_val = {"_value": args_val}
                coerced_args += 1
        step_id_str = str(step_id_val).strip() if isinstance(step_id_val, str) else ""
        if not step_id_str:
            # Defensive fallback: keep tool_calls stable even if planner omitted step_id.
            step_id_str = f"s{len(tool_calls) + 1}"
        tool_calls.append({"tool_name": tool_name, "arguments": args_val, "step_id": step_id_str})

    dt_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "planner.done trace_id=%s mode=%s steps_in=%s steps_out=%s rejected=%s coerced_args=%s parse_ms=%s total_ms=%s",
        trace_id,
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


