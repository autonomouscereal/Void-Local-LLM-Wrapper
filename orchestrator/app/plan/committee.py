from __future__ import annotations

import json
from typing import Dict, Any, List

from ..json_parser import JSONParser
from ..pipeline.compression_orchestrator import co_pack, frames_to_string
from ..committee_client import CommitteeClient, committee_jsonify


PLAN_SCHEMA: Dict[str, Any] = {
    "request_id": str,
    "plan": [
        {
            "id": str,
            "tool": str,
            "inputs": dict,
            "needs": [str],
            "provides": dict,
        }
    ],
}


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


async def _call_llm_json(prompt: str, trace_id: str) -> Dict[str, Any]:
    """
    Legacy planner helper now routed through the central committee path.
    The committee debate produces free-form text; we then parse that text
    into PLAN_SCHEMA via JSONParser. No direct calls to Ollama here.
    """
    # First, route through the main planner committee to get a rich text answer.
    client = CommitteeClient()
    env = await client.run(
        [
            {"role": "system", "content": "You are a multimodal planner. Output ONLY JSON per the schema."},
            {"role": "user", "content": prompt},
        ],
        trace_id=trace_id,
    )
    result: Dict[str, Any]
    if not isinstance(env, dict) or not env.get("ok"):
        # Legacy planner committee failure: log and return a minimal plan with
        # an embedded error so callers can surface it if they choose to.
        import logging
        logging.getLogger("orchestrator.plan.committee").error(
            "legacy_planner committee failed (trace_id=%s): env=%r",
            trace_id,
            env,
        )
        result = {
            "request_id": "",
            "plan": [],
            "error": (env or {}).get("error")
            if isinstance(env, dict)
            else {
                "code": "planner_committee_error",
                "message": str(env),
            },
        }
    else:
        res = env.get("result") or {}
        text = res.get("text") or ""
        # Then, run the raw planner text through committee.jsonify to enforce PLAN_SCHEMA.
        parsed = await committee_jsonify(
            text or "{}",
            expected_schema=PLAN_SCHEMA,
            trace_id=trace_id,
            temperature=0.0,
        )
        result = parsed if isinstance(parsed, dict) else {"request_id": "", "plan": []}
    return result


async def _plan_with_role(system: str, user_text: str, request_id: str) -> Dict[str, Any]:
    # Build CO frames for planner/committee via internal pack (prompt-only, no enforcement)
    co_env = {
        "schema_version": 1,
        "trace_id": request_id,
        "call_kind": "planner",
        "model_caps": {"num_ctx": 8192},
        "user_turn": {"role": "user", "content": user_text},
        "history": [],
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
    prompt = (
        frames_text
        + "\n\n"
        + f"Return JSON exactly matching this schema: {json.dumps(PLAN_SCHEMA, ensure_ascii=False)}\n"
        + f"User request: {user_text}\n"
        + f"Your role: {system}\n"
        + "Include ALL relevant steps for your modality; do not omit sub-steps.\n"
        + "### [PLANNER / SYSTEM]\n"
        + "- Think in explicit plans, not single shots. For each step, decide its purpose, which tool to call (or if it is reasoning-only), and what it depends on.\n"
        + "- Plan minimal but complete steps, prefer tools over pure text when external data, media generation, or locks are involved.\n"
        + "- When prior tool results are available in the CO/TOOLS frames, read their summaries (success/fail, key args, artifact ids, short diagnosis) and incorporate failures into the new plan.\n"
        + "- If a previous step failed (ok=false or error present), do NOT assume it will be retried automatically; instead, add a new step that fixes the inputs or uses a different tool.\n"
        + "### [LOCK ENGINE DIRECTIVES]\n"
        + "- Fetch existing lock bundles with locks.get_bundle when the user references prior characters, outfits, props, scenery, or voices.\n"
        + "- Use locks.build_region_locks when new reference images define clothing/props/background regions to preserve.\n"
        + "- Adjust lock modes instead of re-prompting: locks.update_region_modes for vision regions (shape/texture/color hard|soft|off) and locks.update_audio_modes for tempo/key/stem/lyrics.\n"
        + "- When user requests partial changes (e.g., \"same coat shape, new color\"), keep shape hard and relax color/texture via the update tools; carry forward the bundle ID into downstream image.dispatch/film2.run/music/tts calls.\n"
        + "- Always pass lock_bundle and quality_profile into image.dispatch, film2.run, music.*, and tts.speak so downstream QA can enforce thresholds.\n"
        + "### [HTTP TOOL]\n"
        + "- Use http.request (or api.request where exposed) to call external APIs when needed. Provide url, method, headers/query/body, and set expect_json=false if the endpoint returns plain text.\n"
        + "- Treat http.request results as envelopes: ok=true means success; ok=false responses include error.status and error.details.remote_* fields. Plan a new step with corrected args instead of assuming automatic retries.\n"
        + "### [EXECUTOR / VALIDATION INVARIANTS]\n"
        + "- The executor validates once and runs each tool step once. It will NOT automatically retry or repair; all retries must be explicit new steps you plan.\n"
        + "- Do not count on hidden retries or side effects; always assume tools are run exactly as you specify them.\n"
        + "### [FINALITY / SYSTEM]\n"
        + "Do not output assistant content until committee consensus; one final answer only."
    )
    out = await _call_llm_json(prompt, trace_id=request_id)
    out["request_id"] = request_id
    return out


async def _plan_with_retry(system: str, user_text: str, request_id: str, attempts: int = 2) -> Dict[str, Any]:
    """
    Call the modality-specific planner with a small number of retries.

    This helper MUST NOT raise; on repeated failure it returns a minimal plan
    with an embedded error so callers and HTTP layers can surface what went
    wrong instead of silently dropping the exception.
    """
    last_error: str | None = None
    result: Dict[str, Any] = {"request_id": request_id, "plan": []}
    for i in range(attempts):
        try:
            result = await _plan_with_role(system, user_text, request_id)
            break
        except Exception as ex:  # pragma: no cover - defensive logging
            last_error = str(ex)
            # Best-effort log; committee/planner traces live under the main orchestrator logger.
            import logging, traceback as _tb  # local import to avoid cycles
            logging.getLogger("orchestrator.plan.committee").error(
                "planner._plan_with_retry failed (attempt %d/%d, system=%s, request_id=%s): %s\n%s",
                i + 1,
                attempts,
                system,
                request_id,
                last_error,
                _tb.format_exc(),
            )
            if i == attempts - 1:
                # Final fallback: surface the error in-band so upstream callers can
                # include it in their envelopes instead of fabricating an empty plan.
                result = {
                    "request_id": request_id,
                    "plan": [],
                    "error": {
                        "code": "planner_error",
                        "message": last_error or "planner failed with unknown error",
                        "system": system,
                    },
                }
    return result


async def make_full_plan(user_text: str) -> Dict[str, Any]:
    import uuid as _uuid
    rid = str(_uuid.uuid4())
    # Sequential planning: run IMAGE, VIDEO, AUDIO planners one after another for stability.
    parts: List[Dict[str, Any]] = []
    img_plan = await _plan_with_retry(SYSTEM_IMAGE, user_text, rid)
    if isinstance(img_plan, dict):
        parts.append(img_plan)
    vid_plan = await _plan_with_retry(SYSTEM_VIDEO, user_text, rid)
    if isinstance(vid_plan, dict):
        parts.append(vid_plan)
    aud_plan = await _plan_with_retry(SYSTEM_AUDIO, user_text, rid)
    if isinstance(aud_plan, dict):
        parts.append(aud_plan)
    merged = {"request_id": rid, "plan": []}
    seen = set()
    for p in parts:
        if not isinstance(p, dict):
            continue
        for step in p.get("plan", []):
            key = (step.get("tool"), json.dumps(step.get("inputs", {}), ensure_ascii=False, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            merged["plan"].append(step)
    # Post-process: unify all video intents under film2.run (no bare video.*)
    has_video = any(isinstance(s, dict) and isinstance(s.get("tool"), str) and s.get("tool", "").startswith("video.") for s in (merged.get("plan") or []))
    if has_video:
        merged["plan"] = [s for s in merged.get("plan") or [] if not (isinstance(s, dict) and isinstance(s.get("tool"), str) and s.get("tool", "").startswith("video."))]
        merged["plan"].append({"id": "film-1", "tool": "film2.run", "inputs": {"prompt": user_text}, "needs": [], "provides": {}})
    if not (merged.get("plan") or []):
        merged["plan"] = [{"id": "img-1", "tool": "image.dispatch", "inputs": {"prompt": user_text}, "needs": [], "provides": {}}]
    return merged


