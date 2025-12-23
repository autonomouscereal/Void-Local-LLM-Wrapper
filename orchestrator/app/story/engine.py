from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

from ..tracing.runtime import trace_event
from ..committee_client import committee_ai_text, committee_jsonify
from ..json_parser import JSONParser


log = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, min_val: int = 1, max_val: int = 1000) -> int:
    raw = os.getenv(name, None)
    s = str(raw).strip() if raw is not None else ""
    val = int(s) if (s and s.lstrip("+-").isdigit()) else int(default)
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    return val


def _env_float(name: str, default: float, *, min_val: float = 0.0, max_val: float = 2.0) -> float:
    raw = os.getenv(name, None)
    s = str(raw).strip() if raw is not None else ""
    # Avoid exceptions: only accept simple float forms.
    ok = False
    if s:
        t = s.replace(".", "", 1)
        t = t.replace("e", "", 1).replace("E", "", 1)
        t = t.replace("+", "", 1).replace("-", "", 1)
        ok = t.isdigit()
    val = float(s) if ok else float(default)
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    return val


STORY_COMMITTEE_ROUNDS = _env_int("STORY_COMMITTEE_ROUNDS", 1, min_val=1, max_val=10)
# By default, do NOT hard-cap story iterations. If you want a cap, set STORY_MAX_PASSES > 0.
STORY_MAX_PASSES = _env_int("STORY_MAX_PASSES", 0, min_val=0, max_val=1000)
STORY_TEMPERATURE = _env_float("STORY_TEMPERATURE", 0.5, min_val=0.0, max_val=2.0)


def _story_schema() -> Dict[str, Any]:
    """
    Schema used for committee_jsonify.

    We keep the "classic" keys used by film2.run today (prompt/duration_hint_s/acts/characters/locations/objects),
    and add richer fields for a real story engine without breaking callers.
    """
    return {
        "prompt": str,
        "duration_hint_s": float,
        "logline": str,
        "genre": list,
        "tone": str,
        "themes": list,
        "constraints": list,
        "characters": list,
        "locations": list,
        "objects": list,
        "acts": list,
        "continuity": dict,
        "film_plan": dict,
        "notes": dict,
    }


def _issue_schema() -> Dict[str, Any]:
    return {
        "issues": list,
        "summary": str,
        "must_fix": bool,
        # "done" is the authoritative committee signal for whether the story meets user intent + length.
        "done": bool,
        "length_ok": bool,
        "coverage_ratio": float,
        "next_action": str,
    }


def _norm_prompt(user_prompt: str) -> str:
    return (user_prompt or "").strip()


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _jsonify_then_parse(raw_text: str, expected_schema: Any, *, trace_id: str, rounds: int | None = None, temperature: float | None = None) -> Dict[str, Any]:
    """
    Required JSON flow for any LLM output used as structured data:
      1) LLM produces raw text (possibly messy)
      2) committee_jsonify(raw_text, expected_schema) -> best-effort structured JSON candidate
      3) JSONParser.parse(candidate, expected_schema) -> schema-coerced dict (100% JSON-safe)
    """
    candidate = await committee_jsonify(
        raw_text=(raw_text or "{}"),
        expected_schema=expected_schema,
        trace_id=trace_id,
        rounds=rounds,
        temperature=temperature,
    )
    parser = JSONParser()
    obj = parser.parse(candidate if candidate is not None else "{}", expected_schema)
    return obj if isinstance(obj, dict) else {}


async def _committee_generate_story_raw(user_prompt: str, duration_hint_s: float, *, trace_id: Optional[str]) -> str:
    """
    Ask the committee to propose a story graph. We intentionally ask for JSON-only, then run committee_jsonify
    to get strict structure for downstream consumers.
    """
    prompt_text = _norm_prompt(user_prompt)
    dur = float(duration_hint_s) if isinstance(duration_hint_s, (int, float)) else 0.0
    if dur <= 0.0:
        dur = 600.0
    sys_msg = (
        "You are the Story Engine for a film generator.\n"
        "You MUST produce a story that follows the user's request exactly, but you may fill in missing details.\n"
        "Continuity is critical: injuries, inventory, relationships, time travel rules, and causal consequences MUST remain consistent.\n"
        "Output MUST be a single JSON object (no markdown, no code fences, no prose) describing a full story graph.\n"
        "The story MUST be temporally coherent end-to-end.\n"
        "If the user requests a short duration, keep the story tight. If long, include deeper arcs.\n"
    )
    user_msg = json.dumps(
        {
            "user_prompt": prompt_text,
            "duration_hint_s": dur,
            "requirements": {
                "acts": "3+ acts as needed; each act has scenes; each scene has beats; each beat has dialogue and events",
                "continuity": "explicitly track character and object state changes via events; ensure later beats reflect them",
                "film_plan": "include optional per-beat shot prompts usable by image/video tools",
                "ids": "stable ids for acts/scenes/beats/lines/characters/objects/locations",
            },
        },
        ensure_ascii=False,
    )
    t0 = time.perf_counter()
    env = await committee_ai_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        trace_id=(trace_id or "story_draft"),
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=STORY_TEMPERATURE,
    )
    result_block = env.get("result") if isinstance(env, dict) else {}
    txt = ""
    if isinstance(result_block, dict) and isinstance(result_block.get("text"), str):
        txt = str(result_block.get("text") or "")
    else:
        txt = str(env.get("text") or "")
    log.info(
        "story.committee.draft_raw done trace_id=%s dur_ms=%d raw=%s",
        trace_id,
        int((time.perf_counter() - t0) * 1000),
        txt,
    )
    return txt


async def _committee_audit_story(story: Dict[str, Any], user_prompt: str, *, trace_id: Optional[str]) -> Dict[str, Any]:
    """
    Committee-based continuity + intent audit. Returns {issues: [...], summary: str, must_fix: bool}.
    """
    # Provide basic quantitative coverage, but rely on committee judgement for "done".
    total_hint = float(story.get("duration_hint_s") or 0.0) if isinstance(story, dict) else 0.0
    total_beats_time = 0.0
    acts = story.get("acts") if isinstance(story, dict) and isinstance(story.get("acts"), list) else []
    for act in acts:
        if not isinstance(act, dict):
            continue
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat in beats:
                if not isinstance(beat, dict):
                    continue
                total_beats_time += float(beat.get("time_hint_s") or 0.0)
    ratio = (total_beats_time / total_hint) if (total_hint > 0.0 and total_beats_time > 0.0) else 0.0

    sys_msg = (
        "You are a strict story continuity AND completion auditor.\n"
        "Given a story graph JSON and the user prompt, find ALL violations:\n"
        "- breaks user intent\n"
        "- continuity contradictions (injuries, objects, locations, relationships)\n"
        "- temporal contradictions and time travel inconsistencies\n"
        "- missing causal links / dangling plot threads that will confuse viewers\n"
        "- character motives that don't track\n"
        "\n"
        "You must ALSO judge whether the story is COMPLETE for the requested duration.\n"
        "A 20-second story can be simple; a 2-hour story must have appropriately deep arcs.\n"
        "You will be given duration_hint_s and current_beats_time_s/coverage_ratio, but you MUST decide completion.\n"
        "\n"
        "Return JSON only with fields:\n"
        "- done: bool (authoritative: true ONLY if story is ready to proceed)\n"
        "- length_ok: bool (true if duration/content depth matches the request)\n"
        "- coverage_ratio: float (echo the provided ratio or your estimate)\n"
        "- next_action: string one of: 'accept'|'extend'|'revise'|'extend_then_revise'\n"
        "- must_fix: bool (true if severe issues exist)\n"
        "- summary: string\n"
        "- issues: list of issue objects (code, severity, message, pointers, suggested_fix)\n"
        "Return issues with precise pointers (act_id/scene_id/beat_id/line_id/char_id/obj_id).\n"
    )
    user_msg = json.dumps(
        {
            "user_prompt": _norm_prompt(user_prompt),
            "duration_hint_s": float(total_hint),
            "current_beats_time_s": float(total_beats_time),
            "coverage_ratio": float(ratio),
            "story": story,
        },
        ensure_ascii=False,
    )
    t0 = time.perf_counter()
    env = await committee_ai_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        trace_id=(trace_id or "story_audit"),
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=0.2,
    )
    raw = ""
    result_block = env.get("result") if isinstance(env, dict) else {}
    if isinstance(result_block, dict) and isinstance(result_block.get("text"), str):
        raw = str(result_block.get("text") or "")
    parsed = await _jsonify_then_parse(
        raw_text=raw,
        expected_schema=_issue_schema(),
        trace_id=(trace_id or "story_audit.jsonify"),
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=0.0,
    )
    if not parsed:
        return {"issues": [], "summary": "", "must_fix": False, "done": False, "length_ok": False, "coverage_ratio": float(ratio), "next_action": "extend"}
    issues = parsed.get("issues") if isinstance(parsed.get("issues"), list) else []
    parsed.setdefault("coverage_ratio", float(ratio))
    if not isinstance(parsed.get("next_action"), str):
        parsed["next_action"] = "revise" if bool(parsed.get("must_fix")) else ("accept" if bool(parsed.get("done")) else "extend")
    log.info(
        "story.committee.audit done trace_id=%s issues=%d must_fix=%s dur_ms=%d",
        trace_id,
        len(issues),
        bool(parsed.get("must_fix")) if isinstance(parsed, dict) else False,
        int((time.perf_counter() - t0) * 1000),
    )
    return parsed


async def _committee_revise_story(story: Dict[str, Any], user_prompt: str, audit: Dict[str, Any], *, trace_id: Optional[str]) -> Dict[str, Any]:
    """
    Committee revision step. Produces a full updated story graph (same schema).
    """
    sys_msg = (
        "You are the Story Engine revision pass.\n"
        "You MUST fix the issues while preserving the user's intent and improving coherence.\n"
        "Do not hand-wave: update the story graph so later beats reflect state changes.\n"
        "Output JSON only."
    )
    user_msg = json.dumps(
        {"user_prompt": _norm_prompt(user_prompt), "story": story, "audit": audit},
        ensure_ascii=False,
    )
    t0 = time.perf_counter()
    env = await committee_ai_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        trace_id=(trace_id or "story_revise"),
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=STORY_TEMPERATURE,
    )
    raw = ""
    result_block = env.get("result") if isinstance(env, dict) else {}
    if isinstance(result_block, dict) and isinstance(result_block.get("text"), str):
        raw = str(result_block.get("text") or "")
    parsed = await _jsonify_then_parse(
        raw_text=raw,
        expected_schema=_story_schema(),
        trace_id=(trace_id or "story_revise.jsonify"),
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=STORY_TEMPERATURE,
    )
    acts_n = len(parsed.get("acts") or []) if isinstance(parsed, dict) and isinstance(parsed.get("acts"), list) else 0
    log.info(
        "story.committee.revise done trace_id=%s acts=%d dur_ms=%d",
        trace_id,
        acts_n,
        int((time.perf_counter() - t0) * 1000),
    )
    return parsed if parsed else dict(story or {})


async def draft_story_graph(user_prompt: str, duration_hint_s: Optional[float], trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Committee-driven story generator.

    This is intentionally heavy: it relies on repeated committee calls to draft, audit, and
    revise until the story is coherent and follows the user intent.
    """
    t0 = time.monotonic()
    prompt_text = _norm_prompt(user_prompt)
    duration = float(duration_hint_s) if isinstance(duration_hint_s, (int, float)) else 0.0
    if duration <= 0.0:
        duration = 600.0
    log.info("story.draft start trace_id=%s duration_hint_s=%s prompt_len=%d", trace_id, float(duration), len(prompt_text))
    raw = await _committee_generate_story_raw(prompt_text, duration, trace_id=(trace_id or "film2.story_draft"))
    story = await _jsonify_then_parse(
        raw_text=raw,
        expected_schema=_story_schema(),
        trace_id=(trace_id or "film2.story_draft.jsonify"),
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=STORY_TEMPERATURE,
    )
    if not isinstance(story, dict):
        story = {}
    story.setdefault("prompt", prompt_text)
    story.setdefault("duration_hint_s", float(duration))
    story.setdefault("acts", [])
    story.setdefault("characters", [])
    story.setdefault("locations", [])
    story.setdefault("objects", [])
    # Iterative expansion + audit/revise loop (can be large).
    # We stop ONLY when the committee audit says to accept (authoritative), not after an arbitrary number of loops.
    pass_index = 0
    consecutive_no_progress = 0
    last_beats_time_s: Optional[float] = None
    last_beats_count: Optional[int] = None
    max_passes = int(STORY_MAX_PASSES)
    while True:
        if max_passes > 0 and pass_index >= max_passes:
            trace_event("story.max_passes_reached", {"trace_id": trace_id, "max_passes": int(max_passes)})
            break
        # Measure current duration coverage (sum beat time hints) for logging + context.
        total_hint = float(story.get("duration_hint_s") or 0.0)
        total_beats_time = 0.0
        total_beats_count = 0
        acts = story.get("acts") if isinstance(story.get("acts"), list) else []
        for act in acts:
            if not isinstance(act, dict):
                continue
            scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
            for scene in scenes:
                if not isinstance(scene, dict):
                    continue
                beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
                for beat in beats:
                    if not isinstance(beat, dict):
                        continue
                    total_beats_count += 1
                    total_beats_time += float(beat.get("time_hint_s") or 0.0)
        ratio = (total_beats_time / total_hint) if (total_hint > 0.0 and total_beats_time > 0.0) else 0.0

        log.info(
            "story.draft.pass start trace_id=%s pass_index=%d beats_time_s=%.2f duration_hint_s=%.2f ratio=%.3f",
            trace_id,
            int(pass_index),
            float(total_beats_time),
            float(total_hint),
            float(ratio),
        )

        # Audit after each iteration (authoritative "done" signal).
        audit = await _committee_audit_story(story, prompt_text, trace_id=(trace_id or f"film2.story_audit.{pass_index}"))
        issues = audit.get("issues") if isinstance(audit, dict) and isinstance(audit.get("issues"), list) else []
        must_fix = bool(audit.get("must_fix")) if isinstance(audit, dict) else False
        done = bool(audit.get("done")) if isinstance(audit, dict) else False
        length_ok = bool(audit.get("length_ok")) if isinstance(audit, dict) else False
        next_action = audit.get("next_action") if isinstance(audit, dict) else None
        next_action_s = str(next_action) if isinstance(next_action, str) else ""

        trace_event(
            "story.committee_audit",
            {
                "trace_id": trace_id,
                "pass_index": int(pass_index),
                "issues": len(issues),
                "must_fix": must_fix,
                "done": done,
                "length_ok": length_ok,
                "coverage_ratio": float(audit.get("coverage_ratio") or ratio) if isinstance(audit, dict) else float(ratio),
                "next_action": next_action_s,
            },
        )

        # Authoritative stop: committee says accept.
        if done and next_action_s == "accept":
            break

        # Fix pass when committee requests revision (or marks must_fix).
        if must_fix or next_action_s in ("revise", "extend_then_revise"):
            story = await _committee_revise_story(story, prompt_text, audit, trace_id=(trace_id or f"film2.story_revise.{pass_index}"))
            if not isinstance(story, dict):
                break
            story.setdefault("prompt", prompt_text)
            story.setdefault("duration_hint_s", float(duration))
            pass_index += 1
            continue

        # Extend pass when committee requests extension (or says length not ok).
        if (not length_ok) or next_action_s in ("extend", "extend_then_revise"):
            sys_msg = (
                "You are the Story Engine continuation pass.\n"
                "You MUST EXTEND the existing story graph to meet the requested duration and depth.\n"
                "Rules:\n"
                "- Keep all existing acts/scenes/beats/dialogue/events unchanged unless required for continuity.\n"
                "- Add NEW acts/scenes/beats to reach the duration. Deeper arcs for long films.\n"
                "- Maintain strict continuity: state_change events must be reflected later.\n"
                "- Maintain stable ids: never change existing ids; new ids must be unique and stable.\n"
                "- Output MUST be a single JSON object (no markdown, no prose).\n"
            )
            user_msg = json.dumps(
                {
                    "user_prompt": prompt_text,
                    "duration_hint_s": float(total_hint),
                    "current_beats_time_s": float(total_beats_time),
                    "coverage_ratio": float(ratio),
                    "audit_summary": audit,
                    "story": story,
                },
                ensure_ascii=False,
            )
            env2 = await committee_ai_text(
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                trace_id=(trace_id or f"film2.story_extend.{pass_index}"),
                rounds=STORY_COMMITTEE_ROUNDS,
                temperature=STORY_TEMPERATURE,
            )
            raw2 = ""
            rb2 = env2.get("result") if isinstance(env2, dict) else {}
            if isinstance(rb2, dict) and isinstance(rb2.get("text"), str):
                raw2 = str(rb2.get("text") or "")
            else:
                raw2 = str(env2.get("text") or "")
            extended = await _jsonify_then_parse(
                raw_text=raw2,
                expected_schema=_story_schema(),
                trace_id=(trace_id or f"film2.story_extend.jsonify.{pass_index}"),
                rounds=STORY_COMMITTEE_ROUNDS,
                temperature=STORY_TEMPERATURE,
            )
            if isinstance(extended, dict) and extended:
                story = extended
                story.setdefault("prompt", prompt_text)
                story.setdefault("duration_hint_s", float(duration))
                trace_event(
                    "story.extended",
                    {
                        "trace_id": trace_id,
                        "pass_index": int(pass_index),
                        "beats_time_s": float(total_beats_time),
                        "duration_hint_s": float(total_hint),
                    },
                )
                # Stall guard: if committee keeps asking to extend but the story doesn't grow, prevent infinite loops.
                grew = True
                if last_beats_time_s is not None and last_beats_count is not None:
                    grew = (float(total_beats_time) > float(last_beats_time_s)) or (int(total_beats_count) > int(last_beats_count))
                if not grew:
                    consecutive_no_progress += 1
                else:
                    consecutive_no_progress = 0
                last_beats_time_s = float(total_beats_time)
                last_beats_count = int(total_beats_count)
                if consecutive_no_progress >= 5:
                    trace_event(
                        "story.stalled",
                        {
                            "trace_id": trace_id,
                            "pass_index": int(pass_index),
                            "consecutive_no_progress": int(consecutive_no_progress),
                            "last_audit": audit,
                        },
                    )
                    break
                pass_index += 1
                continue

        # Default: if committee doesn't require fix/extend, stop.
        break
    acts_n = len(story.get("acts") or []) if isinstance(story.get("acts"), list) else 0
    chars_n = len(story.get("characters") or []) if isinstance(story.get("characters"), list) else 0
    log.info(
        "story.draft complete trace_id=%s dur_ms=%d prompt_len=%d acts=%d characters=%d",
        trace_id,
        int((time.monotonic() - t0) * 1000),
        len(prompt_text),
        acts_n,
        chars_n,
    )
    return story


async def check_story_consistency(story: Dict[str, Any], user_prompt: str, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Perform basic structural checks on a story graph.

    This function returns a list of issues. Each issue is a dict with at least "code" and "message".
    The initial implementation only verifies that acts/scenes/beats exist and that total timing is
    approximately consistent with the duration hint.
    """
    issues: List[Dict[str, Any]] = []
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    if not acts:
        issues.append({"code": "no_acts", "message": "Story has no acts."})
        return issues
    total_hint = float(story.get("duration_hint_s") or 0.0)
    total_beats = 0
    total_beats_time = 0.0
    # Basic structural and duration checks
    for act in acts:
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        if not scenes:
            issues.append({"code": "no_scenes", "message": "Act has no scenes.", "act_id": act.get("act_id")})
            continue
        for scene in scenes:
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            if not beats:
                issues.append(
                    {
                        "code": "no_beats",
                        "message": "Scene has no beats.",
                        "act_id": act.get("act_id"),
                        "scene_id": scene.get("scene_id"),
                    }
                )
                continue
            for beat in beats:
                total_beats += 1
                total_beats_time += float(beat.get("time_hint_s") or 0.0)
    if total_hint > 0.0 and total_beats_time > 0.0:
        ratio = total_beats_time / total_hint
        if ratio < 0.8 or ratio > 1.2:
            issues.append(
                {
                    "code": "duration_mismatch",
                    "message": "Beat timing does not match duration hint.",
                    "duration_hint_s": total_hint,
                    "beats_time_s": total_beats_time,
                }
            )
    # Character/object state timeline checks
    character_state: Dict[str, Dict[str, Any]] = {}
    for act in acts:
        act_id = act.get("act_id")
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        for scene in scenes:
            scene_id = scene.get("scene_id")
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat in beats:
                events = beat.get("events") if isinstance(beat.get("events"), list) else []
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("type") != "state_change":
                        continue
                    target = ev.get("target")
                    state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                    if not isinstance(target, str) or not state_delta:
                        continue
                    state_entry = character_state.get(target) or {}
                    for key, val in state_delta.items():
                        prev_val = state_entry.get(key)
                        if prev_val is not None and prev_val != val:
                            issues.append(
                                {
                                    "code": "state_inconsistent",
                                    "message": "Conflicting state change for target.",
                                    "target": target,
                                    "state_key": key,
                                    "prev": prev_val,
                                    "new": val,
                                    "act_id": act_id,
                                    "scene_id": scene_id,
                                }
                            )
                        state_entry[key] = val
                    character_state[target] = state_entry
    # Committee audit (deep continuity + intent). Merge issues.
    audit = await _committee_audit_story(story if isinstance(story, dict) else {}, _norm_prompt(user_prompt), trace_id=(trace_id or "story_check.audit"))
    ci = audit.get("issues") if isinstance(audit, dict) and isinstance(audit.get("issues"), list) else []
    for it in ci:
        if isinstance(it, dict):
            it.setdefault("code", "committee_issue")
            issues.append(it)
    trace_event(
        "story.consistency_audit",
        {
            "trace_id": trace_id,
            "issues_deterministic": int(len(issues) - len(ci)),
            "issues_committee": int(len(ci)),
            "must_fix": bool(audit.get("must_fix")) if isinstance(audit, dict) else False,
        },
    )
    return issues


async def ensure_tts_locks_and_dialogue_audio(
    story: Dict[str, Any],
    locks_arg: Dict[str, Any],
    profile_name: str,
    trace_id: Optional[str],
    tts_runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Ensure basic TTS lock entries and pre-generate dialogue audio for lines in the story.

    Returns an updated lock bundle and a dialogue_index mapping line_id -> metadata.
    """
    updated_locks = dict(locks_arg) if isinstance(locks_arg, dict) else {}
    dialogue_index: Dict[str, Any] = {}
    if not isinstance(story, dict):
        return updated_locks, dialogue_index
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    characters = story.get("characters") if isinstance(story.get("characters"), list) else []
    character_ids = [c.get("char_id") for c in characters if isinstance(c, dict) and isinstance(c.get("char_id"), str)]
    # Ensure a basic TTS lock branch exists per character
    tts_branch = updated_locks.get("tts")
    if not isinstance(tts_branch, dict):
        tts_branch = {}
        updated_locks["tts"] = tts_branch
    char_voice_locks = tts_branch.get("characters")
    if not isinstance(char_voice_locks, dict):
        char_voice_locks = {}
        tts_branch["characters"] = char_voice_locks
    for c_id in character_ids:
        if not isinstance(c_id, str) or not c_id:
            continue
        if c_id not in char_voice_locks:
            char_voice_locks[c_id] = {
                "char_id": c_id,
                "voice_profile": {
                    "description": f"Default voice profile for {c_id} based on story context",
                },
            }
    # Collect dialogue lines
    for act in acts:
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        for scene in scenes:
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat in beats:
                lines = beat.get("dialogue") if isinstance(beat.get("dialogue"), list) else []
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    line_id = line.get("line_id")
                    speaker = line.get("speaker")
                    text = line.get("text")
                    if not isinstance(line_id, str) or not line_id or not isinstance(text, str) or not text.strip():
                        continue
                    if not isinstance(speaker, str) or not speaker:
                        continue
                    dialogue_index[line_id] = {
                        "char_id": speaker,
                        "text": text,
                        "audio_path": None,
                        "duration_s": None,
                        "error": None,
                    }
    if not dialogue_index:
        return updated_locks, dialogue_index
    # Pre-generate audio via tts.speak for each unique line
    for line_id, entry in dialogue_index.items():
        speaker = entry.get("char_id")
        text = entry.get("text")
        if not isinstance(speaker, str) or not isinstance(text, str):
            continue
        args_tts: Dict[str, Any] = {
            "text": text,
            "trace_id": trace_id,
            "quality_profile": profile_name,
        }
        if isinstance(updated_locks, dict) and updated_locks:
            args_tts["lock_bundle"] = updated_locks
        args_tts["character_id"] = speaker
        res = await tts_runner({"name": "tts.speak", "arguments": args_tts})
        if isinstance(res, Dict) and isinstance(res.get("result"), dict):
            r = res.get("result") or {}
            meta_audio = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            ids_audio = r.get("ids") if isinstance(r.get("ids"), dict) else {}
            audio_id = ids_audio.get("audio_id")
            url = meta_audio.get("url")
            if isinstance(audio_id, str) and audio_id:
                entry["audio_path"] = audio_id
            elif isinstance(url, str) and url:
                entry["audio_path"] = url
            dur = meta_audio.get("full_duration_s")
            if isinstance(dur, (int, float)):
                entry["duration_s"] = float(dur)
            trace_event(
                "film2.tts_dialogue_generated",
                {
                    "trace_id": trace_id,
                    "line_id": line_id,
                    "char_id": speaker,
                    "text": (text if isinstance(text, str) else ""),
                    "audio_path": entry.get("audio_path"),
                    "duration_s": entry.get("duration_s"),
                    "quality_profile": profile_name,
                },
            )
        elif isinstance(res, Dict) and res.get("error") is not None:
            entry["error"] = str(res.get("error"))
            trace_event(
                "film2.tts_dialogue_failed",
                {
                    "trace_id": trace_id,
                    "line_id": line_id,
                    "char_id": speaker,
                    "error": str(res.get("error")),
                },
            )
    return updated_locks, dialogue_index


async def fix_story(story: Dict[str, Any], issues: List[Dict[str, Any]], user_prompt: str | None = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    AI-driven story revision.

    We keep the deterministic duration normalization, but the primary fix mechanism is committee revision.
    """
    if not issues:
        return story
    adjusted = dict(story or {})
    acts = adjusted.get("acts") if isinstance(adjusted.get("acts"), list) else []
    duration_hint = float(adjusted.get("duration_hint_s") or 0.0)
    has_duration_issue = False
    for issue in issues:
        code = issue.get("code")
        if code == "duration_mismatch":
            has_duration_issue = True
    if duration_hint > 0.0 and has_duration_issue:
        all_beats: List[Dict[str, Any]] = []
        for act in acts:
            scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
            for scene in scenes:
                beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
                for beat in beats:
                    all_beats.append(beat)
        if all_beats:
            per_beat = duration_hint / float(len(all_beats))
            for beat in all_beats:
                beat["time_hint_s"] = per_beat
    # For state_inconsistent issues, prefer to adjust later beats to match the first occurrence.
    for issue in issues:
        if issue.get("code") != "state_inconsistent":
            continue
        target = issue.get("target")
        state_key = issue.get("state_key")
        prev_val = issue.get("prev")
        new_val = issue.get("new")
        if not isinstance(target, str) or not isinstance(state_key, str):
            continue
        # Normalize by keeping the earliest value and updating later state_delta entries.
        canonical_val = prev_val
        for act in acts:
            scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
            for scene in scenes:
                beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
                for beat in beats:
                    events = beat.get("events") if isinstance(beat.get("events"), list) else []
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        if ev.get("type") != "state_change":
                            continue
                        if ev.get("target") != target:
                            continue
                        state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                        if state_key in state_delta:
                            state_delta[state_key] = canonical_val
                            ev["state_delta"] = state_delta
    adjusted["acts"] = acts
    # Committee revision using the issues we found (and the prompt if available)
    prompt_txt = _norm_prompt(user_prompt or adjusted.get("prompt") or "")
    audit = {"issues": issues, "summary": "deterministic_consistency_issues", "must_fix": True}
    revised = await _committee_revise_story(adjusted, prompt_txt, audit, trace_id=(trace_id or "story_fix"))
    if isinstance(revised, dict) and revised:
        return revised
    return adjusted


def derive_scenes_and_shots(story: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Derive flat scene and shot lists from the story graph.

    Scenes are flattened from acts[*].scenes. Shots are derived by creating one shot per beat.
    """
    scenes_out: List[Dict[str, Any]] = []
    shots_out: List[Dict[str, Any]] = []
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    scene_order = 0
    shot_index = 0
    # Maintain a simple per-character state timeline while deriving shots.
    character_state: Dict[str, Dict[str, Any]] = {}
    for act in acts:
        act_id = act.get("act_id") or f"act_{len(scenes_out)+1}"
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        for scene in scenes:
            scene_id = scene.get("scene_id") or f"{act_id}_sc_{scene_order+1}"
            scene_order += 1
            scene_entry: Dict[str, Any] = {
                "scene_id": scene_id,
                "act_id": act_id,
                "summary": scene.get("summary") or "",
                "order": scene_order,
            }
            scenes_out.append(scene_entry)
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat in beats:
                events = beat.get("events") if isinstance(beat.get("events"), list) else []
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("type") != "state_change":
                        continue
                    target = ev.get("target")
                    state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                    if not isinstance(target, str) or not state_delta:
                        continue
                    state_entry = character_state.get(target) or {}
                    for key, val in state_delta.items():
                        state_entry[key] = val
                    character_state[target] = state_entry
                shot_id = f"{scene_id}_sh_{shot_index+1}"
                shot_index += 1
                description = beat.get("description") or ""
                duration_s = float(beat.get("time_hint_s") or 0.0)
                char_ids = beat.get("characters") or []
                shot_states: Dict[str, Dict[str, Any]] = {}
                for cid in char_ids:
                    if isinstance(cid, str) and cid in character_state:
                        shot_states[cid] = dict(character_state[cid])
                shot_entry: Dict[str, Any] = {
                    "shot_id": shot_id,
                    "scene_id": scene_id,
                    "act_id": act_id,
                    "description": description,
                    "duration_s": duration_s,
                    "characters": char_ids,
                    "locations": beat.get("locations") or [],
                    "objects": beat.get("objects") or [],
                    "events": events,
                    "dialogue": beat.get("dialogue") or [],
                    "character_states": shot_states,
                }
                shots_out.append(shot_entry)
    return scenes_out, shots_out



