from __future__ import annotations

from typing import Any, Dict, List
import logging
import json

from ..pipeline.assets import collect_urls as assets_collect_urls
from ..plan.catalog import PLANNER_VISIBLE_TOOLS
from ..pipeline.catalog import build_allowed_tool_names as catalog_allowed
from ..tools_schema import get_builtin_tools_schema
from ..committee_client import committee_ai_text, committee_jsonify, STATE_DIR
from ..trace_utils import emit_trace


def build_delta_plan(scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collapse primitive scores into an accept/decline decision with a simple patch plan.

    This function now understands per-segment QA passed in under scores["segments"]
    and will emit segment-aware patch steps with segment_id when segments fall
    below hard-coded thresholds. It remains deliberately simple and deterministic
    so that the wrapper can be swapped out for an LLM-based committee later.
    """
    segments: List[Dict[str, Any]] = []
    raw_segments = scores.get("segments")
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if isinstance(item, dict):
                segments.append(item)
    any_violation = False
    patch_plan: List[Dict[str, Any]] = []
    segment_threshold = 0.8
    for seg in segments:
        seg_id = seg.get("id")
        qa_block = seg.get("qa") if isinstance(seg.get("qa"), dict) else {}
        scores_block = qa_block if isinstance(qa_block, dict) else {}
        worst_score = None
        for val in scores_block.values():
            if isinstance(val, (int, float)):
                v = float(val)
                if worst_score is None or v < worst_score:
                    worst_score = v
        if isinstance(seg_id, str) and seg_id and isinstance(worst_score, float) and worst_score < segment_threshold:
            any_violation = True
            domain = seg.get("domain")
            tool_name = None
            if domain == "image":
                tool_name = "image.refine.segment"
            elif domain in ("video", "film2"):
                tool_name = "video.refine.clip"
            elif domain in ("music", "audio"):
                tool_name = "music.refine.window"
            elif domain == "tts":
                tool_name = "tts.refine.segment"
            if tool_name is not None:
                patch_plan.append(
                    {
                        "tool": tool_name,
                        "segment_id": seg_id,
                        "args": {},
                    }
                )
    if not any_violation:
        return {
            "accept": True,
            "reasons": [],
            "patch_plan": [],
            "targets": {},
        }
    return {
        "accept": False,
        "reasons": ["segment_qa_below_threshold"],
        "patch_plan": patch_plan,
        "targets": {"segment_min": segment_threshold},
    }


def _allowed_tools_for_mode(mode: str | None) -> List[str]:
    """
    Return the allowed tool names for planner/committee use for a given mode.

    All planner-visible tools are allowed in all modes; mode may still affect prompts, not tool lists.
    """
    allowed_set = catalog_allowed(get_builtin_tools_schema)
    return sorted(
        [
            n
            for n in allowed_set
            if isinstance(n, str) and n.strip() and n in PLANNER_VISIBLE_TOOLS
        ]
    )


async def postrun_committee_decide(
    trace_id: str,
    user_text: str,
    tool_results: Dict[str, Any] | List[Dict[str, Any]],
    qa_metrics: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """
    Post-run committee decision: returns {"action": "go|revise|fail", "rationale": str, "patch_plan": [ {tool,args} ]}.
    Reuses the same backend route as planner; strict JSON response required.
    """
    # Summarize tools used and artifact urls
    tools_used: List[str] = []
    for tr in (tool_results or []):
        if isinstance(tr, dict):
            nm = (tr.get("name") or tr.get("tool") or "")
            if isinstance(nm, str) and nm.strip():
                tools_used.append(nm.strip())
    tools_used = list(dict.fromkeys(tools_used))
    artifact_urls = assets_collect_urls(tool_results or [], lambda u: u)
    # Allowed tools for this mode (front-door only)
    allowed_for_mode = _allowed_tools_for_mode(mode)
    # Build committee prompt
    sys_hdr = (
        "### [COMMITTEE POSTRUN / SYSTEM]\n"
        "Decide whether to accept artifacts (go) or run one small revision (revise), or fail. "
        "Return strict JSON only: {\"action\":\"go|revise|fail\",\"rationale\":\"...\"," 
        "\"patch_plan\":[{\"tool\":\"<name>\",\"args\":{...}}]}.\n"
        "- Inspect qa_metrics.counts and qa_metrics.domain.* to judge quality. Example triggers: "
        "missing required modalities; images hands_ok_ratio < 0.85 or face/id/region/scene locks below profile thresholds; "
        "videos seam_ok_ratio < 0.90; audio clipping_ratio > 0.0, tempo/key/stem/lyrics/voice lock scores below threshold, or LUFS far from -14.\n"
        "- When any hard lock metric (face_lock, region_shape_min, scene_lock, voice_lock, tempo_lock, key_lock, stem_balance_lock, lyrics_lock) is below the quality_profile threshold, prefer action=\"revise\" with a patch_plan that increases lock strength or updates mode/inputs via locks.update_region_modes or locks.update_audio_modes; escalate to \"fail\" if refinement budget is exhausted.\n"
        "- When tool_results contain errors (tr.error or tr.result.error), treat them as canonical envelope errors: read error.code, error.status, and error.details to understand what failed and why before deciding.\n"
        "- Keep patch_plan minimal (<=2 steps) and prefer lock adjustment tools or a single regeneration step that reuses existing bundles. Tools must be chosen only from the planner-visible front-door set and allowed for the current mode. In chat/analysis mode, do not include any action tools in patch_plan.\n"
        "- In chat/analysis mode, do not include any action tools in patch_plan.\n"
        "- The executor validates once and runs each tool step once. It does NOT retry or repair automatically; any retries must be explicit new steps in patch_plan based on the observed errors and QA metrics.\n"
        "- Document threshold violations, tool errors, and key metrics in rationale so humans understand why content was revised or failed.\n"
    )
    allowed_list = ", ".join(allowed_for_mode or [])
    mode_note = f"mode={mode}"
    # Compute per-tool failure counts so the committee can avoid proposing
    # the same failing tools indefinitely.
    fail_counts: Dict[str, int] = {}
    for tr in (tool_results or []):
        if not isinstance(tr, dict):
            continue
        err = tr.get("error")
        if not err:
            res_tr = tr.get("result") if isinstance(tr.get("result"), dict) else {}
            err = res_tr.get("error")
        if isinstance(err, (dict, str)):
            nm = str((tr.get("name") or tr.get("tool") or "")).strip()
            if nm:
                fail_counts[nm] = int(fail_counts.get(nm, 0)) + 1
    user_blob = {
        "user_text": (user_text or ""),
        "qa_metrics": (qa_metrics or {}),
        "tools_used": tools_used,
        "artifact_urls": artifact_urls[:8],
        "allowed_tools_for_mode": allowed_for_mode,
        "mode": mode,
        "tool_failures": fail_counts,
    }
    msgs = [
        {
            "role": "system",
            "content": sys_hdr
            + f"Allowed tools: {allowed_list}\nCurrent {mode_note}.\n"
            + "If a tool has already failed multiple times in this trace (see tool_failures), you MUST NOT propose running it again; "
              "prefer either a different tool or action=\"fail\" instead of looping.",
        },
        {"role": "user", "content": json.dumps(user_blob, ensure_ascii=False)},
        {"role": "system", "content": "Return ONLY a single JSON object with keys: action, rationale, patch_plan."},
    ]
    # Two-model postrun committee: gather decisions from all participants and merge.
    # Post-run decision is now derived from a single committee debate call.
    decisions: List[Dict[str, Any]] = []
    env_decide = await committee_ai_text(
        msgs,
        trace_id=str(trace_id or "committee_postrun"),
        temperature=0.0,
    )
    if isinstance(env_decide, dict) and env_decide.get("ok"):
        res_decide = env_decide.get("result") or {}
        txt_decide = res_decide.get("text") or ""
        # IMPORTANT: patch args can be dict OR JSON string OR other scalar-ish values.
        # Use `object` passthrough so we do not drop/overwrite LLM-provided fields.
        schema_decide = {
            "action": str,
            "rationale": str,
            "patch_plan": [
                {
                    "tool": str,
                    "name": str,
                    "id": str,
                    "args": object,
                    "arguments": object,
                    "meta": dict,
                }
            ],
        }
        obj = await committee_jsonify(
            txt_decide or "{}",
            expected_schema=schema_decide,
            trace_id=str(trace_id or "committee_postrun"),
            temperature=0.0,
        )
        action = (obj.get("action") or "").strip().lower() or "go"
        rationale = (obj.get("rationale") or "") if isinstance(obj.get("rationale"), str) else ""
        steps = obj.get("patch_plan") or []
        decisions.append(
            {
                "member": "committee",
                "action": action,
                "rationale": rationale,
                "patch_plan": steps if isinstance(steps, list) else [],
            }
        )
        emit_trace(
            STATE_DIR,
            str(trace_id or "committee_postrun"),
            "committee.postrun.member",
            {"member": "committee", "action": action},
        )

    # Default outcome if committee fails to produce a decision.
    out: Dict[str, Any]
    if not decisions:
        # Committee path failed or produced no usable decision. Surface a clear
        # marker and attach any underlying error from env_decide instead of
        # returning a plain "committee_unavailable" with no details, and emit
        # a high‑visibility log so this failure is never silent.
        err = (env_decide or {}).get("error") if isinstance(env_decide, dict) else {
            "code": "committee_postrun_invalid_env",
            "message": str(env_decide),
        }
        logging.getLogger("orchestrator.committee.postrun").error(
            "postrun_committee_decide failed (trace_id=%s): env=%r error=%r",
            str(trace_id or "committee_postrun"),
            env_decide,
            err,
        )
        out = {
            "action": "go",
            "rationale": "committee_unavailable",
            "patch_plan": [],
            "committee_error": err,
        }
        emit_trace(
            STATE_DIR,
            str(trace_id or "committee_postrun"),
            "committee.postrun.ok",
            {"action": out.get("action"), "rationale": out.get("rationale"), "error": err},
        )
    else:
        # Merge decisions: prefer more conservative actions when disagreeing.
        rank = {"go": 0, "revise": 1, "fail": 2}
        best = max(decisions, key=lambda d: rank.get(d.get("action") or "go", 0))
        merged_action = best.get("action") or "go"
        # Merge rationales and patch plans (cap patch_plan length to 2 steps).
        merged_rationale_parts: List[str] = []
        merged_patch_plan: List[Dict[str, Any]] = []
        for d in decisions:
            r = d.get("rationale") or ""
            if isinstance(r, str) and r.strip():
                merged_rationale_parts.append(r.strip())
            steps = d.get("patch_plan") or []
            if isinstance(steps, list):
                for st in steps:
                    if isinstance(st, dict) and len(merged_patch_plan) < 2:
                        merged_patch_plan.append(st)
        out = {
            "action": merged_action,
            "rationale": "\n\n".join(merged_rationale_parts)[:2000],
            "patch_plan": merged_patch_plan[:2],
        }
        emit_trace(
            STATE_DIR,
            str(trace_id or "committee_postrun"),
            "committee.postrun.ok",
            {"action": out.get("action"), "rationale": out.get("rationale")},
        )
    return out


