from __future__ import annotations

from typing import Any, Dict, List
import os
import logging
import json

from ..metrics.tool_results import collect_urls as assets_collect_urls
from ...plan.catalog import PLANNER_VISIBLE_TOOLS
from ...pipeline.catalog import build_allowed_tool_names as catalog_allowed
from ...tools_schema import get_builtin_tools_schema
from ...committee_client import committee_ai_text, committee_jsonify
from ...json_parser import JSONParser
from ...trace_utils import emit_trace

log = logging.getLogger(__name__)

# Trace storage root is sourced from environment to avoid importing runtime-heavy modules
# (and to keep this module usable in isolation).
_STATE_DIR_ENV = (os.getenv("STATE_DIR", "") or "").strip()
if _STATE_DIR_ENV:
    STATE_DIR = _STATE_DIR_ENV
else:
    STATE_DIR = os.path.join(os.getenv("UPLOAD_DIR", "/workspace/uploads"), "state")

_COMMITTEE_SEND_EMBEDDINGS = (os.getenv("COMMITTEE_SEND_EMBEDDINGS", "") or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _strip_embeddings_for_llm(obj: Any) -> Any:
    """
    Temporary LLM-context sanitizer:
    - DOES NOT affect tool args/envelopes.
    - Removes large embedding vectors from the payload to reduce token bloat.
    - Keeps a small descriptor so humans/LLM still know what existed.
    """
    if isinstance(obj, list):
        return [_strip_embeddings_for_llm(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        kk = str(k)
        lk = kk.lower()
        if lk in ("embedding", "embeddings", "id_embedding", "voice_embedding", "vec", "vector", "vectors"):
            if isinstance(v, list):
                out[kk] = {"_omitted": True, "kind": lk, "len": len(v)}
            elif isinstance(v, dict):
                out[kk] = {"_omitted": True, "kind": lk, "keys": sorted([str(x) for x in v.keys()])}
            else:
                out[kk] = {"_omitted": True, "kind": lk}
            continue
        out[kk] = _strip_embeddings_for_llm(v)
    return out


def _collect_embedding_context(obj: Any, *, path: str = "") -> List[Dict[str, Any]]:
    """
    Collect context about embeddings in a nested payload without assuming schema.
    Returns a list of {path, kind, len, entity_id, role, model, source}.
    """
    found: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_collect_embedding_context(v, path=f"{path}[{i}]"))
        return found
    if not isinstance(obj, dict):
        return found
    # Pull nearby identifiers for context
    entity_id = obj.get("entity_id") if isinstance(obj.get("entity_id"), (str, int)) else None
    role = obj.get("role") if isinstance(obj.get("role"), str) else None
    model = obj.get("model") if isinstance(obj.get("model"), str) else None
    for k, v in obj.items():
        kk = str(k)
        lk = kk.lower()
        next_path = f"{path}.{kk}" if path else kk
        if lk in ("embedding", "id_embedding", "voice_embedding", "vec", "vector"):
            if isinstance(v, list):
                found.append(
                    {
                        "path": next_path,
                        "kind": lk,
                        "len": len(v),
                        "entity_id": str(entity_id) if entity_id is not None else None,
                        "role": role,
                        "model": model,
                    }
                )
            else:
                found.append({"path": next_path, "kind": lk, "len": None, "entity_id": str(entity_id) if entity_id is not None else None, "role": role, "model": model})
            continue
        if lk == "embeddings" and isinstance(v, dict):
            # Common pattern: embeddings.id_embedding
            if isinstance(v.get("id_embedding"), list):
                found.append(
                    {
                        "path": next_path + ".id_embedding",
                        "kind": "id_embedding",
                        "len": len(v.get("id_embedding") or []),
                        "entity_id": str(entity_id) if entity_id is not None else None,
                        "role": role,
                        "model": (v.get("id_model") if isinstance(v.get("id_model"), str) else model),
                    }
                )
        found.extend(_collect_embedding_context(v, path=next_path))
    return found


def _committee_metric_guide() -> Dict[str, Any]:
    """
    Human/LLM guide so numeric values are not ambiguous.
    This is intentionally compact but explicit about meaning and thresholds.
    """
    return {
        "notes": [
            "Embeddings are opaque vectors used by lock scoring services (face/voice/etc). They are NOT directly interpretable by humans.",
            "Treat lock scores and QA scores as the decision signals. Use embeddings only to understand what reference/identity is being enforced and to propose lock/tool changes that preserve identity.",
            "Thresholds (quality gates) are provided in qa_metrics.thresholds when available; violations are listed in qa_metrics.threshold_violations.",
        ],
        "lock_scores": {
            "face_lock": "0..1, higher is better identity match (min/strict aggregation often used).",
            "region_shape_min": "0..1, higher is better; min across required regions.",
            "region_texture_min": "0..1, higher is better; min across required regions.",
            "scene_score": "0..1, higher is better scene/background adherence.",
            "style_score": "0..1, higher is better style adherence.",
            "pose_score": "0..1, higher is better pose adherence.",
            "voice_score/voice_lock": "0..1, higher is better voice identity match.",
            "tempo_score/key_score/stem_balance_score/lyrics_score": "0..1, higher is better audio lock adherence.",
        },
        "video_scores": {
            "temporal_stability": "0..1 heuristic; higher is sharper/more stable (less jitter/flicker).",
            "seam_ok_ratio": "0..1, higher is better across joins.",
        },
        "revision_playbook": {
            "how_to_decide": [
                "Prefer action='revise' when a small targeted fix can likely satisfy thresholds in <= 2 steps.",
                "Use action='fail' when: missing required modalities, repeated tool failures, or violations are severe and would require multiple reruns.",
                "When qa_metrics.threshold_violations is non-empty, you should either revise (if remaining budget) or fail; do NOT return go while violations exist.",
            ],
            "preferred_fix_order": [
                "1) Adjust locks (locks.update_region_modes / locks.update_audio_modes) to make the failing constraint 'hard' or increase strength.",
                "2) If locks already hard or the artifact is fundamentally wrong, run one refine step (image.refine.segment / video.refine.clip / music.refine.window).",
            ],
            "patch_templates": {
                "locks.update_region_modes": {
                    "when": "Any of: images.face_lock, images.region_shape_min, images.region_texture, images.scene_lock below threshold.",
                    "args_example": {"character_id": "<character_id>", "updates": {"face": "hard", "regions": "hard", "scene": "hard", "style": "hard"}},
                },
                "locks.update_audio_modes": {
                    "when": "Any of: audio.voice_lock/voice_score, tempo_lock, key_lock, stem_balance_lock, lyrics_lock below threshold.",
                    "args_example": {"character_id": "<character_id>", "update": {"voice_lock_mode": "hard", "tempo_lock_mode": "hard", "key_lock_mode": "hard", "stem_lock_mode": "hard", "lyrics_lock_mode": "hard"}},
                },
                "image.refine.segment": {
                    "when": "Image locks or image QA fail and a new render is needed.",
                    "args_example": {"segment_id": "<segment_id>", "refine_mode": "improve_quality", "quality_profile": "<profile>", "lock_bundle": "<reuse_from_segment_or_character>"},
                },
                "video.refine.clip": {
                    "when": "Clip-level metrics fail; choose refine_mode based on which score failed.",
                    "args_examples": [
                        {"segment_id": "<clip_segment_id>", "refine_mode": "fix_faces"},
                        {"segment_id": "<clip_segment_id>", "refine_mode": "stabilize_motion"},
                        {"segment_id": "<clip_segment_id>", "refine_mode": "fix_lipsync"},
                        {"segment_id": "<clip_segment_id>", "refine_mode": "improve_quality"},
                    ],
                },
                "music.refine.window": {
                    "when": "Music window lock metrics fail; refine that specific window.",
                    "args_example": {"segment_id": "<segment_id>", "window_id": "<window_id>", "quality_profile": "<profile>", "lock_bundle": "<reuse_from_character>"},
                },
            },
            "how_to_explain_embeddings": [
                "If embeddings are present, always name what they represent: face identity vs voice identity vs other.",
                "Use embedding_context entries (path/kind/entity_id/role/model) to reference them, not raw vector contents.",
                "Tie every decision to thresholds: cite qa_metrics.thresholds + qa_metrics.threshold_violations + key domain scores.",
            ],
        },
    }


def build_delta_plan(scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collapse primitive scores into an accept/decline decision with a simple patch plan.

    This function understands per-segment QA passed in under scores["segments"]
    and will emit segment-aware patch steps with segment_id when segments fall
    below hard-coded thresholds. It remains deliberately simple and deterministic.
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
    for segment in segments:
        segment_id = segment.get("segment_id")
        qa_block = segment.get("qa") if isinstance(segment.get("qa"), dict) else {}
        scores_block = qa_block if isinstance(qa_block, dict) else {}
        worst_score = None
        for val in scores_block.values():
            if isinstance(val, (int, float)):
                v = float(val)
                if worst_score is None or v < worst_score:
                    worst_score = v
        if isinstance(segment_id, str) and segment_id and isinstance(worst_score, float) and worst_score < segment_threshold:
            any_violation = True
            domain = segment.get("domain")
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
                patch_plan.append({"tool_name": tool_name, "tool": tool_name, "segment_id": segment_id, "args": {}})
    if not any_violation:
        return {"accept": True, "reasons": [], "patch_plan": [], "targets": {}}
    return {
        "accept": False,
        "reasons": ["segment_qa_below_threshold"],
        "patch_plan": patch_plan,
        "targets": {"segment_min": segment_threshold},
    }


def _allowed_tools_for_mode(mode: str | None) -> List[str]:
    allowed_set = catalog_allowed(get_builtin_tools_schema)
    return sorted([n for n in allowed_set if isinstance(n, str) and n.strip() and n in PLANNER_VISIBLE_TOOLS])


async def postrun_committee_decide(
    trace_id: str,
    user_text: str,
    tool_results: Dict[str, Any] | List[Dict[str, Any]],
    qa_metrics: Dict[str, Any],
    mode: str,
    *,
    state_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Post-run committee decision: returns {"action": "go|revise|fail", "rationale": str, "patch_plan": [ {tool,args} ]}.
    Reuses the same backend route as planner; strict JSON response required.
    """
    log.info(
        "postrun_committee_decide.start trace_id=%s mode=%s user_len=%d tool_results_type=%s qa_keys=%s",
        trace_id,
        mode,
        len(user_text or ""),
        type(tool_results).__name__,
        sorted(list(qa_metrics.keys())) if isinstance(qa_metrics, dict) else type(qa_metrics).__name__,
    )
    tools_used: List[str] = []
    for tr in (tool_results or []):
        if isinstance(tr, dict):
            tool_name = tr.get("tool_name")
            if isinstance(tool_name, str) and tool_name.strip():
                tools_used.append(tool_name.strip())
    tools_used = list(dict.fromkeys(tools_used))
    artifact_urls = assets_collect_urls(tool_results or [], str)
    allowed_for_mode = _allowed_tools_for_mode(mode)
    sys_hdr = (
        "### [COMMITTEE POSTRUN / SYSTEM]\n"
        "Decide whether to accept artifacts (go) or run one small revision (revise), or fail. "
        "Return strict JSON only: {\"action\":\"go|revise|fail\",\"rationale\":\"...\","
        "\"patch_plan\":[{\"tool_name\":\"<name>\",\"args\":{...}}]}.\n"
        "- IMPORTANT: The user payload includes a user_text field. Sometimes this is a JSON blob with scope=\"tool_step_only\". "
        "When scope=\"tool_step_only\", you MUST evaluate ONLY that single tool step. "
        "Use original_user_request only as global context; the PRIMARY goal is tool_intent_primary + tool_args_full + locks_and_bundles and the QA metrics/locks.\n"
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
    fail_counts: Dict[str, int] = {}
    for tr in (tool_results or []):
        if not isinstance(tr, dict):
            log.debug("postrun_committee_decide: skipping non-dict tool_result type=%s trace_id=%s", type(tr).__name__, trace_id)
            continue
        err = tr.get("error")
        if not err:
            res_tr = tr.get("result") if isinstance(tr.get("result"), dict) else {}
            err = res_tr.get("error")
        if isinstance(err, (dict, str)):
            tool_name = tr.get("tool_name")
            if isinstance(tool_name, str) and tool_name.strip():
                tool_name = tool_name.strip()
                fail_counts[tool_name] = int(fail_counts.get(tool_name, 0)) + 1
    user_blob = {
        "user_text": (user_text or ""),
        "qa_metrics": (qa_metrics or {}),
        "tools_used": tools_used,
        "artifact_urls": artifact_urls[:8],
        "allowed_tools_for_mode": allowed_for_mode,
        "mode": mode,
        "tool_failures": fail_counts,
    }
    # Provide explicit context so embeddings/scores are never "random numbers".
    # IMPORTANT: We do NOT modify tool_results/qa_metrics for downstream execution/envelopes.
    embedding_context = _collect_embedding_context({"qa_metrics": qa_metrics, "tool_results": tool_results})
    user_blob["metric_guide"] = _committee_metric_guide()
    user_blob["embedding_context"] = embedding_context[:50]
    user_blob["committee_send_embeddings"] = bool(_COMMITTEE_SEND_EMBEDDINGS)
    if not _COMMITTEE_SEND_EMBEDDINGS:
        user_blob = _strip_embeddings_for_llm(user_blob)
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
    env_decide = await committee_ai_text(messages=msgs, trace_id=trace_id, temperature=0.0)
    resolved_state_dir = (str(state_dir or "").strip() if state_dir is not None else STATE_DIR)
    if isinstance(env_decide, dict) and env_decide.get("ok"):
        res_decide = env_decide.get("result") or {}
        txt_decide = res_decide.get("text") or ""
        schema_decide = {
            "action": str,
            "rationale": str,
            "patch_plan": [
                {
                    "tool_name": str,
                    "name": str,
                    "id": str,
                    "args": object,
                    "arguments": object,
                    "meta": dict,
                }
            ],
        }
        obj_raw = await committee_jsonify(
            raw_text=txt_decide or "{}",
            expected_schema=schema_decide,
            trace_id=trace_id,
            temperature=0.0,
        )
        parser = JSONParser()
        try:
            obj = parser.parse(obj_raw if obj_raw is not None else "{}", schema_decide)
            if not isinstance(obj, dict):
                log.warning("postrun_committee_decide: JSONParser returned non-dict type=%s trace_id=%s", type(obj).__name__, trace_id)
                obj = {}
        except Exception as ex:
            log.warning("postrun_committee_decide: JSONParser.parse failed ex=%s trace_id=%s obj_raw_prefix=%s", ex, trace_id, (str(obj_raw) if obj_raw else "")[:200], exc_info=True)
            obj = {}
        emit_trace(
            trace_id=trace_id,
            kind="committee",
            payload={
                "mode": mode,
                "allowed_tools": allowed_for_mode,
                "tools_used": tools_used,
                "artifact_urls": artifact_urls[:8],
                "decision": obj,
                "raw_text": txt_decide,
            },
            state_dir=resolved_state_dir,
        )
        return obj
    emit_trace(
        trace_id=trace_id,
        kind="committee",
        payload={"mode": mode, "decision": None, "error": env_decide},
        state_dir=resolved_state_dir,
    )
    return {"action": "fail", "rationale": "committee_unavailable", "patch_plan": []}


