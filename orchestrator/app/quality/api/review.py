from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, List

from void_envelopes import ToolEnvelope  # type: ignore

from ...json_parser import JSONParser
from ...tracing.runtime import trace_event
from ..decisions.committee import build_delta_plan
from ..telemetry.ledger import append_ledger
from ..metrics.review_scoring import score_review_image, score_review_audio, score_review_video, select_review_artifacts

log = logging.getLogger(__name__)


def review_score(body: Dict[str, Any]):
    expected = {"trace_id": str, "conversation_id": str, "kind": str, "path": str, "prompt": str}
    try:
        payload = JSONParser().parse(json.dumps(body or {}), expected)
    except Exception as ex:
        log.error(f"review.score: JSONParser.parse failed body_keys={sorted((body or {}).keys()) if isinstance(body, dict) else type(body).__name__!r} ex={ex!r}", exc_info=True)
        return ToolEnvelope.failure(code="parse_error", message=f"Failed to parse request body: {ex}", trace_id="", conversation_id="", status=400, details={"exception": str(ex)})
    trace_id = payload.get("trace_id")
    conversation_id = payload.get("conversation_id")
    kind_raw = payload.get("kind")
    kind = str(kind_raw).lower() if kind_raw is not None else ""
    path = payload.get("path")
    prompt = payload.get("prompt")
    if not isinstance(trace_id, str) or not trace_id:
        log.error(f"review.score: missing trace_id conversation_id={conversation_id!r} kind={kind!r} - upstream caller must pass trace_id. Continuing with empty trace_id but this is an error.")
        trace_id = ""  # Continue processing but log the error
    if kind not in ("image", "audio", "music", "video", "film2"):
        trace_event("review.score.invalid_kind", {"trace_id": trace_id, "conversation_id": conversation_id, "kind": kind})
        return ToolEnvelope.failure(code="invalid_kind", message="invalid kind", trace_id=trace_id, conversation_id=conversation_id, status=400, details={"kind": kind})
    if not path:
        trace_event("review.score.missing_path", {"trace_id": trace_id, "conversation_id": conversation_id, "kind": kind})
        return ToolEnvelope.failure(code="missing_path", message="missing path", trace_id=trace_id, conversation_id=conversation_id, status=400, details={"field": "path"})
    try:
        trace_event("review.score.start", {"trace_id": trace_id, "conversation_id": conversation_id, "kind": kind, "path": path})
        scores: Dict[str, Any] = {}
        if kind == "image":
            scores = score_review_image(path=path, prompt=prompt)
        elif kind in ("video", "film2"):
            scores = score_review_video(path=path, prompt=prompt)
        else:
            scores = score_review_audio(path=path)
        if not isinstance(scores, dict):
            log.warning(f"review.score: scoring function returned non-dict type={type(scores).__name__!r} kind={kind!r} path={path!r} trace_id={trace_id!r}")
            scores = {}
        if not scores:
            log.warning(f"review.score: scoring function returned empty scores kind={kind!r} path={path!r} trace_id={trace_id!r}")
        trace_event("review.score.complete", {"trace_id": trace_id, "conversation_id": conversation_id, "kind": kind, "scores": scores})
        return ToolEnvelope.success(result={"scores": scores, "trace_id": trace_id, "conversation_id": conversation_id}, trace_id=trace_id, conversation_id=conversation_id)
    except Exception as ex:
        log.error(f"review.score.exception trace_id={trace_id!r} conversation_id={conversation_id!r} kind={kind!r} path={path!r} ex={ex!r}", exc_info=True)
        trace_event("review.score.exception", {"trace_id": trace_id, "conversation_id": conversation_id, "kind": kind, "path": path, "exception": str(ex)})
        return ToolEnvelope.failure(code="review_score_exception", message=f"Exception during review scoring: {ex}", trace_id=trace_id, conversation_id=conversation_id, status=500, details={"exception": str(ex), "kind": kind, "path": path})


def review_plan(body: Dict[str, Any]):
    try:
        payload = JSONParser().parse(json.dumps(body or {}), {"trace_id": str, "conversation_id": str, "scores": dict})
    except Exception as ex:
        log.error(f"review.plan: JSONParser.parse failed body_keys={sorted((body or {}).keys()) if isinstance(body, dict) else type(body).__name__!r} ex={ex!r}", exc_info=True)
        return ToolEnvelope.failure(code="parse_error", message=f"Failed to parse request body: {ex}", trace_id="", conversation_id="", status=400, details={"exception": str(ex)})
    trace_id = payload.get("trace_id")
    conversation_id = payload.get("conversation_id")
    if not isinstance(trace_id, str) or not trace_id:
        log.error(f"review.plan: missing trace_id conversation_id={conversation_id!r} - upstream caller must pass trace_id. Continuing with empty trace_id but this is an error.")
        trace_id = ""  # Continue processing but log the error
    try:
        trace_event("review.plan.start", {"trace_id": trace_id, "conversation_id": conversation_id})
        scores_input = payload.get("scores") or {}
        if not isinstance(scores_input, dict):
            log.warning(f"review.plan: scores is not a dict type={type(scores_input).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
            scores_input = {}
        plan = build_delta_plan(scores_input)
        if not isinstance(plan, dict):
            log.warning(f"review.plan: build_delta_plan returned non-dict type={type(plan).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
            plan = {"accept": True, "reasons": [], "patch_plan": [], "targets": {}}
        trace_event("review.plan.complete", {"trace_id": trace_id, "conversation_id": conversation_id, "plan": plan})
        return ToolEnvelope.success(result={"plan": plan, "trace_id": trace_id, "conversation_id": conversation_id}, trace_id=trace_id, conversation_id=conversation_id)
    except Exception as ex:
        log.error(f"review.plan.exception trace_id={trace_id!r} conversation_id={conversation_id!r} ex={ex!r}", exc_info=True)
        trace_event("review.plan.exception", {"trace_id": trace_id, "conversation_id": conversation_id, "exception": str(ex)})
        return ToolEnvelope.failure(code="review_plan_exception", message=f"Exception during review planning: {ex}", trace_id=trace_id, conversation_id=conversation_id, status=500, details={"exception": str(ex)})


def review_loop(body: Dict[str, Any], *, return_envelope: bool = True):
    expected = {"trace_id": str, "conversation_id": str, "artifacts": list, "prompt": str, "created_ms_floor": int}
    try:
        payload = JSONParser().parse(json.dumps(body or {}), expected)
    except Exception as ex:
        log.error(f"review.loop: JSONParser.parse failed body_keys={sorted((body or {}).keys()) if isinstance(body, dict) else type(body).__name__!r} ex={ex!r}", exc_info=True)
        if return_envelope:
            return ToolEnvelope.failure(code="parse_error", message=f"Failed to parse request body: {ex}", trace_id="", conversation_id="", status=400, details={"exception": str(ex)})
        return {"loop_idx": 0, "accepted": False, "plan": {}, "scores": {}, "per_item": [], "artifacts_selected": {}, "artifact_updates": [], "artifacts": [], "trace_id": "", "conversation_id": "", "error": f"parse_error: {ex}"}
    trace_id = payload.get("trace_id")
    conversation_id = payload.get("conversation_id")
    prompt = payload.get("prompt")
    try:
        created_ms_floor = int(payload.get("created_ms_floor") or 0)
    except (ValueError, TypeError) as ex:
        log.warning(f"review.loop: invalid created_ms_floor value={payload.get('created_ms_floor')!r} trace_id={trace_id!r} conversation_id={conversation_id!r} ex={ex!r}")
        created_ms_floor = 0
    if not isinstance(trace_id, str) or not trace_id:
        log.error(f"review.loop: missing trace_id conversation_id={conversation_id!r} - upstream caller must pass trace_id. Continuing with empty trace_id but this is an error.")
        trace_id = ""  # Continue processing but log the error
    try:
        trace_event("review.loop.start", {"trace_id": trace_id, "conversation_id": conversation_id, "artifacts_count": len(payload.get("artifacts") or []), "created_ms_floor": created_ms_floor})
        # Normalize artifacts once (correct datatypes), so the rest of the function can be simple.
        artifacts_schema = {
            "kind": str,
            "path": str,
            "tags": list,
            "tag": str,
            "summary": str,
            "meta": {
                "created_ms": int,
                "created_at": int,
                "review_count": int,
                "review": {"count": int},
            },
        }
        try:
            artifacts_parsed = JSONParser().parse(json.dumps(payload.get("artifacts") or []), {"artifacts": [artifacts_schema]})
            artifacts_list = artifacts_parsed.get("artifacts") if isinstance(artifacts_parsed, dict) else None
        except Exception as ex:
            log.warning(f"review.loop: artifacts JSONParser.parse failed trace_id={trace_id!r} conversation_id={conversation_id!r} ex={ex!r}", exc_info=True)
            artifacts_list = None
        if not isinstance(artifacts_list, list):
            if artifacts_list is not None:
                log.warning(f"review.loop: artifacts_list is not a list type={type(artifacts_list).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
            artifacts_list = []
        # HARD RULE: only review artifacts from THIS tool call scope (by created_ms floor).
        if isinstance(artifacts_list, list) and created_ms_floor > 0:
            artifacts_list = [a for a in artifacts_list if isinstance(a, dict) and int(((a.get("meta") or {}).get("created_ms") or 0)) >= created_ms_floor]
        # Single pass only: avoid infinite review loops.
        loop_idx = 0
        scores: Dict[str, Any] = {}
        per_item: List[Dict[str, Any]] = []
        artifact_updates: List[Dict[str, Any]] = []

        sel = select_review_artifacts(artifacts_list)
        if not isinstance(sel, dict):
            log.warning(f"review.loop: select_review_artifacts returned invalid type={type(sel).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r} artifacts_count={len(artifacts_list)}")
            sel = {"selected": {"image": [], "video": [], "audio": []}, "debug": {}}
        selected = sel.get("selected")
        if not isinstance(selected, dict):
            log.warning(f"review.loop: select_review_artifacts.selected is not a dict type={type(selected).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
            selected = {"image": [], "video": [], "audio": []}
        debug_sel = sel.get("debug")

        img_list = selected.get("image")
        vid_list = selected.get("video")
        aud_list = selected.get("audio")
        
        # Log selection summary
        img_count = len(img_list) if isinstance(img_list, list) else 0
        vid_count = len(vid_list) if isinstance(vid_list, list) else 0
        aud_count = len(aud_list) if isinstance(aud_list, list) else 0
        total_selected = img_count + vid_count + aud_count
        if total_selected == 0:
            log.info(f"review.loop: no artifacts selected for review trace_id={trace_id!r} conversation_id={conversation_id!r} artifacts_count={len(artifacts_list)}")
        else:
            log.debug(f"review.loop: selected artifacts for review trace_id={trace_id!r} conversation_id={conversation_id!r} images={img_count} videos={vid_count} audio={aud_count}")

        # Apply review tags/metadata back onto the artifact list so downstream sees review_count and scores.
        now_ms = int(time.time() * 1000)
        annotated_artifacts: List[Dict[str, Any]] = list(artifacts_list or [])

        for img_path in (img_list or []):
            if not img_path:
                log.warning(f"review.loop: empty img_path in img_list trace_id={trace_id!r} conversation_id={conversation_id!r} img_list_len={len(img_list) if isinstance(img_list, list) else 0}")
                continue
            scores_obj: Dict[str, Any] = {}
            try:
                scores_obj = score_review_image(path=img_path, prompt=prompt)
                if isinstance(scores_obj, dict):
                    scores.update(scores_obj)
                    per_item.append({"kind": "image", "path": img_path, "scores": scores_obj})
                else:
                    log.warning(f"review.loop: score_review_image returned non-dict type={type(scores_obj).__name__!r} path={img_path!r} trace_id={trace_id!r}")
                    scores_obj = {}
            except Exception as ex:
                log.warning(f"review.loop: score_review_image failed path={img_path!r} ex={ex!r}", exc_info=True)
                scores_obj = {}
            # Only update artifacts if scoring succeeded (scores_obj is not empty)
            if scores_obj:
                artifact_found = False
                for artifact_entry in annotated_artifacts:
                    if not isinstance(artifact_entry, dict):
                        log.warning(f"review.loop: invalid artifact_entry type={type(artifact_entry).__name__!r} path={img_path!r} trace_id={trace_id!r}")
                        continue
                    if artifact_entry.get("path") != img_path:
                        continue
                    artifact_found = True
                    meta = artifact_entry.get("meta") if isinstance(artifact_entry.get("meta"), dict) else {}
                    review = meta.get("review") if isinstance(meta.get("review"), dict) else {}
                    prev_count = int(review.get("count") or meta.get("review_count") or 0)
                    new_count = prev_count + 1
                    review_out = dict(review)
                    review_out["count"] = int(new_count)
                    review_out["last_ms"] = int(now_ms)
                    review_out["last_kind"] = "image"
                    review_out["scores"] = scores_obj
                    meta_out = dict(meta)
                    meta_out["review"] = review_out
                    meta_out["review_count"] = int(new_count)
                    tags_out: List[str] = [str(t) for t in (artifact_entry.get("tags") or [])]
                    for t in (f"review_count:{new_count}", "reviewed"):
                        if t not in tags_out:
                            tags_out.append(t)
                    artifact_entry["meta"] = meta_out
                    artifact_entry["tags"] = tags_out
                    artifact_updates.append({"path": artifact_entry.get("path"), "kind": "image", "review_count": new_count, "scores": scores_obj, "review_ms": now_ms})
                    break
                if not artifact_found:
                    log.warning(f"review.loop: artifact not found for path={img_path!r} trace_id={trace_id!r} conversation_id={conversation_id!r} kind=image")
            else:
                log.warning(f"review.loop: scoring failed for path={img_path!r} trace_id={trace_id!r} conversation_id={conversation_id!r} kind=image scores_obj_empty=True")

        for vid_path in (vid_list or []):
            if not vid_path:
                log.warning(f"review.loop: empty vid_path in vid_list trace_id={trace_id!r} conversation_id={conversation_id!r} vid_list_len={len(vid_list) if isinstance(vid_list, list) else 0}")
                continue
            scores_obj: Dict[str, Any] = {}
            try:
                scores_obj = score_review_video(path=vid_path, prompt=prompt)
                if isinstance(scores_obj, dict):
                    scores.update(scores_obj)
                    per_item.append({"kind": "video", "path": vid_path, "scores": scores_obj})
                else:
                    log.warning(f"review.loop: score_review_video returned non-dict type={type(scores_obj).__name__!r} path={vid_path!r} trace_id={trace_id!r}")
                    scores_obj = {}
            except Exception as ex:
                log.warning(f"review.loop: score_review_video failed path={vid_path!r} ex={ex!r}", exc_info=True)
                scores_obj = {}
            # Only update artifacts if scoring succeeded (scores_obj is not empty)
            if scores_obj:
                artifact_found = False
                for artifact_entry in annotated_artifacts:
                    if not isinstance(artifact_entry, dict):
                        log.warning(f"review.loop: invalid artifact_entry type={type(artifact_entry).__name__!r} path={vid_path!r} trace_id={trace_id!r}")
                        continue
                    if artifact_entry.get("path") != vid_path:
                        continue
                    artifact_found = True
                    meta = artifact_entry.get("meta") if isinstance(artifact_entry.get("meta"), dict) else {}
                    review = meta.get("review") if isinstance(meta.get("review"), dict) else {}
                    prev_count = int(review.get("count") or meta.get("review_count") or 0)
                    new_count = prev_count + 1
                    review_out = dict(review)
                    review_out["count"] = int(new_count)
                    review_out["last_ms"] = int(now_ms)
                    review_out["last_kind"] = "video"
                    review_out["scores"] = scores_obj
                    meta_out = dict(meta)
                    meta_out["review"] = review_out
                    meta_out["review_count"] = int(new_count)
                    tags_out: List[str] = [str(t) for t in (artifact_entry.get("tags") or [])]
                    for t in (f"review_count:{new_count}", "reviewed"):
                        if t not in tags_out:
                            tags_out.append(t)
                    artifact_entry["meta"] = meta_out
                    artifact_entry["tags"] = tags_out
                    artifact_updates.append({"path": artifact_entry.get("path"), "kind": "video", "review_count": new_count, "scores": scores_obj, "review_ms": now_ms})
                    break
                if not artifact_found:
                    log.warning(f"review.loop: artifact not found for path={vid_path!r} trace_id={trace_id!r} conversation_id={conversation_id!r} kind=video")
            else:
                log.warning(f"review.loop: scoring failed for path={vid_path!r} trace_id={trace_id!r} conversation_id={conversation_id!r} kind=video scores_obj_empty=True")

        for aud_path in (aud_list or []):
            if not aud_path:
                log.warning(f"review.loop: empty aud_path in aud_list trace_id={trace_id!r} conversation_id={conversation_id!r} aud_list_len={len(aud_list) if isinstance(aud_list, list) else 0}")
                continue
            scores_obj: Dict[str, Any] = {}
            try:
                scores_obj = score_review_audio(path=aud_path)
                if isinstance(scores_obj, dict):
                    scores.update(scores_obj)
                    per_item.append({"kind": "audio", "path": aud_path, "scores": scores_obj})
                else:
                    log.warning(f"review.loop: score_review_audio returned non-dict type={type(scores_obj).__name__!r} path={aud_path!r} trace_id={trace_id!r}")
                    scores_obj = {}
            except Exception as ex:
                log.warning(f"review.loop: score_review_audio failed path={aud_path!r} ex={ex!r}", exc_info=True)
                scores_obj = {}
            # Only update artifacts if scoring succeeded (scores_obj is not empty)
            if scores_obj:
                artifact_found = False
                for artifact_entry in annotated_artifacts:
                    if not isinstance(artifact_entry, dict):
                        log.warning(f"review.loop: invalid artifact_entry type={type(artifact_entry).__name__!r} path={aud_path!r} trace_id={trace_id!r}")
                        continue
                    if artifact_entry.get("path") != aud_path:
                        continue
                    artifact_found = True
                    meta = artifact_entry.get("meta") if isinstance(artifact_entry.get("meta"), dict) else {}
                    review = meta.get("review") if isinstance(meta.get("review"), dict) else {}
                    prev_count = int(review.get("count") or meta.get("review_count") or 0)
                    new_count = prev_count + 1
                    review_out = dict(review)
                    review_out["count"] = int(new_count)
                    review_out["last_ms"] = int(now_ms)
                    review_out["last_kind"] = "audio"
                    review_out["scores"] = scores_obj
                    meta_out = dict(meta)
                    meta_out["review"] = review_out
                    meta_out["review_count"] = int(new_count)
                    tags_out: List[str] = [str(t) for t in (artifact_entry.get("tags") or [])]
                    for t in (f"review_count:{new_count}", "reviewed"):
                        if t not in tags_out:
                            tags_out.append(t)
                    artifact_entry["meta"] = meta_out
                    artifact_entry["tags"] = tags_out
                    artifact_updates.append({"path": artifact_entry.get("path"), "kind": "audio", "review_count": new_count, "scores": scores_obj, "review_ms": now_ms})
                    break
                if not artifact_found:
                    log.warning(f"review.loop: artifact not found for path={aud_path!r} trace_id={trace_id!r} conversation_id={conversation_id!r} kind=audio")
            else:
                log.warning(f"review.loop: scoring failed for path={aud_path!r} trace_id={trace_id!r} conversation_id={conversation_id!r} kind=audio scores_obj_empty=True")

        # Log scoring summary before building plan
        if not scores:
            log.info(f"review.loop: no scores collected trace_id={trace_id!r} conversation_id={conversation_id!r} per_item_count={len(per_item)}")
        else:
            log.debug(f"review.loop: scores collected trace_id={trace_id!r} conversation_id={conversation_id!r} scores_keys={sorted(scores.keys()) if isinstance(scores, dict) else []} per_item_count={len(per_item)}")

        plan = build_delta_plan(scores)
        if not isinstance(plan, dict):
            log.warning(f"review.loop: build_delta_plan returned invalid type={type(plan).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r} scores_keys={sorted(scores.keys()) if isinstance(scores, dict) else type(scores).__name__!r}")
            plan = {"accept": True, "patch_plan": [], "reasons": []}
        try:
            append_ledger(
                {
                    "phase": f"review.loop#{loop_idx}",
                    "scores": scores,
                    "decision": plan,
                    "artifacts_selected": debug_sel,
                    "per_item": per_item,
                    "artifact_updates": artifact_updates,
                }
            )
        except Exception as ex:
            log.warning(f"review.loop: append_ledger failed trace_id={trace_id!r} conversation_id={conversation_id!r} ex={ex!r}", exc_info=True)
        accepted = bool(plan.get("accept") is True)
        result_obj = {
            "loop_idx": loop_idx,
            "accepted": accepted,
            "plan": plan,
            "scores": scores,
            "per_item": per_item,
            "artifacts_selected": debug_sel,
            "artifact_updates": artifact_updates,
            "artifacts": annotated_artifacts,
            "trace_id": trace_id,
            "conversation_id": conversation_id,
        }
        trace_event("review.loop.complete", {"trace_id": trace_id, "conversation_id": conversation_id, "loop_idx": loop_idx, "accepted": accepted, "scores_count": len(scores), "artifacts_count": len(annotated_artifacts)})
        if return_envelope:
            return ToolEnvelope.success(result=result_obj, trace_id=trace_id, conversation_id=conversation_id)
        return result_obj
    except Exception as ex:
        log.error(f"review.loop.exception trace_id={trace_id!r} conversation_id={conversation_id!r} ex={ex!r}", exc_info=True)
        trace_event("review.loop.exception", {"trace_id": trace_id, "conversation_id": conversation_id, "exception": str(ex)})
        error_result = {
            "loop_idx": loop_idx,
            "accepted": False,
            "plan": {},
            "scores": {},
            "per_item": [],
            "artifacts_selected": {},
            "artifact_updates": [],
            "artifacts": [],
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "error": str(ex),
        }
        if return_envelope:
            return ToolEnvelope.failure(code="review_loop_exception", message=f"Exception during review loop: {ex}", trace_id=trace_id, conversation_id=conversation_id, status=500, details={"exception": str(ex)})
        return error_result




