from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from void_envelopes import ToolEnvelope  # type: ignore

from ...json_parser import JSONParser
from ..decisions.committee import build_delta_plan
from ..telemetry.ledger import append_ledger
from ..metrics.review_scoring import score_review_image, score_review_audio, pick_first_review_artifacts


def review_score(body: Dict[str, Any]) -> ToolEnvelope:
    rid = uuid.uuid4().hex
    expected = {"kind": str, "path": str, "prompt": str}
    payload = JSONParser().parse(json.dumps(body or {}), expected)
    kind = (payload.get("kind") or "").strip().lower()
    path = payload.get("path") or ""
    prompt = (payload.get("prompt") or "").strip()
    if kind not in ("image", "audio", "music"):
        return ToolEnvelope.failure("invalid_kind", "invalid kind", status=400, request_id=rid, details={"kind": kind})
    if not isinstance(path, str) or not path.strip():
        return ToolEnvelope.failure("missing_path", "missing path", status=400, request_id=rid, details={"field": "path"})
    if kind == "image":
        scores = score_review_image(path=path, prompt=prompt)
    else:
        scores = score_review_audio(path=path)
    return ToolEnvelope.success({"scores": scores}, request_id=rid)


def review_plan(body: Dict[str, Any]) -> ToolEnvelope:
    rid = uuid.uuid4().hex
    payload = JSONParser().parse(json.dumps(body or {}), {"scores": dict})
    plan = build_delta_plan(payload.get("scores") or {})
    return ToolEnvelope.success({"plan": plan}, request_id=rid)


def review_loop(body: Dict[str, Any]) -> ToolEnvelope:
    rid = uuid.uuid4().hex
    expected = {"artifacts": list, "prompt": str}
    payload = JSONParser().parse(json.dumps(body or {}), expected)
    arts = payload.get("artifacts") or []
    prompt = (payload.get("prompt") or "").strip()
    # Single pass only: avoid infinite review loops.
    loop_idx = 0
    scores: Dict[str, Any] = {}
    img_path, aud_path = pick_first_review_artifacts(arts)
    if isinstance(img_path, str) and img_path:
        scores.update(score_review_image(path=img_path, prompt=prompt))
    if isinstance(aud_path, str) and aud_path:
        scores.update(score_review_audio(path=aud_path))
    plan = build_delta_plan(scores)
    append_ledger({"phase": f"review.loop#{loop_idx}", "scores": scores, "decision": plan})
    if plan.get("accept") is True:
        return ToolEnvelope.success({"loop_idx": loop_idx, "accepted": True, "plan": plan}, request_id=rid)
    return ToolEnvelope.success({"loop_idx": loop_idx, "accepted": False, "plan": plan}, request_id=rid)




