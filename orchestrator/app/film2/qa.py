from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from app.routes.toolrun import http_tool_run  # type: ignore


@dataclass
class QaReport:
    fail_rate: float
    issues: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"fail_rate": float(self.fail_rate), "issues": list(self.issues or [])}


def qa_storyboards(sboards_dir, char_bible) -> QaReport:
    issues: List[Dict[str, Any]] = []
    total = 0
    fails = 0
    for sb in (sboards_dir or []):
        if not isinstance(sb, dict):
            continue
        total += 1
        if not sb.get("path"):
            fails += 1
            issues.append({"shot": sb.get("id"), "issue": "missing_image"})
        if not sb.get("style"):
            issues.append({"shot": sb.get("id"), "issue": "style_missing"})
    rate = (fails / float(total)) if total else 0.0
    return QaReport(fail_rate=rate, issues=issues)


def qa_animatic(animatic_path, shots, char_bible) -> QaReport:
    total = sum(int(s.get("duration_ms", 0)) for s in (shots or []))
    dur = int((animatic_path or {}).get("duration", 0)) * 1000
    dur_ok = abs(dur - total) <= int(total * 0.03) if total else True
    issues: List[Dict[str, Any]] = []
    if not dur_ok:
        issues.append({"issue": "duration_mismatch"})
    return QaReport(fail_rate=(0.0 if dur_ok else 0.2), issues=issues)


async def apply_autofix(
    shots: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Single-pass autofix for Film2 storyboard-style shots.

    For each shot with reported issues (including identity/layout issues surfaced
    by /film2/qa), this will invoke image.dispatch at most once in hero mode and
    attach the new image back to the shot. No loops or recursive refinement.
    """
    issues = report.get("issues") or []
    issues_by_shot: Dict[str, List[Dict[str, Any]]] = {}
    for item in issues:
        sid = str(item.get("shot") or item.get("shot_id") or "").strip()
        if not sid:
            continue
        issues_by_shot.setdefault(sid, []).append(item)

    fixed: List[Dict[str, Any]] = []
    for shot in shots or []:
        if not isinstance(shot, dict):
            fixed.append(shot)
            continue
        sid = str(shot.get("id") or shot.get("shot_id") or "").strip()
        shot_issues = issues_by_shot.get(sid) or []
        if not shot_issues:
            fixed.append(shot)
            continue

        prompt = str(
            shot.get("prompt")
            or shot.get("description")
            or shot.get("caption")
            or ""
        )
        char_id = str(
            shot.get("character_id")
            or shot.get("lock_character_id")
            or ""
        ).strip()

        dispatch_args: Dict[str, Any] = {
            "mode": "gen",
            "prompt": prompt,
            "quality_profile": "hero",
            "film_shot_id": sid,
        }
        if char_id:
            dispatch_args["character_id"] = char_id

        try:
            res = await http_tool_run("image.dispatch", dispatch_args)
            result_block = res.get("result") or {}
            # Adapt to the actual result shape; fall back to generic keys.
            new_image = (
                result_block.get("image_url")
                or result_block.get("image")
                or result_block.get("view_url")
            )
            if isinstance(new_image, str) and new_image.strip():
                shot["image"] = new_image
            shot["qa_autofix"] = {"applied": True, "issues": shot_issues}
        except Exception:
            # Best-effort: keep the original shot if autofix fails.
            shot["qa_autofix"] = {"applied": False, "issues": shot_issues}
        fixed.append(shot)

    return fixed


