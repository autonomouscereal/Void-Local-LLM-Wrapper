from __future__ import annotations

import os
from typing import Any, Dict

from .artifacts import add_artifact, new_manifest, save_manifest
from .bible import merge_answers_into_bibles, write_character_bible, write_story_bible
from .clarifications import collect_one_shot
from .planner import build_scenes, build_shots
from .qa import apply_autofix, qa_animatic, qa_storyboards
from .refs import extract_refs_from_storyboards, inject_refs_into_final
from .renderers import (
    assemble_final,
    render_animatic,
    render_final_shots,
    render_storyboards,
    render_thumbnails,
)
from .resume import Phase, is_shot_done, load_checkpoint, mark_shot_done, save_checkpoint
from .timeline import build_timeline, export_edl, export_srt


def _autofix_enabled() -> bool:
    """
    Gate Film2 autofix behavior behind an environment variable so it can be
    enabled/disabled without code changes.

    FILM2_AUTOFIX:
      - "1", "true", "yes", "on" (case-insensitive) => enabled
      - anything else or unset => disabled
    """
    val = os.getenv("FILM2_AUTOFIX", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


async def run_film(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job: {"prompt": str, "answers": Optional[dict], "preset": {...}, "cid": str}
    Deterministic, resumable state machine. Returns phase + artifacts list.
    """
    cid = str(job.get("cid") or "film_job")
    manifest = new_manifest(cid)
    cp = load_checkpoint(cid)

    # Phase 0: Clarifications (one-shot)
    if cp.phase < Phase.CLARIFY:
        questions = collect_one_shot(job.get("prompt", ""), job.get("preset", {}))
        if questions and not job.get("answers"):
            return {"phase": "clarify", "questions": questions, "artifacts": manifest.list()}
        answers = job.get("answers", {})
        save_checkpoint(cid, Phase.CLARIFY)
    else:
        answers = job.get("answers", {})

    # Phase 1: Bibles
    if cp.phase < Phase.BIBLES:
        story_bible = write_story_bible(job.get("prompt", ""), job.get("preset", {}))
        char_bible = write_character_bible(job.get("prompt", ""), job.get("preset", {}))
        merge_answers_into_bibles(story_bible, char_bible, answers)
        add_artifact(manifest, "story_bible.json", story_bible)
        add_artifact(manifest, "character_bible.json", char_bible)
        save_checkpoint(cid, Phase.BIBLES)
    else:
        # try to read prior artifacts; fallback to regenerating
        story_bible = write_story_bible(job.get("prompt", ""), job.get("preset", {}))
        char_bible = write_character_bible(job.get("prompt", ""), job.get("preset", {}))

    # Phase 2: Planner
    if cp.phase < Phase.PLANNER:
        scenes = build_scenes(story_bible)
        shots = build_shots(scenes, char_bible, coverage=job.get("preset", {}).get("coverage", "basic"))
        add_artifact(manifest, "scenes.json", scenes)
        add_artifact(manifest, "shots.json", shots)
        save_checkpoint(cid, Phase.PLANNER)
    else:
        scenes = build_scenes(story_bible)
        shots = build_shots(scenes, char_bible, coverage=job.get("preset", {}).get("coverage", "basic"))

    # Phase 3: Storyboards (with QA + single-pass autofix)
    if cp.phase < Phase.STORYBOARDS:
        thumbs = render_thumbnails(shots)
        sboards = render_storyboards(shots, thumbs)
        add_artifact(manifest, "thumbnails/", thumbs, dir=True)
        add_artifact(manifest, "storyboards/", sboards, dir=True)
        qa1 = qa_storyboards(sboards, char_bible)
        qa1_dict = qa1.to_dict()
        fail_rate = float(getattr(qa1, "fail_rate", 0.0) or qa1_dict.get("fail_rate") or 0.0)
        # Single hero-style redraw pass for weak storyboards, gated by FILM2_AUTOFIX.
        if _autofix_enabled() and fail_rate > 0.15:
            sboards = await apply_autofix(sboards, qa1_dict)
            add_artifact(manifest, "storyboards/", sboards, dir=True, overwrite=True)
        save_checkpoint(cid, Phase.STORYBOARDS, extra={"qa_storyboards": qa1_dict})
    else:
        sboards = render_storyboards(shots, render_thumbnails(shots))

    # Phase 4: Animatic (with QA)
    if cp.phase < Phase.ANIMATIC:
        animatic = render_animatic(shots, sboards)
        add_artifact(manifest, "animatic.mp4", animatic)
        qa2 = qa_animatic(animatic, shots, char_bible)
        save_checkpoint(cid, Phase.ANIMATIC, extra={"qa_animatic": qa2.to_dict()})
    else:
        animatic = render_animatic(shots, sboards)

    # Phase 5: Final (with reference locking)
    if cp.phase < Phase.FINAL:
        refs = extract_refs_from_storyboards(sboards)
        finals = render_final_shots(shots, refs)
        for s in finals:
            if is_shot_done(cid, s.get("id")):
                continue
            out = s.get("video")
            add_artifact(manifest, f"final_shots/{s['id']}.mp4", out)
            mark_shot_done(cid, s.get("id"))
        final = assemble_final(finals)
        add_artifact(manifest, "final.mp4", final)
        save_checkpoint(cid, Phase.FINAL)
    else:
        finals = render_final_shots(shots, extract_refs_from_storyboards(sboards))
        final = assemble_final(finals)

    # Phase 6: Exports (single timeline → SRT & EDL)
    if cp.phase < Phase.EXPORTS:
        timeline = build_timeline(shots, final_duration=int(final.get("duration", 0)))
        srt = export_srt(timeline)
        edl = export_edl(timeline)
        add_artifact(manifest, "captions.srt", srt)
        add_artifact(manifest, "edl.json", edl)
        save_manifest(manifest)
        save_checkpoint(cid, Phase.EXPORTS)

    return {"phase": "done", "artifacts": manifest.list(), "cid": cid}


