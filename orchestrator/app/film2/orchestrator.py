from __future__ import annotations

from typing import Dict, Any
from .clarifications import collect_one_shot
from .bible import write_story_bible, write_character_bible, merge_answers_into_bibles
from .planner import build_scenes, build_shots
from .refs import extract_refs_from_storyboards, inject_refs_into_final
from .qa import qa_storyboards, qa_animatic, apply_autofix
from .renderers import (
    render_thumbnails,
    render_storyboards,
    render_animatic,
    render_final_shots,
    assemble_final,
)
from .timeline import build_timeline, export_srt, export_edl
from .resume import Phase, load_checkpoint, save_checkpoint, mark_shot_done, is_shot_done
from .artifacts import add_artifact, new_manifest, save_manifest


def run_film(job: Dict[str, Any]) -> Dict[str, Any]:
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

    # Phase 3: Storyboards (with QA)
    if cp.phase < Phase.STORYBOARDS:
        thumbs = render_thumbnails(shots)
        sboards = render_storyboards(shots, thumbs)
        add_artifact(manifest, "thumbnails/", thumbs, dir=True)
        add_artifact(manifest, "storyboards/", sboards, dir=True)
        qa1 = qa_storyboards(sboards, char_bible)
        if getattr(qa1, "fail_rate", 0.0) > 0.15:
            sboards = apply_autofix(sboards, qa1)
            add_artifact(manifest, "storyboards/", sboards, dir=True, overwrite=True)
        save_checkpoint(cid, Phase.STORYBOARDS, extra={"qa_storyboards": qa1.to_dict()})
    else:
        sboards = render_storyboards(shots, render_thumbnails(shots))

    # Phase 4: Animatic (with QA)
    if cp.phase < Phase.ANIMATIC:
        animatic = render_animatic(shots, sboards)
        add_artifact(manifest, "animatic.mp4", animatic)
        qa2 = qa_animatic(animatic, shots, char_bible)
        if getattr(qa2, "fail_rate", 0.0) > 0.10:
            animatic = apply_autofix(animatic, qa2)
            add_artifact(manifest, "animatic.mp4", animatic, overwrite=True)
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


