from __future__ import annotations

"""
Film2 + Story lock helpers.

This file was previously `orchestrator/app/film2/locks.py` but it conceptually
belongs in the canonical `app/locks/` subsystem:
- it ensures per-character persisted lock bundles exist (visual/music/tts/sfx/film2)
- it enriches a runtime locks payload using a story graph
- it provides thin helpers that apply ref packs to storyboard/TTS/music args

Naming rationale:
- "film2_story_locks" is specific to the Film2 runtime story→locks enrichment path
  and avoids the generic/overloaded `locks.py` filename.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable

from ..ref_library.apply import ref_pack_for_ref_id
from .store import get_lock_bundle as _lock_load
from .runtime import (
    migrate_visual_bundle as _lock_migrate_visual,
    migrate_music_bundle as _lock_migrate_music,
    migrate_tts_bundle as _lock_migrate_tts,
    migrate_sfx_bundle as _lock_migrate_sfx,
    migrate_film2_bundle as _lock_migrate_film2,
    ensure_visual_lock_bundle as _lock_ensure_visual,
)
from ..tracing.runtime import trace_event


log = logging.getLogger(__name__)


def _apply_ref_library_ref(ref_library_ref_id: str | None) -> Dict[str, Any]:
    """
    Apply a reference library ref_id to get a ref pack.
    
    Args:
        ref_library_ref_id: A ref_id from the ref_library system (not an artifact_id)
    
    Returns:
        Dict with ref pack (e.g., {"images": [...]} or {"track": ..., "stems": [...]})
    """
    if not ref_library_ref_id:
        return {}
    return ref_pack_for_ref_id(ref_library_ref_id)


def build_ref_pack(char_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a ref pack from character metadata.
    
    All IDs here are ref_library ref_ids (not artifact_ids).
    They point to ref library manifests that contain multiple files.
    """
    pack: Dict[str, Any] = {"image": {}, "voice": {}, "music": {}}
    if not isinstance(char_meta, dict):
        return pack
    image_ref_library_ref_id = char_meta.get("image_ref_id")
    voice_ref_library_ref_id = char_meta.get("voice_ref_id")
    music_ref_library_ref_id = char_meta.get("music_ref_id")
    if image_ref_library_ref_id:
        pack["image"] = _apply_ref_library_ref(image_ref_library_ref_id)
    if voice_ref_library_ref_id:
        pack["voice"] = _apply_ref_library_ref(voice_ref_library_ref_id)
    if music_ref_library_ref_id:
        pack["music"] = _apply_ref_library_ref(music_ref_library_ref_id)
    return pack


def apply_storyboard_locks(shot: Dict[str, Any], ref_pack: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "mode": "gen",
        "prompt": shot.get("prompt") or "",
        "size": shot.get("size", "1024x1024"),
        "refs": (ref_pack or {}).get("image") or {},
        "seed": seed,
    }


def apply_animatic_vo_locks(line: Dict[str, Any], ref_pack: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "text": line.get("text") or "",
        "voice_id": line.get("voice_ref_id"),
        "voice_refs": (ref_pack or {}).get("voice") or {},
        "seed": seed,
    }


def apply_music_cue_locks(cue: Dict[str, Any], ref_pack: Dict[str, Any], seed: int) -> Dict[str, Any]:
    # artifact_id is the music identifier
    artifact_id = cue.get("artifact_id")
    return {
        "prompt": cue.get("prompt") or "",
        "bpm": cue.get("bpm"),
        "length_s": cue.get("length_s", 30),
        "artifact_id": artifact_id,
        "music_refs": (ref_pack or {}).get("music") or {},
        "seed": seed,
    }


async def ensure_story_character_bundles(locks_arg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Ensure that all characters referenced in a story/film2 locks payload have
    visual lock bundles available, migrating any existing bundles into the
    latest schema.

    Returns a mapping {character_id: bundle}.
    """
    t0 = time.perf_counter()
    character_entries = locks_arg.get("characters") if isinstance(locks_arg.get("characters"), list) else []
    character_ids: List[str] = []
    for entry in character_entries:
        character_id = entry.get("character_id")
        if isinstance(character_id, str) and character_id.strip():
            character_ids.append(character_id.strip())
    out: Dict[str, Dict[str, Any]] = {}
    for character_id in character_ids:
        bundle_existing = await _lock_load(character_id)
        if isinstance(bundle_existing, dict):
            bundle_existing = _lock_migrate_visual(bundle_existing)
            bundle_existing = _lock_migrate_music(bundle_existing)
            bundle_existing = _lock_migrate_tts(bundle_existing)
            bundle_existing = _lock_migrate_sfx(bundle_existing)
            bundle_existing = _lock_migrate_film2(bundle_existing)
            out[character_id] = bundle_existing
        else:
            # No bundle yet for this character: create a minimal visual skeleton
            # using the canonical locks.runtime helper.
            bundle_new = await _lock_ensure_visual(character_id, None)
            bundle_new = _lock_migrate_visual(bundle_new)
            out[character_id] = bundle_new
    log.info(
        "film2.locks.ensure_story_character_bundles done characters=%d dur_ms=%d",
        len(out),
        int((time.perf_counter() - t0) * 1000),
    )
    return out


async def ensure_visual_locks_for_story(
    story: Dict[str, Any],
    locks_arg: Dict[str, Any],
    profile_name: str,
    trace_id: Optional[str],
) -> Dict[str, Any]:
    """
    Best-effort helper to enrich a locks payload with basic story/film2 metadata.

    This function ensures that:
    - All characters known from locks_arg and the story graph have a persisted
      visual lock bundle (via the canonical locks.runtime helpers), and
    - The film2.project block is populated with basic structural metadata.

    It never raises; on any failure it simply returns the original locks
    mapping unchanged.
    """
    t0 = time.perf_counter()
    updated: Dict[str, Any] = dict(locks_arg) if isinstance(locks_arg, dict) else {}
    if not isinstance(story, dict):
        return updated
    # ------------------------------------------------------------------
    # 1) Ensure character entries exist in the locks payload
    # ------------------------------------------------------------------
    char_entries = updated.get("characters")
    if not isinstance(char_entries, list):
        char_entries = []
        updated["characters"] = char_entries
    existing_ids = {
        character.get("character_id")
        for character in char_entries
        if isinstance(character, dict) and isinstance(character.get("character_id"), str) and character.get("character_id").strip()
    }
    # Characters declared on the story graph (story.engine)
    story_characters = story.get("characters") if isinstance(story.get("characters"), list) else []
    for character in story_characters:
        if not isinstance(character, dict):
            continue
        character_id = character.get("character_id")
        if not isinstance(character_id, str) or not character_id.strip():
            continue
        character_id = character_id.strip()
        if character_id in existing_ids:
            continue
        # Create a best-effort character entry using story metadata so that
        # downstream visual/music/TTS pipelines can lock against it.
        entry: Dict[str, Any] = {"character_id": character_id}
        character_name = character.get("name")
        descr = character.get("description")
        traits = character.get("traits")
        if isinstance(character_name, str) and character_name.strip():
            entry["name"] = character_name.strip()
        if isinstance(descr, str) and descr.strip():
            entry["description"] = descr.strip()
        if isinstance(traits, dict):
            entry["traits"] = traits
        char_entries.append(entry)
        existing_ids.add(character_id)
    # Also scan beat-level character references to catch any IDs not present
    # in the top-level story.characters list (defensive).
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
                char_ids = beat.get("characters") if isinstance(beat.get("characters"), list) else []
                for character_id in char_ids:
                    if not isinstance(character_id, str) or not character_id.strip():
                        continue
                    character_id = character_id.strip()
                    if character_id in existing_ids:
                        continue
                    char_entries.append({"character_id": character_id})
                    existing_ids.add(character_id)

    # ------------------------------------------------------------------
    # 2) Ensure persisted per-character bundles exist and are migrated
    # ------------------------------------------------------------------
    await ensure_story_character_bundles(updated)

    # ------------------------------------------------------------------
    # 3) Populate film2.project from the story structure
    # ------------------------------------------------------------------
    film2_branch = updated.get("film2")
    if not isinstance(film2_branch, dict):
        film2_branch = {}
        updated["film2"] = film2_branch
    project = film2_branch.get("project")
    if not isinstance(project, dict):
        project = {}
        film2_branch["project"] = project
    # Populate minimal project fields if absent
    prompt_val = story.get("prompt")
    if "prompt" not in project and isinstance(prompt_val, str):
        project["prompt"] = prompt_val
    dur = story.get("duration_hint_s")
    if "duration_s" not in project and isinstance(dur, (int, float)):
        project["duration_s"] = float(dur)
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    acts_count = len(acts)
    scenes_count = 0
    beats_count = 0
    for act in acts:
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        scenes_count += len(scenes)
        for scene in scenes:
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            beats_count += len(beats)
    project.setdefault("acts_count", acts_count)
    project.setdefault("scenes_count", scenes_count)
    project.setdefault("beats_count", beats_count)
    if trace_id:
        trace_event(
            "film2.locks_story_project_enriched",
            {
                "trace_id": trace_id,
                "quality_profile": profile_name,
                "acts_count": project.get("acts_count"),
                "scenes_count": project.get("scenes_count"),
                "beats_count": project.get("beats_count"),
            },
        )
    # High-signal log for observability (counts only, no huge blobs).
    chars = updated.get("characters") if isinstance(updated.get("characters"), list) else []
    log.info(
        "film2.locks.ensure_visual_locks_for_story done trace_id=%s chars=%d acts=%d dur_ms=%d",
        trace_id,
        len(chars),
        len(story.get("acts") or []) if isinstance(story.get("acts"), list) else 0,
        int((time.perf_counter() - t0) * 1000),
    )
    return updated


async def generate_scene_storyboards(
    scenes: List[Dict[str, Any]],
    locks_arg: Dict[str, Any],
    profile_name: str,
    trace_id: Optional[str],
    tool_runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Generate a simple storyboard image per scene using image.dispatch, when possible.
    """
    t0 = time.perf_counter()
    out_scenes: List[Dict[str, Any]] = []
    storyboard_size = "1024x1024"
    film2_branch = locks_arg.get("film2") if isinstance(locks_arg, dict) else {}
    project = film2_branch.get("project") if isinstance(film2_branch, dict) else {}
    sz = project.get("storyboard_size") if isinstance(project, dict) else None
    if isinstance(sz, str) and sz.strip():
        storyboard_size = sz.strip()
    log.info(
        "film2.storyboards.scenes start trace_id=%s scenes=%d storyboard_size=%s quality_profile=%s",
        trace_id,
        len(scenes or []),
        storyboard_size,
        profile_name,
    )
    for scene in scenes or []:
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get("scene_id")
        summary = scene.get("summary") or ""
        prompt = f"Storyboard scene: {summary}".strip()
        log.info("film2.storyboards.scene dispatch trace_id=%s scene_id=%s prompt_len=%d", trace_id, scene_id, len(prompt))
        args = {
            "prompt": prompt,
            "size": scene.get("size") or storyboard_size,
            "quality_profile": profile_name,
            "trace_id": trace_id,
            "lock_bundle": locks_arg if isinstance(locks_arg, dict) else {},
            "mode": "gen",
        }
        res = await tool_runner({"name": "image.dispatch", "arguments": args})
        out = dict(scene)
        if isinstance(res, dict) and isinstance(res.get("result"), dict):
            meta = (res.get("result") or {}).get("meta")
            if isinstance(meta, dict):
                out["storyboard"] = {
                    "scene_id": scene_id,
                    "prompt": prompt,
                    "image_url": meta.get("orch_view_url") or meta.get("view_url") or meta.get("url") or meta.get("data_url"),
                    "seed": meta.get("seed"),
                }
        out_scenes.append(out)
    ok = 0
    for sc in out_scenes:
        if isinstance(sc, dict) and isinstance(sc.get("storyboard"), dict):
            ok += 1
    log.info(
        "film2.storyboards.scenes done trace_id=%s ok=%d/%d dur_ms=%d",
        trace_id,
        ok,
        len(out_scenes),
        int((time.perf_counter() - t0) * 1000),
    )
    return out_scenes


async def generate_shot_storyboards(
    shots: List[Dict[str, Any]],
    locks_arg: Dict[str, Any],
    profile_name: str,
    trace_id: Optional[str],
    tool_runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Generate a storyboard image per derived story shot using image.dispatch.
    """
    t0 = time.perf_counter()
    out_shots: List[Dict[str, Any]] = []
    storyboard_size = "1024x1024"
    film2_branch = locks_arg.get("film2") if isinstance(locks_arg, dict) else {}
    project = film2_branch.get("project") if isinstance(film2_branch, dict) else {}
    sz = project.get("storyboard_size") if isinstance(project, dict) else None
    if isinstance(sz, str) and sz.strip():
        storyboard_size = sz.strip()
    log.info(
        "film2.storyboards.shots start trace_id=%s shots=%d storyboard_size=%s quality_profile=%s",
        trace_id,
        len(shots or []),
        storyboard_size,
        profile_name,
    )
    for shot in shots or []:
        if not isinstance(shot, dict):
            continue
        shot_id = shot.get("shot_id")
        descr = shot.get("description") or ""
        prompt = f"Storyboard shot: {descr}".strip()
        log.info("film2.storyboards.shot dispatch trace_id=%s shot_id=%s prompt_len=%d", trace_id, shot_id, len(prompt))
        args = {
            "prompt": prompt,
            "size": shot.get("size") or storyboard_size,
            "quality_profile": profile_name,
            "trace_id": trace_id,
            "lock_bundle": locks_arg if isinstance(locks_arg, dict) else {},
            "mode": "gen",
        }
        res = await tool_runner({"name": "image.dispatch", "arguments": args})
        out = dict(shot)
        if isinstance(res, dict) and isinstance(res.get("result"), dict):
            meta = (res.get("result") or {}).get("meta")
            if isinstance(meta, dict):
                out["storyboard"] = {
                    "shot_id": shot_id,
                    "prompt": prompt,
                    "image_url": meta.get("orch_view_url") or meta.get("view_url") or meta.get("url") or meta.get("data_url"),
                    "seed": meta.get("seed"),
                }
        out_shots.append(out)
    ok = 0
    for sh in out_shots:
        if isinstance(sh, dict) and isinstance(sh.get("storyboard"), dict):
            ok += 1
    log.info(
        "film2.storyboards.shots done trace_id=%s ok=%d/%d dur_ms=%d",
        trace_id,
        ok,
        len(out_shots),
        int((time.perf_counter() - t0) * 1000),
    )
    return out_shots


