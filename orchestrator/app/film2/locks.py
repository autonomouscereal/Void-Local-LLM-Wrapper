from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable, Awaitable
from ..refs.api import post_refs_apply
from ..locks.store import get_lock_bundle as _lock_load
from ..locks.runtime import (
    migrate_visual_bundle as _lock_migrate_visual,
    migrate_music_bundle as _lock_migrate_music,
    migrate_tts_bundle as _lock_migrate_tts,
    migrate_sfx_bundle as _lock_migrate_sfx,
    migrate_film2_bundle as _lock_migrate_film2,
    ensure_visual_lock_bundle as _lock_ensure_visual,
)
from ..datasets.trace import append_sample as _trace_append


def _apply_ref(ref_id: str | None) -> Dict[str, Any]:
    if not ref_id:
        return {}
    res = post_refs_apply({"ref_id": ref_id})
    if isinstance(res, tuple):
        payload, code = res
        if code != 200:
            return {}
        return payload.get("ref_pack") or {}
    return (res or {}).get("ref_pack") or {}


def build_ref_pack(char_meta: Dict[str, Any]) -> Dict[str, Any]:
    pack: Dict[str, Any] = {"image": {}, "voice": {}, "music": {}}
    if not isinstance(char_meta, dict):
        return pack
    img_id = char_meta.get("image_ref_id")
    voc_id = char_meta.get("voice_ref_id")
    mus_id = char_meta.get("music_ref_id")
    if img_id:
        pack["image"] = _apply_ref(img_id)
    if voc_id:
        pack["voice"] = _apply_ref(voc_id)
    if mus_id:
        pack["music"] = _apply_ref(mus_id)
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
    return {
        "prompt": cue.get("prompt") or "",
        "bpm": cue.get("bpm"),
        "length_s": cue.get("length_s", 30),
        "music_id": cue.get("music_ref_id"),
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
    character_entries = locks_arg.get("characters") if isinstance(locks_arg.get("characters"), list) else []
    character_ids: List[str] = []
    for entry in character_entries:
        char_id = entry.get("id")
        if isinstance(char_id, str) and char_id.strip():
            character_ids.append(char_id.strip())
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
    return out


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
    out_scenes: List[Dict[str, Any]] = []
    for scene in scenes or []:
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get("scene_id")
        summary = scene.get("summary") or ""
        prompt = f"Scene {scene_id}: {summary}"
        args_img: Dict[str, Any] = {
            "prompt": prompt,
            "width": 1024,
            "height": 576,
            "quality_profile": profile_name,
        }
        if isinstance(locks_arg, dict) and locks_arg:
            args_img["lock_bundle"] = locks_arg
        storyboard_path: Optional[str] = None
        res = await tool_runner({"name": "image.dispatch", "arguments": args_img})
        if isinstance(res, dict) and isinstance(res.get("result"), dict):
            r = res.get("result") or {}
            arts = r.get("artifacts") if isinstance(r.get("artifacts"), list) else []
            if arts:
                first = arts[0]
                if isinstance(first, dict):
                    path_val = first.get("path") or first.get("view_url") or first.get("url")
                    if isinstance(path_val, str) and path_val:
                        storyboard_path = path_val
                        _trace_append(
                            "film2",
                            {
                                "event": "scene_storyboard_generated",
                                "scene_id": scene_id,
                                "image_path": path_val,
                                "prompt": prompt,
                                "quality_profile": profile_name,
                            },
                        )
        scene_out = dict(scene)
        if storyboard_path:
            scene_out["storyboard_image"] = storyboard_path
        out_scenes.append(scene_out)
    return out_scenes


async def generate_shot_storyboards(
    shots: List[Dict[str, Any]],
    locks_arg: Dict[str, Any],
    profile_name: str,
    trace_id: Optional[str],
    tool_runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Generate a simple storyboard image per shot using image.dispatch, when possible.
    """
    out_shots: List[Dict[str, Any]] = []
    for shot in shots or []:
        if not isinstance(shot, dict):
            continue
        shot_id = shot.get("shot_id")
        descr = shot.get("description") or ""
        # Enrich storyboard prompt with character state hints when available.
        state_map = shot.get("character_states") if isinstance(shot.get("character_states"), dict) else {}
        if state_map:
            state_phrases: List[str] = []
            for cid, st in state_map.items():
                if not isinstance(st, dict):
                    continue
                if st.get("left_arm") == "missing":
                    state_phrases.append("a character with a missing left arm")
                if st.get("left_arm") == "robot":
                    state_phrases.append("a character with a robotic left arm")
            if state_phrases:
                extra = "; ".join(sorted(set(state_phrases)))
                descr = f"{descr} ({extra})"
        prompt = f"Shot {shot_id}: {descr}"
        args_img: Dict[str, Any] = {
            "prompt": prompt,
            "width": 1024,
            "height": 576,
            "quality_profile": profile_name,
        }
        if isinstance(locks_arg, dict) and locks_arg:
            args_img["lock_bundle"] = locks_arg
        storyboard_path: Optional[str] = None
        res = await tool_runner({"name": "image.dispatch", "arguments": args_img})
        if isinstance(res, dict) and isinstance(res.get("result"), dict):
            r = res.get("result") or {}
            arts = r.get("artifacts") if isinstance(r.get("artifacts"), list) else []
            if arts:
                first = arts[0]
                if isinstance(first, dict):
                    path_val = first.get("path") or first.get("view_url") or first.get("url")
                    if isinstance(path_val, str) and path_val:
                        storyboard_path = path_val
                        _trace_append(
                            "film2",
                            {
                                "event": "shot_storyboard_generated",
                                "shot_id": shot_id,
                                "image_path": path_val,
                                "prompt": prompt,
                                "quality_profile": profile_name,
                            },
                        )
        shot_out = dict(shot)
        if storyboard_path:
            shot_out["storyboard_image"] = storyboard_path
        out_shots.append(shot_out)
    return out_shots


