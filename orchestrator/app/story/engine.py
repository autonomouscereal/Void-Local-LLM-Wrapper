from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

from ..tracing.runtime import trace_event
from ..committee_client import committee_ai_text, committee_jsonify
from ..json_parser import JSONParser
from ..locks.runtime import (
    ensure_visual_lock_bundle,
    ensure_tts_lock_bundle,
    migrate_visual_bundle,
    migrate_music_bundle,
    migrate_tts_bundle,
    migrate_sfx_bundle,
    migrate_film2_bundle,
)
from ..locks.store import get_lock_bundle, upsert_lock_bundle


log = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: int = 1000) -> int:
    raw = os.getenv(name, None)
    s = str(raw).strip() if raw is not None else ""
    val = int(s) if (s and s.lstrip("+-").isdigit()) else int(default)
    if val < min_value:
        val = min_value
    if val > max_value:
        val = max_value
    return val


def _env_float(name: str, default: float, *, min_value: float = 0.0, max_value: float = 2.0) -> float:
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
    if val < min_value:
        val = min_value
    if val > max_value:
        val = max_value
    return val


STORY_COMMITTEE_ROUNDS = _env_int("STORY_COMMITTEE_ROUNDS", 1, min_value=1, max_value=10)
# By default, do NOT hard-cap story iterations. If you want a cap, set STORY_MAX_PASSES > 0.
STORY_MAX_PASSES = _env_int("STORY_MAX_PASSES", 0, min_value=0, max_value=1000)
STORY_TEMPERATURE = _env_float("STORY_TEMPERATURE", 0.5, min_value=0.0, max_value=2.0)





def _story_schema_template() -> Dict[str, Any]:
    """
    Returns a complete template showing the exact JSON structure expected for story graphs.
    This matches what derive_scenes_and_shots, ensure_tts_locks_and_dialogue_audio, and check_story_consistency actually use.
    
    Key points:
    - act_id and scene_id are optional (generated if missing)
    - beat_id is optional (only used in merge operations)
    - characters array contains objects with character_id (required for TTS locks) and character_name
    - locations: array of objects with location_name (fully descriptive name used as key for matching), location_description, location_atmosphere
    - objects: array of objects with object_name (fully descriptive name used as key for matching), object_description
    - locations and objects in beats are arrays of strings (location_name/object_name values that match the location_name/object_name from locations/objects arrays)
    - dialogue uses 'speaker' field (character_id), plus line_id (unique identifier for TTS dialogue_index mapping) and dialogue_text
    - music_hints in scenes/beats provide scene_mood, scene_tone, scene_energy, scene_location_atmosphere for music matching (music created AFTER story)
    - events use event_type and event_target (not 'type' and 'target')
    - scenes use scene_summary (not 'summary')
    - beats use beat_description (not 'description')
    """
    return {
        "prompt": "string: user's original prompt",
        "duration_hint_s": 0.0,
        "logline": "string: one-line summary (optional)",
        "genre": ["string"],
        "tone": "string: overall story tone",
        "themes": ["string"],
        "constraints": ["string"],
        "characters": [
            {
                "character_id": "string: unique identifier (REQUIRED - used for TTS locks and dialogue speaker field)",
                "character_name": "string: character name"
            }
        ],
        "locations": [
            {
                "location_name": "string: fully descriptive location name (REQUIRED - used as key for matching in beats, e.g. 'Post-Apocalyptic Urban Ruin', 'Ancient Temple Chamber')",
                "location_description": "string: detailed location description",
                "location_atmosphere": "string: location atmosphere/mood (used for music matching, e.g. 'stormy', 'eerie', 'triumphant')"
            }
        ],
        "objects": [
            {
                "object_name": "string: fully descriptive object name (REQUIRED - used as key for matching in beats, e.g. 'Chaos Emerald (Red)', 'Shattered Concrete Slab')",
                "object_description": "string: detailed object description"
            }
        ],
        "acts": [
            {
                "act_id": "string: optional unique identifier (generated if missing)",
                "scenes": [
                    {
                        "scene_id": "string: optional unique identifier (generated if missing)",
                        "scene_summary": "string: optional scene summary",
                        "music_hint": {
                            "scene_mood": "string: scene mood (e.g. 'tense', 'triumphant', 'melancholic')",
                            "scene_tone": "string: scene tone (e.g. 'dark', 'bright', 'mysterious')",
                            "scene_energy": "string: energy level (e.g. 'low', 'medium', 'high')",
                            "scene_location_atmosphere": "string: atmosphere from primary location (for music matching)"
                        },
                        "beats": [
                            {
                                "beat_id": "string: optional unique identifier (generated if missing)",
                                "beat_description": "string: optional beat description",
                                "time_hint_s": 0.0,
                                "characters": ["string: character_id values from characters array (OPTIONAL - empty array [] if no characters)"],
                                "locations": ["string: location_name strings that match location_name from locations array (OPTIONAL - empty array [] if no locations)"],
                                "objects": ["string: object_name strings that match object_name from objects array (OPTIONAL - empty array [] if no objects)"],
                                "events": [
                                    {
                                        "event_type": "string: 'state_change' or other event types",
                                        "event_target": "string: character_id or object_name (for state_change events only - REQUIRED for state_change, optional for other event types)",
                                        "state_delta": {"health": 50, "stance": "ready"}  # for state_change events only - dict with arbitrary key-value pairs representing state changes (e.g. {\"health\": 50, \"energy\": \"high\"}, NOT {\"key\": \"health\", \"value\": 50})
                                    }
                                ],
                                "dialogue": [
                                    {
                                        "line_id": "string: unique identifier (REQUIRED for TTS dialogue_index mapping)",
                                        "speaker": "string: character_id (REQUIRED - must match a character_id from characters array)",
                                        "dialogue_text": "string: dialogue text"
                                    }
                                ],
                                "music_hint": {
                                    "beat_mood": "string: beat mood (optional, inherits from scene if not provided)",
                                    "beat_tone": "string: beat tone (optional, inherits from scene if not provided)",
                                    "beat_energy": "string: beat energy level (optional, inherits from scene if not provided)"
                                }
                            }
                        ]
                    }
                ]
            }
        ],
        "continuity": {
            "character_states": {},
            "object_states": {},
            "time_travel_rules": {},
            "causal_consequences": []
        },
        "film_plan": {
            "shot_prompts": [
                {
                    "prompt": "string: shot description"
                }
            ]
        },
        "notes": {}
    }





def _issue_schema_template() -> Dict[str, Any]:
    """
    Returns a complete template showing the exact JSON structure expected for audit responses.
    This includes both chunk-level results and final synthesis results.
    """
    return {
        "issues": [
            {
                "code": "string: issue code (e.g. 'state_inconsistent', 'continuity_break', 'duration_mismatch', 'missing_causal_link')",
                "severity": "string: 'critical', 'major', 'minor', 'warning'",
                "message": "string: detailed description of the issue",
                "act_id": "string: optional - act identifier where issue occurs",
                "scene_id": "string: optional - scene identifier where issue occurs",
                "beat_id": "string: optional - beat identifier where issue occurs",
                "line_id": "string: optional - dialogue line identifier where issue occurs",
                "character_id": "string: optional - character identifier involved in issue",
                "location_name": "string: optional - location name involved in issue",
                "object_name": "string: optional - object name involved in issue",
                "target": "string: optional - target identifier for state_change events",
                "state_key": "string: optional - state key for state inconsistencies",
                "prev": "any: optional - previous value for state inconsistencies",
                "new": "any: optional - new value for state inconsistencies",
                "suggested_fix": "string: optional - suggested fix for the issue"
            }
        ],
        "summary": "string: overall audit summary (REQUIRED for final synthesis)",
        "must_fix": "bool: true if severe issues exist that must be fixed (REQUIRED for final synthesis)",
        "done": "bool: authoritative signal - true ONLY if story is ready to proceed, complete for requested duration, all continuity issues resolved, follows user intent, no critical issues remain (REQUIRED for final synthesis)",
        "length_ok": "bool: true if duration/content depth matches the request (REQUIRED for final synthesis)",
        "coverage_ratio": "float: ratio of current duration to requested duration (REQUIRED for final synthesis)",
        "next_action": "string: one of 'accept'|'extend'|'revise'|'extend_then_revise' - MUST be 'accept' when done=true (REQUIRED for final synthesis)",
        # Chunk-level fields (optional, only present in per-act chunk results):
        "local_summary": "string: optional - brief summary of this act's quality and issues found (for chunk results only)",
        "continuity_concerns": ["string: optional - list of potential continuity issues with other acts (for chunk results only)"]
    }


def _norm_prompt(user_prompt: str) -> str:
    return (user_prompt or "").strip()


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _ensure_story_character_lock_bundles(story: Dict[str, Any], *, trace_id: Optional[str]) -> None:
    """
    Ensure that all characters in the story have lock bundles created (visual, TTS, etc.).
    This is called after story generation, revision, merge, and extension to ensure
    new characters have proper lock bundles for film2 processing.
    
    NOTE: trace_id must be provided - no generation, no fallback, no failure.
    """
    log.debug("_ensure_story_character_lock_bundles start trace_id=%s story_type=%s", trace_id, type(story).__name__)
    if not isinstance(story, dict):
        log.warning("_ensure_story_character_lock_bundles: story is not a dict trace_id=%s story_type=%s", trace_id, type(story).__name__)
        return
    characters = story.get("characters") if isinstance(story.get("characters"), list) else []
    log.debug("_ensure_story_character_lock_bundles characters_count=%d trace_id=%s", len(characters), trace_id)
    if not characters:
        log.debug("_ensure_story_character_lock_bundles: no characters found trace_id=%s", trace_id)
        return
    
    # Also collect character_ids from beats (defensive - catch any not in top-level list)
    character_ids_from_beats: set[str] = set()
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    log.debug("_ensure_story_character_lock_bundles scanning acts_count=%d trace_id=%s", len(acts), trace_id)
    for act_idx, act in enumerate(acts):
        if not isinstance(act, dict):
            log.debug("_ensure_story_character_lock_bundles skipping invalid act act_idx=%d trace_id=%s", act_idx, trace_id)
            continue
        act_id = act.get("act_id") or ("act_%d" % act_idx)
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        log.debug("_ensure_story_character_lock_bundles act act_id=%s scenes_count=%d trace_id=%s", act_id, len(scenes), trace_id)
        for scene_idx, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                log.debug("_ensure_story_character_lock_bundles skipping invalid scene act_id=%s scene_idx=%d trace_id=%s", act_id, scene_idx, trace_id)
                continue
            scene_id = scene.get("scene_id") or ("scene_%d" % scene_idx)
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            log.debug("_ensure_story_character_lock_bundles scene act_id=%s scene_id=%s beats_count=%d trace_id=%s", act_id, scene_id, len(beats), trace_id)
            for beat_idx, beat in enumerate(beats):
                if not isinstance(beat, dict):
                    log.debug("_ensure_story_character_lock_bundles skipping invalid beat act_id=%s scene_id=%s beat_idx=%d trace_id=%s", act_id, scene_id, beat_idx, trace_id)
                    continue
                char_ids = beat.get("characters") if isinstance(beat.get("characters"), list) else []
                log.debug("_ensure_story_character_lock_bundles beat act_id=%s scene_id=%s beat_idx=%d characters_count=%d trace_id=%s", act_id, scene_id, beat_idx, len(char_ids), trace_id)
                for char_id in char_ids:
                    if isinstance(char_id, str) and char_id.strip():
                        character_ids_from_beats.add(char_id.strip())
                        log.debug("_ensure_story_character_lock_bundles found character in beat character_id=%s act_id=%s scene_id=%s beat_idx=%d trace_id=%s", char_id.strip(), act_id, scene_id, beat_idx, trace_id)
    
    # Collect all unique character IDs
    all_character_ids: set[str] = set()
    for char_idx, character in enumerate(characters):
        if not isinstance(character, dict):
            log.debug("_ensure_story_character_lock_bundles skipping invalid character char_idx=%d trace_id=%s", char_idx, trace_id)
            continue
        character_id = character.get("character_id")
        character_name = character.get("character_name") or character.get("name") or ""
        if isinstance(character_id, str) and character_id.strip():
            all_character_ids.add(character_id.strip())
            log.debug("_ensure_story_character_lock_bundles found character in top-level character_id=%s character_name=%s trace_id=%s", character_id.strip(), character_name, trace_id)
        else:
            log.warning("_ensure_story_character_lock_bundles character missing character_id char_idx=%d character_name=%s trace_id=%s", char_idx, character_name, trace_id)
    # Add any from beats
    log.debug("_ensure_story_character_lock_bundles characters_from_beats_count=%d trace_id=%s", len(character_ids_from_beats), trace_id)
    all_character_ids.update(character_ids_from_beats)
    log.info("_ensure_story_character_lock_bundles total_unique_characters=%d top_level=%d from_beats=%d trace_id=%s", len(all_character_ids), len(characters), len(character_ids_from_beats), trace_id)
    
    if not all_character_ids:
        log.warning("_ensure_story_character_lock_bundles: no character IDs found trace_id=%s", trace_id)
        return
    
    t0 = time.perf_counter()
    created_count = 0
    updated_count = 0
    
    for char_idx, character_id in enumerate(sorted(all_character_ids)):
        char_t0 = time.perf_counter()
        log.debug("_ensure_story_character_lock_bundles processing character char_idx=%d/%d character_id=%s trace_id=%s", char_idx + 1, len(all_character_ids), character_id, trace_id)
        try:
            # Load existing bundle
            log.debug("_ensure_story_character_lock_bundles loading bundle character_id=%s trace_id=%s", character_id, trace_id)
            existing_bundle = await get_lock_bundle(character_id)
            has_existing = isinstance(existing_bundle, dict) and existing_bundle
            log.debug("_ensure_story_character_lock_bundles bundle loaded character_id=%s has_existing=%s trace_id=%s", character_id, has_existing, trace_id)
            if has_existing:
                # Migrate existing bundle to latest schema
                log.debug("_ensure_story_character_lock_bundles migrating existing bundle character_id=%s trace_id=%s", character_id, trace_id)
                existing_bundle = migrate_visual_bundle(existing_bundle)
                existing_bundle = migrate_music_bundle(existing_bundle)
                existing_bundle = migrate_tts_bundle(existing_bundle)
                existing_bundle = migrate_sfx_bundle(existing_bundle)
                existing_bundle = migrate_film2_bundle(existing_bundle)
                log.debug("_ensure_story_character_lock_bundles migration complete character_id=%s trace_id=%s", character_id, trace_id)
            else:
                existing_bundle = None
                log.debug("_ensure_story_character_lock_bundles no existing bundle character_id=%s trace_id=%s", character_id, trace_id)
            
            # Ensure visual branch exists (creates/updates bundle)
            log.debug("_ensure_story_character_lock_bundles ensuring visual bundle character_id=%s trace_id=%s", character_id, trace_id)
            bundle = await ensure_visual_lock_bundle(character_id, existing_bundle)
            log.debug("_ensure_story_character_lock_bundles visual bundle ensured character_id=%s has_face=%s has_pose=%s has_style=%s trace_id=%s", 
                     character_id, bool(bundle.get("face")), bool(bundle.get("pose")), bool(bundle.get("style")), trace_id)
            # Ensure TTS branch exists (updates bundle with TTS branch)
            log.debug("_ensure_story_character_lock_bundles ensuring TTS bundle character_id=%s trace_id=%s", character_id, trace_id)
            bundle = await ensure_tts_lock_bundle(character_id, bundle)
            log.debug("_ensure_story_character_lock_bundles TTS bundle ensured character_id=%s has_tts=%s has_audio=%s trace_id=%s", 
                     character_id, bool(bundle.get("tts")), bool(bundle.get("audio")), trace_id)
            # Migrate again after ensuring both branches
            log.debug("_ensure_story_character_lock_bundles final migration character_id=%s trace_id=%s", character_id, trace_id)
            bundle = migrate_visual_bundle(bundle)
            bundle = migrate_music_bundle(bundle)
            bundle = migrate_tts_bundle(bundle)
            bundle = migrate_sfx_bundle(bundle)
            bundle = migrate_film2_bundle(bundle)
            log.debug("_ensure_story_character_lock_bundles final migration complete character_id=%s trace_id=%s", character_id, trace_id)
            # Final upsert to ensure everything is persisted
            log.debug("_ensure_story_character_lock_bundles upserting bundle character_id=%s trace_id=%s", character_id, trace_id)
            await upsert_lock_bundle(character_id, bundle)
            log.debug("_ensure_story_character_lock_bundles bundle upserted character_id=%s trace_id=%s", character_id, trace_id)
            
            if existing_bundle is None:
                created_count += 1
                log.info("_ensure_story_character_lock_bundles created new bundle character_id=%s dur_ms=%d trace_id=%s", 
                        character_id, int((time.perf_counter() - char_t0) * 1000), trace_id)
            else:
                updated_count += 1
                log.info("_ensure_story_character_lock_bundles updated existing bundle character_id=%s dur_ms=%d trace_id=%s", 
                        character_id, int((time.perf_counter() - char_t0) * 1000), trace_id)
        except Exception as ex:
            log.warning(
                "story.ensure_character_bundles failed character_id=%s char_idx=%d/%d dur_ms=%d trace_id=%s ex=%s",
                character_id,
                char_idx + 1,
                len(all_character_ids),
                int((time.perf_counter() - char_t0) * 1000),
                trace_id,
                ex,
                exc_info=True,
            )
    
    log.info(
        "story.ensure_character_bundles done trace_id=%s characters=%d created=%d updated=%d dur_ms=%d",
        trace_id,
        len(all_character_ids),
        created_count,
        updated_count,
        int((time.perf_counter() - t0) * 1000),
    )


async def _apply_story_state_changes_to_locks(story: Dict[str, Any], *, trace_id: Optional[str]) -> None:
    """
    Process all state_change events in the story and apply visual state changes to character lock bundles.
    This ensures that character appearance changes (missing arms, age, injuries, etc.) are reflected in lock bundles.
    
    NOTE: trace_id must be provided - no generation, no fallback, no failure.
    """
    log.info("_apply_story_state_changes_to_locks start trace_id=%s", trace_id)
    if not isinstance(story, dict):
        log.warning("_apply_story_state_changes_to_locks: story is not a dict trace_id=%s story_type=%s", trace_id, type(story).__name__)
        return
    
    # Collect all state changes by character_id
    character_state_changes: Dict[str, Dict[str, Any]] = {}
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    log.debug("_apply_story_state_changes_to_locks scanning acts_count=%d trace_id=%s", len(acts), trace_id)
    
    for act_idx, act in enumerate(acts):
        if not isinstance(act, dict):
            continue
        act_id = act.get("act_id") or ("act_%d" % act_idx)
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        for scene_idx, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue
            scene_id = scene.get("scene_id") or ("scene_%d" % scene_idx)
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat_idx, beat in enumerate(beats):
                if not isinstance(beat, dict):
                    continue
                beat_id = beat.get("beat_id") or ("beat_%d" % beat_idx)
                events = beat.get("events") if isinstance(beat.get("events"), list) else []
                for ev_idx, ev in enumerate(events):
                    if not isinstance(ev, dict):
                        continue
                    event_type = ev.get("event_type") or ev.get("type") or ""
                    if event_type != "state_change":
                        continue
                    event_target = ev.get("event_target") or ev.get("target") or ""
                    state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                    if not isinstance(event_target, str) or not event_target.strip() or not state_delta:
                        continue
                    character_id = event_target.strip()
                    # Only process character state changes (not object state changes)
                    characters = story.get("characters") if isinstance(story.get("characters"), list) else []
                    is_character = any(
                        char.get("character_id") == character_id 
                        for char in characters 
                        if isinstance(char, dict) and isinstance(char.get("character_id"), str)
                    )
                    if not is_character:
                        log.debug("_apply_story_state_changes_to_locks skipping non-character target character_id=%s act_id=%s scene_id=%s beat_id=%s trace_id=%s", 
                                 character_id, act_id, scene_id, beat_id, trace_id)
                        continue
                    # Merge state changes (later events override earlier ones)
                    if character_id not in character_state_changes:
                        character_state_changes[character_id] = {}
                    character_state_changes[character_id].update(state_delta)
                    log.debug("_apply_story_state_changes_to_locks found state_change character_id=%s state_delta=%s act_id=%s scene_id=%s beat_id=%s trace_id=%s", 
                             character_id, state_delta, act_id, scene_id, beat_id, trace_id)
                    # Trace individual state change events
                    trace_event(
                        "story.state_change_detected",
                        {
                            "trace_id": trace_id,
                            "character_id": character_id,
                            "state_delta": state_delta,
                            "act_id": act_id,
                            "scene_id": scene_id,
                            "beat_id": beat_id,
                        }
                    )
    
    if not character_state_changes:
        log.info("_apply_story_state_changes_to_locks no character state changes found trace_id=%s", trace_id)
        return
    
    log.info("_apply_story_state_changes_to_locks processing state_changes characters_count=%d trace_id=%s", len(character_state_changes), trace_id)
    t0 = time.perf_counter()
    updated_count = 0
    
    for character_id, state_updates in character_state_changes.items():
        char_t0 = time.perf_counter()
        log.info("_apply_story_state_changes_to_locks processing character_id=%s state_updates=%s trace_id=%s", character_id, state_updates, trace_id)
        try:
            # Load existing bundle
            existing_bundle = await get_lock_bundle(character_id)
            if not isinstance(existing_bundle, dict):
                log.warning("_apply_story_state_changes_to_locks no existing bundle for character_id=%s trace_id=%s", character_id, trace_id)
                continue
            
            # Apply visual state changes to lock bundle
            # Map state_delta keys to visual lock fields
            visual_updates = {}
            state_tags = []
            if "left_arm" in state_updates:
                visual_updates["left_arm"] = state_updates["left_arm"]
                if state_updates["left_arm"] == "missing":
                    state_tags.append("missing-left-arm")
                elif state_updates["left_arm"] == "robot":
                    state_tags.append("robotic-left-arm")
            if "right_arm" in state_updates:
                visual_updates["right_arm"] = state_updates["right_arm"]
                if state_updates["right_arm"] == "missing":
                    state_tags.append("missing-right-arm")
                elif state_updates["right_arm"] == "robot":
                    state_tags.append("robotic-right-arm")
            if "age" in state_updates:
                age_val = state_updates["age"]
                visual_updates["age"] = age_val
                if isinstance(age_val, (int, float)) and age_val > 0:
                    state_tags.append("age-%d" % int(age_val))
            if "appearance" in state_updates:
                visual_updates["appearance"] = state_updates["appearance"]
                app_val = str(state_updates["appearance"]).lower().replace(" ", "-")
                if app_val:
                    state_tags.append("appearance-%s" % app_val)
            # Add other visual state fields as needed
            # Health, stance, energy_level are state but not visual - skip for visual updates
            for key in ["health", "stance", "energy_level", "inventory", "relationships"]:
                if key in state_updates:
                    # Store in state_changes but don't add to visual_updates
                    pass
            
            if visual_updates:
                # Update visual branch of lock bundle
                visual_branch = existing_bundle.get("visual") if isinstance(existing_bundle.get("visual"), dict) else {}
                if not visual_branch:
                    visual_branch = {}
                    existing_bundle["visual"] = visual_branch
                
                # Store state changes in visual branch (store ALL state changes, not just visual)
                state_changes = visual_branch.get("state_changes") if isinstance(visual_branch.get("state_changes"), dict) else {}
                state_changes.update(state_updates)  # Store all state changes for reference
                visual_branch["state_changes"] = state_changes
                log.debug("_apply_story_state_changes_to_locks stored state_changes character_id=%s state_changes=%s trace_id=%s", 
                         character_id, list(state_changes.keys()), trace_id)
                
                # Update style to reflect visual changes
                style = visual_branch.get("style") if isinstance(visual_branch.get("style"), dict) else {}
                if not style:
                    style = {}
                    visual_branch["style"] = style
                
                # Update style_tags to include state information for prompt visibility
                style_tags = style.get("style_tags") if isinstance(style.get("style_tags"), list) else []
                if not isinstance(style_tags, list):
                    style_tags = []
                # Add state tags if not already present
                for tag in state_tags:
                    if tag not in style_tags:
                        style_tags.append(tag)
                style["style_tags"] = style_tags
                log.debug("_apply_story_state_changes_to_locks updated style_tags character_id=%s style_tags=%s trace_id=%s", 
                         character_id, style_tags, trace_id)
                
                # Build description from state changes for human readability
                desc_parts = []
                if state_changes.get("left_arm") == "missing":
                    desc_parts.append("missing left arm")
                elif state_changes.get("left_arm") == "robot":
                    desc_parts.append("robotic left arm")
                if state_changes.get("right_arm") == "missing":
                    desc_parts.append("missing right arm")
                elif state_changes.get("right_arm") == "robot":
                    desc_parts.append("robotic right arm")
                if "age" in state_changes and isinstance(state_changes["age"], (int, float)):
                    desc_parts.append("age %d" % int(state_changes["age"]))
                if state_changes.get("appearance"):
                    desc_parts.append(str(state_changes["appearance"]))
                
                if desc_parts:
                    # Store in a metadata field for reference (not in style.description as that's for rendering style)
                    metadata = visual_branch.get("metadata") if isinstance(visual_branch.get("metadata"), dict) else {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    metadata["character_state_description"] = ", ".join(desc_parts)
                    visual_branch["metadata"] = metadata
                    log.debug("_apply_story_state_changes_to_locks updated metadata description character_id=%s description=%s trace_id=%s", 
                             character_id, metadata["character_state_description"], trace_id)
                
                # Migrate bundle to ensure schema consistency
                existing_bundle = migrate_visual_bundle(existing_bundle)
                existing_bundle = migrate_music_bundle(existing_bundle)
                existing_bundle = migrate_tts_bundle(existing_bundle)
                existing_bundle = migrate_sfx_bundle(existing_bundle)
                existing_bundle = migrate_film2_bundle(existing_bundle)
                
                # Save updated bundle
                await upsert_lock_bundle(character_id, existing_bundle)
                updated_count += 1
                log.info("_apply_story_state_changes_to_locks updated bundle character_id=%s visual_updates=%s state_tags=%s dur_ms=%d trace_id=%s", 
                        character_id, visual_updates, state_tags, int((time.perf_counter() - char_t0) * 1000), trace_id)
            else:
                log.debug("_apply_story_state_changes_to_locks no visual updates for character_id=%s state_updates=%s trace_id=%s", 
                         character_id, state_updates, trace_id)
        except Exception as ex:
            log.warning("_apply_story_state_changes_to_locks failed character_id=%s state_updates=%s dur_ms=%d trace_id=%s ex=%s", 
                       character_id, state_updates, int((time.perf_counter() - char_t0) * 1000), trace_id, ex, exc_info=True)
    
    log.info("_apply_story_state_changes_to_locks done trace_id=%s characters_updated=%d dur_ms=%d", 
            trace_id, updated_count, int((time.perf_counter() - t0) * 1000))
    
    # Trace state changes for monitoring
    if updated_count > 0:
        trace_event(
            "story.state_changes_applied",
            {
                "trace_id": trace_id,
                "characters_updated": updated_count,
                "total_state_changes": len(character_state_changes),
                "dur_ms": int((time.perf_counter() - t0) * 1000),
            }
        )


async def _jsonify_then_parse(raw_text: str, expected_schema: Any, *, trace_id: Optional[str], rounds: int | None = None, temperature: float | None = None) -> Dict[str, Any]:
    """
    Required JSON flow for any LLM output used as structured data:
      1) LLM produces raw text (possibly messy)
      2) committee_jsonify(raw_text, expected_schema) -> best-effort structured JSON candidate (uses JSONFixer committee)
      3) JSONParser.parse(candidate, expected_schema) -> schema-coerced dict (100% JSON-safe)
    
    This ensures proper JSON structure at every step using the committee system.
    NOTE: trace_id must be provided by caller - no generation, no fallback, no failure.
    If trace_id is None, it will be passed through to downstream functions which may fail.
    """
    t0 = time.perf_counter()
    raw_text_len = len(raw_text) if raw_text else 0
    log.debug("_jsonify_then_parse start trace_id=%s raw_text_len=%d rounds=%s temperature=%s schema_type=%s", 
             trace_id, raw_text_len, rounds, temperature, type(expected_schema).__name__)
    
    # Step 1: Use committee_jsonify to get structured JSON from potentially messy LLM output
    # committee_jsonify uses the JSONFixer committee to clean and structure the JSON
    log.debug("_jsonify_then_parse calling committee_jsonify trace_id=%s", trace_id)
    jsonify_t0 = time.perf_counter()
    candidate = await committee_jsonify(
        raw_text=(raw_text or "{}"),
        expected_schema=expected_schema,
        trace_id=trace_id,
        rounds=rounds,
        temperature=temperature,
    )
    jsonify_dur = time.perf_counter() - jsonify_t0
    candidate_type = type(candidate).__name__
    candidate_len = len(str(candidate)) if candidate else 0
    log.debug("_jsonify_then_parse committee_jsonify done trace_id=%s candidate_type=%s candidate_len=%d dur_ms=%d", 
             trace_id, candidate_type, candidate_len, int(jsonify_dur * 1000))
    
    # Step 2: Use JSONParser to validate and coerce to exact schema
    # This provides final validation and schema coercion
    parser = JSONParser()
    log.debug("_jsonify_then_parse calling JSONParser.parse trace_id=%s", trace_id)
    parse_t0 = time.perf_counter()
    try:
        # Convert candidate back to JSON string for parser (if it's a dict)
        candidate_str = json.dumps(candidate, ensure_ascii=False) if isinstance(candidate, dict) else (str(candidate) if candidate else "{}")
        candidate_str_len = len(candidate_str)
        log.debug("_jsonify_then_parse candidate_str_len=%d trace_id=%s", candidate_str_len, trace_id)
        obj = parser.parse(candidate_str, expected_schema)
        parse_dur = time.perf_counter() - parse_t0
        if not isinstance(obj, dict):
            log.warning("_jsonify_then_parse: JSONParser returned non-dict type=%s trace_id=%s obj_type=%s", type(obj).__name__, trace_id, type(obj).__name__)
            obj = {}
        else:
            obj_keys = list(obj.keys()) if isinstance(obj, dict) else []
            obj_keys_count = len(obj_keys)
            log.debug("_jsonify_then_parse JSONParser.parse success trace_id=%s obj_keys_count=%d dur_ms=%d", 
                     trace_id, obj_keys_count, int(parse_dur * 1000))
            log.debug("_jsonify_then_parse JSONParser.parse obj_keys=%s trace_id=%s", obj_keys[:10] if obj_keys else [], trace_id)
    except Exception as ex:
        parse_dur = time.perf_counter() - parse_t0
        candidate_preview = (str(candidate) if candidate else "")[:200]
        log.warning("_jsonify_then_parse: JSONParser.parse failed ex=%s trace_id=%s candidate_prefix=%s dur_ms=%d", 
                   ex, trace_id, candidate_preview, int(parse_dur * 1000), exc_info=True)
        obj = {}
    
    total_dur = time.perf_counter() - t0
    obj_keys_count = len(obj.keys()) if isinstance(obj, dict) else 0
    log.info("_jsonify_then_parse done trace_id=%s obj_keys_count=%d total_dur_ms=%d jsonify_dur_ms=%d parse_dur_ms=%d", 
            trace_id, obj_keys_count, int(total_dur * 1000), int(jsonify_dur * 1000), int(parse_dur * 1000))
    return obj


async def _committee_generate_story_raw(user_prompt: str, duration_hint_s: float, *, trace_id: Optional[str]) -> str:
    """
    Ask the committee to propose a story graph. Uses committee_jsonify to ensure proper JSON structure.
    NOTE: Critical instructions are in USER message because committee overrides system messages.
    NOTE: trace_id must be provided - no generation, no fallback, no failure.
    """
    t0 = time.perf_counter()
    log.debug("_committee_generate_story_raw start trace_id=%s user_prompt_len=%d duration_hint_s=%s", 
             trace_id, len(user_prompt) if user_prompt else 0, duration_hint_s)
    prompt_text = _norm_prompt(user_prompt)
    dur = float(duration_hint_s) if isinstance(duration_hint_s, (int, float)) else 0.0
    if dur <= 0.0:
        log.debug("_committee_generate_story_raw duration_hint_s invalid, using default trace_id=%s duration_hint_s=%s", trace_id, duration_hint_s)
        dur = 600.0
    log.debug("_committee_generate_story_raw normalized prompt_len=%d duration_s=%.2f trace_id=%s", len(prompt_text), dur, trace_id)
    # Minimal system message - committee will override it anyway
    sys_msg = "You are the Story Engine for a film generator."
    # ALL CRITICAL INSTRUCTIONS IN USER MESSAGE (committee overrides system)
    schema_template = _story_schema_template()
    user_msg_content = f"""### [CRITICAL INSTRUCTIONS - READ CAREFULLY]

You MUST produce a story that follows the user's request exactly, but you may fill in missing details.

Continuity is critical: injuries, inventory, relationships, time travel rules, and causal consequences MUST remain consistent.

Output MUST be a single JSON object (no markdown, no code fences, no prose) describing a full story graph.

The story MUST be temporally coherent end-to-end.

If the user requests a short duration, keep the story tight. If long, include deeper arcs.

CRITICAL: Use ONLY ASCII characters. NO emojis, NO special Unicode symbols, NO non-ASCII characters. Replace any non-ASCII with ASCII equivalents.

### [EXACT JSON STRUCTURE REQUIRED]
You MUST match this exact structure. All field names and nesting must be identical:

{json.dumps(schema_template, ensure_ascii=False, indent=2)}

### [KEY REQUIREMENTS]
- Top level: prompt, duration_hint_s, logline, genre, tone, themes, constraints, characters, locations, objects, acts, continuity, film_plan, notes
- **CRITICAL: CREATE characters, locations, and objects as needed for your story.** If the user mentions characters, locations, or objects that don't exist yet, you MUST add them to the top-level arrays. Do not reference characters/locations/objects that aren't in the arrays.
- Characters: array of objects with character_id (string, REQUIRED) and character_name (string) - character_id is used for TTS locks and dialogue speaker field. **CREATE new characters as needed for the story. Each character that appears in dialogue or events MUST have an entry in the characters array.**
- Locations: array of objects with location_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Post-Apocalyptic Urban Ruin'), location_description (string), location_atmosphere (string - used for music matching, e.g. 'stormy', 'eerie'). **CREATE new locations as needed for the story. Each location referenced in beats MUST have an entry in the locations array.**
- Objects: array of objects with object_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Chaos Emerald (Red)'), object_description (string). **CREATE new objects as needed for the story. Each object referenced in beats or state_change events MUST have an entry in the objects array.**
- Locations and objects in beats: arrays of strings (location_name/object_name values that MUST match location_name/object_name from the locations/objects arrays)
- Acts: array of objects with act_id (string, optional - generated if missing) and scenes (array)
- Scenes: array of objects with scene_id (string, optional - generated if missing), scene_summary (string, optional), music_hint (object with scene_mood, scene_tone, scene_energy, scene_location_atmosphere - for music matching), and beats (array)
- Beats: array of objects with beat_description (string, optional), time_hint_s (float, REQUIRED), characters (array, OPTIONAL - can be empty [] if no characters), locations (array, OPTIONAL - can be empty [] if no locations), objects (array, OPTIONAL - can be empty [] if no objects), events (array), dialogue (array), music_hint (object with beat_mood, beat_tone, beat_energy - optional, inherits from scene)
- Events: array of objects with event_type (string, REQUIRED). 
  * For state_change events: event_target (string, REQUIRED - must be a character_id from characters array OR an object_name from objects array, NOT a location_name) and state_delta (dict, REQUIRED)
  * state_delta structure: A dictionary with direct key-value pairs representing the state changes. The keys are the state property names (e.g. "health", "stance", "energy_level", "left_arm", "age", "appearance") and values are the new state values.
  * **IMPORTANT: Use state_delta to track visual/physical changes that affect character appearance:**
    - Physical changes: "left_arm": "missing" or "left_arm": "robot" (for prosthetic limbs), "right_arm": "missing", "age": 25 (for aging), "appearance": "scarred" (for injuries/scars)
    - State changes: "health": 50, "stance": "ready", "energy_level": "high"
  * CORRECT state_delta examples: {"health": 50}, {"stance": "ready", "energy_level": "high"}, {"left_arm": "missing"}, {"age": 30, "appearance": "weathered"}, {"eruption_level": "medium"}
  * INCORRECT state_delta examples: {"key": "health", "value": 50} (WRONG - don't use nested "key"/"value" structure), {"key": "position", "value": "mid-battle stance"} (WRONG)
  * event_target MUST be a character_id (e.g. "shadow", "charizard") OR an object_name (e.g. "Volcanic Fissure (Active)"), NOT a location_name
  * For non-state_change events: event_target is optional (can be omitted or empty string)
- Dialogue: array of objects with line_id (string, REQUIRED - unique identifier for TTS dialogue_index mapping), speaker (string, REQUIRED - must be a character_id from characters array), dialogue_text (string)
- Music: Music is created AFTER story generation. Music matching uses descriptive information from scenes/beats: scene_mood, scene_tone, scene_energy, scene_location_atmosphere. Do NOT include actual music in story structure, only music_hint objects.

### [USER REQUEST]
{json.dumps({
    "user_prompt": prompt_text,
    "duration_hint_s": dur,
}, ensure_ascii=False)}

### [YOUR TASK]
Produce the complete story graph as a single JSON object matching the exact structure above. Output ONLY the JSON object, no other text.

**CRITICAL REMINDERS:**
1. **CREATE all characters, locations, and objects that appear in your story** - do not reference entities that aren't in the top-level arrays
2. **Use state_change events to track character appearance changes** - if a character loses an arm, ages, gets injured, etc., add a state_change event with appropriate state_delta (e.g., {"left_arm": "missing"} or {"age": 45})
3. **Ensure continuity** - if a character's state changes (e.g., loses an arm), that change must be reflected in all subsequent beats where that character appears

CRITICAL: Do NOT wrap your output in wrapper keys like 'response', 'data', 'result', 'output', 'content', or 'json'. Output the story object directly at the top level."""
    user_msg = user_msg_content
    t0 = time.perf_counter()
    env = await committee_ai_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=STORY_TEMPERATURE,
    )
    result_block = env.get("result") if isinstance(env, dict) else {}
    txt = ""
    if isinstance(result_block, dict) and isinstance(result_block.get("text"), str):
        txt = str(result_block.get("text") or "")
    else:
        txt = str(env.get("text") or "")
    
    # Use committee_jsonify to ensure proper JSON structure before returning
    # Keep trace_id constant - don't mutate it
    log.info("story.committee.draft_raw calling committee_jsonify trace_id=%s txt_len=%d", trace_id, len(txt))
    jsonify_t0 = time.perf_counter()
    structured = await committee_jsonify(
        raw_text=txt,
        expected_schema=_story_schema_template(),
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=0.0,
    )
    jsonify_dur = time.perf_counter() - jsonify_t0
    structured_ok = isinstance(structured, dict)
    log.info("story.committee.draft_raw committee_jsonify done trace_id=%s ok=%r dur_ms=%d", trace_id, structured_ok, int(jsonify_dur * 1000))
    # Convert back to JSON string for return (will be parsed again in draft_story_graph)
    txt_structured = json.dumps(structured, ensure_ascii=False) if isinstance(structured, dict) else txt
    
    log.info(
        "story.committee.draft_raw done trace_id=%s dur_ms=%d",
        trace_id,
        int((time.perf_counter() - t0) * 1000),
    )
    return txt_structured


async def _committee_audit_story(story: Dict[str, Any], user_prompt: str, *, trace_id: Optional[str]) -> Dict[str, Any]:
    """
    Committee-based continuity + intent audit with chunked evaluation for large stories.
    Evaluates full story in chunks, then synthesizes results.
    Returns {issues: [...], summary: str, must_fix: bool, done: bool}.
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

    story_summary = _story_extract_summary(story)
    
    # For large stories, evaluate in chunks. For small stories, evaluate full story.
    all_issues = []
    chunk_results = []
    
    log.info("_committee_audit_story starting chunked evaluation trace_id=%s acts_count=%d", trace_id, len(acts))
    
    # Process story in act-sized chunks
    for act_idx, act in enumerate(acts):
        if not isinstance(act, dict):
            log.debug("_committee_audit_story skipping invalid act in chunk loop act_idx=%d trace_id=%s", act_idx, trace_id)
            continue
        act_id = act.get("act_id") or ("act_%d" % act_idx)
        log.debug("_committee_audit_story processing chunk act_idx=%d act_id=%s trace_id=%s", act_idx, act_id, trace_id)
        act_chunk = _story_get_chunk_by_ids(story, act_ids=[act_id] if act_id else None)
        act_chunk_type = type(act_chunk).__name__
        act_chunk_keys = list(act_chunk.keys()) if isinstance(act_chunk, dict) else []
        log.debug("_committee_audit_story act_chunk extracted act_id=%s chunk_type=%s chunk_keys_count=%d trace_id=%s", 
                 act_id, act_chunk_type, len(act_chunk_keys), trace_id)
        
        # Minimal system message - committee overrides it
        sys_msg = "You are a strict story continuity AND completion auditor."
        # ALL CRITICAL INSTRUCTIONS IN USER MESSAGE (committee overrides system)
        issue_schema_template = _issue_schema_template()
        user_msg_content = f"""### [CRITICAL INSTRUCTIONS - READ CAREFULLY]

You are evaluating ONE ACT of a larger story.

Given the story summary (full structure) and this act's full content, find ALL violations:
- breaks user intent
- continuity contradictions (injuries, objects, locations, relationships)
- temporal contradictions and time travel inconsistencies
- missing causal links / dangling plot threads
- character motives that don't track
- inconsistencies with other acts (check summary for context)
- missing required fields (line_id, speaker, dialogue_text for dialogue; event_type, event_target for events)
- invalid references (character_id not in characters array; location_name/object_name not in locations/objects arrays)

CRITICAL: Use ONLY ASCII characters. NO emojis, NO special Unicode symbols, NO non-ASCII characters.

### [EXACT JSON STRUCTURE REQUIRED]
Your response MUST match this exact structure. All field names and nesting must be identical:

{json.dumps(issue_schema_template, ensure_ascii=False, indent=2)}

### [ISSUE OBJECT REQUIREMENTS]
Each issue object MUST include:
- code: string identifier (e.g. 'state_inconsistent', 'continuity_break', 'duration_mismatch', 'missing_causal_link', 'invalid_reference', 'missing_field')
- severity: 'critical', 'major', 'minor', or 'warning'
- message: detailed description explaining the issue
- Precise pointers: Include ALL relevant identifiers (act_id, scene_id, beat_id, line_id, character_id, location_name, object_name, target, state_key, prev, new)
- suggested_fix: optional but recommended - specific fix suggestion

### [EXAMPLES]
Example issue for state inconsistency:
{{
  "code": "state_inconsistent",
  "severity": "critical",
  "message": "Character health changes from 100 to 50 in beat_3, but later beat_5 assumes health is still 100",
  "act_id": "act_1",
  "scene_id": "scene_2",
  "beat_id": "beat_5",
  "character_id": "shadow",
  "target": "shadow",
  "state_key": "health",
  "prev": 100,
  "new": 50,
  "suggested_fix": "Update beat_5 to reflect health=50, or add state_change event in beat_3 to set health=50"
}}

Example issue for invalid reference:
{{
  "code": "invalid_reference",
  "severity": "major",
  "message": "Beat references location_name 'Ancient Temple' but this location_name is not defined in locations array",
  "act_id": "act_2",
  "scene_id": "scene_4",
  "beat_id": "beat_12",
  "location_name": "Ancient Temple",
  "suggested_fix": "Add location with location_name 'Ancient Temple' to locations array, or fix the reference in beat_12"
}}

Example issue for invalid state_delta structure:
{{
  "code": "invalid_reference",
  "severity": "major",
  "message": "Event in beat_2 has invalid state_delta structure: uses {{'key': 'position', 'value': 'mid-battle stance'}} instead of direct key-value pairs like {{'position': 'mid-battle stance'}}",
  "act_id": "act_1",
  "scene_id": "scene_1",
  "beat_id": "beat_2",
  "suggested_fix": "Change state_delta from {{'key': 'position', 'value': 'mid-battle stance'}} to {{'position': 'mid-battle stance'}}"
}}

Example issue for invalid event_target:
{{
  "code": "invalid_reference",
  "severity": "major",
  "message": "Event in beat_3 has event_target 'Shattered Battlefield' which is a location_name, not a character_id or object_name. event_target can only be character_id or object_name, not location_name.",
  "act_id": "act_1",
  "scene_id": "scene_1",
  "beat_id": "beat_3",
  "location_name": "Shattered Battlefield",
  "suggested_fix": "Remove this event_target or change it to a valid character_id or object_name. If you want to track location state, create an object with object_name matching the location and use that as event_target."
}}

### [DATA]
{json.dumps({
    "user_prompt": _norm_prompt(user_prompt),
    "duration_hint_s": float(total_hint),
    "current_beats_time_s": float(total_beats_time),
    "coverage_ratio": float(ratio),
    "story_summary": story_summary,
    "act_chunk": act_chunk,
}, ensure_ascii=False)}

### [YOUR RESPONSE]
Output ONLY the JSON object matching the exact structure above. Include ALL issues found. No other text."""
        user_msg = user_msg_content
        # Keep trace_id constant - don't mutate it
        t0_chunk = time.perf_counter()
        env = await committee_ai_text(
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            trace_id=trace_id,
            rounds=STORY_COMMITTEE_ROUNDS,
            temperature=0.2,
        )
        raw = ""
        result_block = env.get("result") if isinstance(env, dict) else {}
        if isinstance(result_block, dict) and isinstance(result_block.get("text"), str):
            raw = str(result_block.get("text") or "")
        chunk_parsed = await _jsonify_then_parse(
            raw_text=raw,
            expected_schema=issue_schema_template,
            trace_id=trace_id,
            rounds=STORY_COMMITTEE_ROUNDS,
            temperature=0.0,
        )
        if isinstance(chunk_parsed, dict):
            chunk_issues = chunk_parsed.get("issues") if isinstance(chunk_parsed.get("issues"), list) else []
            all_issues.extend(chunk_issues)
            chunk_results.append({
                "act_id": act_id,
                "act_idx": act_idx,
                "issues_count": len(chunk_issues),
                "local_summary": chunk_parsed.get("local_summary") or "",
                "continuity_concerns": chunk_parsed.get("continuity_concerns") or [],
            })
        log.info(
            "story.audit.chunk done trace_id=%s act_idx=%d act_id=%s issues=%d dur_ms=%d",
            trace_id,
            act_idx,
            act_id,
            len(chunk_parsed.get("issues") or []) if isinstance(chunk_parsed, dict) else 0,
            int((time.perf_counter() - t0_chunk) * 1000),
        )
    
    # Synthesize chunk results into final audit
    # Minimal system message - committee overrides it
    sys_msg_synth = "You are synthesizing story audit results."
    # ALL CRITICAL INSTRUCTIONS IN USER MESSAGE (committee overrides system)
    user_msg_synth_content = f"""### [CRITICAL INSTRUCTIONS - READ CAREFULLY]

You are synthesizing story audit results from multiple act evaluations.

You have received:
- Story summary (full structure)
- Per-act audit results with issues and continuity concerns
- Overall metrics (duration_hint_s, current_beats_time_s, coverage_ratio)

### [YOUR TASK]
1. Synthesize all issues into a coherent audit
2. Judge whether the story is COMPLETE for the requested duration
3. Determine if story is ready to proceed

### [OUTPUT FORMAT - CRITICAL]
Return JSON ONLY with these exact fields:
- done: bool (authoritative: true ONLY if story is ready to proceed - set this to true when story is complete and meets all requirements)
- length_ok: bool (true if duration/content depth matches the request)
- coverage_ratio: float (echo the provided ratio or your estimate)
- next_action: string one of: 'accept'|'extend'|'revise'|'extend_then_revise' (use 'accept' when done=true)
- must_fix: bool (true if severe issues exist)
- summary: string (overall audit summary)
- issues: list of ALL issue objects from all chunks (consolidated, deduplicated)

### [DONE FLAG RULES]
- Set done=true ONLY when:
  * Story is complete for requested duration
  * All continuity issues are resolved
  * Story follows user intent
  * No critical issues remain
- When done=true, next_action MUST be 'accept'
- When done=false, next_action should be 'extend' (if length not ok) or 'revise' (if issues exist)

### [DATA]
{json.dumps({
    "user_prompt": _norm_prompt(user_prompt),
    "duration_hint_s": float(total_hint),
    "current_beats_time_s": float(total_beats_time),
    "coverage_ratio": float(ratio),
    "story_summary": story_summary,
    "chunk_results": chunk_results,
    "all_issues": all_issues,
}, ensure_ascii=False)}

### [YOUR RESPONSE]
Output ONLY the JSON object with the fields above. No other text."""
    user_msg_synth = user_msg_synth_content
    # Keep trace_id constant - don't mutate it
    t0_synth = time.perf_counter()
    env_synth = await committee_ai_text(
        messages=[{"role": "system", "content": sys_msg_synth}, {"role": "user", "content": user_msg_synth}],
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=0.2,
    )
    raw_synth = ""
    result_block_synth = env_synth.get("result") if isinstance(env_synth, dict) else {}
    if isinstance(result_block_synth, dict) and isinstance(result_block_synth.get("text"), str):
        raw_synth = str(result_block_synth.get("text") or "")
    parsed = await _jsonify_then_parse(
        raw_text=raw_synth,
        expected_schema=_issue_schema_template(),
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=0.0,
    )
    if not parsed:
        return {"issues": all_issues, "summary": "", "must_fix": False, "done": False, "length_ok": False, "coverage_ratio": float(ratio), "next_action": "extend"}
    # Use synthesized issues if provided, otherwise use collected issues
    if isinstance(parsed.get("issues"), list) and parsed.get("issues"):
        issues = parsed.get("issues")
    else:
        issues = all_issues
    parsed["issues"] = issues
    parsed.setdefault("coverage_ratio", float(ratio))
    if not isinstance(parsed.get("next_action"), str):
        parsed["next_action"] = "revise" if bool(parsed.get("must_fix")) else ("accept" if bool(parsed.get("done")) else "extend")
    log.info(
        "story.committee.audit done trace_id=%s chunks=%d issues=%d must_fix=%s dur_ms=%d",
        trace_id,
        len(chunk_results),
        len(issues),
        bool(parsed.get("must_fix")) if isinstance(parsed, dict) else False,
        int((time.perf_counter() - t0_synth) * 1000),
    )
    return parsed


async def _committee_revise_story(story: Dict[str, Any], user_prompt: str, audit: Dict[str, Any], *, trace_id: Optional[str]) -> Dict[str, Any]:
    """
    Committee revision step. Returns ONLY the changes/deltas, not the full story.
    NOTE: Critical instructions are in USER message because committee overrides system messages.
    NOTE: trace_id must be provided - no generation, no fallback, no failure.
    """
    t0 = time.perf_counter()
    log.debug("_committee_revise_story start trace_id=%s story_type=%s user_prompt_len=%d audit_type=%s", 
             trace_id, type(story).__name__, len(user_prompt) if user_prompt else 0, type(audit).__name__)
    
    audit_issues = audit.get("issues") if isinstance(audit, dict) and isinstance(audit.get("issues"), list) else []
    audit_issues_count = len(audit_issues) if isinstance(audit_issues, list) else 0
    audit_must_fix = bool(audit.get("must_fix")) if isinstance(audit, dict) else False
    log.info("_committee_revise_story audit summary trace_id=%s issues_count=%d must_fix=%s", 
            trace_id, audit_issues_count, audit_must_fix)
    
    story_summary = _story_extract_summary(story)
    story_summary_keys = list(story_summary.keys()) if isinstance(story_summary, dict) else []
    log.debug("_committee_revise_story story_summary extracted trace_id=%s summary_keys_count=%d", 
             trace_id, len(story_summary_keys))
    # Minimal system message - committee overrides it
    sys_msg = "You are the Story Engine revision pass."
    # ALL CRITICAL INSTRUCTIONS IN USER MESSAGE (committee overrides system)
    schema_template = _story_schema_template()
    user_msg_content = f"""### [CRITICAL INSTRUCTIONS - READ CAREFULLY]

You MUST fix the issues while preserving the user's intent and improving coherence.

Do not hand-wave: update the story graph so later beats reflect state changes.

CRITICAL: Return ONLY the changes/deltas, not the full story.
- For modified acts/scenes/beats: include them with their IDs (they will be merged by ID).
- For new acts/scenes/beats: include them with new unique IDs.
- Only include fields that are being changed or added.
- Do NOT include unchanged acts/scenes/beats.
- If fixing a state inconsistency: add state_change events or update later beats to reflect the state
- **CRITICAL: CREATE missing characters, locations, and objects.** If fixing invalid references: add missing locations/objects/characters to top-level arrays OR fix the references. Do not leave dangling references.
- If fixing missing fields: add the required fields (line_id, speaker, dialogue_text for dialogue; event_type, event_target for events)

CRITICAL: Use ONLY ASCII characters. NO emojis, NO special Unicode symbols, NO non-ASCII characters.

### [EXACT JSON STRUCTURE REQUIRED]
Your delta MUST match this exact structure. All field names and nesting must be identical:

{json.dumps(schema_template, ensure_ascii=False, indent=2)}

### [KEY REQUIREMENTS]
- Characters: array of objects with character_id (string, REQUIRED) and character_name (string) - character_id is used for TTS locks and dialogue speaker field
- Locations: array of objects with location_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Post-Apocalyptic Urban Ruin'), location_description (string), location_atmosphere (string - used for music matching, e.g. 'stormy', 'eerie')
- Objects: array of objects with object_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Chaos Emerald (Red)'), object_description (string)
- Locations and objects in beats: arrays of strings (location_name/object_name values that MUST match location_name/object_name from the locations/objects arrays). Can be empty arrays [] if no characters/locations/objects.
- Scenes: scene_summary (string, optional), music_hint (object with scene_mood, scene_tone, scene_energy, scene_location_atmosphere - for music matching)
- Beats: beat_description (string, optional), time_hint_s (float, REQUIRED), characters (array, OPTIONAL - can be empty []), locations (array, OPTIONAL - can be empty []), objects (array, OPTIONAL - can be empty []), events (array), dialogue (array), music_hint (object with beat_mood, beat_tone, beat_energy - optional, inherits from scene)
- Events: array of objects with event_type (string, REQUIRED). 
  * For state_change events: event_target (string, REQUIRED - must be a character_id from characters array OR an object_name from objects array, NOT a location_name) and state_delta (dict, REQUIRED)
  * state_delta structure: A dictionary with direct key-value pairs representing the state changes. The keys are the state property names (e.g. "health", "stance", "energy_level") and values are the new state values.
  * CORRECT state_delta examples: {{"health": 50}}, {{"stance": "ready", "energy_level": "high"}}, {{"eruption_level": "medium"}}
  * INCORRECT state_delta examples: {{"key": "health", "value": 50}} (WRONG - don't use nested "key"/"value" structure), {{"key": "position", "value": "mid-battle stance"}} (WRONG)
  * event_target MUST be a character_id (e.g. "shadow", "charizard") OR an object_name (e.g. "Volcanic Fissure (Active)"), NOT a location_name
  * For non-state_change events: event_target is optional (can be omitted or empty string)
- Dialogue: array of objects with line_id (string, REQUIRED - unique identifier for TTS dialogue_index mapping), speaker (string, REQUIRED - must be a character_id from characters array), dialogue_text (string, REQUIRED)
- Music: Music is created AFTER story generation. Music matching uses descriptive information from scenes/beats: scene_mood, scene_tone, scene_energy, scene_location_atmosphere. Do NOT include actual music in story structure, only music_hint objects.

### [DELTA EXAMPLES]
Example delta creating a new character:
{{
  "characters": [
    {{
      "character_id": "new_character",
      "character_name": "New Character Name"
    }}
  ],
  "acts": [
    {{
      "act_id": "act_1",
      "scenes": [
        {{
          "scene_id": "scene_1",
          "beats": [
            {{
              "beat_id": "beat_3",
              "characters": ["new_character"],
              "dialogue": [
                {{
                  "line_id": "line_010",
                  "speaker": "new_character",
                  "dialogue_text": "Hello, I'm a new character."
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Example delta fixing a state inconsistency:
{{
  "acts": [
    {{
      "act_id": "act_1",
      "scenes": [
        {{
          "scene_id": "scene_2",
          "beats": [
            {{
              "beat_id": "beat_5",
              "events": [
                {{
                  "event_type": "state_change",
                  "event_target": "shadow",
                  "state_delta": {{"health": 50}}
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Example delta fixing invalid state_delta structure:
{{
  "acts": [
    {{
      "act_id": "act_1",
      "scenes": [
        {{
          "scene_id": "scene_1",
          "beats": [
            {{
              "beat_id": "beat_2",
              "events": [
                {{
                  "event_type": "state_change",
                  "event_target": "shadow",
                  "state_delta": {{"position": "mid-battle stance"}}
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}
Note: state_delta uses direct key-value pairs: {{"position": "mid-battle stance"}}, NOT nested structure like {{"key": "position", "value": "mid-battle stance"}}

Example delta fixing invalid event_target:
{{
  "acts": [
    {{
      "act_id": "act_1",
      "scenes": [
        {{
          "scene_id": "scene_1",
          "beats": [
            {{
              "beat_id": "beat_3",
              "events": [
                {{
                  "event_type": "state_change",
                  "event_target": "Volcanic Fissure (Active)",
                  "state_delta": {{"eruption_level": "medium"}}
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}
Note: event_target "Volcanic Fissure (Active)" is valid because it's an object_name. If it were "Shattered Battlefield" (a location_name), it would be INVALID.

Example delta adding missing location:
{{
  "locations": [
    {{
      "location_name": "Ancient Temple",
      "location_description": "A mysterious ancient temple with glowing runes",
      "location_atmosphere": "eerie"
    }}
  ],
  "acts": [
    {{
      "act_id": "act_2",
      "scenes": [
        {{
          "scene_id": "scene_4",
          "beats": [
            {{
              "beat_id": "beat_12",
              "locations": ["Ancient Temple"]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Example delta fixing missing dialogue fields:
{{
  "acts": [
    {{
      "act_id": "act_1",
      "scenes": [
        {{
          "scene_id": "scene_1",
          "beats": [
            {{
              "beat_id": "beat_3",
              "dialogue": [
                {{
                  "line_id": "line_001",
                  "speaker": "shadow",
                  "dialogue_text": "I must find the Chaos Emerald."
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

### [DATA]
{json.dumps({
    "user_prompt": _norm_prompt(user_prompt),
    "story_summary": story_summary,
    "audit": audit,
}, ensure_ascii=False)}

### [YOUR TASK]
Fix the issues identified in the audit. Return ONLY the JSON delta with changes matching the exact structure above. No other text.

CRITICAL: Do NOT wrap your output in wrapper keys like 'response', 'data', 'result', 'output', 'content', or 'json'. Output the delta object directly at the top level."""
    user_msg = user_msg_content
    # Keep trace_id constant - don't mutate it
    t0 = time.perf_counter()
    env = await committee_ai_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=STORY_TEMPERATURE,
    )
    raw = ""
    result_block = env.get("result") if isinstance(env, dict) else {}
    if isinstance(result_block, dict) and isinstance(result_block.get("text"), str):
        raw = str(result_block.get("text") or "")
    # Use story schema for deltas (JSONParser handles missing fields gracefully)
    parsed = await _jsonify_then_parse(
        raw_text=raw,
        expected_schema=_story_schema_template(),
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=STORY_TEMPERATURE,
    )
    delta_acts = parsed.get("acts") if isinstance(parsed, dict) and isinstance(parsed.get("acts"), list) else []
    acts_n = len(delta_acts)
    log.info(
        "story.committee.revise done trace_id=%s delta_acts=%d dur_ms=%d",
        trace_id,
        acts_n,
        int((time.perf_counter() - t0) * 1000),
    )
    # Return delta (may be empty dict if no changes)
    return parsed if isinstance(parsed, dict) else {}


def _story_measure_coverage(story: Dict[str, Any]):
    """
    Compute coverage and beat counts for a story graph.
    Returns: (total_hint_s, total_beats_time_s, total_beats_count, coverage_ratio)
    """
    total_hint = float(story.get("duration_hint_s") or 0.0) if isinstance(story, dict) else 0.0
    total_beats_time = 0.0
    total_beats_count = 0
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
                total_beats_count += 1
                total_beats_time += float(beat.get("time_hint_s") or 0.0)
    ratio = (total_beats_time / total_hint) if (total_hint > 0.0 and total_beats_time > 0.0) else 0.0
    return total_hint, total_beats_time, total_beats_count, ratio


def _story_extract_summary(story: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a structured summary of the story with metadata and structure.
    Used for context when LLM needs overview but not full details.
    """
    if not isinstance(story, dict):
        return {}
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    characters = story.get("characters") if isinstance(story.get("characters"), list) else []
    locations = story.get("locations") if isinstance(story.get("locations"), list) else []
    objects = story.get("objects") if isinstance(story.get("objects"), list) else []
    
    # Build structured summary with IDs and brief metadata
    act_summaries = []
    for act in acts:
        if not isinstance(act, dict):
            continue
        act_id = act.get("act_id") or ""
        act_title = act.get("title") or ""
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        scene_summaries = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = scene.get("scene_id") or ""
            scene_title = scene.get("title") or ""
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            beat_summaries = []
            for beat in beats:
                if not isinstance(beat, dict):
                    continue
                beat_id = beat.get("beat_id") or ""
                beat_summary = beat.get("beat_summary") or beat.get("beat_description") or beat.get("summary") or beat.get("description") or ""
                beat_summaries.append({"beat_id": beat_id, "summary": beat_summary[:200] if beat_summary else ""})
            scene_summaries.append({
                "scene_id": scene_id,
                "title": scene_title,
                "beat_count": len(beats),
                "beats": beat_summaries
            })
        act_summaries.append({
            "act_id": act_id,
            "title": act_title,
            "scene_count": len(scenes),
            "scenes": scene_summaries
        })
    
    char_summaries = [{"character_id": char.get("character_id"), "character_name": char.get("character_name") or char.get("name") or ""} for char in characters if isinstance(char, dict) and char.get("character_id")]
    loc_summaries = [{"location_name": loc.get("location_name") if isinstance(loc, dict) else (str(loc) if isinstance(loc, str) else ""), "location_atmosphere": loc.get("location_atmosphere") or loc.get("atmosphere") if isinstance(loc, dict) else ""} for loc in locations if (isinstance(loc, dict) and (loc.get("location_name") or loc.get("name"))) or isinstance(loc, str)]
    obj_summaries = [{"object_name": obj.get("object_name") if isinstance(obj, dict) else (str(obj) if isinstance(obj, str) else ""), "object_description": obj.get("object_description") or obj.get("description") if isinstance(obj, dict) else ""} for obj in objects if (isinstance(obj, dict) and (obj.get("object_name") or obj.get("name"))) or isinstance(obj, str)]
    
    return {
        "prompt": story.get("prompt") or "",
        "duration_hint_s": story.get("duration_hint_s") or 0.0,
        "logline": story.get("logline") or "",
        "genre": story.get("genre") or [],
        "tone": story.get("tone") or "",
        "themes": story.get("themes") or [],
        "act_count": len(acts),
        "acts": act_summaries,
        "characters": char_summaries,
        "locations": loc_summaries,
        "objects": obj_summaries,
    }


def _story_get_chunk_by_ids(story: Dict[str, Any], act_ids: List[str] = None, scene_ids: List[str] = None, beat_ids: List[str] = None) -> Dict[str, Any]:
    """
    Extract specific chunks of the story by IDs for detailed evaluation.
    Returns full content for specified acts/scenes/beats.
    """
    if not isinstance(story, dict):
        return {}
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    
    chunk_acts = []
    for act in acts:
        if not isinstance(act, dict):
            continue
        act_id = act.get("act_id") or ""
        if act_ids and act_id not in act_ids:
            continue
        
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        chunk_scenes = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = scene.get("scene_id") or ""
            if scene_ids and scene_id not in scene_ids:
                continue
            
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            chunk_beats = []
            for beat in beats:
                if not isinstance(beat, dict):
                    continue
                beat_id = beat.get("beat_id") or ""
                if beat_ids and beat_id not in beat_ids:
                    continue
                chunk_beats.append(beat)
            
            if chunk_beats or (not beat_ids and not scene_ids):
                chunk_scene = dict(scene)
                chunk_scene["beats"] = chunk_beats
                chunk_scenes.append(chunk_scene)
        
        if chunk_scenes or (not scene_ids and not act_ids):
            chunk_act = dict(act)
            chunk_act["scenes"] = chunk_scenes
            chunk_acts.append(chunk_act)
    
    return {"acts": chunk_acts}


def _story_get_merge_context(story: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get full context needed for merging: affected acts/scenes/beats from base story + delta.
    Returns full content of items that will be merged, plus summary of rest.
    """
    if not isinstance(story, dict) or not isinstance(delta, dict):
        return {"summary": _story_extract_summary(story), "delta": delta}
    
    # Find all IDs mentioned in delta
    delta_acts = delta.get("acts") if isinstance(delta.get("acts"), list) else []
    affected_act_ids = []
    affected_scene_ids = []
    affected_beat_ids = []
    
    for act in delta_acts:
        if not isinstance(act, dict):
            continue
        act_id = act.get("act_id")
        if act_id:
            affected_act_ids.append(act_id)
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = scene.get("scene_id")
            if scene_id:
                affected_scene_ids.append(scene_id)
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat in beats:
                if not isinstance(beat, dict):
                    continue
                beat_id = beat.get("beat_id")
                if beat_id:
                    affected_beat_ids.append(beat_id)
    
    # Get full chunks for affected items
    affected_chunks = _story_get_chunk_by_ids(story, act_ids=affected_act_ids, scene_ids=affected_scene_ids, beat_ids=affected_beat_ids)
    
    return {
        "summary": _story_extract_summary(story),
        "affected_chunks": affected_chunks,
        "delta": delta,
    }


async def _story_merge_delta_with_llm(base_story: Dict[str, Any], delta: Dict[str, Any], user_prompt: str, *, trace_id: Optional[str]) -> Dict[str, Any]:
    """
    Use LLM to intelligently merge delta into base story with full context.
    NOTE: Critical instructions are in USER message because committee overrides system messages.
    NOTE: trace_id must be provided - no generation, no fallback, no failure.
    """
    t0 = time.perf_counter()
    log.debug("_story_merge_delta_with_llm start trace_id=%s base_story_type=%s delta_type=%s user_prompt_len=%d", 
             trace_id, type(base_story).__name__, type(delta).__name__, len(user_prompt) if user_prompt else 0)
    
    base_acts = base_story.get("acts") if isinstance(base_story, dict) and isinstance(base_story.get("acts"), list) else []
    base_characters = base_story.get("characters") if isinstance(base_story, dict) and isinstance(base_story.get("characters"), list) else []
    delta_acts = delta.get("acts") if isinstance(delta, dict) and isinstance(delta.get("acts"), list) else []
    delta_characters = delta.get("characters") if isinstance(delta, dict) and isinstance(delta.get("characters"), list) else []
    log.info("_story_merge_delta_with_llm merge context trace_id=%s base_acts_count=%d base_characters_count=%d delta_acts_count=%d delta_characters_count=%d", 
            trace_id, len(base_acts), len(base_characters), len(delta_acts), len(delta_characters))
    
    merge_context = _story_get_merge_context(base_story, delta)
    merge_context_keys = list(merge_context.keys()) if isinstance(merge_context, dict) else []
    log.debug("_story_merge_delta_with_llm merge_context extracted trace_id=%s context_keys_count=%d context_keys=%s", 
             trace_id, len(merge_context_keys), merge_context_keys[:10] if merge_context_keys else [])
    
    # Minimal system message - committee overrides it
    sys_msg = "You are the Story Engine merge coordinator."
    # ALL CRITICAL INSTRUCTIONS IN USER MESSAGE (committee overrides system)
    schema_template = _story_schema_template()
    user_msg_content = f"""### [CRITICAL INSTRUCTIONS - READ CAREFULLY]

You will receive:
- A summary of the full story structure
- Full content of story chunks that are being modified/merged
- A delta containing changes/extensions to apply

### [YOUR TASK]
1. Merge the delta into the base story intelligently
2. Preserve all unchanged content from base story
3. Apply changes from delta (update by ID, append new items)
4. Maintain continuity and coherence
5. Return the FULL merged story (not just changes)
6. Ensure all references are valid (character_id in characters array, location_name/object_name in locations/objects arrays)
7. Ensure all required fields are present (line_id, speaker, dialogue_text for dialogue; event_type, event_target for state_change events)

### [MERGE RULES]
- If delta contains an act/scene/beat with an existing ID, update that item (merge fields, don't replace entire item)
- If delta contains an act/scene/beat with a new ID, append it
- Preserve all items from base story that aren't in delta
- Maintain all IDs and structure
- Ensure continuity is preserved across the merge
- **CRITICAL: For characters/locations/objects:**
  - If delta contains a character with an existing character_id, update that character (merge fields)
  - If delta contains a character with a new character_id, append it to the characters array
  - If delta contains a location with an existing location_name, update that location (merge fields)
  - If delta contains a location with a new location_name, append it to the locations array
  - If delta contains an object with an existing object_name, update that object (merge fields)
  - If delta contains an object with a new object_name, append it to the objects array
- For locations/objects: match by location_name/object_name (not IDs)
- For characters: match by character_id
- Merge arrays intelligently (e.g., combine dialogue arrays, don't replace)
- **CRITICAL: Preserve all state_change events from both base and delta.** State changes are cumulative - later events override earlier ones for the same character.

CRITICAL: Use ONLY ASCII characters. NO emojis, NO special Unicode symbols, NO non-ASCII characters.

### [EXACT JSON STRUCTURE REQUIRED]
Your merged story MUST match this exact structure. All field names and nesting must be identical:

{json.dumps(schema_template, ensure_ascii=False, indent=2)}

### [KEY REQUIREMENTS]
- Characters: array of objects with character_id (string, REQUIRED) and character_name (string) - character_id is used for TTS locks and dialogue speaker field
- Locations: array of objects with location_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Post-Apocalyptic Urban Ruin'), location_description (string), location_atmosphere (string - used for music matching, e.g. 'stormy', 'eerie')
- Objects: array of objects with object_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Chaos Emerald (Red)'), object_description (string)
- Locations and objects in beats: arrays of strings (location_name/object_name values that MUST match location_name/object_name from the locations/objects arrays). Can be empty arrays [] if no characters/locations/objects.
- Scenes: scene_summary (string, optional), music_hint (object with scene_mood, scene_tone, scene_energy, scene_location_atmosphere - for music matching)
- Beats: beat_description (string, optional), time_hint_s (float, REQUIRED), characters (array, OPTIONAL - can be empty []), locations (array, OPTIONAL - can be empty []), objects (array, OPTIONAL - can be empty []), events (array), dialogue (array), music_hint (object with beat_mood, beat_tone, beat_energy - optional, inherits from scene)
- Events: array of objects with event_type (string, REQUIRED). 
  * For state_change events: event_target (string, REQUIRED - must be a character_id from characters array OR an object_name from objects array, NOT a location_name) and state_delta (dict, REQUIRED)
  * state_delta structure: A dictionary with direct key-value pairs representing the state changes. The keys are the state property names (e.g. "health", "stance", "energy_level") and values are the new state values.
  * CORRECT state_delta examples: {{"health": 50}}, {{"stance": "ready", "energy_level": "high"}}, {{"eruption_level": "medium"}}
  * INCORRECT state_delta examples: {{"key": "health", "value": 50}} (WRONG - don't use nested "key"/"value" structure), {{"key": "position", "value": "mid-battle stance"}} (WRONG)
  * event_target MUST be a character_id (e.g. "shadow", "charizard") OR an object_name (e.g. "Volcanic Fissure (Active)"), NOT a location_name
  * For non-state_change events: event_target is optional (can be omitted or empty string)
- Dialogue: array of objects with line_id (string, REQUIRED - unique identifier for TTS dialogue_index mapping), speaker (string, REQUIRED - must be a character_id from characters array), dialogue_text (string, REQUIRED)
- Music: Music is created AFTER story generation. Music matching uses descriptive information from scenes/beats: scene_mood, scene_tone, scene_energy, scene_location_atmosphere. Do NOT include actual music in story structure, only music_hint objects.

### [MERGE EXAMPLES]
Example: If base story has beat_3 with dialogue [{{"line_id": "line_001", ...}}] and delta has beat_3 with dialogue [{{"line_id": "line_002", ...}}], merge should result in beat_3 with dialogue [{{"line_id": "line_001", ...}}, {{"line_id": "line_002", ...}}]

Example: If base story has location_name "Ancient Temple" and delta updates it with new location_description, merge should update the existing location object, not create a duplicate.

### [OUTPUT FORMAT]
Output JSON ONLY with the complete merged story. The output must be a valid JSON object matching the exact structure above. Include ALL acts, scenes, beats, characters, locations, objects from both base and delta.

CRITICAL: Do NOT wrap your output in wrapper keys like 'response', 'data', 'result', 'output', 'content', or 'json'. Output the story object directly at the top level.

### [DATA]
{json.dumps({
    "user_prompt": _norm_prompt(user_prompt),
    "merge_context": merge_context,
}, ensure_ascii=False)}

### [YOUR RESPONSE]
Output ONLY the complete merged story as JSON matching the exact structure above. No other text."""
    user_msg = user_msg_content
    
    # Keep trace_id constant - don't mutate it
    t0 = time.perf_counter()
    env = await committee_ai_text(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=0.2,
    )
    raw = ""
    result_block = env.get("result") if isinstance(env, dict) else {}
    if isinstance(result_block, dict) and isinstance(result_block.get("text"), str):
        raw = str(result_block.get("text") or "")
    else:
        raw = str(env.get("text") or "")
    
    merged = await _jsonify_then_parse(
        raw_text=raw,
        expected_schema=_story_schema_template(),
        trace_id=trace_id,
        rounds=STORY_COMMITTEE_ROUNDS,
        temperature=0.0,
    )
    
    merge_dur = time.perf_counter() - t0
    
    # Log merge results
    merged_acts = merged.get("acts") if isinstance(merged, dict) and isinstance(merged.get("acts"), list) else []
    merged_characters = merged.get("characters") if isinstance(merged, dict) and isinstance(merged.get("characters"), list) else []
    merged_locations = merged.get("locations") if isinstance(merged, dict) and isinstance(merged.get("locations"), list) else []
    merged_objects = merged.get("objects") if isinstance(merged, dict) and isinstance(merged.get("objects"), list) else []
    log.info("_story_merge_delta_with_llm done trace_id=%s dur_ms=%d merged_type=%s acts=%d characters=%d locations=%d objects=%d", 
            trace_id, int(merge_dur * 1000), type(merged).__name__, len(merged_acts), len(merged_characters), len(merged_locations), len(merged_objects))
    
    # Trace merge completion
    trace_event(
        "story.merge_delta_complete",
        {
            "trace_id": trace_id,
            "dur_ms": int(merge_dur * 1000),
            "acts_count": len(merged_acts),
            "characters_count": len(merged_characters),
            "locations_count": len(merged_locations),
            "objects_count": len(merged_objects),
        }
    )
    
    return merged if isinstance(merged, dict) and merged else base_story


def _story_merge_delta(base_story: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a story delta (changes/extensions) into the base story.
    Delta can contain:
    - New acts/scenes/beats (to append)
    - Modified acts/scenes/beats (to update by ID)
    - New characters/locations/objects (to append)
    - Modified characters/locations/objects (to update by ID)
    - Top-level field updates
    """
    if not isinstance(base_story, dict):
        base_story = {}
    if not isinstance(delta, dict) or not delta:
        return dict(base_story)
    
    merged = dict(base_story)
    
    # Merge top-level fields (delta overrides base)
    for key in ("prompt", "duration_hint_s", "logline", "tone", "continuity", "film_plan", "notes"):
        if key in delta:
            merged[key] = delta[key]
    
    # Merge lists that are additive (genre, themes, constraints)
    for key in ("genre", "themes", "constraints"):
        base_list = merged.get(key) if isinstance(merged.get(key), list) else []
        delta_list = delta.get(key) if isinstance(delta.get(key), list) else []
        # Add new items that aren't already in base
        for item in delta_list:
            if item not in base_list:
                base_list.append(item)
        merged[key] = base_list
    
    # Merge acts: update by act_id, or append if new
    base_acts = merged.get("acts") if isinstance(merged.get("acts"), list) else []
    delta_acts = delta.get("acts") if isinstance(delta.get("acts"), list) else []
    act_by_id = {act.get("act_id"): act for act in base_acts if isinstance(act, dict) and act.get("act_id")}
    
    for delta_act in delta_acts:
        if not isinstance(delta_act, dict):
            continue
        act_id = delta_act.get("act_id")
        if act_id and act_id in act_by_id:
            # Merge into existing act
            base_act = act_by_id[act_id]
            # Merge scenes within act
            base_scenes = base_act.get("scenes") if isinstance(base_act.get("scenes"), list) else []
            delta_scenes = delta_act.get("scenes") if isinstance(delta_act.get("scenes"), list) else []
            scene_by_id = {scene.get("scene_id"): scene for scene in base_scenes if isinstance(scene, dict) and scene.get("scene_id")}
            
            for delta_scene in delta_scenes:
                if not isinstance(delta_scene, dict):
                    continue
                scene_id = delta_scene.get("scene_id")
                if scene_id and scene_id in scene_by_id:
                    # Merge into existing scene
                    base_scene = scene_by_id[scene_id]
                    # Merge beats within scene
                    base_beats = base_scene.get("beats") if isinstance(base_scene.get("beats"), list) else []
                    delta_beats = delta_scene.get("beats") if isinstance(delta_scene.get("beats"), list) else []
                    beat_by_id = {beat.get("beat_id"): beat for beat in base_beats if isinstance(beat, dict) and beat.get("beat_id")}
                    
                    for delta_beat in delta_beats:
                        if not isinstance(delta_beat, dict):
                            continue
                        beat_id = delta_beat.get("beat_id")
                        if beat_id and beat_id in beat_by_id:
                            # Update existing beat
                            beat_by_id[beat_id].update(delta_beat)
                        else:
                            # New beat, append
                            base_beats.append(delta_beat)
                    base_scene["beats"] = base_beats
                else:
                    # New scene, append
                    base_scenes.append(delta_scene)
            base_act["scenes"] = base_scenes
        else:
            # New act, append
            base_acts.append(delta_act)
            if act_id:
                act_by_id[act_id] = delta_act
    
    merged["acts"] = base_acts
    
    # Merge characters: update by character_id or append
    base_characters = merged.get("characters") if isinstance(merged.get("characters"), list) else []
    delta_characters = delta.get("characters") if isinstance(delta.get("characters"), list) else []
    char_by_id = {char.get("character_id"): char for char in base_characters if isinstance(char, dict) and char.get("character_id")}
    new_characters_count = 0
    updated_characters_count = 0
    for delta_char in delta_characters:
        if not isinstance(delta_char, dict):
            continue
        char_id = delta_char.get("character_id")
        if char_id and char_id in char_by_id:
            char_by_id[char_id].update(delta_char)
            updated_characters_count += 1
        else:
            base_characters.append(delta_char)
            if char_id:
                char_by_id[char_id] = delta_char
                new_characters_count += 1
    merged["characters"] = base_characters
    if new_characters_count > 0 or updated_characters_count > 0:
        log.debug("_story_merge_delta characters merged new=%d updated=%d total=%d", new_characters_count, updated_characters_count, len(base_characters))
    
    # Merge locations: update by name (descriptive name used as key) or append
    # Locations can be objects with name field, or strings (for backwards compatibility)
    base_locations = merged.get("locations") if isinstance(merged.get("locations"), list) else []
    delta_locations = delta.get("locations") if isinstance(delta.get("locations"), list) else []
    # Normalize base_locations: convert strings to objects
    normalized_base_locations = []
    loc_by_name = {}
    for loc in base_locations:
        if isinstance(loc, dict):
            loc_name = loc.get("location_name") or loc.get("name")
            if loc_name:
                normalized_base_locations.append(loc)
                loc_by_name[loc_name] = loc
        elif isinstance(loc, str):
            # Backwards compatibility: convert string to object
            loc_obj = {"location_name": loc, "location_description": "", "location_atmosphere": ""}
            normalized_base_locations.append(loc_obj)
            loc_by_name[loc] = loc_obj
    # Merge delta locations
    new_locations_count = 0
    updated_locations_count = 0
    for delta_loc in delta_locations:
        if isinstance(delta_loc, str):
            # Backwards compatibility: convert string to object
            delta_loc = {"location_name": delta_loc, "location_description": "", "location_atmosphere": ""}
        if not isinstance(delta_loc, dict):
            continue
        loc_name = delta_loc.get("location_name") or delta_loc.get("name")
        if not isinstance(loc_name, str) or not loc_name.strip():
            continue
        loc_name = loc_name.strip()
        if loc_name in loc_by_name:
            # Update existing location (merge fields, don't replace)
            existing_loc = loc_by_name[loc_name]
            existing_loc.update(delta_loc)
            updated_locations_count += 1
        else:
            # New location, append
            normalized_base_locations.append(delta_loc)
            loc_by_name[loc_name] = delta_loc
            new_locations_count += 1
    merged["locations"] = normalized_base_locations
    if new_locations_count > 0 or updated_locations_count > 0:
        log.debug("_story_merge_delta locations merged new=%d updated=%d total=%d", new_locations_count, updated_locations_count, len(normalized_base_locations))
    
    # Merge objects: update by name (descriptive name used as key) or append
    # Objects can be objects with name field, or strings (for backwards compatibility)
    base_objects = merged.get("objects") if isinstance(merged.get("objects"), list) else []
    delta_objects = delta.get("objects") if isinstance(delta.get("objects"), list) else []
    # Normalize base_objects: convert strings to objects
    normalized_base_objects = []
    obj_by_name = {}
    for obj in base_objects:
        if isinstance(obj, dict):
            obj_name = obj.get("object_name") or obj.get("name")
            if obj_name:
                normalized_base_objects.append(obj)
                obj_by_name[obj_name] = obj
        elif isinstance(obj, str):
            # Backwards compatibility: convert string to object
            obj_obj = {"object_name": obj, "object_description": ""}
            normalized_base_objects.append(obj_obj)
            obj_by_name[obj] = obj_obj
    # Merge delta objects
    new_objects_count = 0
    updated_objects_count = 0
    for delta_obj in delta_objects:
        if isinstance(delta_obj, str):
            # Backwards compatibility: convert string to object
            delta_obj = {"object_name": delta_obj, "object_description": ""}
        if not isinstance(delta_obj, dict):
            continue
        obj_name = delta_obj.get("object_name") or delta_obj.get("name")
        if not isinstance(obj_name, str) or not obj_name.strip():
            continue
        obj_name = obj_name.strip()
        if obj_name in obj_by_name:
            # Update existing object (merge fields, don't replace)
            existing_obj = obj_by_name[obj_name]
            existing_obj.update(delta_obj)
            updated_objects_count += 1
        else:
            # New object, append
            normalized_base_objects.append(delta_obj)
            obj_by_name[obj_name] = delta_obj
            new_objects_count += 1
    merged["objects"] = normalized_base_objects
    if new_objects_count > 0 or updated_objects_count > 0:
        log.debug("_story_merge_delta objects merged new=%d updated=%d total=%d", new_objects_count, updated_objects_count, len(normalized_base_objects))
    
    return merged


def _story_update_stall_guard(
    *,
    prior_beats_time_s: Optional[float],
    prior_beats_count: Optional[int],
    cur_beats_time_s: float,
    cur_beats_count: int,
    consecutive_no_progress: int,
):
    grew = True
    if prior_beats_time_s is not None and prior_beats_count is not None:
        grew = (float(cur_beats_time_s) > float(prior_beats_time_s)) or (int(cur_beats_count) > int(prior_beats_count))
    if not grew:
        consecutive_no_progress = int(consecutive_no_progress) + 1
    else:
        consecutive_no_progress = 0
    return grew, consecutive_no_progress, float(cur_beats_time_s), int(cur_beats_count)


async def draft_story_graph(user_prompt: str, duration_hint_s: Optional[float], trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Committee-driven story generator.

    This is intentionally heavy: it relies on repeated committee calls to draft, audit, and
    revise until the story is coherent and follows the user intent.
    """
    t0 = time.monotonic()
    log.debug("draft_story_graph start trace_id=%s user_prompt_len=%d duration_hint_s=%s", 
             trace_id, len(user_prompt) if user_prompt else 0, duration_hint_s)
    prompt_text = _norm_prompt(user_prompt)
    duration = float(duration_hint_s) if isinstance(duration_hint_s, (int, float)) else 0.0
    if duration <= 0.0:
        log.debug("draft_story_graph duration_hint_s invalid, using default trace_id=%s duration_hint_s=%s", trace_id, duration_hint_s)
        duration = 600.0
    # NOTE: trace_id must be provided - no generation, no fallback, no failure
    log.info("story.draft start trace_id=%s duration_hint_s=%.2f prompt_len=%d", trace_id, float(duration), len(prompt_text))
    raw = await _committee_generate_story_raw(prompt_text, duration, trace_id=trace_id)
    # Keep trace_id constant - don't mutate it
    story = await _jsonify_then_parse(
        raw_text=raw,
        expected_schema=_story_schema_template(),
        trace_id=trace_id,
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
    
    story_acts_count = len(story.get("acts") or [])
    story_characters_count = len(story.get("characters") or [])
    story_locations_count = len(story.get("locations") or [])
    story_objects_count = len(story.get("objects") or [])
    log.info("draft_story_graph initial story parsed trace_id=%s acts_count=%d characters_count=%d locations_count=%d objects_count=%d", 
            trace_id, story_acts_count, story_characters_count, story_locations_count, story_objects_count)
    if story_characters_count > 0:
        character_ids = [char.get("character_id") for char in story.get("characters", []) if isinstance(char, dict) and char.get("character_id")]
        log.info("draft_story_graph initial characters trace_id=%s character_ids=%s", trace_id, character_ids[:10])
    if story_locations_count > 0:
        location_names = [loc.get("location_name") for loc in story.get("locations", []) if isinstance(loc, dict) and loc.get("location_name")]
        log.info("draft_story_graph initial locations trace_id=%s location_names=%s", trace_id, location_names[:10])
    if story_objects_count > 0:
        object_names = [obj.get("object_name") for obj in story.get("objects", []) if isinstance(obj, dict) and obj.get("object_name")]
        log.info("draft_story_graph initial objects trace_id=%s object_names=%s", trace_id, object_names[:10])
    
    # Trace initial story creation
    trace_event(
        "story.initial_created",
        {
            "trace_id": trace_id,
            "acts_count": story_acts_count,
            "characters_count": story_characters_count,
            "locations_count": story_locations_count,
            "objects_count": story_objects_count,
        }
    )
    
    # Ensure lock bundles exist for all characters
    log.debug("draft_story_graph ensuring character lock bundles trace_id=%s", trace_id)
    await _ensure_story_character_lock_bundles(story, trace_id=trace_id)
    log.debug("draft_story_graph character lock bundles ensured trace_id=%s", trace_id)
    # Apply state changes from story to lock bundles
    await _apply_story_state_changes_to_locks(story, trace_id=trace_id)
    log.debug("draft_story_graph initial state changes applied to locks trace_id=%s", trace_id)
    
    # Iterative expansion + audit/revise loop (can be large).
    # We stop ONLY when the committee audit says to accept (authoritative), not after an arbitrary number of loops.
    pass_index = 0
    consecutive_no_progress = 0
    last_beats_time_s: Optional[float] = None
    last_beats_count: Optional[int] = None
    max_passes = int(STORY_MAX_PASSES)
    stop_requested = False
    stop_reason = ""
    log.info("draft_story_graph starting iteration loop trace_id=%s max_passes=%d", trace_id, max_passes)
    # No `break`: the loop exits only via its condition.
    while (not stop_requested) and (max_passes <= 0 or pass_index < max_passes):
        # Measure current duration coverage (sum beat time hints) for logging + context.
        total_hint, total_beats_time, total_beats_count, ratio = _story_measure_coverage(story)

        log.info(
            "story.draft.pass start trace_id=%s pass_index=%d beats_time_s=%.2f duration_hint_s=%.2f ratio=%.3f",
            trace_id,
            int(pass_index),
            float(total_beats_time),
            float(total_hint),
            float(ratio),
        )

        # Audit at start of each iteration (authoritative "done" signal).
        # Keep trace_id constant - don't mutate it
        log.info("draft_story_graph calling audit trace_id=%s pass_index=%d", trace_id, pass_index)
        audit = await _committee_audit_story(story, prompt_text, trace_id=trace_id)
        issues = audit.get("issues") if isinstance(audit, dict) and isinstance(audit.get("issues"), list) else []
        must_fix = bool(audit.get("must_fix")) if isinstance(audit, dict) else False
        done = bool(audit.get("done")) if isinstance(audit, dict) else False
        length_ok = bool(audit.get("length_ok")) if isinstance(audit, dict) else False
        next_action = audit.get("next_action") if isinstance(audit, dict) else None
        next_action_s = str(next_action) if isinstance(next_action, str) else ""
        audit_summary = audit.get("summary") or "" if isinstance(audit, dict) else ""
        log.info("draft_story_graph audit complete trace_id=%s pass_index=%d issues_count=%d must_fix=%s done=%s length_ok=%s next_action=%s", 
                trace_id, pass_index, len(issues), must_fix, done, length_ok, next_action_s)
        log.debug("draft_story_graph audit summary trace_id=%s pass_index=%d summary=%s", 
                 trace_id, pass_index, audit_summary[:300] if audit_summary else "")

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
            log.info("draft_story_graph stopping: accept trace_id=%s pass_index=%d", trace_id, pass_index)
            stop_requested = True
            stop_reason = "accept"
            break

        # Fix pass when committee requests revision (or marks must_fix).
        if (not stop_requested) and (must_fix or next_action_s in ("revise", "extend_then_revise")):
            log.info("draft_story_graph starting revision trace_id=%s pass_index=%d must_fix=%s next_action=%s", 
                    trace_id, pass_index, must_fix, next_action_s)
            # Keep trace_id constant - don't mutate it
            log.info("draft_story_graph calling _committee_revise_story trace_id=%s pass_index=%d", trace_id, pass_index)
            revise_t0 = time.perf_counter()
            delta = await _committee_revise_story(
                story,
                prompt_text,
                audit,
                trace_id=trace_id,
            )
            revise_dur = time.perf_counter() - revise_t0
            log.info("draft_story_graph _committee_revise_story done trace_id=%s pass_index=%d dur_ms=%d", trace_id, pass_index, int(revise_dur * 1000))
            delta_type = type(delta).__name__
            delta_keys = list(delta.keys()) if isinstance(delta, dict) else []
            delta_acts = delta.get("acts") if isinstance(delta, dict) and isinstance(delta.get("acts"), list) else []
            delta_characters = delta.get("characters") if isinstance(delta, dict) and isinstance(delta.get("characters"), list) else []
            delta_locations = delta.get("locations") if isinstance(delta, dict) and isinstance(delta.get("locations"), list) else []
            delta_objects = delta.get("objects") if isinstance(delta, dict) and isinstance(delta.get("objects"), list) else []
            log.info("draft_story_graph revision delta received trace_id=%s pass_index=%d delta_type=%s delta_keys_count=%d delta_acts_count=%d delta_characters_count=%d delta_locations_count=%d delta_objects_count=%d", 
                    trace_id, pass_index, delta_type, len(delta_keys), len(delta_acts), len(delta_characters), len(delta_locations), len(delta_objects))
            if delta_characters:
                new_character_ids = [char.get("character_id") for char in delta_characters if isinstance(char, dict) and char.get("character_id")]
                log.info("draft_story_graph revision new characters trace_id=%s pass_index=%d character_ids=%s", trace_id, pass_index, new_character_ids[:10])
                # Trace new character creation
                trace_event(
                    "story.characters_created",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "context": "revision",
                        "character_ids": new_character_ids[:20],  # Limit to first 20
                        "count": len(new_character_ids),
                    }
                )
            if delta_locations:
                new_location_names = [loc.get("location_name") for loc in delta_locations if isinstance(loc, dict) and loc.get("location_name")]
                log.info("draft_story_graph revision new locations trace_id=%s pass_index=%d location_names=%s", trace_id, pass_index, new_location_names[:10])
                # Trace new location creation
                trace_event(
                    "story.locations_created",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "context": "revision",
                        "location_names": new_location_names[:20],  # Limit to first 20
                        "count": len(new_location_names),
                    }
                )
            if delta_objects:
                new_object_names = [obj.get("object_name") for obj in delta_objects if isinstance(obj, dict) and obj.get("object_name")]
                log.info("draft_story_graph revision new objects trace_id=%s pass_index=%d object_names=%s", trace_id, pass_index, new_object_names[:10])
                # Trace new object creation
                trace_event(
                    "story.objects_created",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "context": "revision",
                        "object_names": new_object_names[:20],  # Limit to first 20
                        "count": len(new_object_names),
                    }
                )
            if isinstance(delta, dict) and delta:
                # Use LLM to intelligently merge delta into base story
                log.info("draft_story_graph merging revision delta trace_id=%s pass_index=%d", trace_id, pass_index)
                merge_t0 = time.perf_counter()
                story = await _story_merge_delta_with_llm(
                    base_story=story,
                    delta=delta,
                    user_prompt=prompt_text,
                    trace_id=trace_id,
                )
                merge_dur = time.perf_counter() - merge_t0
                log.info("draft_story_graph revision merge complete trace_id=%s pass_index=%d dur_ms=%d", trace_id, pass_index, int(merge_dur * 1000))
                # Ensure lock bundles exist for any new characters added during revision merge
                await _ensure_story_character_lock_bundles(story, trace_id=trace_id)
                log.debug("draft_story_graph revision lock bundles ensured trace_id=%s pass_index=%d", trace_id, pass_index)
                # Apply state changes from story to lock bundles
                await _apply_story_state_changes_to_locks(story, trace_id=trace_id)
                log.debug("draft_story_graph revision state changes applied trace_id=%s pass_index=%d", trace_id, pass_index)
                
                # Trace revision merge
                trace_event(
                    "story.revision_merged",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "dur_ms": int(merge_dur * 1000),
                        "acts_count": len(story.get("acts") or []),
                        "characters_count": len(story.get("characters") or []),
                    }
                )
            else:
                log.warning("draft_story_graph revision delta invalid trace_id=%s pass_index=%d delta_type=%s", 
                           trace_id, pass_index, delta_type)
            if not isinstance(story, dict):
                log.error("draft_story_graph revision failed: story is not dict trace_id=%s pass_index=%d", trace_id, pass_index)
                # Never break: stop deterministically with the best available state.
                stop_requested = True
                stop_reason = "revise_failed"
                break
            story.setdefault("prompt", prompt_text)
            story.setdefault("duration_hint_s", float(duration))
            post_revise_acts = len(story.get("acts") or [])
            post_revise_characters = len(story.get("characters") or [])
            log.info("draft_story_graph revision complete trace_id=%s pass_index=%d acts_count=%d characters_count=%d", 
                    trace_id, pass_index, post_revise_acts, post_revise_characters)

        # Extend pass when committee requests extension (or says length not ok).
        if (not stop_requested) and ((not length_ok) or next_action_s in ("extend", "extend_then_revise")):
            log.info("draft_story_graph starting extension trace_id=%s pass_index=%d length_ok=%s next_action=%s", 
                    trace_id, pass_index, length_ok, next_action_s)
            story_summary = _story_extract_summary(story)
            log.debug("draft_story_graph story_summary extracted for extension trace_id=%s pass_index=%d", trace_id, pass_index)
            # Minimal system message - committee overrides it
            sys_msg = "You are the Story Engine continuation pass."
            # ALL CRITICAL INSTRUCTIONS IN USER MESSAGE (committee overrides system)
            schema_template = _story_schema_template()
            user_msg_content = f"""### [CRITICAL INSTRUCTIONS - READ CAREFULLY]

You MUST EXTEND the existing story graph to meet the requested duration and depth.

CRITICAL: Return ONLY the new content/extensions, not the full story.
- Add NEW acts/scenes/beats to reach the duration. Deeper arcs for long films.
- Maintain strict continuity: state_change events must be reflected later.
- Maintain stable ids: never change existing ids; new ids must be unique and stable.
- Only include NEW acts/scenes/beats with new unique IDs.
- Do NOT include existing acts/scenes/beats.
- Use existing characters/locations/objects from story_summary - reference them by character_id/location_name/object_name
- If you need new characters/locations/objects, add them to the top-level arrays
- Ensure all required fields are present (line_id, speaker, dialogue_text for dialogue; event_type, event_target for state_change events)

CRITICAL: Use ONLY ASCII characters. NO emojis, NO special Unicode symbols, NO non-ASCII characters.

### [EXACT JSON STRUCTURE REQUIRED]
Your extension delta MUST match this exact structure. All field names and nesting must be identical:

{json.dumps(schema_template, ensure_ascii=False, indent=2)}

### [KEY REQUIREMENTS]
- **CRITICAL: CREATE new characters, locations, and objects as needed for extensions.** If your extension requires new characters, locations, or objects, you MUST add them to the top-level arrays. Do not reference entities that aren't in the arrays.
- Characters: array of objects with character_id (string, REQUIRED) and character_name (string) - character_id is used for TTS locks and dialogue speaker field. **CREATE new characters as needed for the extension. Each new character that appears in dialogue or events MUST have an entry in the characters array.**
- Locations: array of objects with location_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Post-Apocalyptic Urban Ruin'), location_description (string), location_atmosphere (string - used for music matching, e.g. 'stormy', 'eerie'). **CREATE new locations as needed for the extension. Each new location referenced in beats MUST have an entry in the locations array.**
- Objects: array of objects with object_name (string, REQUIRED - fully descriptive name used as key for matching, e.g. 'Chaos Emerald (Red)'), object_description (string). **CREATE new objects as needed for the extension. Each new object referenced in beats or state_change events MUST have an entry in the objects array.**
- Locations and objects in beats: arrays of strings (location_name/object_name values that MUST match location_name/object_name from the locations/objects arrays). Can be empty arrays [] if no characters/locations/objects.
- Scenes: scene_summary (string, optional), music_hint (object with scene_mood, scene_tone, scene_energy, scene_location_atmosphere - for music matching)
- Beats: beat_description (string, optional), time_hint_s (float, REQUIRED), characters (array, OPTIONAL - can be empty []), locations (array, OPTIONAL - can be empty []), objects (array, OPTIONAL - can be empty []), events (array), dialogue (array), music_hint (object with beat_mood, beat_tone, beat_energy - optional, inherits from scene)
- Events: array of objects with event_type (string, REQUIRED). 
  * For state_change events: event_target (string, REQUIRED - must be a character_id from characters array OR an object_name from objects array, NOT a location_name) and state_delta (dict, REQUIRED)
  * state_delta structure: A dictionary with direct key-value pairs representing the state changes. The keys are the state property names (e.g. "health", "stance", "energy_level") and values are the new state values.
  * CORRECT state_delta examples: {{"health": 50}}, {{"stance": "ready", "energy_level": "high"}}, {{"eruption_level": "medium"}}
  * INCORRECT state_delta examples: {{"key": "health", "value": 50}} (WRONG - don't use nested "key"/"value" structure), {{"key": "position", "value": "mid-battle stance"}} (WRONG)
  * event_target MUST be a character_id (e.g. "shadow", "charizard") OR an object_name (e.g. "Volcanic Fissure (Active)"), NOT a location_name
  * For non-state_change events: event_target is optional (can be omitted or empty string)
- Dialogue: array of objects with line_id (string, REQUIRED - unique identifier for TTS dialogue_index mapping), speaker (string, REQUIRED - must be a character_id from characters array), dialogue_text (string, REQUIRED)
- Music: Music is created AFTER story generation. Music matching uses descriptive information from scenes/beats: scene_mood, scene_tone, scene_energy, scene_location_atmosphere. Do NOT include actual music in story structure, only music_hint objects.

### [EXTENSION EXAMPLES]
Example extension adding a new act:
{{
  "acts": [
    {{
      "act_id": "act_4",
      "scenes": [
        {{
          "scene_id": "scene_10",
          "scene_summary": "The final confrontation",
          "music_hint": {{
            "scene_mood": "triumphant",
            "scene_tone": "epic",
            "scene_energy": "high",
            "scene_location_atmosphere": "stormy"
          }},
          "beats": [
            {{
              "beat_id": "beat_25",
              "beat_description": "Shadow prepares for the final battle",
              "time_hint_s": 5.0,
              "characters": ["shadow"],
              "locations": ["Post-Apocalyptic Urban Ruin"],
              "objects": [],
              "events": [],
              "dialogue": [
                {{
                  "line_id": "line_050",
                  "speaker": "shadow",
                  "dialogue_text": "This ends now."
                }}
              ],
              "music_hint": {{
                "beat_mood": "determined",
                "beat_tone": "dark",
                "beat_energy": "high"
              }}
            }}
          ]
        }}
      ]
    }}
  ]
}}

### [OUTPUT FORMAT]
Output MUST be a single JSON object (no markdown, no prose). The JSON should contain only new/extended content, structured the same way as the story schema but with only new items.

CRITICAL: Do NOT wrap your output in wrapper keys like 'response', 'data', 'result', 'output', 'content', or 'json'. Output the extension delta object directly at the top level.

### [DATA]
{json.dumps({
    "user_prompt": prompt_text,
    "duration_hint_s": float(total_hint),
    "current_beats_time_s": float(total_beats_time),
    "coverage_ratio": float(ratio),
    "audit_summary": audit,
    "story_summary": story_summary,
}, ensure_ascii=False)}

### [YOUR TASK]
Extend the story to meet the duration requirements. Return ONLY the JSON delta with new content matching the exact structure above. No other text."""
            user_msg = user_msg_content
            # Keep trace_id constant - don't mutate it
            log.info("draft_story_graph calling committee_ai_text for extension trace_id=%s pass_index=%d", trace_id, pass_index)
            env2 = await committee_ai_text(
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                trace_id=trace_id,
                rounds=STORY_COMMITTEE_ROUNDS,
                temperature=STORY_TEMPERATURE,
            )
            env2_ok = env2.get("ok") is not False if isinstance(env2, dict) else False
            log.info("draft_story_graph committee_ai_text for extension done trace_id=%s pass_index=%d ok=%r", trace_id, pass_index, env2_ok)
            raw2 = ""
            rb2 = env2.get("result") if isinstance(env2, dict) else {}
            if isinstance(rb2, dict) and isinstance(rb2.get("text"), str):
                raw2 = str(rb2.get("text") or "")
            else:
                raw2 = str(env2.get("text") or "")
            # Use story schema for extension delta (JSONParser handles missing fields gracefully)
            log.info("draft_story_graph calling _jsonify_then_parse for extension trace_id=%s pass_index=%d", trace_id, pass_index)
            extended_delta = await _jsonify_then_parse(
                raw_text=raw2,
                expected_schema=_story_schema_template(),
                trace_id=trace_id,
                rounds=STORY_COMMITTEE_ROUNDS,
                temperature=STORY_TEMPERATURE,
            )
            extended_delta_type = type(extended_delta).__name__
            extended_delta_keys = list(extended_delta.keys()) if isinstance(extended_delta, dict) else []
            extended_delta_acts = extended_delta.get("acts") if isinstance(extended_delta, dict) and isinstance(extended_delta.get("acts"), list) else []
            extended_delta_characters = extended_delta.get("characters") if isinstance(extended_delta, dict) and isinstance(extended_delta.get("characters"), list) else []
            extended_delta_locations = extended_delta.get("locations") if isinstance(extended_delta, dict) and isinstance(extended_delta.get("locations"), list) else []
            extended_delta_objects = extended_delta.get("objects") if isinstance(extended_delta, dict) and isinstance(extended_delta.get("objects"), list) else []
            log.info("draft_story_graph extension delta received trace_id=%s pass_index=%d delta_type=%s delta_keys_count=%d delta_acts_count=%d delta_characters_count=%d delta_locations_count=%d delta_objects_count=%d", 
                    trace_id, pass_index, extended_delta_type, len(extended_delta_keys), len(extended_delta_acts), len(extended_delta_characters), len(extended_delta_locations), len(extended_delta_objects))
            if extended_delta_characters:
                new_character_ids = [char.get("character_id") for char in extended_delta_characters if isinstance(char, dict) and char.get("character_id")]
                log.info("draft_story_graph extension new characters trace_id=%s pass_index=%d character_ids=%s", trace_id, pass_index, new_character_ids[:10])
                # Trace new character creation
                trace_event(
                    "story.characters_created",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "context": "extension",
                        "character_ids": new_character_ids[:20],  # Limit to first 20
                        "count": len(new_character_ids),
                    }
                )
            if extended_delta_locations:
                new_location_names = [loc.get("location_name") for loc in extended_delta_locations if isinstance(loc, dict) and loc.get("location_name")]
                log.info("draft_story_graph extension new locations trace_id=%s pass_index=%d location_names=%s", trace_id, pass_index, new_location_names[:10])
                # Trace new location creation
                trace_event(
                    "story.locations_created",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "context": "extension",
                        "location_names": new_location_names[:20],  # Limit to first 20
                        "count": len(new_location_names),
                    }
                )
            if extended_delta_objects:
                new_object_names = [obj.get("object_name") for obj in extended_delta_objects if isinstance(obj, dict) and obj.get("object_name")]
                log.info("draft_story_graph extension new objects trace_id=%s pass_index=%d object_names=%s", trace_id, pass_index, new_object_names[:10])
                # Trace new object creation
                trace_event(
                    "story.objects_created",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "context": "extension",
                        "object_names": new_object_names[:20],  # Limit to first 20
                        "count": len(new_object_names),
                    }
                )
            if isinstance(extended_delta, dict) and extended_delta:
                # Use LLM to intelligently merge extension delta into base story
                log.info("draft_story_graph merging extension delta trace_id=%s pass_index=%d", trace_id, pass_index)
                merge_ext_t0 = time.perf_counter()
                story = await _story_merge_delta_with_llm(
                    base_story=story,
                    delta=extended_delta,
                    user_prompt=prompt_text,
                    trace_id=trace_id,
                )
                merge_ext_dur = time.perf_counter() - merge_ext_t0
                log.info("draft_story_graph extension merge complete trace_id=%s pass_index=%d dur_ms=%d", trace_id, pass_index, int(merge_ext_dur * 1000))
                # Ensure lock bundles exist for any new characters added during extension merge
                await _ensure_story_character_lock_bundles(story, trace_id=trace_id)
                # Apply state changes from story to lock bundles
                await _apply_story_state_changes_to_locks(story, trace_id=trace_id)
                story.setdefault("prompt", prompt_text)
                
                # Trace extension merge
                trace_event(
                    "story.extension_merged",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "dur_ms": int(merge_ext_dur * 1000),
                        "acts_count": len(story.get("acts") or []),
                        "characters_count": len(story.get("characters") or []),
                        "locations_count": len(story.get("locations") or []),
                        "objects_count": len(story.get("objects") or []),
                    }
                )
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
                log.debug("draft_story_graph checking stall guard trace_id=%s pass_index=%d prior_beats_time_s=%s prior_beats_count=%s cur_beats_time_s=%.2f cur_beats_count=%d consecutive_no_progress=%d", 
                         trace_id, pass_index, last_beats_time_s, last_beats_count, total_beats_time, total_beats_count, consecutive_no_progress)
                grew, consecutive_no_progress, last_beats_time_s, last_beats_count = _story_update_stall_guard(
                    prior_beats_time_s=last_beats_time_s,
                    prior_beats_count=last_beats_count,
                    cur_beats_time_s=float(total_beats_time),
                    cur_beats_count=int(total_beats_count),
                    consecutive_no_progress=int(consecutive_no_progress),
                )
                log.info("draft_story_graph stall guard result trace_id=%s pass_index=%d grew=%s consecutive_no_progress=%d last_beats_time_s=%.2f last_beats_count=%d", 
                        trace_id, pass_index, grew, consecutive_no_progress, last_beats_time_s, last_beats_count)
                if consecutive_no_progress >= 5:
                    log.warning("draft_story_graph stalling detected trace_id=%s pass_index=%d consecutive_no_progress=%d", 
                               trace_id, pass_index, consecutive_no_progress)
                    trace_event(
                        "story.stalled",
                        {
                            "trace_id": trace_id,
                            "pass_index": int(pass_index),
                            "consecutive_no_progress": int(consecutive_no_progress),
                            "last_audit": audit,
                        },
                    )
                    stop_requested = True
                    stop_reason = "stalled"
                else:
                    # CRITICAL: Audit immediately after extend to check if done, don't wait for next loop iteration
                    # Keep trace_id constant - don't mutate it
                    log.debug("draft_story_graph post-extend audit trace_id=%s pass_index=%d", trace_id, pass_index)
                    post_extend_audit = await _committee_audit_story(story, prompt_text, trace_id=trace_id)
                    post_done = bool(post_extend_audit.get("done")) if isinstance(post_extend_audit, dict) else False
                    post_next_action = post_extend_audit.get("next_action") if isinstance(post_extend_audit, dict) else None
                    post_next_action_s = str(post_next_action) if isinstance(post_next_action, str) else ""
                    post_issues_count = len(post_extend_audit.get("issues") or []) if isinstance(post_extend_audit, dict) else 0
                    log.info("draft_story_graph post-extend audit complete trace_id=%s pass_index=%d done=%s next_action=%s issues_count=%d", 
                            trace_id, pass_index, post_done, post_next_action_s, post_issues_count)
                    if post_done and post_next_action_s == "accept":
                        log.info("draft_story_graph stopping: accept after extension trace_id=%s pass_index=%d", trace_id, pass_index)
                        stop_requested = True
                        stop_reason = "accept"
                        audit = post_extend_audit  # Use the post-extend audit as the final audit
            else:
                log.warning("draft_story_graph extension failed: invalid delta trace_id=%s pass_index=%d delta_type=%s", 
                           trace_id, pass_index, extended_delta_type)
                stop_requested = True
                stop_reason = "extend_failed"

        # Default: if committee doesn't require fix/extend, stop.
        if not stop_requested and not (must_fix or (not length_ok) or next_action_s in ("extend", "extend_then_revise", "revise")):
            log.info("draft_story_graph stopping: no action required trace_id=%s pass_index=%d must_fix=%s length_ok=%s next_action=%s", 
                    trace_id, pass_index, must_fix, length_ok, next_action_s)
            stop_requested = True
            stop_reason = "no_action"

        if not stop_requested:
            pass_index += 1

    if not stop_reason and max_passes > 0 and pass_index >= max_passes:
        log.warning("draft_story_graph stopping: max_passes reached trace_id=%s pass_index=%d max_passes=%d", 
                   trace_id, pass_index, max_passes)
        stop_reason = "max_passes"
        trace_event("story.max_passes_reached", {"trace_id": trace_id, "max_passes": int(max_passes)})
    
    final_acts = len(story.get("acts") or [])
    final_characters = len(story.get("characters") or [])
    final_locations = len(story.get("locations") or [])
    final_objects = len(story.get("objects") or [])
    final_total_hint, final_total_beats_time, final_total_beats_count, final_ratio = _story_measure_coverage(story)
    total_dur = time.monotonic() - t0
    log.info("story.draft complete trace_id=%s total_dur_s=%.2f passes=%d stop_reason=%s acts_count=%d characters_count=%d locations_count=%d objects_count=%d beats_count=%d beats_time_s=%.2f duration_hint_s=%.2f ratio=%.3f", 
            trace_id, total_dur, pass_index, stop_reason, final_acts, final_characters, final_locations, final_objects, final_total_beats_count, final_total_beats_time, final_total_hint, final_ratio)
    log.debug("draft_story_graph final story structure trace_id=%s story_keys=%s", 
             trace_id, list(story.keys())[:20] if isinstance(story, dict) else [])
    
    # Trace final story completion
    trace_event(
        "story.draft_complete",
        {
            "trace_id": trace_id,
            "total_dur_s": float(total_dur),
            "passes": pass_index,
            "stop_reason": stop_reason,
            "acts_count": final_acts,
            "characters_count": final_characters,
            "locations_count": final_locations,
            "objects_count": final_objects,
            "beats_count": final_total_beats_count,
            "beats_time_s": float(final_total_beats_time),
            "duration_hint_s": float(final_total_hint),
            "coverage_ratio": float(final_ratio),
        }
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
                    event_type = ev.get("event_type") or ev.get("type") or ""
                    if event_type != "state_change":
                        continue
                    target = ev.get("event_target") or ev.get("target") or ""
                    state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                    if not isinstance(target, str) or not state_delta:
                        continue
                    state_entry = character_state.get(target) or {}
                    for key, val in state_delta.items():
                        prev = state_entry.get(key)
                        if prev is not None and prev != val:
                            issues.append(
                                {
                                    "code": "state_inconsistent",
                                    "message": "Conflicting state change for target.",
                                    "target": target,
                                    "state_key": key,
                                    "prev": prev,
                                    "new": val,
                                    "act_id": act_id,
                                    "scene_id": scene_id,
                                }
                            )
                        state_entry[key] = val
                    character_state[target] = state_entry
    # Committee audit (deep continuity + intent). Merge issues.
    # NOTE: trace_id must be provided - no generation, no fallback, no failure
    audit = await _committee_audit_story(story if isinstance(story, dict) else {}, _norm_prompt(user_prompt), trace_id=trace_id)
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
    trace_id: str,
    conversation_id: str,
    tts_runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    character_bundles: Optional[Dict[str, Dict[str, Any]]] = None,
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
    character_ids = [character.get("character_id") for character in characters if isinstance(character, dict) and isinstance(character.get("character_id"), str)]
    # Ensure a basic TTS lock branch exists per character
    tts_branch = updated_locks.get("tts")
    if not isinstance(tts_branch, dict):
        tts_branch = {}
        updated_locks["tts"] = tts_branch
    char_voice_locks = tts_branch.get("characters")
    if not isinstance(char_voice_locks, dict):
        char_voice_locks = {}
        tts_branch["characters"] = char_voice_locks
    for temp_character_id in character_ids:
        if not isinstance(temp_character_id, str) or not temp_character_id:
            continue
        if temp_character_id not in char_voice_locks:
            char_voice_locks[temp_character_id] = {
                "character_id": temp_character_id,
                "voice_profile": {
                    "description": f"Default voice profile for {temp_character_id} based on story context",
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
                    text = line.get("dialogue_text") or line.get("text") or ""
                    if not isinstance(line_id, str) or not line_id or not isinstance(text, str) or not text.strip():
                        continue
                    if not isinstance(speaker, str) or not speaker:
                        continue
                    dialogue_index[line_id] = {
                        "character_id": speaker,
                        "text": text,
                        "audio_path": None,
                        "duration_s": None,
                        "error": None,
                    }
    if not dialogue_index:
        return updated_locks, dialogue_index
    # Pre-generate audio via tts.speak for each unique line
    for line_id, entry in dialogue_index.items():
        speaker = entry.get("character_id")
        text = entry.get("text")
        if not isinstance(speaker, str) or not isinstance(text, str):
            continue
        args_tts: Dict[str, Any] = {
            "text": text,
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "quality_profile": profile_name,
        }
        # Prefer per-character bundles (stable voices/locks); fallback to story-level locks bundle.
        if isinstance(character_bundles, dict) and isinstance(character_bundles.get(speaker), dict) and character_bundles.get(speaker):
            args_tts["lock_bundle"] = character_bundles.get(speaker)
            args_tts.setdefault("voice_id", speaker)
        elif isinstance(updated_locks, dict) and updated_locks:
            args_tts["lock_bundle"] = updated_locks
        args_tts["character_id"] = speaker
        res = await tts_runner({"name": "tts.speak", "arguments": args_tts})
        if isinstance(res, Dict) and isinstance(res.get("result"), dict):
            r = res.get("result") or {}
            meta_audio = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            external_ids = r.get("ids") if isinstance(r.get("ids"), dict) else {}  # external API identifiers
            audio_id = external_ids.get("audio_id")
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
                    "character_id": speaker,
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
                    "character_id": speaker,
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
        prev = issue.get("prev")
        new = issue.get("new")
        if not isinstance(target, str) or not isinstance(state_key, str):
            continue
        # Normalize by keeping the earliest value and updating later state_delta entries.
        canonical = prev
        for act in acts:
            scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
            for scene in scenes:
                beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
                for beat in beats:
                    events = beat.get("events") if isinstance(beat.get("events"), list) else []
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        event_type = ev.get("event_type") or ev.get("type") or ""
                        if event_type != "state_change":
                            continue
                        event_target = ev.get("event_target") or ev.get("target") or ""
                        if event_target != target:
                            continue
                        state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                        if state_key in state_delta:
                            state_delta[state_key] = canonical
                            ev["state_delta"] = state_delta
    adjusted["acts"] = acts
    # Committee revision using the issues we found (and the prompt if available)
    prompt_txt = _norm_prompt(user_prompt or adjusted.get("prompt") or "")
    audit = {"issues": issues, "summary": "deterministic_consistency_issues", "must_fix": True}
    # NOTE: trace_id must be provided - no generation, no fallback, no failure
    revised = await _committee_revise_story(adjusted, prompt_txt, audit, trace_id=trace_id)
    if isinstance(revised, dict) and revised:
        return revised
    return adjusted


def derive_scenes_and_shots(story: Dict[str, Any], trace_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Derive flat scene and shot lists from the story graph.

    Scenes are flattened from acts[*].scenes. Shots are derived by creating one shot per beat.
    Character state is tracked across beats and included in shot character_states.
    
    Args:
        story: Story graph dictionary
        trace_id: Optional trace ID for logging
    """
    scenes_out: List[Dict[str, Any]] = []
    shots_out: List[Dict[str, Any]] = []
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    log.debug("derive_scenes_and_shots start trace_id=%s acts_count=%d", trace_id, len(acts))
    scene_order = 0
    shot_index = 0
    # Maintain a simple per-character state timeline while deriving shots.
    character_state: Dict[str, Dict[str, Any]] = {}
    state_change_events_processed = 0
    for act_idx, act in enumerate(acts):
        if not isinstance(act, dict):
            log.debug("derive_scenes_and_shots skipping invalid act act_idx=%d trace_id=%s", act_idx, trace_id)
            continue
        act_id = act.get("act_id") or ("act_%d" % (act_idx + 1))
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        log.debug("derive_scenes_and_shots processing act act_id=%s scenes_count=%d trace_id=%s", act_id, len(scenes), trace_id)
        for scene_idx, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                log.debug("derive_scenes_and_shots skipping invalid scene act_id=%s scene_idx=%d trace_id=%s", act_id, scene_idx, trace_id)
                continue
            scene_id = scene.get("scene_id") or ("%s_sc_%d" % (act_id, scene_order + 1))
            scene_order += 1
            scene_entry: Dict[str, Any] = {
                "scene_id": scene_id,
                "act_id": act_id,
                "summary": scene.get("scene_summary") or scene.get("summary") or "",
                "order": scene_order,
            }
            scenes_out.append(scene_entry)
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat_idx, beat in enumerate(beats):
                beat_id = beat.get("beat_id") or ("beat_%d" % beat_idx)
                events = beat.get("events") if isinstance(beat.get("events"), list) else []
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    event_type = ev.get("event_type") or ev.get("type") or ""
                    if event_type != "state_change":
                        continue
                    target = ev.get("event_target") or ev.get("target") or ""
                    state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                    if not isinstance(target, str) or not target.strip() or not state_delta:
                        continue
                    target = target.strip()
                    state_entry = character_state.get(target) or {}
                    for key, val in state_delta.items():
                        state_entry[key] = val
                    character_state[target] = state_entry
                    state_change_events_processed += 1
                    log.debug("derive_scenes_and_shots processed state_change trace_id=%s target=%s state_delta=%s scene_id=%s beat_id=%s", 
                             trace_id, target, state_delta, scene_id, beat_id)
                shot_id = "%s_sh_%d" % (scene_id, shot_index + 1)
                shot_index += 1
                description = beat.get("beat_description") or beat.get("description") or ""
                duration_s = float(beat.get("time_hint_s") or 0.0)
                char_ids = beat.get("characters") or []
                shot_states: Dict[str, Dict[str, Any]] = {}
                for character_id in char_ids:
                    if isinstance(character_id, str) and character_id in character_state:
                        shot_states[character_id] = dict(character_state[character_id])
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
    # Count shots with character states
    shots_with_states = sum(1 for shot in shots_out if isinstance(shot, dict) and shot.get("character_states"))
    
    if state_change_events_processed > 0:
        log.info("derive_scenes_and_shots state_changes_processed=%d characters_with_state=%d shots_with_states=%d trace_id=%s", 
                state_change_events_processed, len(character_state), shots_with_states, trace_id)
        # Trace state changes in shot derivation
        trace_event(
            "story.shots_derived_with_state",
            {
                "trace_id": trace_id,
                "state_changes_processed": state_change_events_processed,
                "characters_with_state": len(character_state),
                "scenes_count": len(scenes_out),
                "shots_count": len(shots_out),
                "shots_with_states": shots_with_states,
            }
        )
    log.info("derive_scenes_and_shots done trace_id=%s scenes=%d shots=%d shots_with_states=%d", 
            trace_id, len(scenes_out), len(shots_out), shots_with_states)
    
    # Trace shot derivation completion
    trace_event(
        "story.shots_derived",
        {
            "trace_id": trace_id,
            "scenes_count": len(scenes_out),
            "shots_count": len(shots_out),
            "shots_with_states": shots_with_states,
        }
    )
    
    return scenes_out, shots_out



