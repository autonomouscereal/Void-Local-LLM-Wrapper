from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

from ..tracing.runtime import trace_event


def draft_story_graph(user_prompt: str, duration_hint_s: Optional[float]) -> Dict[str, Any]:
    """
    Build a minimal but structured story graph from a user prompt and optional duration hint.

    This implementation is intentionally deterministic and lightweight so it can run without
    external model calls. It produces a three-act structure with one scene per act and a small
    number of beats per scene. Downstream tools can refine or override this as needed.
    """
    prompt_text = (user_prompt or "").strip()
    duration = float(duration_hint_s) if isinstance(duration_hint_s, (int, float)) else 0.0
    if duration <= 0.0:
        duration = 600.0
    total_beats = 9
    beat_duration = duration / float(total_beats)
    acts: List[Dict[str, Any]] = []
    characters: List[Dict[str, Any]] = []
    locations: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    beat_index = 0
    act_descriptions = [
        "Act 1: setup and introduction.",
        "Act 2: rising conflict and complications.",
        "Act 3: climax and resolution.",
    ]
    # Minimal multi-character/location/object lists derived from prompt shape
    main_char = {
        "char_id": "char_1",
        "name": "Protagonist",
        "description": f"Main character of the film based on: {prompt_text}",
        "traits": {"role": "hero"},
    }
    ally_char = {
        "char_id": "char_2",
        "name": "Ally",
        "description": "A close ally or friend who supports the protagonist.",
        "traits": {"role": "ally"},
    }
    antagonist_char = {
        "char_id": "char_3",
        "name": "Antagonist",
        "description": "Primary antagonist or opposing force.",
        "traits": {"role": "antagonist"},
    }
    characters.extend([main_char, ally_char, antagonist_char])
    locations.extend(
        [
            {
                "loc_id": "loc_1",
                "name": "Primary Location",
                "description": "Key location for the story.",
            },
            {
                "loc_id": "loc_2",
                "name": "Secondary Location",
                "description": "Secondary setting where conflicts escalate.",
            },
            {
                "loc_id": "loc_3",
                "name": "Climactic Location",
                "description": "Final confrontation or resolution setting.",
            },
        ]
    )
    objects.extend(
        [
            {
                "obj_id": "obj_1",
                "name": "Key Object",
                "description": "An important prop or artifact in the story.",
            },
            {
                "obj_id": "obj_2",
                "name": "Secondary Object",
                "description": "Secondary prop that influences the conflict.",
            },
        ]
    )
    for act_num in range(3):
        act_id = f"act_{act_num+1}"
        act_summary = act_descriptions[act_num]
        scene_id = f"{act_id}_sc_1"
        beats: List[Dict[str, Any]] = []
        for beat_in_act in range(3):
            beat_id = f"beat_{beat_index+1}"
            beat_desc = f"{act_summary} Beat {beat_index+1} derived from prompt."
            # Simple placeholder dialogue lines rotating speakers
            line_id = f"line_{beat_index+1}"
            if act_num == 0:
                speaker_id = main_char["char_id"]
            elif act_num == 1:
                speaker_id = ally_char["char_id"]
            else:
                speaker_id = antagonist_char["char_id"]
            dialogue: List[Dict[str, Any]] = [
                {
                    "line_id": line_id,
                    "speaker": speaker_id,
                    "text": f"Line {beat_index+1} inspired by: {prompt_text[:80]}",
                }
            ]
            # Basic character/location/object assignment per act
            if act_num == 0:
                beat_chars = [main_char["char_id"], ally_char["char_id"]]
                beat_locs = ["loc_1"]
                beat_objs = ["obj_1"]
            elif act_num == 1:
                beat_chars = [main_char["char_id"], ally_char["char_id"], antagonist_char["char_id"]]
                beat_locs = ["loc_2"]
                beat_objs = ["obj_1", "obj_2"]
            else:
                beat_chars = [main_char["char_id"], antagonist_char["char_id"]]
                beat_locs = ["loc_3"]
                beat_objs = ["obj_1"]
            events: List[Dict[str, Any]] = []
            # Inject a simple state_change timeline for main_char across the film
            if beat_index == 3:
                events.append(
                    {
                        "event_id": "evt_char1_loses_arm",
                        "type": "state_change",
                        "target": main_char["char_id"],
                        "state_delta": {"left_arm": "missing"},
                    }
                )
            if beat_index == 6:
                events.append(
                    {
                        "event_id": "evt_char1_gets_robot_arm",
                        "type": "state_change",
                        "target": main_char["char_id"],
                        "state_delta": {"left_arm": "robot"},
                    }
                )
            beats.append(
                {
                    "beat_id": beat_id,
                    "description": beat_desc,
                    "time_hint_s": beat_duration,
                    "characters": beat_chars,
                    "locations": beat_locs,
                    "objects": beat_objs,
                    "events": events,
                    "dialogue": dialogue,
                }
            )
            beat_index += 1
        scene = {
            "scene_id": scene_id,
            "summary": act_summary,
            "beats": beats,
        }
        acts.append(
            {
                "act_id": act_id,
                "summary": act_summary,
                "scenes": [scene],
            }
        )
    story: Dict[str, Any] = {
        "prompt": prompt_text,
        "duration_hint_s": duration,
        "acts": acts,
        "characters": characters,
        "locations": locations,
        "objects": objects,
    }
    return story


def check_story_consistency(story: Dict[str, Any], user_prompt: str) -> List[Dict[str, Any]]:
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
                    if ev.get("type") != "state_change":
                        continue
                    target = ev.get("target")
                    state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                    if not isinstance(target, str) or not state_delta:
                        continue
                    state_entry = character_state.get(target) or {}
                    for key, val in state_delta.items():
                        prev_val = state_entry.get(key)
                        if prev_val is not None and prev_val != val:
                            issues.append(
                                {
                                    "code": "state_inconsistent",
                                    "message": "Conflicting state change for target.",
                                    "target": target,
                                    "state_key": key,
                                    "prev": prev_val,
                                    "new": val,
                                    "act_id": act_id,
                                    "scene_id": scene_id,
                                }
                            )
                        state_entry[key] = val
                    character_state[target] = state_entry
    return issues


async def ensure_tts_locks_and_dialogue_audio(
    story: Dict[str, Any],
    locks_arg: Dict[str, Any],
    profile_name: str,
    trace_id: Optional[str],
    tts_runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
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
    character_ids = [c.get("char_id") for c in characters if isinstance(c, dict) and isinstance(c.get("char_id"), str)]
    # Ensure a basic TTS lock branch exists per character
    tts_branch = updated_locks.get("tts")
    if not isinstance(tts_branch, dict):
        tts_branch = {}
        updated_locks["tts"] = tts_branch
    char_voice_locks = tts_branch.get("characters")
    if not isinstance(char_voice_locks, dict):
        char_voice_locks = {}
        tts_branch["characters"] = char_voice_locks
    for c_id in character_ids:
        if not isinstance(c_id, str) or not c_id:
            continue
        if c_id not in char_voice_locks:
            char_voice_locks[c_id] = {
                "char_id": c_id,
                "voice_profile": {
                    "description": f"Default voice profile for {c_id} based on story context",
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
                    text = line.get("text")
                    if not isinstance(line_id, str) or not line_id or not isinstance(text, str) or not text.strip():
                        continue
                    if not isinstance(speaker, str) or not speaker:
                        continue
                    dialogue_index[line_id] = {
                        "char_id": speaker,
                        "text": text,
                        "audio_path": None,
                        "duration_s": None,
                        "error": None,
                    }
    if not dialogue_index:
        return updated_locks, dialogue_index
    # Pre-generate audio via tts.speak for each unique line
    for line_id, entry in dialogue_index.items():
        speaker = entry.get("char_id")
        text = entry.get("text")
        if not isinstance(speaker, str) or not isinstance(text, str):
            continue
        args_tts: Dict[str, Any] = {
            "text": text,
            "trace_id": trace_id,
            "quality_profile": profile_name,
        }
        if isinstance(updated_locks, dict) and updated_locks:
            args_tts["lock_bundle"] = updated_locks
        args_tts["character_id"] = speaker
        res = await tts_runner({"name": "tts.speak", "arguments": args_tts})
        if isinstance(res, Dict) and isinstance(res.get("result"), dict):
            r = res.get("result") or {}
            meta_audio = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            ids_audio = r.get("ids") if isinstance(r.get("ids"), dict) else {}
            audio_id = ids_audio.get("audio_id")
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
                    "char_id": speaker,
                    "text": (text[:128] if isinstance(text, str) else ""),
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
                    "char_id": speaker,
                    "error": str(res.get("error")),
                },
            )
    return updated_locks, dialogue_index


def fix_story(story: Dict[str, Any], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply simple, deterministic fixes to address basic structural issues.

    The initial implementation only adjusts beat duration when there is a duration mismatch.
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
        prev_val = issue.get("prev")
        new_val = issue.get("new")
        if not isinstance(target, str) or not isinstance(state_key, str):
            continue
        # Normalize by keeping the earliest value and updating later state_delta entries.
        canonical_val = prev_val
        for act in acts:
            scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
            for scene in scenes:
                beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
                for beat in beats:
                    events = beat.get("events") if isinstance(beat.get("events"), list) else []
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        if ev.get("type") != "state_change":
                            continue
                        if ev.get("target") != target:
                            continue
                        state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                        if state_key in state_delta:
                            state_delta[state_key] = canonical_val
                            ev["state_delta"] = state_delta
    adjusted["acts"] = acts
    return adjusted


def derive_scenes_and_shots(story: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Derive flat scene and shot lists from the story graph.

    Scenes are flattened from acts[*].scenes. Shots are derived by creating one shot per beat.
    """
    scenes_out: List[Dict[str, Any]] = []
    shots_out: List[Dict[str, Any]] = []
    acts = story.get("acts") if isinstance(story.get("acts"), list) else []
    scene_order = 0
    shot_index = 0
    # Maintain a simple per-character state timeline while deriving shots.
    character_state: Dict[str, Dict[str, Any]] = {}
    for act in acts:
        act_id = act.get("act_id") or f"act_{len(scenes_out)+1}"
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        for scene in scenes:
            scene_id = scene.get("scene_id") or f"{act_id}_sc_{scene_order+1}"
            scene_order += 1
            scene_entry: Dict[str, Any] = {
                "scene_id": scene_id,
                "act_id": act_id,
                "summary": scene.get("summary") or "",
                "order": scene_order,
            }
            scenes_out.append(scene_entry)
            beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
            for beat in beats:
                events = beat.get("events") if isinstance(beat.get("events"), list) else []
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("type") != "state_change":
                        continue
                    target = ev.get("target")
                    state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                    if not isinstance(target, str) or not state_delta:
                        continue
                    state_entry = character_state.get(target) or {}
                    for key, val in state_delta.items():
                        state_entry[key] = val
                    character_state[target] = state_entry
                shot_id = f"{scene_id}_sh_{shot_index+1}"
                shot_index += 1
                description = beat.get("description") or ""
                duration_s = float(beat.get("time_hint_s") or 0.0)
                char_ids = beat.get("characters") or []
                shot_states: Dict[str, Dict[str, Any]] = {}
                for cid in char_ids:
                    if isinstance(cid, str) and cid in character_state:
                        shot_states[cid] = dict(character_state[cid])
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
    return scenes_out, shots_out



