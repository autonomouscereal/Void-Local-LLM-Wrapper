from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional, Callable

from .store import upsert_lock_bundle
from .builder import build_image_bundle


def _read_file_b64(path: Optional[str]) -> Optional[str]:
    if not (isinstance(path, str) and path):
        return None
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


def _get_visual(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort accessor for the visual branch of a lock bundle.

    Returns an empty dict if the branch is missing or malformed.
    """
    vis = bundle.get("visual")
    return vis if isinstance(vis, dict) else {}


def _get_music(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort accessor for the music branch of a lock bundle.

    Returns an empty dict if the branch is missing or malformed.
    """
    mus = bundle.get("music")
    return mus if isinstance(mus, dict) else {}


def _get_tts(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort accessor for the tts branch of a lock bundle.

    Returns an empty dict if the branch is missing or malformed.
    """
    tts = bundle.get("tts")
    return tts if isinstance(tts, dict) else {}


def _get_sfx(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort accessor for the sfx branch of a lock bundle.

    Returns an empty dict if the branch is missing or malformed.
    """
    sfx = bundle.get("sfx")
    return sfx if isinstance(sfx, dict) else {}


def _get_film2(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort accessor for the film2 branch of a lock bundle.

    Returns an empty dict if the branch is missing or malformed.
    """
    film2 = bundle.get("film2")
    return film2 if isinstance(film2, dict) else {}


def visual_get_entities(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of visual entities from a lock bundle (may be empty)."""
    vis = _get_visual(bundle)
    ents = vis.get("entities")
    return ents if isinstance(ents, list) else []


def visual_freeze_entities(bundle: Dict[str, Any], entity_ids: List[str]) -> None:
    """
    Mark the given entities as hard-locked for all visual attributes.

    This is an internal helper intended to be used by image/film pipelines
    prior to calling bundle_to_image_locks. It does not perform any I/O.
    """
    if not entity_ids:
        return
    vis = _get_visual(bundle)
    ents = vis.get("entities")
    if not isinstance(ents, list):
        return
    target_ids = {eid for eid in entity_ids if isinstance(eid, str)}
    for ent in ents:
        if not isinstance(ent, dict):
            continue
        if ent.get("entity_id") not in target_ids:
            continue
        c = ent.setdefault("constraints", {})
        if not isinstance(c, dict):
            c = {}
            ent["constraints"] = c
        c["lock_mode"] = "hard"
        c["lock_color"] = True
        c["lock_texture"] = True
        c["lock_shape"] = True
        c["lock_style"] = True


def visual_refresh_all_except(bundle: Dict[str, Any], keep_ids: List[str]) -> None:
    """
    Relax all entities except those in keep_ids so they can be freely redrawn.

    Entities listed in keep_ids are left untouched; callers can combine this
    with visual_freeze_entities for "keep these, refresh everything else".
    """
    vis = _get_visual(bundle)
    ents = vis.get("entities")
    if not isinstance(ents, list):
        return
    keep = {eid for eid in keep_ids if isinstance(eid, str)}
    for ent in ents:
        if not isinstance(ent, dict):
            continue
        if ent.get("entity_id") in keep:
            continue
        c = ent.setdefault("constraints", {})
        if not isinstance(c, dict):
            c = {}
            ent["constraints"] = c
        # Only relax non-hard locks; callers can explicitly override if needed.
        if c.get("lock_mode") != "hard":
            c["lock_mode"] = "off"
            c["lock_color"] = False
            c["lock_texture"] = False
            c["lock_shape"] = False
            c["lock_style"] = False


def visual_relax_entity_color_texture(bundle: Dict[str, Any], entity_id: str) -> None:
    """
    Relax color/texture constraints for a single entity while keeping shape.

    Useful for edits like "keep shirt shape, change texture/color".
    """
    if not isinstance(entity_id, str) or not entity_id:
        return
    vis = _get_visual(bundle)
    ents = vis.get("entities")
    if not isinstance(ents, list):
        return
    for ent in ents:
        if not isinstance(ent, dict):
            continue
        if ent.get("entity_id") != entity_id:
            continue
        c = ent.setdefault("constraints", {})
        if not isinstance(c, dict):
            c = {}
            ent["constraints"] = c
        c["lock_color"] = False
        c["lock_texture"] = False
        # Preserve or enable shape locking by default
        if c.get("lock_shape") is None:
            c["lock_shape"] = True


def migrate_visual_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort migration helper to populate the visual.* branch from legacy
    top-level fields (face/pose/style/regions/scene).

    This is intentionally conservative and side-effect-free for callers that
    already maintain a visual branch.
    """
    if not isinstance(bundle, dict):
        return bundle
    # Ensure visual container
    visual = bundle.get("visual")
    if not isinstance(visual, dict):
        visual = {}
        bundle["visual"] = visual
    # Style
    if "style" not in visual and isinstance(bundle.get("style"), dict):
        legacy_style = bundle.get("style") or {}
        visual["style"] = {
            "style_tags": list(legacy_style.get("prompt_tags") or []),
            "clip_style_embedding": None,
            "palette": dict(legacy_style.get("palette") or {}),
            "lighting": None,
            "noise_grain_level": None,
            "constraints": {
                "lock_mode": "soft",
                "lock_palette": True,
                "lock_lighting": False,
            },
        }
    # Scene
    if "scene" not in visual and isinstance(bundle.get("scene"), dict):
        legacy_scene = bundle.get("scene") or {}
        visual["scene"] = {
            "scene_tags": list(legacy_scene.get("camera_style_tags") or []),
            "layout_embedding": None,
            "depth_stats": {
                "mean_depth": None,
                "variance": None,
            },
            "background_entity_id": None,
            "constraints": {
                "lock_mode": legacy_scene.get("lock_mode") or "soft",
                "lock_layout": False,
                "lock_depth": False,
            },
        }
    # Faces
    vis_faces = visual.get("faces")
    if (not isinstance(vis_faces, list) or not vis_faces) and isinstance(bundle.get("face"), dict):
        legacy_face = bundle.get("face") or {}
        img_path = legacy_face.get("image_path") if isinstance(legacy_face.get("image_path"), str) else None
        extra_paths = legacy_face.get("extra_image_paths") if isinstance(legacy_face.get("extra_image_paths"), list) else []
        refs: List[Dict[str, Any]] = []
        if img_path:
            refs.append({"image_path": img_path})
        for p in extra_paths:
            if isinstance(p, str) and p:
                refs.append({"image_path": p})
        visual["faces"] = [
            {
                "entity_id": "face_0",
                "role": "primary_face",
                "priority": 10,
                "region": {
                    "mask_path": legacy_face.get("mask") if isinstance(legacy_face.get("mask"), str) else None,
                    "bbox": None,
                    "z_index": 10,
                },
                "embeddings": {
                    "id_embedding": legacy_face.get("embedding"),
                    "clip_embedding": None,
                },
                "constraints": {
                    "lock_mode": "hard",
                    "lock_identity": True,
                    "lock_expression": False,
                    "lock_head_pose": False,
                    "lock_hair": True,
                    "allow_movement_px": 0,
                    "allow_scale_delta": 0.0,
                    "strength": legacy_face.get("strength", 0.75),
                },
                "refs": refs,
            }
        ]
    # Pose
    if "pose" not in visual and isinstance(bundle.get("pose"), dict):
        legacy_pose = bundle.get("pose") or {}
        if legacy_pose.get("skeleton") is not None:
            visual["pose"] = {
                "skeleton": legacy_pose.get("skeleton"),
                "constraints": {
                    "lock_mode": "soft",
                    "max_joint_deviation_deg": 25.0,
                },
            }
    # Regions -> entities
    vis_entities = visual.get("entities")
    if (not isinstance(vis_entities, list) or not vis_entities) and isinstance(bundle.get("regions"), dict):
        entities: List[Dict[str, Any]] = []
        for region_id, region_data in (bundle.get("regions") or {}).items():
            if not isinstance(region_data, dict):
                continue
            mask_path = region_data.get("mask_path")
            if not isinstance(mask_path, str):
                continue
            entities.append(
                {
                    "entity_id": str(region_id),
                    "entity_type": "object",
                    "role": region_data.get("role"),
                    "priority": 5,
                    "region": {
                        "mask_path": mask_path,
                        "bbox": region_data.get("bbox"),
                        "z_index": 0,
                    },
                    "embeddings": {
                        "clip": None,
                        "dino": None,
                        "texture": region_data.get("texture_embedding"),
                        "shape": region_data.get("shape_embedding"),
                    },
                    "constraints": {
                        "lock_mode": (region_data.get("lock_mode") or {}).get("mode") if isinstance(region_data.get("lock_mode"), dict) else region_data.get("lock_mode") or "soft",
                        "lock_color": True,
                        "lock_texture": True,
                        "lock_shape": True,
                        "lock_style": False,
                        "allow_movement_px": 3,
                        "allow_scale_delta": 0.03,
                        "strength": region_data.get("strength", 0.75),
                    },
                    "refs": [],
                }
            )
        if entities:
            visual["entities"] = entities
    # Ensure schema_version is at least 3
    sv_raw = bundle.get("schema_version")
    sv = int(sv_raw) if isinstance(sv_raw, (int, float)) else 0
    if sv < 3:
        bundle["schema_version"] = 3
    return bundle


def merge_lock_bundles(existing: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge an updated lock bundle into an existing one, preserving schema
    invariants and migrating into the latest visual/music/tts/sfx/film2 shape.
    """
    base: Dict[str, Any] = dict(existing or {})
    for key in ("schema_version", "character_id"):
        if update.get(key) is not None:
            base[key] = update[key]
    for section in ("face", "pose", "style", "audio"):
        existing_section = base.get(section) if isinstance(base.get(section), dict) else {}
        update_section = update.get(section) if isinstance(update.get(section), dict) else {}
        merged = dict(existing_section or {})
        for subkey, value in update_section.items():
            merged[subkey] = value
        if merged:
            base[section] = merged
    existing_regions = base.get("regions") if isinstance(base.get("regions"), dict) else {}
    update_regions = update.get("regions") if isinstance(update.get("regions"), dict) else {}
    if update_regions:
        merged_regions = dict(existing_regions or {})
        for region_id, region_data in update_regions.items():
            if isinstance(region_data, dict):
                merged_regions[region_id] = dict(region_data)
        base["regions"] = merged_regions
    existing_scene = base.get("scene") if isinstance(base.get("scene"), dict) else {}
    update_scene = update.get("scene") if isinstance(update.get("scene"), dict) else {}
    if update_scene:
        merged_scene = dict(existing_scene or {})
        merged_scene.update(update_scene)
        base["scene"] = merged_scene
    # Ensure any newer branches (visual/music/tts/sfx/film2) are present and schema_version bumped.
    base = migrate_visual_bundle(base)
    base = migrate_music_bundle(base)
    base = migrate_tts_bundle(base)
    base = migrate_sfx_bundle(base)
    base = migrate_film2_bundle(base)
    return base


def music_get_voices(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of music voices from a lock bundle (may be empty)."""
    mus = _get_music(bundle)
    voices = mus.get("voices")
    return voices if isinstance(voices, list) else []


def music_get_instruments(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of music instruments from a lock bundle (may be empty)."""
    mus = _get_music(bundle)
    inst = mus.get("instruments")
    return inst if isinstance(inst, list) else []


def music_get_sections(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of music sections from a lock bundle (may be empty)."""
    mus = _get_music(bundle)
    secs = mus.get("sections")
    return secs if isinstance(secs, list) else []


def music_get_motifs(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of music motifs from a lock bundle (may be empty)."""
    mus = _get_music(bundle)
    motifs = mus.get("motifs")
    return motifs if isinstance(motifs, list) else []


def music_get_events(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of music events from a lock bundle (may be empty)."""
    mus = _get_music(bundle)
    events = mus.get("events")
    return events if isinstance(events, list) else []


def tts_get_global(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Return the tts.global block (may be empty)."""
    tts = _get_tts(bundle)
    glob = tts.get("global")
    return glob if isinstance(glob, dict) else {}


def tts_get_voices(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of TTS voices from a lock bundle (may be empty)."""
    tts = _get_tts(bundle)
    voices = tts.get("voices")
    return voices if isinstance(voices, list) else []


def tts_get_styles(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of TTS styles from a lock bundle (may be empty)."""
    tts = _get_tts(bundle)
    styles = tts.get("styles")
    return styles if isinstance(styles, list) else []


def tts_get_segments(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of TTS segments from a lock bundle (may be empty)."""
    tts = _get_tts(bundle)
    segments = tts.get("segments")
    return segments if isinstance(segments, list) else []


def tts_get_events(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of TTS events from a lock bundle (may be empty)."""
    tts = _get_tts(bundle)
    events = tts.get("events")
    return events if isinstance(events, list) else []


def sfx_get_global(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Return the sfx.global block (may be empty)."""
    sfx = _get_sfx(bundle)
    glob = sfx.get("global")
    return glob if isinstance(glob, dict) else {}


def sfx_get_assets(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of SFX assets from a lock bundle (may be empty)."""
    sfx = _get_sfx(bundle)
    assets = sfx.get("assets")
    return assets if isinstance(assets, list) else []


def sfx_get_layers(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of SFX layers from a lock bundle (may be empty)."""
    sfx = _get_sfx(bundle)
    layers = sfx.get("layers")
    return layers if isinstance(layers, list) else []


def sfx_get_events(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of SFX events from a lock bundle (may be empty)."""
    sfx = _get_sfx(bundle)
    events = sfx.get("events")
    return events if isinstance(events, list) else []


def sfx_get_ambiences(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of SFX ambiences from a lock bundle (may be empty)."""
    sfx = _get_sfx(bundle)
    amb = sfx.get("ambiences")
    return amb if isinstance(amb, list) else []


def film2_get_project(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Return the film2.project block (may be empty)."""
    film2 = _get_film2(bundle)
    proj = film2.get("project")
    return proj if isinstance(proj, dict) else {}


async def ensure_visual_lock_bundle(character_id: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure that a character-scoped lock bundle exists with a minimal visual
    skeleton suitable for image/film generation flows.

    If an existing bundle is provided it is returned unchanged (after an
    upsert); otherwise a new skeleton bundle is created and persisted.
    """
    if not isinstance(character_id, str) or not character_id.strip():
        return bundle or {}
    if not isinstance(bundle, dict):
        bundle = {
            "schema_version": 2,
            "character_id": character_id.strip(),
            "face": {},
            "pose": {},
            "style": {},
            "audio": {},
            "regions": {},
            "scene": {
                "background_embedding": None,
                "camera_style_tags": [],
                "lighting_tags": [],
                "lock_mode": "soft",
            },
        }
    await upsert_lock_bundle(character_id.strip(), bundle)
    return bundle


async def ensure_tts_lock_bundle(character_id: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure that a character-scoped lock bundle exists with a minimal TTS/audio
    skeleton suitable for speech generation flows.

    If an existing bundle is provided it is returned unchanged (after an
    upsert); otherwise a new skeleton bundle is created and persisted.
    """
    if not isinstance(character_id, str) or not character_id.strip():
        return bundle or {}
    if not isinstance(bundle, dict):
        bundle = {
            "schema_version": 2,
            "character_id": character_id.strip(),
            "tts": {},
            "audio": {},
        }
    await upsert_lock_bundle(character_id.strip(), bundle)
    return bundle


def summarize_visual_bundle_for_context(
    ctx_add: Callable[[str, str, str, Optional[str], Optional[int], List[str], Dict[str, Any]], None],
    character_id: str,
    bundle: Dict[str, Any],
) -> None:
    """
    Emit lightweight text/symbol summaries of visual locks into OmniContext.

    This creates per-entity artifacts keyed by entity_id so natural-language
    references like "that flower on the left" can be resolved to entities.
    """
    if not isinstance(character_id, str) or not character_id:
        return
    vis = bundle.get("visual")
    if not isinstance(vis, dict):
        return
    entities = vis.get("entities")
    if not isinstance(entities, list):
        return
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        ent_id = ent.get("entity_id")
        if not isinstance(ent_id, str) or not ent_id:
            continue
        ent_type = ent.get("entity_type") or "object"
        role = ent.get("role") or ""
        region = ent.get("region") if isinstance(ent.get("region"), dict) else {}
        bbox = region.get("bbox")
        tags: List[str] = ["entity", f"entity_type:{ent_type}"]
        if isinstance(role, str) and role:
            tags.append(f"role:{role}")
        text_desc = f"{ent_type} {role} (entity_id={ent_id}) for character {character_id}"
        meta = {
            "text": text_desc,
            "entity_id": ent_id,
            "entity_type": ent_type,
            "role": role,
            "bbox": bbox,
        }
        # Use entity_id as a synthetic path key; context index does not require an actual file.
        ctx_add(character_id, "image.entity", ent_id, None, None, tags, meta)


def summarize_music_bundle_for_context(
    ctx_add: Callable[[str, str, str, Optional[str], Optional[int], List[str], Dict[str, Any]], None],
    character_id: str,
    bundle: Dict[str, Any],
) -> None:
    """
    Emit lightweight summaries of music locks (voices/sections/motifs) into OmniContext.
    """
    if not isinstance(character_id, str) or not character_id:
        return
    mus = bundle.get("music")
    if not isinstance(mus, dict):
        return
    # Voices
    voices = mus.get("voices")
    if isinstance(voices, list):
        for v in voices:
            if not isinstance(v, dict):
                continue
            vid = v.get("voice_id")
            if not isinstance(vid, str) or not vid:
                continue
            role = v.get("role") or ""
            tags: List[str] = ["music.voice"]
            if isinstance(role, str) and role:
                tags.append(f"role:{role}")
            text_desc = f"music voice {vid} ({role}) for character {character_id}"
            meta = {
                "text": text_desc,
                "voice_id": vid,
                "role": role,
                "style_tags": list(v.get("style_tags") or []),
            }
            ctx_add(character_id, "audio.voice", vid, None, None, tags, meta)
    # Sections
    sections = mus.get("sections")
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sid = sec.get("section_id")
            if not isinstance(sid, str) or not sid:
                continue
            stype = sec.get("type") or ""
            tags = ["music.section"]
            if isinstance(stype, str) and stype:
                tags.append(f"type:{stype}")
            text_desc = f"music section {sid} ({stype}) for character {character_id}"
            meta = {
                "text": text_desc,
                "section_id": sid,
                "type": stype,
                "order_index": sec.get("order_index"),
                "bar_start": sec.get("bar_start"),
                "bar_end": sec.get("bar_end"),
            }
            ctx_add(character_id, "audio.section", sid, None, None, tags, meta)
    # Motifs
    motifs = mus.get("motifs")
    if isinstance(motifs, list):
        for m in motifs:
            if not isinstance(m, dict):
                continue
            mid = m.get("motif_id")
            if not isinstance(mid, str) or not mid:
                continue
            role = m.get("role") or ""
            tags = ["music.motif"]
            if isinstance(role, str) and role:
                tags.append(f"role:{role}")
            text_desc = f"music motif {mid} ({role}) for character {character_id}"
            meta = {
                "text": text_desc,
                "motif_id": mid,
                "role": role,
                "source_section_id": m.get("source_section_id"),
            }
            ctx_add(character_id, "audio.motif", mid, None, None, tags, meta)


def summarize_tts_bundle_for_context(
    ctx_add: Callable[[str, str, str, Optional[str], Optional[int], List[str], Dict[str, Any]], None],
    character_id: str,
    bundle: Dict[str, Any],
) -> None:
    """
    Summarize TTS voices/styles/segments into OmniContext.
    """
    if not isinstance(character_id, str) or not character_id:
        return
    tts = bundle.get("tts")
    if not isinstance(tts, dict):
        return
    # Voices
    voices = tts.get("voices")
    if isinstance(voices, list):
        for v in voices:
            if not isinstance(v, dict):
                continue
            vid = v.get("voice_id")
            if not isinstance(vid, str) or not vid:
                continue
            role = v.get("role") or ""
            tags = ["tts.voice"]
            if isinstance(role, str) and role:
                tags.append(f"role:{role}")
            text_desc = f"tts voice {vid} ({role}) for character {character_id}"
            meta = {
                "text": text_desc,
                "voice_id": vid,
                "role": role,
                "style_tags": list(v.get("style_tags") or []),
            }
            ctx_add(character_id, "tts.voice", vid, None, None, tags, meta)
    # Styles
    styles = tts.get("styles")
    if isinstance(styles, list):
        for s in styles:
            if not isinstance(s, dict):
                continue
            sid = s.get("style_id")
            if not isinstance(sid, str) or not sid:
                continue
            name = s.get("name") or ""
            tags = ["tts.style"]
            text_desc = f"tts style {sid} ({name})"
            meta = {
                "text": text_desc,
                "style_id": sid,
                "name": name,
                "tags": list(s.get("tags") or []),
            }
            ctx_add(character_id, "tts.style", sid, None, None, tags, meta)
    # Segments
    segments = tts.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_id = seg.get("segment_id")
            if not isinstance(seg_id, str) or not seg_id:
                continue
            text_ref = seg.get("text_ref") if isinstance(seg.get("text_ref"), dict) else {}
            txt = text_ref.get("text") or ""
            tags = ["tts.segment"]
            text_desc = f"tts segment {seg_id} text={txt!r}"
            meta = {
                "text": text_desc,
                "segment_id": seg_id,
                "scene_id": seg.get("scene_id"),
                "line_index": seg.get("line_index"),
            }
            ctx_add(character_id, "tts.segment", seg_id, None, None, tags, meta)


def summarize_sfx_bundle_for_context(
    ctx_add: Callable[[str, str, str, Optional[str], Optional[int], List[str], Dict[str, Any]], None],
    character_id: str,
    bundle: Dict[str, Any],
) -> None:
    """
    Summarize SFX assets/layers/events/ambiences into OmniContext.
    """
    if not isinstance(character_id, str) or not character_id:
        return
    sfx = bundle.get("sfx")
    if not isinstance(sfx, dict):
        return
    assets = sfx.get("assets")
    if isinstance(assets, list):
        for a in assets:
            if not isinstance(a, dict):
                continue
            aid = a.get("asset_id")
            if not isinstance(aid, str) or not aid:
                continue
            tags = ["sfx.asset"]
            tags.extend(list(a.get("tags") or []))
            text_desc = f"sfx asset {aid} for character {character_id}"
            meta = {
                "text": text_desc,
                "asset_id": aid,
                "ucs": a.get("ucs"),
                "tags": a.get("tags"),
            }
            ctx_add(character_id, "sfx.asset", aid, None, None, tags, meta)
    layers = sfx.get("layers")
    if isinstance(layers, list):
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            lid = layer.get("layer_id")
            if not isinstance(lid, str) or not lid:
                continue
            role = layer.get("role") or ""
            tags = ["sfx.layer"]
            if isinstance(role, str) and role:
                tags.append(f"role:{role}")
            text_desc = f"sfx layer {lid} ({role}) for character {character_id}"
            meta = {
                "text": text_desc,
                "layer_id": lid,
                "role": role,
                "components": layer.get("components"),
            }
            ctx_add(character_id, "sfx.layer", lid, None, None, tags, meta)
    events = sfx.get("events")
    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, dict):
                continue
            eid = ev.get("event_id")
            if not isinstance(eid, str) or not eid:
                continue
            role = ev.get("role") or ""
            tags = ["sfx.event"]
            if isinstance(role, str) and role:
                tags.append(f"role:{role}")
            text_desc = f"sfx event {eid} ({role})"
            meta = {
                "text": text_desc,
                "event_id": eid,
                "scene_id": ev.get("scene_id"),
                "shot_id": ev.get("shot_id"),
            }
            ctx_add(character_id, "sfx.event", eid, None, None, tags, meta)
    ambs = sfx.get("ambiences")
    if isinstance(ambs, list):
        for amb in ambs:
            if not isinstance(amb, dict):
                continue
            amb_id = amb.get("ambience_id")
            if not isinstance(amb_id, str) or not amb_id:
                continue
            tags = ["sfx.ambience"]
            tags.extend(list(amb.get("tags") or []))
            text_desc = f"sfx ambience {amb_id} for character {character_id}"
            meta = {
                "text": text_desc,
                "ambience_id": amb_id,
                "ucs": amb.get("ucs"),
                "tags": amb.get("tags"),
            }
            ctx_add(character_id, "sfx.ambience", amb_id, None, None, tags, meta)


def summarize_film2_bundle_for_context(
    ctx_add: Callable[[str, str, str, Optional[str], Optional[int], List[str], Dict[str, Any]], None],
    character_id: str,
    bundle: Dict[str, Any],
) -> None:
    """
    Summarize film scenes/shots/segments into OmniContext.
    """
    if not isinstance(character_id, str) or not character_id:
        return
    film2 = bundle.get("film2")
    if not isinstance(film2, dict):
        return
    scenes = film2.get("scenes")
    if isinstance(scenes, list):
        for sc in scenes:
            if not isinstance(sc, dict):
                continue
            sid = sc.get("scene_id")
            if not isinstance(sid, str) or not sid:
                continue
            slug = ((sc.get("script_ref") or {}) or {}).get("slugline")
            tags = ["film2.scene"]
            text_desc = f"film2 scene {sid} slugline={slug!r}"
            meta = {
                "text": text_desc,
                "scene_id": sid,
                "sequence_id": sc.get("sequence_id"),
                "tags": sc.get("tags"),
            }
            ctx_add(character_id, "film2.scene", sid, None, None, tags, meta)
    shots = film2.get("shots")
    if isinstance(shots, list):
        for sh in shots:
            if not isinstance(sh, dict):
                continue
            shot_id = sh.get("shot_id")
            if not isinstance(shot_id, str) or not shot_id:
                continue
            scene_id = sh.get("scene_id")
            tags = ["film2.shot"]
            text_desc = f"film2 shot {shot_id} in scene {scene_id}"
            meta = {
                "text": text_desc,
                "shot_id": shot_id,
                "scene_id": scene_id,
                "dominant_characters": sh.get("dominant_characters"),
            }
            ctx_add(character_id, "film2.shot", shot_id, None, None, tags, meta)
    segments = film2.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_id = seg.get("segment_id")
            if not isinstance(seg_id, str) or not seg_id:
                continue
            shot_id = seg.get("shot_id")
            tags = ["film2.segment"]
            text_desc = f"film2 segment {seg_id} (shot {shot_id})"
            meta = {
                "text": text_desc,
                "segment_id": seg_id,
                "shot_id": shot_id,
                "scene_id": seg.get("scene_id"),
            }
            ctx_add(character_id, "film2.segment", seg_id, None, None, tags, meta)


def summarize_all_locks_for_context(
    ctx_add: Callable[[str, str, str, Optional[str], Optional[int], List[str], Dict[str, Any]], None],
    character_id: str,
    bundle: Dict[str, Any],
) -> None:
    """
    Convenience helper: summarize all lock branches into context for a character.
    """
    summarize_visual_bundle_for_context(ctx_add, character_id, bundle)
    summarize_music_bundle_for_context(ctx_add, character_id, bundle)
    summarize_tts_bundle_for_context(ctx_add, character_id, bundle)
    summarize_sfx_bundle_for_context(ctx_add, character_id, bundle)
    summarize_film2_bundle_for_context(ctx_add, character_id, bundle)


def film2_get_sequences(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of film2 sequences (may be empty)."""
    film2 = _get_film2(bundle)
    seqs = film2.get("sequences")
    return seqs if isinstance(seqs, list) else []


def film2_get_scenes(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of film2 scenes (may be empty)."""
    film2 = _get_film2(bundle)
    scenes = film2.get("scenes")
    return scenes if isinstance(scenes, list) else []


def film2_get_shots(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of film2 shots (may be empty)."""
    film2 = _get_film2(bundle)
    shots = film2.get("shots")
    return shots if isinstance(shots, list) else []


def film2_get_segments(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of film2 segments (may be empty)."""
    film2 = _get_film2(bundle)
    segments = film2.get("segments")
    return segments if isinstance(segments, list) else []


def film2_get_timeline(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Return the film2.timeline block (may be empty)."""
    film2 = _get_film2(bundle)
    tl = film2.get("timeline")
    return tl if isinstance(tl, dict) else {}


def migrate_music_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort migration helper to populate the music.* branch from any legacy
    music-related fields (e.g. audio stem locks). Currently this is a no-op
    except for ensuring the music container exists and bumping schema_version.
    """
    if not isinstance(bundle, dict):
        return bundle
    mus = bundle.get("music")
    if not isinstance(mus, dict):
        bundle["music"] = {"global": {}, "voices": [], "instruments": [], "motifs": [], "sections": [], "events": []}
    # Ensure schema_version is at least 3
    sv_raw = bundle.get("schema_version")
    sv = int(sv_raw) if isinstance(sv_raw, (int, float)) else 0
    if sv < 3:
        bundle["schema_version"] = 3
    return bundle


def migrate_tts_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort migration helper to populate the tts.* branch from any legacy
    audio fields. Ensures a basic tts structure and bumps schema_version.
    """
    if not isinstance(bundle, dict):
        return bundle
    tts = bundle.get("tts")
    if not isinstance(tts, dict):
        tts = {"global": {}, "voices": [], "styles": [], "segments": [], "events": []}
        bundle["tts"] = tts
    audio = bundle.get("audio") if isinstance(bundle.get("audio"), dict) else {}
    voices = tts.get("voices")
    if (not isinstance(voices, list) or not voices) and audio:
        ve = audio.get("voice_embedding")
        if isinstance(ve, (list, dict)):
            voice_id = "voice_1"
            voice_entry: Dict[str, Any] = {
                "voice_id": voice_id,
                "role": "narrator",
                "character_id": bundle.get("character_id"),
                "style_tags": list(audio.get("timbre_tags") or []),
                "gender": "unspecified",
                "age_hint": "adult",
                "embeddings": {
                    "speaker": ve,
                    "timbre": ve,
                    "gst_style": None,
                },
                "range_midi": {"min": 55, "max": 81},
                "language": "en",
                "vocal_type": "spoken",
                "baseline_prosody": {},
                "emotion_profile": {},
                "pitch_shift_allowed_semitones": 2,
                "rate_scaling_allowed": [0.8, 1.2],
                "formant_preservation": True,
                "constraints": {
                    "lock_mode": "guide",
                    "lock_timbre": True,
                    "lock_language": False,
                    "lock_baseline_prosody": False,
                    "lock_emotion_profile": False,
                },
                "refs": [],
                "linked_music_voice_id": None,
            }
            tts["voices"] = [voice_entry]
            glob = tts.get("global")
            if not isinstance(glob, dict):
                glob = {}
                tts["global"] = glob
            glob.setdefault("default_voice_id", voice_id)
    # Ensure schema_version is at least 3
    sv_raw = bundle.get("schema_version")
    sv = int(sv_raw) if isinstance(sv_raw, (int, float)) else 0
    if sv < 3:
        bundle["schema_version"] = 3
    return bundle


def migrate_sfx_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort migration helper to ensure the sfx.* branch exists.

    Currently this does not infer SFX from legacy fields; it only populates
    a default sfx structure and bumps schema_version.
    """
    if not isinstance(bundle, dict):
        return bundle
    sfx = bundle.get("sfx")
    if not isinstance(sfx, dict):
        sfx = {
            "global": {
                "default_bus": "SFX",
                "default_lufs_target": -18.0,
                "spatial_mode": "stereo",
                "renderer_hint": "bed",
                "constraints": {
                    "lock_mode": "guide",
                    "lock_loudness_norm": True,
                    "lock_bus_routing": True,
                },
            },
            "assets": [],
            "layers": [],
            "events": [],
            "ambiences": [],
        }
        bundle["sfx"] = sfx
    # Ensure schema_version is at least 3
    sv_raw = bundle.get("schema_version")
    sv = int(sv_raw) if isinstance(sv_raw, (int, float)) else 0
    if sv < 3:
        bundle["schema_version"] = 3
    return bundle


def migrate_film2_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort migration helper to ensure the film2.* branch exists.

    This establishes an empty film2 structure suitable for film2 tools; it does
    not attempt to infer structure from external EDLs at this stage.
    """
    if not isinstance(bundle, dict):
        return bundle
    # If legacy film branch exists, prefer it as the source and then normalize to film2
    legacy_film = bundle.get("film")
    if isinstance(legacy_film, dict) and "film2" not in bundle:
        bundle["film2"] = legacy_film
        # Do not keep the legacy film key around
        try:
            del bundle["film"]
        except Exception:
            pass
    film2 = bundle.get("film2")
    if not isinstance(film2, dict):
        film2 = {
            "project": {},
            "sequences": [],
            "scenes": [],
            "shots": [],
            "segments": [],
            "timeline": {},
        }
        bundle["film2"] = film2
    # Ensure schema_version is at least 3
    sv_raw = bundle.get("schema_version")
    sv = int(sv_raw) if isinstance(sv_raw, (int, float)) else 0
    if sv < 3:
        bundle["schema_version"] = 3
    return bundle


def bundle_to_image_locks(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a lock bundle into the compact locks mapping expected by image
    graph builders.

    Supports both legacy top-level fields (face/pose/style/regions/scene)
    and the richer visual.* layout introduced in schema_version>=3.
    """
    locks: Dict[str, Any] = {}

    # Preferred: visual branch (schema_version >= 3)
    vis = _get_visual(bundle)
    if vis:
        # Faces
        faces = vis.get("faces")
        if isinstance(faces, list) and faces:
            face_locks: List[Dict[str, Any]] = []
            for face_ent in faces:
                if not isinstance(face_ent, dict):
                    continue
                region = face_ent.get("region") if isinstance(face_ent.get("region"), dict) else {}
                refs = face_ent.get("refs") if isinstance(face_ent.get("refs"), list) else []
                embeddings = face_ent.get("embeddings") if isinstance(face_ent.get("embeddings"), dict) else {}
                img_path = None
                for ref in refs or []:
                    if isinstance(ref, dict) and isinstance(ref.get("image_path"), str):
                        img_path = ref.get("image_path")
                        break
                constraints = face_ent.get("constraints") if isinstance(face_ent.get("constraints"), dict) else {}
                strength_val = constraints.get("strength")
                if isinstance(strength_val, (int, float)):
                    weight = float(strength_val)
                else:
                    weight = 0.75
                face_locks.append(
                    {
                        "ref_b64": _read_file_b64(img_path),
                        "weight": weight,
                        # Prefer identity embedding when available
                        "embedding": embeddings.get("id_embedding"),
                        "entity_id": face_ent.get("entity_id"),
                        "region": region,
                    }
                )
            if face_locks:
                locks["faces"] = face_locks

        # Pose from visual.pose
        pose_block = vis.get("pose") if isinstance(vis.get("pose"), dict) else {}
        if pose_block.get("skeleton"):
            constraints = pose_block.get("constraints") if isinstance(pose_block.get("constraints"), dict) else {}
            strength_val = constraints.get("strength")
            strength = float(strength_val) if isinstance(strength_val, (int, float)) else 0.7
            locks["poses"] = [
                {
                    "ref_b64": None,
                    "weight": strength,
                    "skeleton": pose_block.get("skeleton"),
                }
            ]

        # Style & palette
        style_block = vis.get("style") if isinstance(vis.get("style"), dict) else {}
        palette = style_block.get("palette") if isinstance(style_block.get("palette"), dict) else {}
        style_tags = style_block.get("style_tags") if isinstance(style_block.get("style_tags"), list) else []
        if style_tags:
            locks["style_tags"] = style_tags
        if palette:
            locks.setdefault("style_palette", palette)

        # Entities -> region locks
        entities = vis.get("entities")
        if isinstance(entities, list) and entities:
            region_locks: Dict[str, Any] = {}
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                ent_id = ent.get("entity_id")
                region = ent.get("region") if isinstance(ent.get("region"), dict) else {}
                mask_path = region.get("mask_path")
                if not isinstance(mask_path, str):
                    continue
                strength_val = (ent.get("constraints") or {}).get("strength")
                strength = float(strength_val) if isinstance(strength_val, (int, float)) else 0.75
                region_locks[str(ent_id or mask_path)] = {
                    "mask_b64": _read_file_b64(mask_path),
                    "role": ent.get("role"),
                    "bbox": region.get("bbox"),
                    "shape_embedding": (ent.get("embeddings") or {}).get("shape"),
                    "texture_embedding": (ent.get("embeddings") or {}).get("texture"),
                    "color_palette": (ent.get("embeddings") or {}).get("palette"),
                    "lock_mode": (ent.get("constraints") or {}).get("lock_mode"),
                    "strength": strength,
                }
            if region_locks:
                locks["regions"] = region_locks

        # Scene
        scene_block = vis.get("scene") if isinstance(vis.get("scene"), dict) else {}
        if scene_block:
            locks["scene"] = scene_block

    # Legacy top-level fields remain supported and are merged in as defaults
    face = bundle.get("face") if isinstance(bundle.get("face"), dict) else {}
    if face and "faces" not in locks:
        img_path = face.get("image_path") if isinstance(face.get("image_path"), str) else None
        locks["faces"] = [
            {
                "ref_b64": _read_file_b64(img_path),
                "weight": float(face.get("strength", 0.75)),
                "embedding": face.get("embedding"),
            }
        ]
    pose = bundle.get("pose") if isinstance(bundle.get("pose"), dict) else {}
    if pose.get("skeleton") and "poses" not in locks:
        locks["poses"] = [
            {
                "ref_b64": None,
                "weight": float(pose.get("strength", 0.7)),
                "skeleton": pose.get("skeleton"),
            }
        ]
    style = bundle.get("style") if isinstance(bundle.get("style"), dict) else {}
    palette = style.get("palette") if isinstance(style.get("palette"), dict) else {}
    prompt_tags = style.get("prompt_tags") if isinstance(style.get("prompt_tags"), list) else []
    if prompt_tags and "style_tags" not in locks:
        locks["style_tags"] = prompt_tags
    if palette and "style_palette" not in locks:
        locks.setdefault("style_palette", palette)
    regions = bundle.get("regions") if isinstance(bundle.get("regions"), dict) else {}
    if regions and "regions" not in locks:
        region_locks: Dict[str, Any] = {}
        for region_id, region_data in regions.items():
            if not isinstance(region_data, dict):
                continue
            mask_path = region_data.get("mask_path")
            if not isinstance(mask_path, str):
                continue
            region_locks[region_id] = {
                "mask_b64": _read_file_b64(mask_path),
                "role": region_data.get("role"),
                "bbox": region_data.get("bbox"),
                "shape_embedding": region_data.get("shape_embedding"),
                "texture_embedding": region_data.get("texture_embedding"),
                "color_palette": region_data.get("color_palette"),
                "lock_mode": region_data.get("lock_mode"),
                "strength": region_data.get("strength"),
            }
        if region_locks:
            locks["regions"] = region_locks
    scene = bundle.get("scene") if isinstance(bundle.get("scene"), dict) else {}
    if scene and "scene" not in locks:
        locks["scene"] = scene
    return locks


QUALITY_PRESETS = {
    "draft": {
        "steps": 16,
        "cfg": 4.5,
        "lock_strength": 0.6,
        "max_refine_passes": 0,
        "face_min": 0.6,
        "region_shape_min": 0.6,
        "region_texture_min": 0.6,
        "scene_min": 0.6,
        "voice_min": 0.7,
        "tempo_min": 0.7,
        "key_min": 0.7,
        "stem_balance_min": 0.7,
        "lyrics_min": 0.7,
    },
    "standard": {
        "steps": 26,
        "cfg": 5.5,
        "lock_strength": 0.75,
        "max_refine_passes": 1,
        "face_min": 0.82,
        "region_shape_min": 0.8,
        "region_texture_min": 0.8,
        "scene_min": 0.8,
        "voice_min": 0.85,
        "tempo_min": 0.85,
        "key_min": 0.85,
        "stem_balance_min": 0.8,
        "lyrics_min": 0.85,
    },
    "hero": {
        "steps": 36,
        "cfg": 6.2,
        "lock_strength": 0.9,
        "max_refine_passes": 3,
        "face_min": 0.9,
        "region_shape_min": 0.9,
        "region_texture_min": 0.9,
        "scene_min": 0.9,
        "voice_min": 0.92,
        "tempo_min": 0.9,
        "key_min": 0.9,
        "stem_balance_min": 0.9,
        "lyrics_min": 0.9,
    },
}


def apply_quality_profile(
    profile: Optional[str],
    lock_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    name = (profile or "standard").lower()
    preset = QUALITY_PRESETS.get(name, QUALITY_PRESETS["standard"])
    adjusted = dict(lock_bundle or {})
    face = dict(adjusted.get("face") or {})
    if face:
        face_strength = float(face.get("strength", preset["lock_strength"]))
        face["strength"] = max(face_strength, preset["lock_strength"])
        adjusted["face"] = face
    pose = dict(adjusted.get("pose") or {})
    if pose:
        pose_strength = float(pose.get("strength", preset["lock_strength"]))
        pose["strength"] = max(pose_strength, preset["lock_strength"] * 0.9)
        adjusted["pose"] = pose
    style = dict(adjusted.get("style") or {})
    if style and isinstance(style.get("palette"), dict):
        style["profile"] = name
        adjusted["style"] = style
    adjusted.setdefault("quality_profile", name)
    adjusted.setdefault("max_refine_passes", int(preset["max_refine_passes"]))
    region_map = adjusted.get("regions") if isinstance(adjusted.get("regions"), dict) else {}
    new_regions: Dict[str, Any] = {}
    for region_id, region_data in (region_map or {}).items():
        if not isinstance(region_data, dict):
            continue
        updated = dict(region_data)
        if updated.get("strength") is not None:
            try:
                updated["strength"] = max(float(updated["strength"]), preset["lock_strength"])
            except Exception:
                pass
        new_regions[region_id] = updated
    if new_regions:
        adjusted["regions"] = new_regions
    return adjusted


def quality_thresholds(profile: Optional[str]) -> Dict[str, float]:
    name = (profile or "standard").lower()
    preset = QUALITY_PRESETS.get(name, QUALITY_PRESETS["standard"])
    return {
        "face_min": float(preset.get("face_min", 0.8)),
        "region_shape_min": float(preset.get("region_shape_min", 0.8)),
        "region_texture_min": float(preset.get("region_texture_min", 0.8)),
        "scene_min": float(preset.get("scene_min", 0.8)),
        "voice_min": float(preset.get("voice_min", 0.85)),
        "tempo_min": float(preset.get("tempo_min", 0.85)),
        "key_min": float(preset.get("key_min", 0.85)),
        "stem_balance_min": float(preset.get("stem_balance_min", 0.8)),
        "lyrics_min": float(preset.get("lyrics_min", 0.85)),
    }


async def update_bundle_from_hero_frame(
    character_id: str,
    hero_image_path: str,
    existing_bundle: Dict[str, Any],
    *,
    locks_root_dir: str,
    detect_pose: bool = True,
    extract_style: bool = True,
) -> Dict[str, Any]:
    hero_options = {
        "detect_pose": detect_pose,
        "extract_style": extract_style,
    }
    hero_bundle = await build_image_bundle(
        character_id=character_id,
        image_url=None,
        options=hero_options,
        locks_root_dir=locks_root_dir,
        image_path=hero_image_path,
    )
    merged = dict(existing_bundle or {})
    hero_face = hero_bundle.get("face") if isinstance(hero_bundle.get("face"), dict) else {}
    hero_pose = hero_bundle.get("pose") if isinstance(hero_bundle.get("pose"), dict) else {}
    hero_style = hero_bundle.get("style") if isinstance(hero_bundle.get("style"), dict) else {}
    hero_scene = hero_bundle.get("scene") if isinstance(hero_bundle.get("scene"), dict) else {}
    if hero_face:
        merged["face"] = _merge_face_lock(existing_bundle.get("face") if isinstance(existing_bundle.get("face"), dict) else {}, hero_face)
    if hero_pose:
        merged["pose"] = _merge_pose_lock(existing_bundle.get("pose") if isinstance(existing_bundle.get("pose"), dict) else {}, hero_pose)
    if hero_style:
        merged["style"] = _merge_style_lock(existing_bundle.get("style") if isinstance(existing_bundle.get("style"), dict) else {}, hero_style)
    if hero_scene:
        merged["scene"] = _merge_scene_lock(existing_bundle.get("scene") if isinstance(existing_bundle.get("scene"), dict) else {}, hero_scene)
    existing_regions = existing_bundle.get("regions") if isinstance(existing_bundle.get("regions"), dict) else {}
    hero_regions = hero_bundle.get("regions") if isinstance(hero_bundle.get("regions"), dict) else {}
    merged_regions: Dict[str, Any] = {}
    for region_id, region_data in existing_regions.items():
        hero_region = hero_regions.get(region_id) if isinstance(hero_regions.get(region_id), dict) else {}
        merged_regions[region_id] = _merge_region_lock(region_data if isinstance(region_data, dict) else {}, hero_region)
    for region_id, region_data in hero_regions.items():
        if region_id not in merged_regions and isinstance(region_data, dict):
            merged_regions[region_id] = region_data
    if merged_regions:
        merged["regions"] = merged_regions
    merged["schema_version"] = 2
    await upsert_lock_bundle(character_id, merged)
    return merged


def _merge_face_lock(existing: Dict[str, Any], hero: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    if hero.get("embedding") is not None:
        merged["embedding"] = hero.get("embedding")
    if hero.get("mask") is not None:
        merged["mask"] = hero.get("mask")
    if hero.get("image_path") is not None:
        merged["image_path"] = hero.get("image_path")
    merged["strength"] = merged.get("strength", hero.get("strength", 0.75))
    return merged


def _merge_pose_lock(existing: Dict[str, Any], hero: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    if hero.get("skeleton") is not None:
        merged["skeleton"] = hero.get("skeleton")
    merged["strength"] = merged.get("strength", hero.get("strength", 0.7))
    return merged


def _merge_style_lock(existing: Dict[str, Any], hero: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    if hero.get("prompt_tags") is not None:
        merged["prompt_tags"] = hero.get("prompt_tags")
    if hero.get("palette") is not None:
        merged["palette"] = hero.get("palette")
    return merged


def _merge_scene_lock(existing: Dict[str, Any], hero: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    if hero.get("background_embedding") is not None:
        merged["background_embedding"] = hero.get("background_embedding")
    if hero.get("camera_style_tags") is not None:
        merged["camera_style_tags"] = hero.get("camera_style_tags")
    if hero.get("lighting_tags") is not None:
        merged["lighting_tags"] = hero.get("lighting_tags")
    merged["lock_mode"] = merged.get("lock_mode") or hero.get("lock_mode") or "soft"
    return merged


def _merge_region_lock(existing: Dict[str, Any], hero: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    merged["role"] = existing.get("role") or hero.get("role")
    merged["lock_mode"] = existing.get("lock_mode") or hero.get("lock_mode") or {
        "shape": "hard",
        "texture": "soft",
        "color": "soft",
    }
    if hero.get("shape_embedding") is not None:
        merged["shape_embedding"] = hero.get("shape_embedding")
    if hero.get("texture_embedding") is not None:
        merged["texture_embedding"] = hero.get("texture_embedding")
    if hero.get("color_palette") is not None:
        merged["color_palette"] = hero.get("color_palette")
    merged["strength"] = merged.get("strength", hero.get("strength", 0.0))
    merged["mask_path"] = merged.get("mask_path") or hero.get("mask_path")
    merged["bbox"] = merged.get("bbox") or hero.get("bbox")
    return merged
