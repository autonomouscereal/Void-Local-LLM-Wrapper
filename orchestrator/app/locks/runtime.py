from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

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


def bundle_to_image_locks(bundle: Dict[str, Any]) -> Dict[str, Any]:
    locks: Dict[str, Any] = {}
    face = bundle.get("face") if isinstance(bundle.get("face"), dict) else {}
    if face:
        img_path = face.get("image_path") if isinstance(face.get("image_path"), str) else None
        locks["faces"] = [{
            "ref_b64": _read_file_b64(img_path),
            "weight": float(face.get("strength", 0.75)),
            "embedding": face.get("embedding"),
        }]
    pose = bundle.get("pose") if isinstance(bundle.get("pose"), dict) else {}
    if pose.get("skeleton"):
        locks["poses"] = [{
            "ref_b64": None,
            "weight": float(pose.get("strength", 0.7)),
            "skeleton": pose.get("skeleton"),
        }]
    style = bundle.get("style") if isinstance(bundle.get("style"), dict) else {}
    palette = style.get("palette") if isinstance(style.get("palette"), dict) else {}
    prompt_tags = style.get("prompt_tags") if isinstance(style.get("prompt_tags"), list) else []
    if prompt_tags:
        locks["style_tags"] = prompt_tags
    if palette:
        locks.setdefault("style_palette", palette)
    regions = bundle.get("regions") if isinstance(bundle.get("regions"), dict) else {}
    if regions:
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
    if scene:
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
