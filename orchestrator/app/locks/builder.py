from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import httpx  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ImageDraw  # type: ignore

_INSIGHTFACE_MODEL = None


def _now_ts() -> int:
    return int(time.time() * 1000)


async def _download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        resp = await client.get(url)
        # Do not raise on HTTP errors; return empty bytes so callers can handle missing data.
        if resp.status_code < 200 or resp.status_code >= 300:
            return b""
        return resp.content


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hex_from_rgb(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _color_name(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    if r > 200 and g < 80 and b < 80:
        return "red"
    if g > 200 and r < 80 and b < 80:
        return "green"
    if b > 200 and r < 80 and g < 80:
        return "blue"
    if r > 200 and g > 200 and b < 120:
        return "yellow"
    if all(c > 200 for c in rgb):
        return "white"
    if all(c < 55 for c in rgb):
        return "black"
    if r > 180 and g > 100 and b < 120:
        return "orange"
    if r > 150 and b > 150 and g < 120:
        return "purple"
    return "mixed"


def _extract_palette(image: Image.Image, top_k: int = 4) -> List[Tuple[int, int, int]]:
    img = image.convert("RGB").resize((96, 96))
    colors = img.getcolors(96 * 96)
    if not colors:
        return []
    ranked = sorted(colors, key=lambda c: c[0], reverse=True)
    selected: List[Tuple[int, int, int]] = []
    for _, rgb in ranked:
        if not isinstance(rgb, tuple) or len(rgb) != 3:
            continue
        selected.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
        if len(selected) >= top_k:
            break
    return selected


def _palette_tags(palette: List[Tuple[int, int, int]]) -> List[str]:
    ctr = Counter()
    for rgb in palette:
        ctr[_color_name(rgb)] += 1
    tags: List[str] = []
    for name, _ in ctr.most_common():
        if name == "mixed":
            continue
        tags.append(name)
    return tags


def _blocking_face_embedding(image_path: str) -> Optional[List[float]]:
    global _INSIGHTFACE_MODEL
    try:
        import numpy as np  # type: ignore
        import insightface  # type: ignore
    except Exception:
        return None
    if _INSIGHTFACE_MODEL is None:
        try:
            app = insightface.app.FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1)
            _INSIGHTFACE_MODEL = app
        except Exception:
            return None
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    try:
        if cv2 is not None:
            img = cv2.imread(image_path)
        else:
            img = Image.open(image_path).convert("RGB")
            import numpy as np  # type: ignore
            img = np.array(img)
    except Exception:
        return None
    faces = _INSIGHTFACE_MODEL.get(img) if _INSIGHTFACE_MODEL else []
    if not faces:
        return None
    emb = faces[0].normed_embedding.tolist()
    return emb


async def _compute_face_embedding(image_path: str) -> Optional[List[float]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _blocking_face_embedding, image_path)


def _save_image_bytes(root: str, character_id: str, image_bytes: bytes, suffix: str) -> str:
    path_root = os.path.join(root, "image", character_id)
    _ensure_dir(path_root)
    filename = f"{_now_ts()}_{suffix}.png"
    full_path = os.path.join(path_root, filename)
    with open(full_path, "wb") as f:
        f.write(image_bytes)
    return full_path


def _save_audio_bytes(root: str, character_id: str, audio_bytes: bytes, suffix: str) -> str:
    path_root = os.path.join(root, "audio", character_id)
    _ensure_dir(path_root)
    filename = f"{_now_ts()}_{suffix}.wav"
    full_path = os.path.join(path_root, filename)
    with open(full_path, "wb") as f:
        f.write(audio_bytes)
    return full_path


async def build_image_bundle(
    character_id: str,
    image_url: Optional[str],
    options: Optional[Dict[str, Any]],
    *,
    locks_root_dir: str,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    opts = options or {}
    if image_path:
        with open(image_path, "rb") as fh:
            raw = fh.read()
    elif image_url:
        raw = await _download_bytes(image_url)
    else:
        raise ValueError("image_url or image_path required")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    palette = _extract_palette(image)
    palette_hex = [_hex_from_rgb(rgb) for rgb in palette]
    palette_tags = _palette_tags(palette)
    stored_path = _save_image_bytes(locks_root_dir, character_id, raw, "ref")
    pose_block: Dict[str, Any] = {}
    if bool(opts.get("detect_pose")):
        pose_block = {"skeleton": None, "strength": 0.7}
    style_block: Dict[str, Any] = {}
    if bool(opts.get("extract_style", True)):
        style_block = {
            "prompt_tags": palette_tags,
            "palette": {
                "primary": palette_hex[0] if palette_hex else None,
                "accent": palette_hex[1] if len(palette_hex) > 1 else None,
                "all": palette_hex,
            },
        }
    # Primary reference embedding from the main image.
    primary_emb = await _compute_face_embedding(stored_path)
    # Optional: additional reference images for the same character. These are
    # stored as extra_image_paths on the legacy face block so migrate_visual_bundle
    # can expose them as visual.faces[0].refs without changing callers.
    extra_paths: List[str] = []
    extra_urls = opts.get("extra_image_urls")
    if isinstance(extra_urls, list):
        for idx, u in enumerate(extra_urls):
            if not isinstance(u, str) or not u.strip():
                continue
            try:
                extra_raw = await _download_bytes(u.strip())
                if not extra_raw:
                    continue
                p = _save_image_bytes(locks_root_dir, character_id, extra_raw, f"ref_extra_{idx}")
                extra_paths.append(p)
            except Exception:
                continue
    # Aggregate all available embeddings into a single mean id_embedding so that
    # multi-view locks are more stable than a single-shot embedding.
    embs: List[List[float]] = []
    if isinstance(primary_emb, list):
        embs.append(primary_emb)
    for p in extra_paths:
        try:
            e = await _compute_face_embedding(p)
        except Exception:
            e = None
        if isinstance(e, list):
            embs.append(e)
    mean_emb: Optional[List[float]] = None
    if embs:
        # Compute elementwise mean across all embedding vectors, assuming they
        # have consistent dimensionality.
        dim = len(embs[0])
        acc = [0.0] * dim
        count = 0
        for vec in embs:
            if not isinstance(vec, list) or len(vec) != dim:
                continue
            for i, v in enumerate(vec):
                try:
                    acc[i] += float(v)
                except Exception:
                    acc[i] += 0.0
            count += 1
        if count > 0:
            mean_emb = [v / float(count) for v in acc]
    face_block: Dict[str, Any] = {
        "embedding": mean_emb if mean_emb is not None else primary_emb,
        "mask": None,
        "image_path": stored_path,
        "strength": opts.get("face_strength", 0.75),
    }
    if extra_paths:
        face_block["extra_image_paths"] = extra_paths
    image_np = np.asarray(image)
    background_embedding = None
    if image_np.size > 0:
        mean_rgb = image_np.reshape(-1, 3).mean(axis=0) / 255.0
        background_embedding = [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])]
    bundle = {
        "schema_version": 2,
        "character_id": character_id,
        "face": face_block,
        "pose": pose_block,
        "style": style_block,
        "audio": {},
        "regions": {},
        "scene": {
            "background_embedding": background_embedding,
            "camera_style_tags": [],
            "lighting_tags": [],
            "lock_mode": "soft",
        },
    }
    return bundle


def _blocking_voice_embedding(audio_path: str) -> Optional[List[float]]:
    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore
    except Exception:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=22050)
    except Exception:
        return None
    if y.size == 0:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    if mfcc.size == 0:
        return None
    vec = mfcc.mean(axis=1).astype(float)
    return vec.tolist()


async def _compute_voice_embedding(audio_path: str) -> Optional[List[float]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _blocking_voice_embedding, audio_path)


def _voice_tags_from_embedding(vec: Optional[List[float]]) -> List[str]:
    if not vec:
        return []
    mean_val = sum(vec) / len(vec)
    tags: List[str] = []
    if mean_val > 0.5:
        tags.append("bright")
    elif mean_val < -0.5:
        tags.append("dark")
    else:
        tags.append("neutral")
    energy = sum(abs(x) for x in vec) / len(vec)
    if energy > 100.0:
        tags.append("energetic")
    else:
        tags.append("calm")
    return tags


async def build_audio_bundle(
    character_id: str,
    audio_url: str,
    *,
    locks_root_dir: str,
) -> Dict[str, Any]:
    raw = await _download_bytes(audio_url)
    stored_path = _save_audio_bytes(locks_root_dir, character_id, raw, "ref")
    embedding = await _compute_voice_embedding(stored_path)
    tags = _voice_tags_from_embedding(embedding)
    audio_block: Dict[str, Any] = {
        "voice_embedding": embedding,
        "timbre_tags": tags,
        "reference_path": stored_path,
        "tempo_bpm": None,
        "tempo_lock_mode": "soft",
        "key": None,
        "key_lock_mode": "soft",
        "stem_profile": {},
        "stem_lock_mode": "soft",
        "lyrics_segments": [],
    }
    bundle = {
        "schema_version": 2,
        "character_id": character_id,
        "face": {},
        "pose": {},
        "style": {},
        "audio": audio_block,
        "regions": {},
        "scene": {
            "background_embedding": None,
            "camera_style_tags": [],
            "lighting_tags": [],
            "lock_mode": "soft",
        },
    }
    return bundle


async def voice_embedding_from_path(audio_path: str) -> Optional[List[float]]:
    return await _compute_voice_embedding(audio_path)


def _region_output_dir(root: str, character_id: str) -> str:
    path_root = os.path.join(root, "region", character_id)
    _ensure_dir(path_root)
    return path_root


def _clamp_bbox(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        return (0, 0, width, height)
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(round(x0)), width))
    y0 = max(0, min(int(round(y0)), height))
    x1 = max(0, min(int(round(x1)), width))
    y1 = max(0, min(int(round(y1)), height))
    if x1 <= x0:
        x1 = min(width, x0 + max(1, width // 4))
    if y1 <= y0:
        y1 = min(height, y0 + max(1, height // 4))
    return (x0, y0, x1, y1)


def _region_default_lock_mode(role: str) -> Dict[str, str]:
    base = {"shape": "hard", "texture": "soft", "color": "soft"}
    if role == "texture":
        base["shape"] = "soft"
    return base


def _save_region_mask(path_root: str, region_id: str, size: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> str:
    mask = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    filename = f"{_now_ts()}_{region_id}_mask.png"
    mask_path = os.path.join(path_root, filename)
    mask.save(mask_path)
    return mask_path


def _region_embeddings(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[List[float], List[float], Dict[str, Any]]:
    cropped = image.crop(bbox)
    w, h = image.size
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0 / max(1.0, float(w))
    cy = (y0 + y1) / 2.0 / max(1.0, float(h))
    rel_w = (x1 - x0) / max(1.0, float(w))
    rel_h = (y1 - y0) / max(1.0, float(h))
    shape_vec = [rel_w, rel_h, cx, cy]
    pixels = np.asarray(cropped.convert("RGB"))
    if pixels.size == 0:
        texture_vec = [0.0, 0.0, 0.0]
        palette = {"primary": None, "secondary": None, "accent": None, "all": []}
        return shape_vec, texture_vec, palette
    mean_rgb = pixels.reshape(-1, 3).mean(axis=0)
    var_rgb = pixels.reshape(-1, 3).var(axis=0)
    texture_vec = [float(mean_rgb[0] / 255.0), float(mean_rgb[1] / 255.0), float(mean_rgb[2] / 255.0),
                   float(var_rgb[0] / 255.0), float(var_rgb[1] / 255.0), float(var_rgb[2] / 255.0)]
    palette_raw = _extract_palette(cropped, top_k=3)
    palette_hex = [_hex_from_rgb(rgb) for rgb in palette_raw]
    palette = {
        "primary": palette_hex[0] if len(palette_hex) > 0 else None,
        "secondary": palette_hex[1] if len(palette_hex) > 1 else None,
        "accent": palette_hex[2] if len(palette_hex) > 2 else None,
        "all": palette_hex,
    }
    return shape_vec, texture_vec, palette


async def build_region_locks(
    character_id: str,
    image_url: str,
    regions: Optional[List[Dict[str, Any]]],
    *,
    locks_root_dir: str,
) -> Dict[str, Any]:
    raw = await _download_bytes(image_url)
    outdir = _region_output_dir(locks_root_dir, character_id)
    filename = f"{_now_ts()}_region_src.png"
    src_path = os.path.join(outdir, filename)
    with open(src_path, "wb") as fh:
        fh.write(raw)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = image.size
    region_defs = regions or []
    if not region_defs:
        region_defs = [{
            "region_id": "region_full",
            "role": "background",
            "bbox": [0, 0, width, height],
        }]
    built_regions: Dict[str, Any] = {}
    for idx, descriptor in enumerate(region_defs):
        region_id = str(descriptor.get("region_id") or f"region_{idx}")
        role = str(descriptor.get("role") or "object")
        bbox_raw = descriptor.get("bbox")
        bbox = _clamp_bbox(bbox_raw if isinstance(bbox_raw, list) else [0, 0, width, height], width, height)
        mask_path = _save_region_mask(outdir, region_id, (width, height), bbox)
        shape_vec, texture_vec, palette = _region_embeddings(image, bbox)
        lock_mode = descriptor.get("lock_mode") if isinstance(descriptor.get("lock_mode"), dict) else _region_default_lock_mode(role)
        strength = float(descriptor.get("strength") or 0.75)
        built_regions[region_id] = {
            "role": role,
            "mask_path": mask_path,
            "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "shape_embedding": shape_vec,
            "texture_embedding": texture_vec,
            "color_palette": palette,
            "lock_mode": lock_mode,
            "strength": strength,
        }
    image_np = np.asarray(image)
    background_embedding = None
    if image_np.size > 0:
        mean_rgb = image_np.reshape(-1, 3).mean(axis=0) / 255.0
        background_embedding = [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])]
    return {
        "schema_version": 2,
        "character_id": character_id,
        "regions": built_regions,
        "scene": {
            "background_embedding": background_embedding,
            "camera_style_tags": [],
            "lighting_tags": [],
            "lock_mode": "soft",
        },
        "region_source": src_path,
    }


def apply_region_mode_updates(bundle: Dict[str, Any], updates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not updates:
        return bundle
    next_bundle = dict(bundle or {})
    if next_bundle.get("schema_version", 1) < 2:
        next_bundle["schema_version"] = 2
    regions = dict(next_bundle.get("regions") or {})
    for update in updates:
        region_id = str(update.get("region_id") or "").strip()
        if not region_id:
            continue
        target = dict(regions.get(region_id) or {})
        if not target:
            continue
        lock_mode_update = update.get("lock_mode")
        if isinstance(lock_mode_update, dict):
            current_mode = dict(target.get("lock_mode") or {})
            current_mode.update({k: v for k, v in lock_mode_update.items() if isinstance(v, str)})
            target["lock_mode"] = current_mode
        if update.get("strength") is not None:
            try:
                target["strength"] = float(update.get("strength"))
            except Exception:
                pass
        if update.get("color_palette"):
            palette = update.get("color_palette")
            if isinstance(palette, dict):
                target["color_palette"] = palette
        regions[region_id] = target
    next_bundle["regions"] = regions
    return next_bundle


def apply_audio_mode_updates(bundle: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    if not update:
        return bundle
    next_bundle = dict(bundle or {})
    if next_bundle.get("schema_version", 1) < 2:
        next_bundle["schema_version"] = 2
    audio = dict(next_bundle.get("audio") or {})
    if update.get("tempo_bpm") is not None:
        try:
            audio["tempo_bpm"] = float(update.get("tempo_bpm"))
        except Exception:
            pass
    if isinstance(update.get("tempo_lock_mode"), str):
        audio["tempo_lock_mode"] = update["tempo_lock_mode"]
    if isinstance(update.get("key"), str):
        audio["key"] = update["key"]
    if isinstance(update.get("key_lock_mode"), str):
        audio["key_lock_mode"] = update["key_lock_mode"]
    stem_profile = update.get("stem_profile")
    if isinstance(stem_profile, dict):
        audio["stem_profile"] = {k: float(v) for k, v in stem_profile.items() if isinstance(v, (int, float))}
    if isinstance(update.get("stem_lock_mode"), str):
        audio["stem_lock_mode"] = update["stem_lock_mode"]
    lyrics_segments = update.get("lyrics_segments")
    if isinstance(lyrics_segments, list):
        normalized_segments: List[Dict[str, Any]] = []
        for segment in lyrics_segments:
            if not isinstance(segment, dict):
                continue
            seg_id = str(segment.get("id") or "").strip()
            if not seg_id:
                continue
            normalized_segments.append({
                "id": seg_id,
                "text": segment.get("text"),
                "lock_mode": segment.get("lock_mode") or "soft",
            })
        if normalized_segments:
            audio["lyrics_segments"] = normalized_segments
    next_bundle["audio"] = audio
    return next_bundle


