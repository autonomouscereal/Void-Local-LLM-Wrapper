from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore
from PIL import Image  # type: ignore

_INSIGHTFACE_APP = None


def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    s = hex_code.strip().lstrip("#")
    if len(s) == 6:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a or [])
    b_list = list(b or [])
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    num = sum(x * y for x, y in zip(a_list, b_list))
    den_a = math.sqrt(sum(x * x for x in a_list))
    den_b = math.sqrt(sum(y * y for y in b_list))
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return max(min(num / (den_a * den_b), 1.0), -1.0)


def _load_insightface():
    global _INSIGHTFACE_APP
    if _INSIGHTFACE_APP is not None:
        return _INSIGHTFACE_APP
    try:
        import insightface  # type: ignore
    except Exception:
        return None
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)
    _INSIGHTFACE_APP = app
    return _INSIGHTFACE_APP


def _blocking_face_embedding(image_path: str) -> Optional[List[float]]:
    app = _load_insightface()
    if app is None:
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
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].normed_embedding.tolist()


async def compute_face_lock_score(image_path: str, reference_embedding: Optional[List[float]]) -> Optional[float]:
    if not (isinstance(reference_embedding, list) and reference_embedding):
        return None
    # Hard-blocking execution (no thread pool/executor allowed).
    emb = _blocking_face_embedding(image_path)
    if not emb:
        return None
    score = _cosine(reference_embedding, emb)
    return float(max(0.0, min((score + 1.0) / 2.0, 1.0)))


def _extract_palette(image: Image.Image, top_k: int = 4) -> List[Tuple[int, int, int]]:
    img = image.convert("RGB").resize((96, 96))
    colors = img.getcolors(96 * 96)
    if not colors:
        return []
    ranked = sorted(colors, key=lambda c: c[0], reverse=True)
    palette: List[Tuple[int, int, int]] = []
    for _, rgb in ranked:
        if isinstance(rgb, tuple) and len(rgb) == 3:
            palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
            if len(palette) >= top_k:
                break
    return palette


def _palette_distance(p1: List[Tuple[int, int, int]], p2: List[Tuple[int, int, int]]) -> float:
    if not p1 or not p2:
        return 0.0
    total = 0.0
    count = 0
    for i, rgb in enumerate(p1):
        target = p2[min(i, len(p2) - 1)]
        dist = math.sqrt(sum((rgb[j] - target[j]) ** 2 for j in range(3)))
        total += dist
        count += 1
    if count == 0:
        return 0.0
    return total / count


async def compute_style_similarity(image_path: str, reference_palette: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(reference_palette, dict):
        return None
    ref_colors_raw = reference_palette.get("all") or []
    if not ref_colors_raw:
        return None
    try:
        ref_colors = [_hex_to_rgb(str(c)) for c in ref_colors_raw if isinstance(c, str)]
    except Exception:
        return None
    if not ref_colors:
        return None
    # Hard-blocking execution (no thread pool/executor allowed).
    image = Image.open(image_path).convert("RGB")
    palette = _extract_palette(image)
    if not palette:
        return None
    dist = _palette_distance(palette, ref_colors)
    max_dist = math.sqrt(3 * (255 ** 2))
    score = max(0.0, min(1.0 - (dist / max_dist), 1.0))
    return float(score)


async def compute_pose_similarity(image_path: str, pose_skeleton: Optional[Dict[str, Any]]) -> Optional[float]:
    if not pose_skeleton:
        return None
    # Pose estimation is not yet implemented in this service; return a neutral score.
    return None


def _region_shape_vector(width: int, height: int, bbox: Tuple[int, int, int, int]) -> List[float]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0 / max(1.0, float(width))
    cy = (y0 + y1) / 2.0 / max(1.0, float(height))
    rel_w = (x1 - x0) / max(1.0, float(width))
    rel_h = (y1 - y0) / max(1.0, float(height))
    return [rel_w, rel_h, cx, cy]


def _vector_distance(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    return float(sum(abs(x - y) for x, y in zip(a, b)) / len(a))


def _palette_similarity(palette_a: List[str], palette_b: List[str]) -> float:
    if not palette_a or not palette_b:
        return 0.0
    overlap = len(set(palette_a) & set(palette_b))
    return float(overlap / max(len(palette_a), len(palette_b)))


def _extract_region_features(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[List[float], List[float], Dict[str, Any]]:
    cropped = image.crop(bbox)
    width, height = image.size
    shape_vec = _region_shape_vector(width, height, bbox)
    pixels = np.asarray(cropped.convert("RGB"))
    if pixels.size == 0:
        return shape_vec, [0.0, 0.0, 0.0], {"primary": None, "secondary": None, "accent": None, "all": []}
    mean_rgb = pixels.reshape(-1, 3).mean(axis=0)
    var_rgb = pixels.reshape(-1, 3).var(axis=0)
    texture_vec = [
        float(mean_rgb[0] / 255.0),
        float(mean_rgb[1] / 255.0),
        float(mean_rgb[2] / 255.0),
        float(var_rgb[0] / 255.0),
        float(var_rgb[1] / 255.0),
        float(var_rgb[2] / 255.0),
    ]
    palette_raw = _extract_palette(cropped, top_k=3)
    palette_hex = [_hex_from_rgb(rgb) for rgb in palette_raw]
    palette = {
        "primary": palette_hex[0] if len(palette_hex) > 0 else None,
        "secondary": palette_hex[1] if len(palette_hex) > 1 else None,
        "accent": palette_hex[2] if len(palette_hex) > 2 else None,
        "all": palette_hex,
    }
    return shape_vec, texture_vec, palette


def _hex_from_rgb(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


async def compute_region_scores(
    image_path: str,
    region_bundle: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    mask_path = region_bundle.get("mask_path")
    bbox = region_bundle.get("bbox")
    shape_embedding = region_bundle.get("shape_embedding") if isinstance(region_bundle.get("shape_embedding"), list) else None
    texture_embedding = region_bundle.get("texture_embedding") if isinstance(region_bundle.get("texture_embedding"), list) else None
    palette_ref = region_bundle.get("color_palette") if isinstance(region_bundle.get("color_palette"), dict) else {}
    if not isinstance(mask_path, str):
        return {}
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {}
    image = Image.open(image_path).convert("RGB")
    bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    shape_vec, texture_vec, palette = _extract_region_features(image, bbox_tuple)
    scores: Dict[str, Optional[float]] = {}
    if shape_embedding:
        dist = _vector_distance([float(x) for x in shape_embedding], shape_vec)
        scores["shape_score"] = max(0.0, 1.0 - dist)
    if texture_embedding:
        dist = _vector_distance([float(x) for x in texture_embedding], texture_vec)
        scores["texture_score"] = max(0.0, 1.0 - (dist / 2.0))
    ref_palette_list = palette_ref.get("all") if isinstance(palette_ref.get("all"), list) else []
    if ref_palette_list:
        scores["color_score"] = _palette_similarity(ref_palette_list, palette.get("all") or [])
    return scores


async def compute_scene_score(image_path: str, scene_bundle: Dict[str, Any]) -> Optional[float]:
    if not isinstance(scene_bundle, dict):
        return None
    background_embedding = scene_bundle.get("background_embedding")
    if not isinstance(background_embedding, list):
        return None
    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image)
    if pixels.size == 0:
        return None
    mean_rgb = pixels.reshape(-1, 3).mean(axis=0) / 255.0
    target = [float(x) for x in background_embedding]
    if len(target) != 3:
        return None
    dist = _vector_distance(target, [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])])
    return max(0.0, 1.0 - dist)

