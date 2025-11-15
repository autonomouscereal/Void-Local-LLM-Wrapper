from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ....analysis.media import analyze_image


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> Optional[float]:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return None
    num = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
    den_a = math.sqrt(sum(float(x) * float(x) for x in vec_a))
    den_b = math.sqrt(sum(float(y) * float(y) for y in vec_b))
    if den_a == 0.0 or den_b == 0.0:
        return None
    return num / (den_a * den_b)


def _normalize_similarity(sim: Optional[float]) -> Optional[float]:
    if sim is None:
        return None
    # Map from [-1,1] → [0,1]
    score = (sim + 1.0) / 2.0
    return max(0.0, min(1.0, score))


def _extract_embedding(ref: Any) -> Optional[List[float]]:
    if isinstance(ref, list):
        floats: List[float] = []
        for x in ref:
            if isinstance(x, (int, float)):
                floats.append(float(x))
            else:
                return None
        return floats or None
    if isinstance(ref, dict):
        for key in ("embedding", "emb", "vector"):
            val = ref.get(key)
            if isinstance(val, list):
                floats: List[float] = []
                for x in val:
                    if isinstance(x, (int, float)):
                        floats.append(float(x))
                    else:
                        return None
                return floats or None
    return None


async def compute_face_lock_score(image_path: str, ref_embedding: List[float]) -> Optional[float]:
    """
    Compute a best-effort face lock score between an image and a reference embedding.

    Uses the clip embedding from analyze_image (if available) and cosine similarity against ref_embedding.
    Returns a score in [0,1] or None if unavailable.
    """
    ref_vec = _extract_embedding(ref_embedding)
    if ref_vec is None:
        return None
    info = analyze_image(image_path)
    if not isinstance(info, dict):
        return None
    emb = (info.get("emb") or {}).get("clip")
    if not isinstance(emb, list):
        return None
    img_vec: List[float] = []
    for x in emb:
        if isinstance(x, (int, float)):
            img_vec.append(float(x))
        else:
            return None
    sim = _cosine_similarity(img_vec, ref_vec)
    return _normalize_similarity(sim)


async def compute_style_similarity(image_path: str, style_ref: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort style similarity.

    If style_ref carries an embedding-like payload, compare to the image clip embedding; otherwise return None.
    """
    ref_vec = _extract_embedding(style_ref)
    if ref_vec is None:
        return None
    info = analyze_image(image_path)
    if not isinstance(info, dict):
        return None
    emb = (info.get("emb") or {}).get("clip")
    if not isinstance(emb, list):
        return None
    img_vec: List[float] = []
    for x in emb:
        if isinstance(x, (int, float)):
            img_vec.append(float(x))
        else:
            return None
    sim = _cosine_similarity(img_vec, ref_vec)
    return _normalize_similarity(sim)


async def compute_pose_similarity(image_path: str, pose_ref: Dict[str, Any]) -> Optional[float]:
    """
    Placeholder pose similarity.

    If pose_ref includes an embedding, reuse the same cosine-similarity heuristic; otherwise return None.
    """
    ref_vec = _extract_embedding(pose_ref)
    if ref_vec is None:
        return None
    info = analyze_image(image_path)
    if not isinstance(info, dict):
        return None
    emb = (info.get("emb") or {}).get("clip")
    if not isinstance(emb, list):
        return None
    img_vec: List[float] = []
    for x in emb:
        if isinstance(x, (int, float)):
            img_vec.append(float(x))
        else:
            return None
    sim = _cosine_similarity(img_vec, ref_vec)
    return _normalize_similarity(sim)


async def compute_region_scores(image_path: str, region_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Best-effort region-level lock scores.

    Uses the global image clip embedding from analyze_image() and compares it
    against any region-level embeddings when available. This is a heuristic
    approximation intended to populate lock metrics; it does not perform true
    mask-aware cropping.
    """
    info = analyze_image(image_path)
    if not isinstance(info, dict):
        return {
            "shape_score": None,
            "texture_score": None,
            "color_score": None,
            "clip_lock": None,
        }
    emb = (info.get("emb") or {}).get("clip")
    img_vec: Optional[List[float]] = None
    if isinstance(emb, list):
        floats: List[float] = []
        for x in emb:
            if isinstance(x, (int, float)):
                floats.append(float(x))
            else:
                floats = []
                break
        img_vec = floats or None
    # Region embeddings (if present)
    shape_ref = region_data.get("shape_embedding") or (region_data.get("embeddings") or {}).get("shape")
    texture_ref = region_data.get("texture_embedding") or (region_data.get("embeddings") or {}).get("texture")
    clip_ref = (region_data.get("embeddings") or {}).get("clip")
    shape_vec = _extract_embedding(shape_ref)
    texture_vec = _extract_embedding(texture_ref)
    clip_vec = _extract_embedding(clip_ref) if clip_ref is not None else None

    # Compute cosine similarities where possible
    shape_score = None
    texture_score = None
    clip_lock = None
    if img_vec is not None and shape_vec is not None:
        shape_score = _normalize_similarity(_cosine_similarity(img_vec, shape_vec))
    if img_vec is not None and texture_vec is not None:
        texture_score = _normalize_similarity(_cosine_similarity(img_vec, texture_vec))
    if img_vec is not None and clip_vec is not None:
        clip_lock = _normalize_similarity(_cosine_similarity(img_vec, clip_vec))
    # Color score is left as None for now; populate later when color histograms are available.
    return {
        "shape_score": shape_score,
        "texture_score": texture_score,
        "color_score": None,
        "clip_lock": clip_lock,
    }


async def compute_scene_score(image_path: str, scene_data: Dict[str, Any]) -> Optional[float]:
    """
    Scene-level score placeholder.

    Reserved for future refinement; returns None for now.
    """
    _ = (image_path, scene_data)
    return None


