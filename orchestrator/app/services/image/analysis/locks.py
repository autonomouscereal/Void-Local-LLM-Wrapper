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
        try:
            return [float(x) for x in ref]
        except Exception:
            return None
    if isinstance(ref, dict):
        for key in ("embedding", "emb", "vector"):
            val = ref.get(key)
            if isinstance(val, list):
                try:
                    return [float(x) for x in val]
                except Exception:
                    return None
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
    try:
        img_vec = [float(x) for x in emb]
    except Exception:
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
    try:
        img_vec = [float(x) for x in emb]
    except Exception:
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
    try:
        img_vec = [float(x) for x in emb]
    except Exception:
        return None
    sim = _cosine_similarity(img_vec, ref_vec)
    return _normalize_similarity(sim)


async def compute_region_scores(image_path: str, region_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Region-level scores placeholder.

    Current implementation does not perform true region-based analysis; it returns an empty metrics dict.
    Structured keys are kept for forward compatibility.
    """
    _ = (image_path, region_data)
    return {
        "shape_score": None,
        "texture_score": None,
        "color_score": None,
    }


async def compute_scene_score(image_path: str, scene_data: Dict[str, Any]) -> Optional[float]:
    """
    Scene-level score placeholder.

    Reserved for future refinement; returns None for now.
    """
    _ = (image_path, scene_data)
    return None


