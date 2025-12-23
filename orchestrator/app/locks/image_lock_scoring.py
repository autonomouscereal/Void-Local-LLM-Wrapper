from __future__ import annotations

"""
Image lock scoring utilities.

This module computes *scores* for lock adherence (identity/style/pose/scene/regions)
given a rendered image path + lock-bundle-derived reference data.

It intentionally lives under `app/locks/` because it must stay aligned with:
- lock bundle schemas (`app/locks/schema.py`)
- lock bundle builders (`app/locks/builder.py`)
- lock bundle runtime conversion (`app/locks/runtime.py`)
"""

import base64
import io
import json
import math
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

from ..analysis.media import analyze_image, cosine_similarity
from ..http.faceid_client import faceid_embed
from ..json_parser import JSONParser

log = logging.getLogger(__name__)


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
                floats = []
                for x in val:
                    if isinstance(x, (int, float)):
                        floats.append(float(x))
                    else:
                        return None
                return floats or None
    return None


def _cosine(a: Iterable[float], b: Iterable[float]) -> Optional[float]:
    """
    Local cosine helper (returns None on invalid inputs) for non-CLIP vectors.
    """
    try:
        a_list = [float(x) for x in (a or [])]
        b_list = [float(x) for x in (b or [])]
    except Exception:
        return None
    if not a_list or not b_list or len(a_list) != len(b_list):
        return None
    num = float(sum(x * y for x, y in zip(a_list, b_list)))
    den_a = float(sum(x * x for x in a_list)) ** 0.5
    den_b = float(sum(y * y for y in b_list)) ** 0.5
    if den_a == 0.0 or den_b == 0.0:
        return None
    return num / (den_a * den_b)


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = a_area + b_area - inter
    return float(inter / den) if den > 0.0 else 0.0


def _coerce_norm_bbox_to_px(bbox: Any, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Accepts bbox in:
      - normalized [x,y,w,h] (schema v3 uses this)
      - normalized [x0,y0,x1,y1]
      - absolute px [x0,y0,x1,y1]
    Returns float px bbox (x0,y0,x1,y1).
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        a = [float(x) for x in bbox]
    except Exception:
        return None
    # normalized heuristic
    is_norm = all(0.0 <= v <= 1.5 for v in a)
    if is_norm:
        x, y, c, d = a
        # Heuristic: if (c,d) look like widths, treat as xywh; else treat as x1y1
        if (c <= 1.0 and d <= 1.0) and (x + c <= 1.5 and y + d <= 1.5) and (c <= 0.8 and d <= 0.8):
            # xywh
            x0 = x * width
            y0 = y * height
            x1 = (x + c) * width
            y1 = (y + d) * height
        else:
            # x0y0x1y1
            x0 = x * width
            y0 = y * height
            x1 = c * width
            y1 = d * height
    else:
        x0, y0, x1, y1 = a
    # clamp
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _load_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    try:
        im = Image.open(image_path)
        return (int(im.size[0]), int(im.size[1]))
    except Exception as ex:
        log.exception("image_lock_scoring: failed to read image size path=%s error=%s", image_path, ex)
        return None


async def compute_multiface_identity_scores(
    image_path: str,
    face_refs: List[Dict[str, Any]],
    *,
    max_detected_faces: int = 16,
) -> Dict[str, Any]:
    """
    Multi-face identity scoring with assignment.

    Inputs:
      - image_path: candidate rendered image
      - face_refs: list of ref face specs, each dict supporting:
          - entity_id (str)
          - priority (int, optional)
          - role (str, optional)
          - embedding (list[float]) OR embeddings.id_embedding
          - model (str, optional): which InsightFace model to use for candidate embeddings
          - bbox (optional): normalized or px bbox for where that face should appear

    Output:
      {
        "aggregate": float | None,           # strict min across ref faces that produced a score
        "by_entity": {entity_id: {...}},     # per-ref face score + assignment metadata
        "detected_faces": [ ... ],           # list of detected faces (bbox/det_score) (no vectors to keep payload light)
      }
    """
    if not isinstance(face_refs, list) or not face_refs:
        return {"aggregate": None, "by_entity": {}, "detected_faces": []}

    # Determine candidate embedding model: prefer explicit per-ref model; else None (best available).
    model = None
    for r in face_refs:
        if isinstance(r, dict) and isinstance(r.get("model"), str) and r.get("model"):
            model = r.get("model")
            break

    detected, _model_used, _env = await faceid_embed(image_path=image_path, model_name=model, max_faces=max_detected_faces)
    if not detected:
        return {"aggregate": None, "by_entity": {}, "detected_faces": []}

    # Preload size for spatial matching (optional).
    size = _load_image_size(image_path)
    width, height = size if size else (0, 0)

    # Prepare candidates with indices
    cand = []
    for i, f in enumerate(detected):
        if not isinstance(f, dict):
            continue
        vec = f.get("embedding")
        if not isinstance(vec, list):
            continue
        bb = f.get("bbox")
        bb_px = None
        if isinstance(bb, list) and len(bb) == 4:
            try:
                bb_px = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            except Exception:
                bb_px = None
        cand.append(
            {
                "i": i,
                "embedding": vec,
                "det_score": f.get("det_score"),
                "bbox_px": bb_px,
            }
        )

    used = set()
    by_entity: Dict[str, Any] = {}

    # Sort refs by priority desc, stable.
    def _ref_key(r: Dict[str, Any]) -> Tuple[int, str]:
        pr = r.get("priority")
        pri = int(pr) if isinstance(pr, (int, float)) else 0
        eid = r.get("entity_id")
        return (-pri, str(eid) if isinstance(eid, str) else "")

    refs_sorted = [r for r in face_refs if isinstance(r, dict)]
    refs_sorted.sort(key=_ref_key)

    for ref in refs_sorted:
        eid = ref.get("entity_id")
        entity_id = str(eid) if isinstance(eid, str) and eid else f"face_{len(by_entity)}"

        # Extract ref embedding
        ref_vec = None
        if isinstance(ref.get("embedding"), list):
            ref_vec = _extract_embedding(ref.get("embedding"))
        if ref_vec is None and isinstance(ref.get("embeddings"), dict):
            ref_vec = _extract_embedding((ref.get("embeddings") or {}).get("id_embedding"))
        if ref_vec is None:
            by_entity[entity_id] = {
                "score": None,
                "reason": "missing_ref_embedding",
                "assigned_face": None,
                "role": ref.get("role"),
                "priority": ref.get("priority"),
            }
            continue

        # Optional region bbox
        ref_bbox_px = None
        bbox_any = ref.get("bbox")
        if bbox_any is None and isinstance(ref.get("region"), dict):
            bbox_any = (ref.get("region") or {}).get("bbox")
        if bbox_any is not None and width > 0 and height > 0:
            ref_bbox_px = _coerce_norm_bbox_to_px(bbox_any, width, height)

        best = None
        for c in cand:
            if c["i"] in used:
                continue
            vec = c.get("embedding")
            if not isinstance(vec, list) or len(vec) != len(ref_vec):
                continue
            sim = cosine_similarity(ref_vec, vec)  # type: ignore[arg-type]
            sim_score = _normalize_similarity(sim)
            if sim_score is None:
                continue

            spatial = None
            if ref_bbox_px is not None and isinstance(c.get("bbox_px"), tuple):
                spatial = _bbox_iou(ref_bbox_px, c["bbox_px"])
            # Combine: identity dominates; bbox only nudges selection when available.
            combined = float(sim_score)
            if spatial is not None:
                combined = float(0.85 * float(sim_score) + 0.15 * float(spatial))

            if best is None or combined > best["combined"]:
                best = {
                    "combined": combined,
                    "sim_score": float(sim_score),
                    "spatial": float(spatial) if spatial is not None else None,
                    "cand_i": c["i"],
                    "cand_bbox": c.get("bbox_px"),
                    "cand_det": c.get("det_score"),
                }

        if best is None:
            by_entity[entity_id] = {
                "score": None,
                "reason": "no_candidate_face_match",
                "assigned_face": None,
                "role": ref.get("role"),
                "priority": ref.get("priority"),
            }
            continue

        used.add(best["cand_i"])
        by_entity[entity_id] = {
            "score": float(best["sim_score"]),
            "combined": float(best["combined"]),
            "spatial": best.get("spatial"),
            "assigned_face": {
                "index": int(best["cand_i"]),
                "bbox": [float(x) for x in best["cand_bbox"]] if isinstance(best.get("cand_bbox"), tuple) else None,
                "det_score": best.get("cand_det"),
            },
            "role": ref.get("role"),
            "priority": ref.get("priority"),
        }

    # Aggregate strict min across available per-entity scores.
    scores = []
    for v in by_entity.values():
        s = v.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))
    agg = float(min(scores)) if scores else None

    # Expose detected faces without embeddings (keep payload small)
    detected_light = []
    for f in detected:
        if not isinstance(f, dict):
            continue
        detected_light.append({"bbox": f.get("bbox"), "det_score": f.get("det_score")})

    return {"aggregate": agg, "by_entity": by_entity, "detected_faces": detected_light}

def _hex_to_rgb(hex_code: str) -> Optional[Tuple[int, int, int]]:
    s = (hex_code or "").strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except Exception:
        return None


def _extract_palette_from_image_path(image_path: str, top_k: int = 4) -> List[Tuple[int, int, int]]:
    try:
        img = Image.open(image_path).convert("RGB").resize((96, 96))
        colors = img.getcolors(96 * 96)
        if not colors:
            return []
        ranked = sorted(colors, key=lambda c: c[0], reverse=True)
        palette: List[Tuple[int, int, int]] = []
        for _, rgb in ranked:
            if isinstance(rgb, tuple) and len(rgb) == 3:
                palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
                if len(palette) >= int(top_k):
                    break
        return palette
    except Exception:
        return []


def _palette_similarity_hex(ref_hex: List[str], cand_rgb: List[Tuple[int, int, int]]) -> Optional[float]:
    """
    Similarity in [0,1] between a reference palette (hex strings) and a candidate palette (RGB tuples).
    """
    if not ref_hex or not cand_rgb:
        return None
    ref_rgb: List[Tuple[int, int, int]] = []
    for h in ref_hex:
        if isinstance(h, str):
            rgb = _hex_to_rgb(h)
            if rgb:
                ref_rgb.append(rgb)
    if not ref_rgb:
        return None
    # Average per-index RGB distance (candidate matched to same index, clamped)
    total = 0.0
    count = 0
    for i, c in enumerate(cand_rgb):
        t = ref_rgb[min(i, len(ref_rgb) - 1)]
        dist = math.sqrt((c[0] - t[0]) ** 2 + (c[1] - t[1]) ** 2 + (c[2] - t[2]) ** 2)
        total += dist
        count += 1
    if count <= 0:
        return None
    avg = total / float(count)
    max_dist = math.sqrt(3 * (255.0 ** 2))
    score = max(0.0, min(1.0 - (avg / max_dist), 1.0))
    return float(score)


def _coerce_bbox_to_px(bbox: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Accepts bbox in either:
      - absolute pixels [x0,y0,x1,y1]
      - normalized [x,y,w,h] or [x0,y0,x1,y1] in 0..1
    Returns clamped pixel bbox (x0,y0,x1,y1) with x1>x0,y1>y0.
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        a = [float(x) for x in bbox]
    except Exception:
        return None
    # Heuristic: normalized if all in [0,1.5]
    is_norm = all(0.0 <= v <= 1.5 for v in a)
    if is_norm:
        x0, y0, x1, y1 = a[0], a[1], a[2], a[3]
        # If provided as [x,y,w,h], convert
        if x1 <= 1.0 and y1 <= 1.0 and (x1 >= 0.0 and y1 >= 0.0) and (x1 <= 1.5 and y1 <= 1.5):
            # ambiguous: treat as x1,y1 if they look like coordinates; otherwise if w/h small assume w/h
            already_corners = (x1 >= x0 and y1 >= y0)
            if not already_corners:
                # treat as w,h
                x1 = x0 + x1
                y1 = y0 + y1
        px0 = int(max(0, min(width - 1, round(x0 * width))))
        py0 = int(max(0, min(height - 1, round(y0 * height))))
        px1 = int(max(0, min(width, round(x1 * width))))
        py1 = int(max(0, min(height, round(y1 * height))))
    else:
        px0 = int(round(a[0])); py0 = int(round(a[1])); px1 = int(round(a[2])); py1 = int(round(a[3]))
        px0 = max(0, min(width - 1, px0))
        py0 = max(0, min(height - 1, py0))
        px1 = max(0, min(width, px1))
        py1 = max(0, min(height, py1))
    if px1 <= px0 or py1 <= py0:
        return None
    return (px0, py0, px1, py1)


def _decode_mask_b64(mask_b64: str) -> Optional[bytes]:
    if not isinstance(mask_b64, str) or not mask_b64.strip():
        return None
    s = mask_b64.strip()
    # Accept data URLs
    if s.startswith("data:") and "base64," in s:
        s = s.split("base64,", 1)[-1].strip()
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        return None


def _extract_pose_keypoints_from_image(image_path: str) -> Optional[Dict[str, List[float]]]:
    """
    Extract a stable subset of pose keypoints from an image using MediaPipe Pose.
    Returns a mapping {name: [x,y]} where x,y are normalized in [0,1].
    """
    # Load image RGB
    img = None
    try:
        bgr = cv2.imread(image_path)
        if bgr is not None:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        img = None
    if img is None:
        try:
            pil = Image.open(image_path).convert("RGB")
            img = np.array(pil)
        except Exception:
            return None
    if img is None:
        return None
    pose = mp.solutions.pose  # type: ignore[attr-defined]
    # Static mode (single image), deterministic settings
    try:
        with pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as p:
            res = p.process(img)
    except Exception:
        return None
    if not getattr(res, "pose_landmarks", None):
        return None
    lms = res.pose_landmarks.landmark
    # MediaPipe landmark indices
    idx = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    out: Dict[str, List[float]] = {}
    for name, i in idx.items():
        try:
            lm = lms[i]
            x = float(getattr(lm, "x", 0.0))
            y = float(getattr(lm, "y", 0.0))
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                out[name] = [x, y]
        except Exception:
            continue
    return out or None


def _load_skeleton_any(pose_ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Accept skeleton in several shapes:
      - {"skeleton": {...}}
      - {"skeleton": "/path/to/file.json"}
      - {"keypoints": {...}} (treated as skeleton-like)
      - {"skeleton": {"keypoints": {...}}}
    """
    if not isinstance(pose_ref, dict):
        return None
    sk = pose_ref.get("skeleton")
    if isinstance(sk, dict):
        return sk
    if isinstance(sk, str) and sk.strip():
        p = sk.strip()
        # if looks like JSON, parse
        if p.startswith("{") and p.endswith("}"):
            parser = JSONParser()
            obj = parser.parse(p, {})
            return obj if isinstance(obj, dict) else None
        # else treat as path
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    parser = JSONParser()
                    obj = parser.parse(f.read(), {})
                return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    if isinstance(pose_ref.get("keypoints"), dict):
        return {"format": "custom", "keypoints": pose_ref.get("keypoints")}
    return None


def _normalize_pose_points(points: Dict[str, List[float]]) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Normalize keypoints for translation/scale invariance:
      - center at mid-hip if available else mean
      - scale by shoulder span else hip span else max radius
    """
    if not isinstance(points, dict) or not points:
        return None
    pts: Dict[str, Tuple[float, float]] = {}
    for k, v in points.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                x = float(v[0]); y = float(v[1])
            except Exception:
                continue
            pts[k] = (x, y)
    if not pts:
        return None
    def mid(a: str, b: str) -> Optional[Tuple[float, float]]:
        if a in pts and b in pts:
            return ((pts[a][0] + pts[b][0]) / 2.0, (pts[a][1] + pts[b][1]) / 2.0)
        return None
    center = mid("left_hip", "right_hip")
    if center is None:
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        center = (sum(xs) / len(xs), sum(ys) / len(ys))
    # scale
    scale = None
    if "left_shoulder" in pts and "right_shoulder" in pts:
        dx = pts["left_shoulder"][0] - pts["right_shoulder"][0]
        dy = pts["left_shoulder"][1] - pts["right_shoulder"][1]
        scale = (dx * dx + dy * dy) ** 0.5
    if (scale is None or scale <= 1e-6) and ("left_hip" in pts and "right_hip" in pts):
        dx = pts["left_hip"][0] - pts["right_hip"][0]
        dy = pts["left_hip"][1] - pts["right_hip"][1]
        scale = (dx * dx + dy * dy) ** 0.5
    if scale is None or scale <= 1e-6:
        # max radius
        mx = 0.0
        for (x, y) in pts.values():
            dx = x - center[0]; dy = y - center[1]
            mx = max(mx, (dx * dx + dy * dy) ** 0.5)
        scale = mx
    if scale is None or scale <= 1e-6:
        return None
    out: Dict[str, Tuple[float, float]] = {}
    for k, (x, y) in pts.items():
        out[k] = ((x - center[0]) / scale, (y - center[1]) / scale)
    return out


async def compute_face_lock_score(image_path: str, ref_embedding: List[float], ref_model: Optional[str] = None) -> Optional[float]:
    """
    Identity lock score in [0,1] based on InsightFace/ArcFace embeddings (same space as lock bundle creation).
    """
    ref_vec = _extract_embedding(ref_embedding)
    if ref_vec is None:
        return None
    faces, _model_used, _env = await faceid_embed(image_path=image_path, model_name=ref_model, max_faces=16)
    if not faces:
        return None
    # Multi-face: score against every detected face and take the best match.
    best: Optional[float] = None
    for f in faces:
        vec = f.get("embedding")
        if not isinstance(vec, list) or len(vec) != len(ref_vec):
            continue
        sim = cosine_similarity(ref_vec, vec)  # type: ignore[arg-type]
        ns = _normalize_similarity(sim)
        if isinstance(ns, (int, float)):
            best = float(ns) if best is None else max(best, float(ns))
    return best


async def compute_style_similarity(image_path: str, style_ref: Dict[str, Any]) -> Optional[float]:
    """
    Style adherence score in [0,1].

    Uses a weighted blend of:
    - CLIP embedding similarity (when style_ref contains an embedding-like vector)
    - palette similarity (when style_ref contains palette hex colors)
    - semantic tag overlap (style tags vs analyzer tags)
    """
    if not isinstance(style_ref, dict):
        return None

    ref_vec = _extract_embedding(style_ref)
    info = analyze_image(image_path)
    if not isinstance(info, dict):
        return None
    sem = info.get("semantics") if isinstance(info.get("semantics"), dict) else {}
    emb = sem.get("clip_emb")
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

    scores: List[float] = []
    weights: List[float] = []
    if ref_vec is not None and img_vec is not None and len(ref_vec) == len(img_vec):
        sim = cosine_similarity(img_vec, ref_vec)
        ns = _normalize_similarity(sim)
        if isinstance(ns, (int, float)):
            scores.append(float(ns)); weights.append(0.65)

    pal = style_ref.get("palette") if isinstance(style_ref.get("palette"), dict) else None
    ref_all = None
    if isinstance(pal, dict):
        ref_all = pal.get("all") or pal.get("primary") or pal.get("secondary") or pal.get("accent")
    if isinstance(ref_all, str):
        ref_hex = [ref_all]
    elif isinstance(ref_all, list):
        ref_hex = [str(x) for x in ref_all if isinstance(x, str)]
    else:
        ref_hex = []
    cand_rgb = _extract_palette_from_image_path(image_path, top_k=4)
    pal_score = _palette_similarity_hex(ref_hex, cand_rgb) if ref_hex else None
    if isinstance(pal_score, (int, float)):
        scores.append(float(pal_score)); weights.append(0.25)

    ref_tags = style_ref.get("style_tags") or style_ref.get("prompt_tags") or []
    if isinstance(ref_tags, list):
        ref_set = {str(t).strip().lower() for t in ref_tags if isinstance(t, str) and t.strip()}
    else:
        ref_set = set()
    sem_tags = sem.get("tags") if isinstance(sem, dict) else []
    if isinstance(sem_tags, list):
        sem_set = {str(t).strip().lower() for t in sem_tags if isinstance(t, str) and t.strip()}
    else:
        sem_set = set()
    if ref_set and sem_set:
        inter = len(ref_set & sem_set)
        union = len(ref_set | sem_set)
        tag_score = float(inter) / float(union) if union else 0.0
        scores.append(tag_score); weights.append(0.10)

    if not scores:
        return None
    wsum = sum(weights) if weights else 0.0
    if wsum <= 0:
        return float(sum(scores) / float(len(scores)))
    return float(sum(s * w for s, w in zip(scores, weights)) / wsum)


async def compute_pose_similarity(image_path: str, pose_ref: Dict[str, Any]) -> Optional[float]:
    """
    Pose adherence score in [0,1] comparing lock skeleton keypoints to MediaPipe Pose keypoints.
    """
    sk = _load_skeleton_any(pose_ref if isinstance(pose_ref, dict) else {})
    if not isinstance(sk, dict):
        return None
    kp = sk.get("keypoints") if isinstance(sk.get("keypoints"), dict) else None
    if not isinstance(kp, dict) or not kp:
        return None

    ref_pts_raw: Dict[str, List[float]] = {}
    for k, v in kp.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                ref_pts_raw[k] = [float(v[0]), float(v[1])]
            except Exception:
                continue
    if not ref_pts_raw:
        return None

    img_pts_raw = _extract_pose_keypoints_from_image(image_path)
    if not isinstance(img_pts_raw, dict) or not img_pts_raw:
        return None

    ref_norm = _normalize_pose_points(ref_pts_raw)
    img_norm = _normalize_pose_points(img_pts_raw)
    if not ref_norm or not img_norm:
        return None

    keys = [k for k in ref_norm.keys() if k in img_norm]
    if len(keys) < 5:
        return None

    dsum = 0.0
    for k in keys:
        ax, ay = ref_norm[k]
        bx, by = img_norm[k]
        dsum += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    avg = dsum / float(len(keys))
    score = 1.0 / (1.0 + avg)
    return float(max(0.0, min(score, 1.0)))


async def compute_region_scores(image_path: str, region_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Region/entity adherence metrics:
      - shape_score: bbox-shape vector similarity
      - texture_score: mean/var RGB feature similarity
      - color_score: palette similarity
      - clip_lock: CLIP similarity between region crop and stored region CLIP embedding
    """
    out: Dict[str, Optional[float]] = {"shape_score": None, "texture_score": None, "color_score": None, "clip_lock": None}
    if not isinstance(region_data, dict):
        return out
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:
        return out
    w, h = im.size
    bbox = _coerce_bbox_to_px(region_data.get("bbox"), w, h)
    if bbox is None:
        return out
    x0, y0, x1, y1 = bbox
    crop = im.crop((x0, y0, x1, y1))

    # Optional mask: mask_b64 (preferred) or mask_path
    mask_img = None
    mask_b64 = region_data.get("mask_b64")
    mask_path = region_data.get("mask_path")
    if isinstance(mask_b64, str) and mask_b64.strip():
        b = _decode_mask_b64(mask_b64)
        if b:
            try:
                mask_img = Image.open(io.BytesIO(b)).convert("L")
            except Exception:
                mask_img = None
    elif isinstance(mask_path, str) and mask_path.strip() and os.path.exists(mask_path.strip()):
        try:
            mask_img = Image.open(mask_path.strip()).convert("L")
        except Exception:
            mask_img = None

    if mask_img is not None:
        try:
            mask_resized = mask_img.resize(crop.size)
            bg = Image.new("RGB", crop.size, (0, 0, 0))
            crop = Image.composite(crop, bg, mask_resized)
        except Exception:
            # Keep scoring alive even if the optional mask path/b64 is malformed.
            log.debug(
                "image_lock_scoring.mask_composite_failed image_path=%r bbox=%r",
                image_path,
                bbox,
                exc_info=True,
            )

    tmp_dir = os.path.join(os.getenv("UPLOAD_DIR", "/workspace/uploads"), "tmp_region_locks")
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.png")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        crop.save(tmp_path, format="PNG")
    except Exception:
        tmp_path = ""

    crop_clip: Optional[List[float]] = None
    if tmp_path and os.path.exists(tmp_path):
        info = analyze_image(tmp_path, prompt=None)
        if isinstance(info, dict):
            sem = info.get("semantics") if isinstance(info.get("semantics"), dict) else {}
            emb = sem.get("clip_emb")
            if isinstance(emb, list):
                floats = []
                ok = True
                for x in emb:
                    if isinstance(x, (int, float)):
                        floats.append(float(x))
                    else:
                        ok = False
                        break
                crop_clip = floats if ok and floats else None

    # shape score
    shape_ref = region_data.get("shape_embedding") or (region_data.get("embeddings") or {}).get("shape")
    shape_vec = _extract_embedding(shape_ref)
    if shape_vec and len(shape_vec) == 4:
        rel_w = float((x1 - x0) / max(1.0, float(w)))
        rel_h = float((y1 - y0) / max(1.0, float(h)))
        cx = float(((x0 + x1) / 2.0) / max(1.0, float(w)))
        cy = float(((y0 + y1) / 2.0) / max(1.0, float(h)))
        cand = [rel_w, rel_h, cx, cy]
        dist = 0.0
        for a, b in zip(shape_vec, cand):
            dist += abs(float(a) - float(b))
        dist = dist / 4.0
        out["shape_score"] = float(max(0.0, min(1.0 - dist, 1.0)))

    # texture score
    texture_ref = region_data.get("texture_embedding") or (region_data.get("embeddings") or {}).get("texture")
    tex_vec = _extract_embedding(texture_ref)
    if tex_vec and len(tex_vec) == 6:
        try:
            px = np.asarray(crop.convert("RGB"))
            if px.size > 0:
                flat = px.reshape(-1, 3).astype("float32")
                mean = flat.mean(axis=0) / 255.0
                var = flat.var(axis=0) / 255.0
                cand6 = [float(mean[0]), float(mean[1]), float(mean[2]), float(var[0]), float(var[1]), float(var[2])]
                dist = 0.0
                for a, b in zip(tex_vec, cand6):
                    dist += abs(float(a) - float(b))
                dist = dist / 6.0
                out["texture_score"] = float(max(0.0, min(1.0 - dist, 1.0)))
        except Exception:
            log.debug(
                "image_lock_scoring.texture_score_failed image_path=%r bbox=%r",
                image_path,
                bbox,
                exc_info=True,
            )

    # color score
    pal_ref = region_data.get("color_palette") or (region_data.get("embeddings") or {}).get("palette") or {}
    ref_hex_list: List[str] = []
    if isinstance(pal_ref, dict):
        raw = pal_ref.get("all") or []
        if isinstance(raw, list):
            ref_hex_list = [str(x) for x in raw if isinstance(x, str)]
    cand_pal = _extract_palette_from_image_path(tmp_path if tmp_path else image_path, top_k=3)
    pal_score = _palette_similarity_hex(ref_hex_list, cand_pal) if ref_hex_list else None
    if isinstance(pal_score, (int, float)):
        out["color_score"] = float(pal_score)

    # clip lock
    clip_ref = (region_data.get("embeddings") or {}).get("clip") or region_data.get("clip_embedding")
    clip_vec = _extract_embedding(clip_ref)
    if crop_clip is not None and clip_vec is not None and len(crop_clip) == len(clip_vec):
        out["clip_lock"] = _normalize_similarity(cosine_similarity(crop_clip, clip_vec))

    return out


async def compute_scene_score(image_path: str, scene_data: Dict[str, Any]) -> Optional[float]:
    """
    Scene adherence score in [0,1].
    Uses:
      - background_embedding similarity (RGB mean, from lock bundle)
      - scene tag overlap (lock bundle vs analyze_image locks tags)
    """
    if not isinstance(scene_data, dict):
        return None

    bg = scene_data.get("background_embedding")
    bg_vec = _extract_embedding(bg) if bg is not None else None
    bg_score: Optional[float] = None
    if bg_vec and len(bg_vec) == 3:
        try:
            im = Image.open(image_path).convert("RGB")
            px = np.asarray(im)
            if px.size > 0:
                mean_rgb = px.reshape(-1, 3).mean(axis=0) / 255.0
                cand = [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])]
                dist = (abs(float(bg_vec[0]) - cand[0]) + abs(float(bg_vec[1]) - cand[1]) + abs(float(bg_vec[2]) - cand[2])) / 3.0
                bg_score = float(max(0.0, min(1.0 - dist, 1.0)))
        except Exception:
            bg_score = None

    tag_score: Optional[float] = None
    ref_tags_raw = scene_data.get("scene_tags") or []
    ref_tags = {str(t).strip().lower() for t in (ref_tags_raw if isinstance(ref_tags_raw, list) else []) if isinstance(t, str) and t.strip()}
    try:
        info = analyze_image(image_path, prompt=None)
    except Exception:
        info = {}
    locks = info.get("locks") if isinstance(info, dict) and isinstance(info.get("locks"), dict) else {}
    cand_tags_raw = locks.get("scene_tags") or locks.get("entity_tags") or []
    cand_tags = {str(t).strip().lower() for t in (cand_tags_raw if isinstance(cand_tags_raw, list) else []) if isinstance(t, str) and t.strip()}
    if ref_tags and cand_tags:
        inter = len(ref_tags & cand_tags)
        union = len(ref_tags | cand_tags)
        tag_score = float(inter) / float(union) if union else 0.0

    if bg_score is None and tag_score is None:
        return None
    if bg_score is not None and tag_score is not None:
        return float(0.7 * bg_score + 0.3 * tag_score)
    return bg_score if bg_score is not None else tag_score


