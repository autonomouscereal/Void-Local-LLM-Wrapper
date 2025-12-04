from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional, List, Tuple
import uuid

import cv2  # type: ignore
import numpy as np  # type: ignore
import requests  # type: ignore
import torch  # type: ignore
import open_clip  # type: ignore
from PIL import Image  # type: ignore
import soundfile as sf  # type: ignore
import pyloudnorm as pyln  # type: ignore
import librosa  # type: ignore
from ..json_parser import JSONParser


VLM_API_URL = os.getenv("VLM_API_URL")  # e.g., http://vlm:8050
OCR_API_URL = os.getenv("OCR_API_URL")  # e.g., http://ocr:8070
VISION_REPAIR_API_URL = os.getenv("VISION_REPAIR_API_URL")  # e.g., http://vision_repair:8095
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")

# Aesthetic head configuration (LAION aesthetic predictor v2 on ViT-L/14)
AESTHETIC_HEAD_PATH = os.getenv("AESTHETIC_HEAD_PATH", "")
AESTHETIC_CLIP_MODEL_NAME = os.getenv("AESTHETIC_CLIP_MODEL_NAME", "ViT-L-14")
AESTHETIC_CLIP_PRETRAINED = os.getenv("AESTHETIC_CLIP_PRETRAINED", "laion2b_s32b_b82k")

_AESTHETIC_CLIP_MODEL: Any = None
_AESTHETIC_PREPROCESS: Any = None
_AESTHETIC_HEAD: Any = None
_AESTHETIC_AVAILABLE: bool = False


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else float(x)


def _image_quality_metrics(path: str) -> Dict[str, float]:
    """
    Cheap, deterministic, non-ML image quality metrics.

    Returns brightness/contrast/sharpness/edge_density in [0,1] plus a
    composite "score" also in [0,1].
    """
    metrics: Dict[str, float] = {
        "brightness": 0.0,
        "contrast": 0.0,
        "sharpness": 0.0,
        "edge_density": 0.0,
        "score": 0.0,
    }
    img = cv2.imread(path)
    if img is None:
        metrics["error"] = "load_failed"
        return metrics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype("float32") / 255.0
    # brightness: mean in [0,1], favor mid-range around 0.5
    brightness = float(gray_f.mean())
    brightness_score = _clamp01(1.0 - abs(brightness - 0.5) / 0.5)
    # contrast: std-dev normalized, typical good contrast ~0.25
    contrast = float(gray_f.std())
    contrast_score = _clamp01(contrast / 0.25)
    # sharpness: variance of Laplacian, normalized
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    sharp_raw = float(lap.var())
    sharpness_score = _clamp01(sharp_raw / 0.02)
    # edge density: fraction of edge pixels, normalized
    edges = cv2.Canny((gray_f * 255).astype("uint8"), 100, 200)
    edge_density = float(edges.mean()) / 255.0
    edge_score = _clamp01(edge_density / 0.3)
    quality = (brightness_score + contrast_score + sharpness_score + edge_score) / 4.0
    metrics.update(
        {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharp_raw,
            "edge_density": edge_density,
            "score": _clamp01(quality),
        }
    )
    return metrics


def _path_to_public_url(path: str) -> Optional[str]:
    """
    Map an uploads-backed filesystem path to a public URL, if possible.
    """
    if not PUBLIC_BASE_URL:
        return None
    try:
        root = os.path.abspath(UPLOAD_DIR)
        full = os.path.abspath(path)
        if not full.startswith(root):
            return None
        rel = os.path.relpath(full, root).replace("\\", "/")
        return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}"
    except Exception:
        return None


def _run_ocr(path: str) -> Dict[str, Any]:
    out = {"ocr_text": "", "has_text": False, "error": None}
    if not OCR_API_URL:
        out["error"] = "ocr_unconfigured"
        return out
    try:
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        r = requests.post(OCR_API_URL.rstrip("/") + "/ocr", json={"b64": b64, "ext": ext})
        from ..json_parser import JSONParser  # local import to avoid cycles at module import time
        parser = JSONParser()
        js = parser.parse_superset(r.text or "{}", {"text": str})["coerced"]
        txt = (js.get("text") or "").strip() if isinstance(js, dict) else ""
        out["ocr_text"] = txt
        out["has_text"] = bool(txt)
    except Exception as ex:
        out["error"] = f"ocr_error:{ex}"
    return out


def _run_yolo(path: str) -> Dict[str, Any]:
    """
    Call the vision_repair service once to get YOLO-style detections and
    derive basic entity/scene tags.
    """
    res: Dict[str, Any] = {
        "yolo": [],
        "face_regions": [],
        "scene_tags": [],
        "entity_tags": [],
        "error": None,
    }
    if not VISION_REPAIR_API_URL:
        res["error"] = "vision_repair_unconfigured"
        return res
    try:
        r = requests.post(
            VISION_REPAIR_API_URL.rstrip("/") + "/v1/image/analyze",
            json={"image_path": path},
            timeout=None,
        )
        from ..json_parser import JSONParser  # local import to avoid cycles at module import time
        parser = JSONParser()
        js = parser.parse_superset(r.text or "{}", {"objects": list, "faces": list})["coerced"]
        objects = js.get("objects") or [] if isinstance(js, dict) else []
        yolo_list: List[Dict[str, Any]] = []
        entity_tags: List[str] = []
        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                label = obj.get("class_name") or obj.get("label")
                conf_raw = obj.get("conf")
                bbox = obj.get("bbox") or obj.get("xyxy")
                try:
                    conf_val = float(conf_raw) if conf_raw is not None else 0.0
                except Exception:
                    conf_val = 0.0
                if not isinstance(label, str) or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                yolo_list.append({"label": label, "conf": conf_val, "bbox": bbox})
                if conf_val >= 0.4:
                    entity_tags.append(label)
        res["yolo"] = yolo_list
        # faces/hands currently unused; wire through if the service starts emitting them
        faces = js.get("faces") or []
        if isinstance(faces, list):
            res["face_regions"] = faces
        # simple tag list from object labels
        if entity_tags:
            dedup: List[str] = []
            seen = set()
            for t in entity_tags:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            res["entity_tags"] = dedup
            res["scene_tags"] = dedup
    except Exception as ex:
        res["error"] = f"vision_repair_error:{ex}"
    return res


def _clip_analyze(path: str, prompt: Optional[str]) -> Dict[str, Any]:
    out = {"clip_score": None, "clip_prompt": None, "clip_emb": None, "error": None}
    p = (prompt or "").strip()
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        image = preprocess(Image.open(path)).unsqueeze(0)
        # Always compute image embedding for semantics, even without prompt
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        out["clip_emb"] = image_features.squeeze().cpu().tolist()
        if p:
            text = tokenizer([p])
            with torch.no_grad():
                text_features = model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                sim = (image_features @ text_features.T).squeeze().cpu().item()
            out["clip_score"] = float(_clamp01((sim + 1.0) / 2.0))  # [-1,1] → [0,1]
            out["clip_prompt"] = p
    except Exception as ex:
        out["error"] = f"clip_error:{ex}"
    return out


def _qwen_vl_analyze(path: str, prompt: Optional[str]) -> Dict[str, Any]:
    """
    Use the VLM (Qwen-VL) service as a VL captioner + prompt-match scorer.

    The service is instructed to respond with a strict JSON payload containing
    caption, keywords, and match_score in [0,1]. If parsing fails, we fall
    back to treating the raw text as the caption and derive tags naively.
    """
    res: Dict[str, Any] = {"caption": "", "tags": [], "match_score": None, "error": None}
    if not VLM_API_URL:
        res["error"] = "vlm_unconfigured"
        return res
    image_url = _path_to_public_url(path)
    if not image_url:
        res["error"] = "no_public_url"
        return res
    user_prompt = (prompt or "").strip()
    base_instr = (
        "Respond in strict JSON with keys "
        '"caption" (string), "keywords" (array of strings), '
        '"match_score" (float between 0 and 1). '
        "caption should be a detailed description of the image. "
    )
    if user_prompt:
        base_instr += (
            "Then rate how well the image matches this description: "
            f"{user_prompt!r} on the 0-1 scale in match_score. "
            "If the description is irrelevant to the image, use a low score near 0.0."
        )
    else:
        base_instr += "No user description is provided; set match_score to 1.0."
    try:
        r = requests.post(
            VLM_API_URL.rstrip("/") + "/analyze",
            json={"image_url": image_url, "prompt": base_instr},
            timeout=None,
        )
        from ..json_parser import JSONParser  # local import to avoid cycles at module import time
        parser = JSONParser()
        js = parser.parse_superset(r.text or "{}", {"text": str})["coerced"]
        text = (js.get("text") or "").strip() if isinstance(js, dict) else ""
        if not text:
            res["error"] = "vlm_empty_text"
            return res
        from ..json_parser import JSONParser  # local import to avoid cycles at module import time
        parser = JSONParser()
        # VLM JSON is free-form but expected to contain at least caption/keywords/tags.
        parsed = parser.parse_superset(text, {"caption": str, "keywords": list, "tags": list})["coerced"]
        caption = parsed.get("caption")
        if isinstance(caption, str):
            res["caption"] = caption.strip()
        keywords = parsed.get("keywords") or parsed.get("tags")
        if isinstance(keywords, list):
            tags: List[str] = []
            seen = set()
            for item in keywords:
                if isinstance(item, str):
                    t = item.strip()
                    if len(t) >= 2 and t not in seen:
                        seen.add(t)
                        tags.append(t)
            res["tags"] = tags
        ms = parsed.get("match_score")
        if isinstance(ms, (int, float)):
            res["match_score"] = float(_clamp01(float(ms)))
    except Exception as ex:
        res["error"] = f"vlm_error:{ex}"
    return res


def _load_aesthetic_head() -> None:
    """
    Lazy-load the LAION aesthetic head and associated CLIP ViT-L/14 backbone.

    If anything fails, _AESTHETIC_AVAILABLE remains False and callers fall back
    to 0.0 aesthetic scores with an error entry in problems.
    """
    global _AESTHETIC_CLIP_MODEL, _AESTHETIC_PREPROCESS, _AESTHETIC_HEAD, _AESTHETIC_AVAILABLE
    if _AESTHETIC_AVAILABLE:
        return
    path = AESTHETIC_HEAD_PATH
    if not path or not os.path.exists(path):
        return
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(  # type: ignore[attr-defined]
            AESTHETIC_CLIP_MODEL_NAME,
            pretrained=AESTHETIC_CLIP_PRETRAINED,
        )
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        obj = torch.load(path, map_location=device)
        # Accept either a full Module or a plain state_dict for a Linear head.
        if isinstance(obj, torch.nn.Module):
            head = obj
        elif isinstance(obj, dict):
            in_features = None
            visual = getattr(model, "visual", None)
            if visual is not None:
                in_features = getattr(visual, "output_dim", None)
            if in_features is None:
                # As a fallback, infer from a dummy forward pass
                dummy = torch.zeros(1, 3, 224, 224, device=device)
                with torch.no_grad():
                    emb = model.encode_image(dummy)
                in_features = emb.shape[-1]
            head = torch.nn.Linear(int(in_features), 1)  # type: ignore[arg-type]
            head.load_state_dict(obj)
        else:
            return
        head.eval()
        head.to(device)
        _AESTHETIC_CLIP_MODEL = model
        _AESTHETIC_PREPROCESS = preprocess
        _AESTHETIC_HEAD = head
        _AESTHETIC_AVAILABLE = True
    except Exception:
        _AESTHETIC_CLIP_MODEL = None
        _AESTHETIC_PREPROCESS = None
        _AESTHETIC_HEAD = None
        _AESTHETIC_AVAILABLE = False


def _clip_aesthetic(path: str) -> Dict[str, Any]:
    """
    Real aesthetic scoring using LAION aesthetic head on CLIP ViT-L/14 embeddings.

    Returns:
      {
        "clip_aesthetic": float in [0,1],
        "raw_score": float (typically ~1-10),
        "model": "laion_aesthetic_v2_L14",
        "error": Optional[str],
      }
    """
    out: Dict[str, Any] = {
        "clip_aesthetic": 0.0,
        "raw_score": 0.0,
        "model": "laion_aesthetic_v2_L14",
    }
    if not _AESTHETIC_AVAILABLE or _AESTHETIC_CLIP_MODEL is None or _AESTHETIC_PREPROCESS is None or _AESTHETIC_HEAD is None:
        out["error"] = "aesthetic_unavailable"
        return out
    try:
        device = next(_AESTHETIC_HEAD.parameters()).device  # type: ignore[union-attr]
        img = Image.open(path).convert("RGB")  # type: ignore[call-arg]
        image_tensor = _AESTHETIC_PREPROCESS(img).unsqueeze(0).to(device)  # type: ignore[operator]
        with torch.no_grad():
            emb = _AESTHETIC_CLIP_MODEL.encode_image(image_tensor)  # type: ignore[union-attr]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            score_tensor = _AESTHETIC_HEAD(emb)  # type: ignore[operator]
            raw = float(score_tensor.view(-1).item())
        # Map approx [1,10] → [0,1]
        norm = _clamp01((raw - 1.0) / 9.0)
        out["raw_score"] = raw
        out["clip_aesthetic"] = norm
    except Exception as ex:
        out["error"] = f"aesthetic_error:{ex}"
    return out


def analyze_image(path: str, prompt: str | None = None) -> Dict[str, Any]:
    """
    Full-stack image analysis using cheap CV, YOLO, OCR, CLIP, and Qwen-VL.

    Returns a structured dict with sections:
      - ok, path, width, height
      - tech_quality: cheap CV metrics
      - semantics: caption/tags/clip/qwen scores
      - objects: YOLO objects
      - text: OCR text
      - locks: coarse hooks for downstream lock/QA engines
      - score: overall/semantic/aesthetic/technical in [0,1]
    """
    out: Dict[str, Any] = {
        "ok": False,
        "path": path,
        "width": 0,
        "height": 0,
        "tech_quality": {
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "edge_density": 0.0,
            "score": 0.0,
        },
        "semantics": {
            "caption": "",
            "tags": [],
            "clip_score": None,
            "clip_prompt": None,
            "clip_emb": None,
            "qwen_vl_score": None,
        },
        "objects": {"yolo": []},
        "text": {"ocr_text": "", "has_text": False},
        "locks": {
            "face_regions": [],
            "scene_tags": [],
            "entity_tags": [],
        },
        "aesthetics": {
            "clip_aesthetic": 0.0,
            "raw_score": 0.0,
            "model": "laion_aesthetic_v2_L14",
        },
        "score": {
            "overall": 0.0,
            "semantic": 0.0,
            "aesthetic": 0.0,
            "technical": 0.0,
        },
        "problems": [],
    }
    if not isinstance(path, str) or not os.path.exists(path):
        return out
    problems: List[Dict[str, Any]] = []
    img = cv2.imread(path)
    if img is None:
        problems.append({"stage": "load", "code": "load_failed", "message": "could not read image"})
        out["problems"] = problems
        out["ok"] = False
        return out
    h, w = img.shape[:2]
    out["width"] = int(w)
    out["height"] = int(h)
    out["ok"] = True
    # Layer 1: technical quality
    try:
        tq = _image_quality_metrics(path)
        out["tech_quality"] = tq
        tech_score = float(tq.get("score") or 0.0)
        if tq.get("error"):
            problems.append({"stage": "tech_quality", "code": str(tq.get("error")), "message": "tech quality metrics degraded"})
    except Exception as ex:
        tech_score = 0.0
        problems.append({"stage": "tech_quality", "code": "exception", "message": str(ex)})
    # Layer 2: YOLO objects (vision_repair service)
    yolo_info = _run_yolo(path)
    out["objects"]["yolo"] = yolo_info.get("yolo") or []
    locks = out["locks"]
    locks["face_regions"] = yolo_info.get("face_regions") or []
    locks["entity_tags"] = yolo_info.get("entity_tags") or []
    locks["scene_tags"] = yolo_info.get("scene_tags") or []
    if yolo_info.get("error"):
        problems.append({"stage": "yolo", "code": str(yolo_info.get("error")), "message": "vision_repair/yolo stage failed"})
    # Layer 3: OCR
    text_info = _run_ocr(path)
    out["text"]["ocr_text"] = text_info.get("ocr_text") or ""
    out["text"]["has_text"] = bool(text_info.get("has_text"))
    # lock-friendly flag for downstream QA/locks
    locks["text_present"] = out["text"]["has_text"]
    if text_info.get("error"):
        problems.append({"stage": "ocr", "code": str(text_info.get("error")), "message": "ocr stage failed"})
    elif not out["text"]["has_text"]:
        problems.append({"stage": "ocr", "code": "no_text", "message": "no OCR text found"})
    # Layer 4: CLIP semantics
    clip_info = _clip_analyze(path, prompt)
    sem = out["semantics"]
    sem["clip_score"] = clip_info.get("clip_score")
    sem["clip_prompt"] = clip_info.get("clip_prompt")
    sem["clip_emb"] = clip_info.get("clip_emb")
    if clip_info.get("error"):
        problems.append({"stage": "clip", "code": str(clip_info.get("error")), "message": "clip stage failed"})
    # Layer 5: Qwen-VL captioning + tags + semantic score
    qwen_info = _qwen_vl_analyze(path, prompt)
    sem["caption"] = qwen_info.get("caption") or ""
    q_tags = qwen_info.get("tags") or []
    if not isinstance(q_tags, list):
        q_tags = []
    # Build combined tags from Qwen, YOLO labels, and OCR keywords
    tags: List[str] = []
    for t in q_tags:
        if isinstance(t, str):
            tt = t.strip()
            if len(tt) >= 2:
                tags.append(tt)
    # YOLO labels as tags
    for obj in out["objects"]["yolo"]:
        if isinstance(obj, dict):
            lbl = obj.get("label")
            if isinstance(lbl, str) and len(lbl) >= 2:
                tags.append(lbl.strip())
    # OCR-derived tokens (lightweight)
    ocr_txt = out["text"]["ocr_text"]
    if isinstance(ocr_txt, str) and ocr_txt.strip():
        for tok in ocr_txt.split():
            t = tok.strip(" ,.;:()[]{}\"'")
            if len(t) >= 3:
                tags.append(t)
    dedup_tags: List[str] = []
    seen_tags = set()
    for t in tags:
        if t not in seen_tags:
            seen_tags.add(t)
            dedup_tags.append(t)
    sem["tags"] = dedup_tags
    sem["qwen_vl_score"] = qwen_info.get("match_score")
    if qwen_info.get("error"):
        problems.append({"stage": "qwen_vl", "code": str(qwen_info.get("error")), "message": "vlm/qwen-vl stage failed"})
    # Layer 6: aesthetics using LAION aesthetic head
    _load_aesthetic_head()
    aest = _clip_aesthetic(path)
    out["aesthetics"] = aest
    aesthetic_score = float(aest.get("clip_aesthetic") or 0.0)
    if aesthetic_score <= 0.0:
        problems.append({"stage": "aesthetics", "code": "low_aesthetic", "message": "aesthetic score is zero or unavailable"})
    # Layer 7: aggregate scores
    clip_score = float(sem.get("clip_score") or 0.0)
    q_score = float(sem.get("qwen_vl_score") or 0.0)
    if clip_score or q_score:
        semantic_score = float(_clamp01(0.6 * clip_score + 0.4 * q_score))
    else:
        semantic_score = 0.0
    out["score"]["technical"] = tech_score
    out["score"]["semantic"] = semantic_score
    out["score"]["aesthetic"] = aesthetic_score
    overall = 0.4 * semantic_score + 0.3 * aesthetic_score + 0.3 * tech_score
    out["score"]["overall"] = float(_clamp01(overall))
    # ok flag: must have loaded and have a non-zero overall score
    out["ok"] = bool(out["ok"] and out["score"]["overall"] > 0.0)
    out["problems"] = problems
    return out


def analyze_image_regions(path: str, prompt: str | None, global_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-region image analysis built on top of the global analyzer.

    This function:
      - builds region proposals from YOLO and a coarse grid
      - crops each region
      - runs the full analyze_image() on each crop
      - aggregates region-level QA metrics (face_lock, hands_ok_ratio, etc.)
    """
    out: Dict[str, Any] = {
        "path": path,
        "prompt": prompt,
        "regions": [],
        "aggregates": {
            "face_lock": None,
            "id_lock": None,
            "hands_ok_ratio": None,
            "text_readable_lock": None,
            "background_quality": None,
        },
        "problems": [],
    }
    problems: List[Dict[str, Any]] = []
    if not isinstance(path, str) or not os.path.exists(path):
        problems.append(
            {"stage": "regions", "code": "missing_path", "message": "image path does not exist"}
        )
        out["problems"] = problems
        return out

    img = cv2.imread(path)
    if img is None:
        problems.append(
            {"stage": "regions", "code": "load_failed", "message": "could not read image for regions"}
        )
        out["problems"] = problems
        return out
    h, w = img.shape[:2]

    # ---- 1. Build region proposals ----
    proposals: List[Dict[str, Any]] = []

    # 1.1 YOLO-based proposals from global_result["objects"]["yolo"]
    objs = (global_result.get("objects") or {}).get("yolo") if isinstance(global_result, dict) else None
    if isinstance(objs, list):
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label") or obj.get("class_name") or ""
            try:
                conf_val = float(obj.get("conf") or 0.0)
            except Exception:
                conf_val = 0.0
            bbox = obj.get("bbox") or obj.get("xyxy")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x0, y0, x1, y1 = bbox
            # heuristic kind inference
            lab_l = str(label).lower()
            kind = "object"
            if "hand" in lab_l:
                kind = "hand"
            elif "face" in lab_l or "head" in lab_l:
                kind = "face"
            elif "person" in lab_l:
                kind = "face"
            proposals.append(
                {
                    "source": "yolo",
                    "kind": kind,
                    "label": label,
                    "conf": conf_val,
                    "bbox": [x0, y0, x1, y1],
                }
            )

    # 1.2 OCR-based proposals are left as a future extension once OCR boxes are available.

    # 1.3 Grid-based proposals: coarse coverage of the whole image
    grid_rows = 4
    grid_cols = 4
    cell_w = max(w // grid_cols, 1)
    cell_h = max(h // grid_rows, 1)
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            x0 = gx * cell_w
            y0 = gy * cell_h
            x1 = w if gx == grid_cols - 1 else (gx + 1) * cell_w
            y1 = h if gy == grid_rows - 1 else (gy + 1) * cell_h
            proposals.append(
                {
                    "source": "grid",
                    "kind": "region",
                    "label": "grid_patch",
                    "conf": 1.0,
                    "bbox": [x0, y0, x1, y1],
                }
            )

    regions: List[Dict[str, Any]] = []
    total_area = float(max(w * h, 1))
    tmp_root = os.path.join(UPLOAD_DIR, "tmp_regions")
    os.makedirs(tmp_root, exist_ok=True)

    # ---- 2. Crop and analyze each region ----
    for idx, prop in enumerate(proposals):
        bbox = prop.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x0_f, y0_f, x1_f, y1_f = bbox
        # clamp and normalize bbox
        try:
            x0 = int(max(0, min(x0_f, w - 1)))
            y0 = int(max(0, min(y0_f, h - 1)))
            x1 = int(max(0, min(x1_f, w)))
            y1 = int(max(0, min(y1_f, h)))
        except Exception as ex:
            problems.append(
                {"stage": "regions", "code": "bbox_cast_error", "message": str(ex)}
            )
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        # expand bbox by 10% margin for context
        bw = x1 - x0
        bh = y1 - y0
        dx = max(int(bw * 0.1), 1)
        dy = max(int(bh * 0.1), 1)
        x0 = max(0, x0 - dx)
        y0 = max(0, y0 - dy)
        x1 = min(w, x1 + dx)
        y1 = min(h, y1 + dy)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        region_area = float((x1 - x0) * (y1 - y0))
        size_frac = region_area / total_area
        region_id = f"r{idx:04d}-{prop.get('source','region')}"
        crop_name = f"{uuid.uuid4().hex}_region.png"
        crop_path = os.path.join(tmp_root, crop_name)
        try:
            ok = cv2.imwrite(crop_path, crop)
        except Exception as ex:
            problems.append(
                {
                    "stage": "regions",
                    "code": "crop_write_error",
                    "message": str(ex),
                    "region_id": region_id,
                }
            )
            continue
        if not ok:
            problems.append(
                {
                    "stage": "regions",
                    "code": "crop_write_failed",
                    "message": "cv2.imwrite returned False",
                    "region_id": region_id,
                }
            )
            continue
        try:
            region_analysis = analyze_image(crop_path, prompt)
        except Exception as ex:
            problems.append(
                {
                    "stage": "regions",
                    "code": "region_analyze_error",
                    "message": str(ex),
                    "region_id": region_id,
                }
            )
            continue
        region_entry = {
            "region_id": region_id,
            "source": prop.get("source"),
            "kind": prop.get("kind"),
            "label": prop.get("label"),
            "conf": prop.get("conf"),
            "bbox": [x0, y0, x1, y1],
            "size_frac": size_frac,
            "crop_path": crop_path,
            "analysis": region_analysis,
        }
        regions.append(region_entry)

    out["regions"] = regions

    # ---- 3. Aggregates ----
    agg = out["aggregates"]

    # Faces / identity
    face_regions = [r for r in regions if r.get("kind") == "face"]
    if face_regions:
        face_scores: List[float] = []
        for r in face_regions:
            sc = (((r.get("analysis") or {}).get("score") or {}).get("overall"))
            if isinstance(sc, (int, float)):
                face_scores.append(float(sc))
        if face_scores:
            face_lock = min(face_scores)
            agg["face_lock"] = face_lock
            agg["id_lock"] = face_lock

    # Hands quality ratio
    hand_regions = [r for r in regions if r.get("kind") == "hand"]
    if hand_regions:
        threshold = 0.7
        good = 0
        total = 0
        for r in hand_regions:
            sc = (((r.get("analysis") or {}).get("score") or {}).get("overall"))
            if isinstance(sc, (int, float)):
                total += 1
                if float(sc) >= threshold:
                    good += 1
        if total > 0:
            agg["hands_ok_ratio"] = float(good) / float(total)

    # Text readability lock: placeholder using grid regions for now
    text_has = bool((global_result.get("text") or {}).get("has_text")) if isinstance(global_result, dict) else False
    if text_has:
        # Use regions from any source where the regional OCR indicates text presence
        region_text_flags: List[bool] = []
        for r in regions:
            ra = r.get("analysis") or {}
            rt = (ra.get("text") or {}).get("has_text")
            if isinstance(rt, bool):
                region_text_flags.append(rt)
        if region_text_flags:
            readable_ratio = float(sum(1 for f in region_text_flags if f)) / float(len(region_text_flags))
            agg["text_readable_lock"] = readable_ratio

    # Background quality: mean over grid-based regions
    grid_regions = [r for r in regions if r.get("source") == "grid"]
    if grid_regions:
        vals: List[float] = []
        for r in grid_regions:
            sc = (((r.get("analysis") or {}).get("score") or {}).get("overall"))
            if isinstance(sc, (int, float)):
                vals.append(float(sc))
        if vals:
            agg["background_quality"] = float(sum(vals) / float(len(vals)))

    out["problems"] = problems
    return out


def _load_audio(path: str) -> Tuple[Optional[List[float]], int]:
    try:
        y, sr = sf.read(path)
        if getattr(y, "ndim", 1) > 1:
            # mixdown to mono
            try:
                y = y.mean(axis=1) if hasattr(y, "mean") else [float(sum(frames)/len(frames)) for frames in y]
            except Exception:
                y = y[:, 0]
        if hasattr(y, "tolist"):
            y = y.tolist()
        return y, int(sr)
    except Exception:
        return None, 0


def analyze_audio(path: str) -> Dict[str, Any]:
    """
    Returns { lufs: float|None, tempo_bpm: float|None, key: str|None, emotion: str|None }.
    Uses pyloudnorm/librosa if available. Graceful fallbacks otherwise.
    """
    out: Dict[str, Any] = {"lufs": None, "tempo_bpm": None, "key": None, "emotion": None, "genre": None, "pitch_mean_hz": None}
    if not isinstance(path, str) or not os.path.exists(path):
        return out
    y, sr = _load_audio(path)
    # LUFS
    try:
        if y and sr:
            meter = pyln.Meter(sr)
            out["lufs"] = float(meter.integrated_loudness(y))
    except Exception:
        pass
    # Tempo + Key + Emotion (heuristic)
    try:
        if y and sr:
            y_np = np.asarray(y, dtype=float)
            # tempo
            tempo, _ = librosa.beat.beat_track(y=y_np, sr=sr)
            out["tempo_bpm"] = float(tempo)
            # key via chroma
            chroma = librosa.feature.chroma_cqt(y=y_np, sr=sr)
            prof_maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            prof_min = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            chroma_mean = chroma.mean(axis=1)
            maj_scores = [np.correlate(np.roll(prof_maj, i), chroma_mean)[0] for i in range(12)]
            min_scores = [np.correlate(np.roll(prof_min, i), chroma_mean)[0] for i in range(12)]
            pitch_class = int(np.argmax(maj_scores + min_scores) % 12)
            mode = "maj" if max(maj_scores) >= max(min_scores) else "min"
            key_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            out["key"] = f"{key_names[pitch_class]} {mode}"
            # pitch mean
            f0 = librosa.yin(y_np, fmin=60, fmax=400, sr=sr)
            out["pitch_mean_hz"] = float(float(f0.mean()) if f0 is not None and hasattr(f0, "mean") else 0.0)
            # emotion heuristic: energy and spectral centroid
            sc = librosa.feature.spectral_centroid(y=y_np, sr=sr).mean()
            rmse = librosa.feature.rms(y=y_np).mean()
            if float(tempo or 0) > 140 or sc > 3000:
                out["emotion"] = "excited"
            elif float(tempo or 0) < 80 and rmse < 0.02:
                out["emotion"] = "calm"
            else:
                out["emotion"] = "neutral"
            # genre heuristic: coarse rule-based using tempo and spectral features
            zcr = librosa.feature.zero_crossing_rate(y_np).mean()
            if float(tempo or 0) > 150 and sc > 3000:
                out["genre"] = "electronic"
            elif float(tempo or 0) > 100 and zcr > 0.1:
                out["genre"] = "rock"
            elif float(tempo or 0) < 80 and sc < 2000 and rmse < 0.02:
                out["genre"] = "ambient"
            else:
                out["genre"] = "pop"
    except Exception:
        pass
    return out


def normalize_lufs(path: str, target_lufs: float) -> Optional[float]:
    """
    Loudness normalize in-place to target LUFS if pyloudnorm/soundfile available.
    Returns applied gain in dB or None.
    """
    try:
        y, sr = sf.read(path)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(y)
        if loud is None:
            return None
        loud_norm = pyln.normalize.loudness(y, loud, target_lufs)
        sf.write(path, loud_norm, sr)
        return float(target_lufs - float(loud))
    except Exception:
        return None


