from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore


app = FastAPI(title="VisionRepair Service", version="0.1.0")

_detector: Optional[Any] = None


def get_detector():
    """
    Lazy-load YOLO detector. Uses a default YOLOv8/9/10-style model name.
    """
    global _detector
    if _detector is None:
        if YOLO is None:
        # Use a reasonably strong default; callers do not depend on exact variant.
        model_name = os.getenv("VISION_REPAIR_YOLO_MODEL", "yolov8x.pt")
        _detector = YOLO(model_name)
    return _detector


def _load_image(path: str) -> np.ndarray:
    if not path:
    img = cv2.imread(path)
    if img is None:
    return img


@app.get("/healthz")
async def healthz():
    try:
        det = get_detector()
        model_info = getattr(det, "name", "yolo")
    except Exception:
        model_info = None
    return {"status": "ok", "detector": model_info}


@app.post("/v1/image/analyze")
async def image_analyze(body: Dict[str, Any]):
    image_path = body.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        return JSONResponse(status_code=400, content={"error": "missing image_path"})
    img = _load_image(image_path)
    try:
        det = get_detector()
        results = det(img, verbose=False)
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": f"detector_error: {ex}"})
    faces: List[Dict[str, Any]] = []
    hands: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    quality: Dict[str, Any] = {}
    # Basic heuristic: return all detections as "objects"; more specific routing can be added later.
    if results:
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        # Try to resolve class names from model metadata when available.
        names_map: Dict[int, str] = {}
        try:
            det = get_detector()
            names_attr = getattr(det, "names", None) or getattr(getattr(det, "model", None), "names", None)
            if isinstance(names_attr, dict):
                names_map = {int(k): str(v) for k, v in names_attr.items()}
        except Exception:
            names_map = {}
        if boxes is not None:
            for b in boxes:
                try:
                    xyxy = b.xyxy[0].tolist()
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    cls_name = names_map.get(cls_id, str(cls_id))
                    obj = {
                        "bbox": xyxy,
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                    objects.append(obj)
                except Exception:
                    continue
    # Simple blur/quality metric: variance of Laplacian.
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality["sharpness"] = float(fm)
    except Exception:
        pass
    return {
        "faces": faces,
        "hands": hands,
        "objects": objects,
        "quality": quality,
    }


@app.post("/v1/image/repair")
async def image_repair(body: Dict[str, Any]):
    """
    Best-effort image repair entrypoint.

    For now, this implements a conservative behavior:
    - Reads the source image.
    - Optionally applies a mild denoise/sharpen filter.
    - Writes a new file and returns its path.
    The orchestrator decides which frames/images to send here and how to interpret regions/locks.
    """
    image_path = body.get("image_path")
    regions = body.get("regions") if isinstance(body.get("regions"), list) else []
    mode = body.get("mode")
    if not isinstance(image_path, str) or not image_path:
        return JSONResponse(status_code=400, content={"error": "missing image_path"})
    img = _load_image(image_path)
    h, w = img.shape[:2]
    # Very simple enhancement: Gaussian blur reduction + unsharp masking-like effect, applied per-region when provided.
    try:
        base_blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        base_sharpened = cv2.addWeighted(img, 1.5, base_blurred, -0.5, 0)
    except Exception:
        base_sharpened = img
    out_img = img.copy()
    if regions:
        for region in regions:
            if not isinstance(region, dict):
                continue
            bbox = region.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            try:
                x1_i = max(0, min(int(x1), w - 1))
                y1_i = max(0, min(int(y1), h - 1))
                x2_i = max(0, min(int(x2), w))
                y2_i = max(0, min(int(y2), h))
                if x2_i <= x1_i or y2_i <= y1_i:
                    continue
                roi = base_sharpened[y1_i:y2_i, x1_i:x2_i]
                out_img[y1_i:y2_i, x1_i:x2_i] = roi
            except Exception:
                continue
    else:
        out_img = base_sharpened
    # Write to a sibling file with suffix.
    base, ext = os.path.splitext(image_path)
    repaired_path = f"{base}.repaired{ext or '.png'}"
    ok = cv2.imwrite(repaired_path, out_img)
    if not ok:
        return JSONResponse(status_code=500, content={"error": "could_not_write_repaired_image"})
    return {
        "repaired_image_path": repaired_path,
        "regions": regions,
        "mode": mode,
    }


