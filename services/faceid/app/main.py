from __future__ import annotations

import base64
import io
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import FastAPI
import httpx  # type: ignore

from void_envelopes import ToolEnvelope
import insightface  # type: ignore

app = FastAPI(title="FaceID Service", version="0.1.0")

log = logging.getLogger(__name__)

# Cache multiple InsightFace FaceAnalysis apps keyed by (model_name, model_root, ctx_id)
_APPS: Dict[Tuple[str, str, int], Any] = {}


def _model_root() -> str:
    """
    InsightFace model root inside the faceid container.

    docker-compose mounts:
      - comfyui_models:/opt/models
    and this service expects InsightFace under /opt/models/insightface by default.
    """
    return (os.environ.get("INSIGHTFACE_MODEL_ROOT") or os.environ.get("INSIGHTFACE_HOME") or "/opt/models/insightface").strip() or "/opt/models/insightface"


def _required_antelope_files(model_root: str) -> List[str]:
    ant_dir = os.path.join(model_root, "models", "antelopev2")
    return [
        os.path.join(ant_dir, "glintr100.onnx"),
        os.path.join(ant_dir, "scrfd_10g_bnkps.onnx"),
    ]


def best_available_face_model(model_root: Optional[str] = None) -> str:
    """
    Choose the best face model available on disk (preference order):
    - antelopev2 (if required files exist under model_root/models/antelopev2)
    - buffalo_l (fallback)

    Override with env `FACEID_FACE_MODEL` (or legacy `LOCKS_FACE_MODEL`) to force a specific model.
    """
    forced = (os.environ.get("FACEID_FACE_MODEL") or os.environ.get("LOCKS_FACE_MODEL") or "").strip()
    if forced:
        return forced
    root = model_root or _model_root()
    req = _required_antelope_files(root)
    if all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in req):
        return "antelopev2"
    return "buffalo_l"


def _ctx_id() -> int:
    # 0 = first GPU, -1 = CPU
    raw = (os.environ.get("FACEID_CTX_ID") or os.environ.get("LOCKS_FACE_CTX_ID") or "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except Exception:
        return 0


def _get_app(model_name: Optional[str] = None, model_root: Optional[str] = None):
    name = (model_name or best_available_face_model(model_root)).strip() or "buffalo_l"
    root = model_root or _model_root()
    ctx = _ctx_id()
    key = (name, root, ctx)
    if key in _APPS:
        return _APPS[key]
    if insightface is None:
        return None
    try:
        app0 = insightface.app.FaceAnalysis(name=name, root=root)
        # Keep det_size aligned with orchestrator-side expectations.
        app0.prepare(ctx_id=ctx, det_size=(640, 640))
        _APPS[key] = app0
        log.info(f"faceid: loaded insightface model={name!r} root={root!r} ctx_id={ctx!r}")
        return app0
    except Exception:
        log.exception("faceid: failed to init FaceAnalysis model=%s root=%s ctx_id=%s", name, root, ctx)
        return None


def _bgr_from_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    # RGB -> BGR for InsightFace parity with OpenCV codepaths.
    return rgb[:, :, ::-1].copy()


async def _load_image_bytes(body: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """
    Load image bytes from one of:
      - image_path (preferred for shared volume /workspace/uploads/*)
      - image_url
      - image_b64 / image_base64 (raw image bytes, base64-encoded)
    Returns (bytes, meta) where meta echoes the chosen source.
    """
    meta: Dict[str, Any] = {}
    image_path = body.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        p = image_path.strip()
        meta["image_path"] = p
        try:
            with open(p, "rb") as f:
                return (f.read(), meta)
        except Exception as ex:
            meta["image_path_error"] = str(ex)
            return (b"", meta)

    image_b64 = body.get("image_b64") or body.get("image_base64")
    if isinstance(image_b64, str) and image_b64.strip():
        meta["image_b64"] = True
        b64 = image_b64.strip()
        # Allow "data:image/png;base64,..." inputs
        if "," in b64 and "base64" in b64[:64].lower():
            b64 = b64.split(",", 1)[1].strip()
        try:
            return (base64.b64decode(b64), meta)
        except Exception as ex:
            meta["image_b64_error"] = str(ex)
            return (b"", meta)

    image_url = body.get("image_url")
    if isinstance(image_url, str) and image_url.strip():
        url = image_url.strip()
        meta["image_url"] = url
        try:
            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                r = await client.get(url)
                if r.status_code < 200 or r.status_code >= 300:
                    meta["image_url_status"] = int(r.status_code)
                    return (b"", meta)
                return (r.content or b"", meta)
        except Exception as ex:
            meta["image_url_error"] = str(ex)
            return (b"", meta)

    return (b"", meta)


def _image_bytes_to_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    if not image_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(img)
        if getattr(rgb, "ndim", 0) != 3 or rgb.shape[-1] != 3:
            return None
        return _bgr_from_rgb_u8(rgb)
    except Exception:
        return None


@app.get("/healthz")
async def healthz():
    root = _model_root()
    model = best_available_face_model(root)
    return {"status": "ok", "model_root": root, "model": model, "insightface": bool(insightface)}


@app.post("/embed")
async def embed(body: Dict[str, Any]):
    trace_id = str(body.get("trace_id") or "").strip() if isinstance(body.get("trace_id"), (str, int)) else ""
    conversation_id = str(body.get("conversation_id") or "").strip() if isinstance(body.get("conversation_id"), (str, int)) else ""
    image_bytes, src_meta = await _load_image_bytes(body or {})
    if not image_bytes:
        return ToolEnvelope.failure(
            code="ValidationError",
            message="missing image input (provide image_path, image_url, or image_b64)",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=400,
            details={"trace_id": trace_id, "conversation_id": conversation_id, "source": src_meta, "stack": "".join(traceback.format_stack())},
        )
    img_bgr = _image_bytes_to_bgr(image_bytes)
    if img_bgr is None:
        return ToolEnvelope.failure(
            code="ValidationError",
            message="failed to decode image bytes",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=400,
            details={"trace_id": trace_id, "conversation_id": conversation_id, "source": src_meta, "bytes": len(image_bytes), "stack": "".join(traceback.format_stack())},
        )

    model_root = body.get("model_root") if isinstance(body.get("model_root"), str) else None
    model_name = body.get("model_name") if isinstance(body.get("model_name"), str) else None
    max_faces_raw = body.get("max_faces")
    try:
        max_faces = int(max_faces_raw) if isinstance(max_faces_raw, (int, float, str)) else 16
    except Exception:
        max_faces = 16
    if max_faces <= 0:
        max_faces = 0

    app0 = _get_app(model_name=model_name, model_root=model_root)
    if app0 is None:
        return ToolEnvelope.failure(
            code="ModelError",
            message="failed to initialize InsightFace FaceAnalysis",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=500,
            details={"trace_id": trace_id, "model_name": model_name, "model_root": model_root, "source": src_meta, "stack": traceback.format_exc()},
        )

    try:
        faces = app0.get(img_bgr)  # type: ignore[misc]
    except Exception:
        log.exception("faceid.embed: app.get failed model=%s source=%s", model_name, src_meta)
        return ToolEnvelope.failure(
            code="InferenceError",
            message="face detection/embedding failed",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=500,
            details={"trace_id": trace_id, "model_name": model_name, "source": src_meta, "stack": traceback.format_exc()},
        )

    out_faces: List[Dict[str, Any]] = []
    for f in faces or []:
        try:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            vec = emb.tolist() if hasattr(emb, "tolist") else (emb if isinstance(emb, list) else None)
            if not isinstance(vec, list) or not vec:
                continue
            det = getattr(f, "det_score", None)
            bbox = getattr(f, "bbox", None)
            bbox_list = bbox.tolist() if hasattr(bbox, "tolist") else (bbox if isinstance(bbox, list) else None)
            out_faces.append(
                {
                    "embedding": vec,
                    "det_score": float(det) if isinstance(det, (int, float)) else None,
                    "bbox": [int(x) for x in bbox_list] if isinstance(bbox_list, list) and len(bbox_list) == 4 else None,
                }
            )
        except Exception:
            continue

    ranked_faces: List[tuple[float, int, Dict[str, Any]]] = []
    for i, face_entry in enumerate(out_faces):
        det_score = face_entry.get("det_score")
        det_score_val = float(det_score) if isinstance(det_score, (int, float)) else 0.0
        ranked_faces.append((det_score_val, int(i), face_entry))
    ranked_faces.sort(reverse=True)
    out_faces = [triple[2] for triple in ranked_faces]
    if max_faces > 0:
        out_faces = out_faces[: int(max_faces)]

    best = out_faces[0]["embedding"] if out_faces else None
    result: Dict[str, Any] = {
        # New canonical payload
        "faces": out_faces,
    # Back-compat response shapes:
    # - `embedding` / `vec`: older orchestrator code paths
    # - `embeddings`: list form for older clients
        "embeddings": ([best] if isinstance(best, list) else []),
        "embedding": best,
        "vec": best,
        # Echo inputs / execution metadata for end-to-end debugging
        "model": (model_name or best_available_face_model(model_root)),
        "max_faces": max_faces,
        "source": src_meta,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
    }
    return ToolEnvelope.success(result=result, trace_id=trace_id, conversation_id=conversation_id)


