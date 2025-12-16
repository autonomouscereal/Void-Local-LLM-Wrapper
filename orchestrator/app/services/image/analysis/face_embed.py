import os
import numpy as np
from PIL import Image
from typing import Optional, List
import logging

try:
    import insightface  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError:
    insightface = None
    FaceAnalysis = None

_INSIGHTFACE_ROOT = os.environ.get("INSIGHTFACE_HOME", "/models/insightface")
_APP = None

log = logging.getLogger(__name__)

def _get_app():
    global _APP
    if _APP is not None:
        return _APP
    if FaceAnalysis is None:
        return None
    try:
        # Ensure root exists or fallback
        if not os.path.exists(_INSIGHTFACE_ROOT):
            # If running locally without docker volume, might fail or default to ~/.insightface
            try:
                os.makedirs(_INSIGHTFACE_ROOT, exist_ok=True)
            except Exception:
                log.debug("face_embed: failed to create INSIGHTFACE_HOME=%s", _INSIGHTFACE_ROOT, exc_info=True)
        app = FaceAnalysis(name="buffalo_l", root=_INSIGHTFACE_ROOT)
        app.prepare(ctx_id=0, det_size=(640, 640))
        _APP = app
        return _APP
    except Exception:
        return None

def compute_face_embedding_from_pil(img: Image.Image) -> Optional[List[float]]:
    """
    Compute a 512-D face embedding from a PIL image using InsightFace (ArcFace).
    Returns the embedding of the highest-scoring face found, or None.
    """
    app = _get_app()
    if app is None:
        return None
    
    # Convert PIL to numpy (RGB) - InsightFace expects BGR or RGB? 
    # InsightFace typically expects BGR if using cv2.imread, but FaceAnalysis.get() handling might vary.
    # Standard practice: Convert to numpy array.
    # "The input image should be a numpy array with shape (H, W, 3) and RGB format." - usually it handles RGB if not using cv2 directly.
    # However, actually InsightFace models often trained on BGR. Let's convert to BGR to be safe if we were using cv2, 
    # but app.get() usually accepts RGB if using standard insightface.app.
    # Let's stick to standard numpy conversion.
    np_img = np.array(img)
    
    # If image is RGBA, drop alpha
    if np_img.shape[-1] == 4:
        np_img = np_img[..., :3]
        
    # InsightFace expects BGR usually when not specified, but let's check docs or assume standard behavior.
    # Most insightface examples use cv2.imread -> BGR.
    # So let's convert RGB (PIL) to BGR.
    np_img_bgr = np_img[:, :, ::-1]

    faces = app.get(np_img_bgr)
    if not faces:
        return None

    # Take the highest-score face
    face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    emb = face.normed_embedding  # 512-D np.array, L2-normalized
    return emb.astype("float32").tolist()

