from __future__ import annotations

import math
import os
from typing import List

import logging

import numpy as np  # type: ignore
import torch  # type: ignore
import laion_clap  # type: ignore

from ...analysis.media import _load_audio

log = logging.getLogger(__name__)


class MusicEvalError(Exception):
    """
    Non-fatal error for music evaluation paths (e.g., CLAP failures).
    """


def cos_sim(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return float(num) / float(da * db + 1e-9)


_MODELS_ROOT = os.environ.get("FILM2_MODELS", "/opt/models")
CLAP_MODEL_DIR = os.path.join(_MODELS_ROOT, "clap")

if not (os.path.isdir(CLAP_MODEL_DIR) and os.listdir(CLAP_MODEL_DIR)):
    CLAP_MODEL_DIR_MISSING = True
else:
    CLAP_MODEL_DIR_MISSING = False

_CLAP_MODEL = None
_CLAP_STATE_PATH = os.path.join(CLAP_MODEL_DIR, "CLAP_HTSAT_base.pt")
if not CLAP_MODEL_DIR_MISSING:
    _CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    if os.path.exists(_CLAP_STATE_PATH) and os.path.getsize(_CLAP_STATE_PATH) > 0:
        state = torch.load(_CLAP_STATE_PATH, map_location="cpu")
        _CLAP_MODEL.load_state_dict(state)
    _CLAP_MODEL.eval()


def embed_music_clap(path: str) -> List[float]:
    if CLAP_MODEL_DIR_MISSING or _CLAP_MODEL is None:
        # Never raise: this is QA-only. Degrade to empty embedding and let callers score as 0.
        log.error("CLAP model dir missing/unavailable; returning empty embedding path=%r", path)
        return []
    y, sr = _load_audio(path)
    if not y or not sr:
        return []
    with torch.no_grad():
        emb = _CLAP_MODEL.get_audio_embedding_from_filelist([path])
    if emb is None:
        return []
    try:
        if hasattr(emb, "__len__") and len(emb) == 0:
            return []
    except Exception:
        log.error("CLAP embedding object does not support len(); cannot validate empty embedding")
    e0 = emb[0]
    if isinstance(e0, torch.Tensor):
        arr = e0.detach().cpu().numpy().astype("float32")
    elif isinstance(e0, np.ndarray):
        arr = e0.astype("float32")
    else:
        arr = np.asarray(e0, dtype="float32")
    return arr.tolist()




