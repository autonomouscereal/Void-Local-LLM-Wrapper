from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore


def resolve_upload_video_path(upload_dir: str, src: str) -> str:
    """
    Resolve a video path to an absolute path under upload_dir.
    Accepts:
      - absolute /workspace/... paths
      - /uploads/... paths
      - relative paths under upload_dir
    """
    if not isinstance(src, str) or not src:
        return ""
    p = src
    if p.startswith("/uploads/"):
        p = "/workspace" + p
    if not p.startswith("/"):
        p = os.path.normpath(os.path.join(upload_dir, p))
    p = os.path.normpath(p)
    if not p.startswith(os.path.normpath(upload_dir)):
        return ""
    return p


def _ffprobe_avg_fps(src_abs: str) -> float:
    fps = 0.0
    p1 = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            src_abs,
        ],
        capture_output=True,
        text=True,
    )
    fr = (p1.stdout or "").strip()
    if "/" in fr:
        a, b = fr.split("/", 1)
        try:
            aa = float(a)
            bb = float(b)
            fps = aa / bb if bb != 0.0 else 0.0
        except Exception:
            fps = 0.0
    else:
        try:
            fps = float(fr) if fr else 0.0
        except Exception:
            fps = 0.0
    if fps <= 0.0:
        fps = 24.0
    return float(fps)


def _ffprobe_duration_s(src_abs: str) -> float:
    dur_s = 0.0
    p2 = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            src_abs,
        ],
        capture_output=True,
        text=True,
    )
    dd = (p2.stdout or "").strip()
    try:
        dur_s = float(dd) if dd else 0.0
    except Exception:
        dur_s = 0.0
    return float(dur_s)


def compute_temporal_clip_metrics(src_abs: str) -> Optional[Dict[str, Any]]:
    """
    Deterministic temporal/sharpness statistics for a clip:
      - fps, duration_s, frames
      - sharpness_median (Laplacian variance median across sampled frames)
      - delta_median (mean absdiff between sampled consecutive frames)
      - temporal_stability score in [0,1]
    """
    if not isinstance(src_abs, str) or not src_abs or (not os.path.isfile(src_abs)):
        return None

    fps = _ffprobe_avg_fps(src_abs)
    dur_s = _ffprobe_duration_s(src_abs)

    cap = cv2.VideoCapture(src_abs)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None

    sample_n = 12
    step = max(1, total // sample_n)
    sharp_vals = []
    delta_vals = []
    prev_gray = None
    idx = 0
    while idx < total and len(sharp_vals) < sample_n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray.astype("float32") / 255.0, cv2.CV_32F)
        sharp_vals.append(float(lap.var()))
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            delta_vals.append(float(diff.mean()) / 255.0)
        prev_gray = gray
        idx += step
    cap.release()

    sharp_med = float(np.median(np.array(sharp_vals, dtype=np.float32))) if sharp_vals else 0.0
    delta_med = float(np.median(np.array(delta_vals, dtype=np.float32))) if delta_vals else 0.0

    # Heuristic scoring (0..1)
    sharp_score = float(max(0.0, min(1.0, sharp_med / 0.02)))
    delta_score = float(1.0 - max(0.0, min(1.0, abs(delta_med - 0.06) / 0.10)))
    temporal_stability = float(max(0.0, min(1.0, 0.65 * sharp_score + 0.35 * delta_score)))

    return {
        "fps": float(fps),
        "duration_s": float(dur_s),
        "frames": int(total),
        "sharpness_median": float(sharp_med),
        "delta_median": float(delta_med),
        "temporal_stability": float(temporal_stability),
    }


