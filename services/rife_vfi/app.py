from __future__ import annotations

from contextlib import asynccontextmanager
import os
import subprocess
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request

from void_envelopes import ToolEnvelope
from void_json import JSONParser


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # FastAPI deprecated @app.on_event("startup"); use lifespan instead.
    _mkdirs()
    yield


APP = FastAPI(title="RIFE VFI Service", version="1.0", lifespan=_lifespan)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads").rstrip("/")
LOG_DIR = os.getenv("LOG_DIR", "/workspace/logs").rstrip("/")

RIFE_VFI_PORT = int(os.getenv("RIFE_VFI_PORT", "8098"))
RIFE_CODE_ROOT = os.getenv("RIFE_CODE_ROOT", "/opt/practical_rife").rstrip("/")
RIFE_MODEL_DIR = os.getenv("RIFE_MODEL_DIR", "/opt/models/rife_vfi/train_log").rstrip("/")

DEFAULT_WORKERS = int(os.getenv("RIFE_UVICORN_WORKERS", "2"))


def _mkdirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_DIR, "vfi"), exist_ok=True)


def _safe_join(root: str, rel_or_abs: str) -> str:
    p = (rel_or_abs or "").strip()
    if not p:
        return os.path.join(root, "INVALID_PATH")
    if p.startswith("/"):
        full = os.path.normpath(p)
    else:
        full = os.path.normpath(os.path.join(root, p))
    root_norm = os.path.normpath(root)
    if not full.startswith(root_norm):
        return os.path.join(root_norm, "INVALID_PATH")
    return full


def _ffprobe_video_info(path: str) -> Tuple[float, float]:
    """
    Returns (fps, duration_s) best-effort.
    """
    fps = 0.0
    dur = 0.0

    # fps
    cmd_fps = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    p1 = subprocess.run(cmd_fps, capture_output=True, text=True)
    s = (p1.stdout or "").strip()
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            aa = float(a)
            bb = float(b)
            fps = aa / bb if bb != 0.0 else 0.0
        except Exception:
            fps = 0.0
    else:
        try:
            fps = float(s) if s else 0.0
        except Exception:
            fps = 0.0

    # duration
    cmd_dur = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    p2 = subprocess.run(cmd_dur, capture_output=True, text=True)
    d = (p2.stdout or "").strip()
    try:
        dur = float(d) if d else 0.0
    except Exception:
        dur = 0.0

    return float(fps), float(dur)


def _pick_multi(
    *,
    src_fps: float,
    target_fps: float,
    explicit_multi: Optional[int],
    explicit_exp: Optional[int],
) -> Dict[str, Any]:
    if isinstance(explicit_exp, int) and explicit_exp > 0:
        return {"mode": "exp", "value": int(explicit_exp)}
    if isinstance(explicit_multi, int) and explicit_multi > 1:
        return {"mode": "multi", "value": int(explicit_multi)}

    ratio = (float(target_fps) / float(src_fps)) if (src_fps and target_fps) else 2.0
    if ratio <= 2.0:
        return {"mode": "multi", "value": 2}
    if ratio <= 4.0:
        return {"mode": "multi", "value": 4}
    if ratio <= 8.0:
        return {"mode": "multi", "value": 8}
    return {"mode": "multi", "value": 16}


@APP.get("/healthz")
def healthz() -> Dict[str, Any]:
    _mkdirs()
    return {
        "ok": True,
        "port": RIFE_VFI_PORT,
        "upload_dir": UPLOAD_DIR,
        "rife_code_root": RIFE_CODE_ROOT,
        "rife_model_dir": RIFE_MODEL_DIR,
        "model_dir_exists": bool(os.path.isdir(RIFE_MODEL_DIR) and os.listdir(RIFE_MODEL_DIR)),
    }


@APP.post("/v1/vfi/interpolate")
async def interpolate(request: Request):
    _mkdirs()

    raw = await request.body()
    body_txt = (raw or b"").decode("utf-8", errors="replace")
    parser = JSONParser()
    data = parser.parse(body_txt or "{}", {})
    if not isinstance(data, dict):
        data = {}

    input_path = data.get("input_path")
    if not isinstance(input_path, str) or not input_path.strip():
        return ToolEnvelope.failure("missing_input_path", "input_path is required", status=422)

    target_fps_raw = data.get("target_fps")
    if not isinstance(target_fps_raw, (int, float)) or float(target_fps_raw) <= 0.0:
        return ToolEnvelope.failure("missing_target_fps", "target_fps must be > 0", status=422)
    target_fps = float(target_fps_raw)

    job_id = data.get("job_id")
    job = str(job_id).strip() if isinstance(job_id, str) and job_id.strip() else uuid.uuid4().hex

    rel_in = input_path.strip()
    abs_in = _safe_join(UPLOAD_DIR, rel_in)
    if not os.path.isfile(abs_in):
        return ToolEnvelope.failure("input_not_found", f"input not found: {rel_in}", status=404, details={"input_path": rel_in})

    out_dir = os.path.join(UPLOAD_DIR, "vfi", job)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.mp4")

    model_dir = data.get("model_dir")
    model = _safe_join("/opt/models", model_dir.strip()) if isinstance(model_dir, str) and model_dir.strip() else RIFE_MODEL_DIR

    explicit_multi = data.get("multi")
    explicit_exp = data.get("exp")
    uhd = data.get("uhd")
    scale = data.get("scale")
    ext = data.get("ext")
    montage = data.get("montage")

    src_fps_override = data.get("src_fps")
    if isinstance(src_fps_override, (int, float)) and float(src_fps_override) > 0.0:
        src_fps = float(src_fps_override)
        src_dur = 0.0
    else:
        src_fps, src_dur = _ffprobe_video_info(abs_in)
    if src_fps <= 0.0:
        src_fps = 24.0
    if target_fps < 1.0:
        return ToolEnvelope.failure("bad_target_fps", "target_fps must be >= 1", status=422)

    chosen = _pick_multi(
        src_fps=src_fps,
        target_fps=float(target_fps),
        explicit_multi=(int(explicit_multi) if isinstance(explicit_multi, int) else None),
        explicit_exp=(int(explicit_exp) if isinstance(explicit_exp, int) else None),
    )

    cmd = ["python3", os.path.join(RIFE_CODE_ROOT, "inference_video.py")]
    cmd.append("--video=" + abs_in)
    cmd.append("--output=" + out_path)
    cmd.append("--model=" + model)
    # Critical: set encoder fps to target_fps so duration remains consistent (no slow-mo).
    cmd.append("--fps=" + str(int(float(target_fps))))
    if chosen["mode"] == "exp":
        cmd.append("--exp=" + str(int(chosen["value"])))
    else:
        cmd.append("--multi=" + str(int(chosen["value"])))
    if isinstance(ext, str) and ext.strip():
        cmd.append("--ext=" + ext.strip())
    if isinstance(montage, bool) and montage:
        cmd.append("--montage")
    if isinstance(uhd, bool) and uhd:
        cmd.append("--UHD")
    if isinstance(scale, (int, float)) and float(scale) > 0.0:
        cmd.append("--scale=" + str(float(scale)))

    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=RIFE_CODE_ROOT, capture_output=True, text=True)
    dur_ms = int((time.monotonic() - t0) * 1000.0)

    ok = bool(os.path.isfile(out_path) and os.path.getsize(out_path) > 0 and int(p.returncode) == 0)
    out_fps, out_dur = _ffprobe_video_info(out_path) if ok else (0.0, 0.0)

    # Guard against accidental duration drift (interpolation should NOT extend).
    # If drift is detected and source duration is known, trim output to match.
    drift_s = float(out_dur - src_dur) if (src_dur and out_dur) else 0.0
    trimmed = False
    if ok and src_dur > 0.0 and out_dur > 0.0:
        tol = max(0.050, 1.5 / max(target_fps, 1.0))
        if drift_s > tol:
            fixed_path = os.path.join(out_dir, "output_trimmed.mp4")
            trim_cmd = ["ffmpeg", "-y", "-i", out_path, "-t", str(float(src_dur)), "-c", "copy", fixed_path]
            t1 = time.monotonic()
            ptrim = subprocess.run(trim_cmd, capture_output=True, text=True)
            trim_ms = int((time.monotonic() - t1) * 1000.0)
            if int(ptrim.returncode) == 0 and os.path.isfile(fixed_path) and os.path.getsize(fixed_path) > 0:
                out_path = fixed_path
                out_fps, out_dur = _ffprobe_video_info(out_path)
                drift_s = float(out_dur - src_dur) if (src_dur and out_dur) else drift_s
                trimmed = True
            else:
                ok = False
                return ToolEnvelope.failure(
                    "duration_drift",
                    "interpolation produced longer output and trim failed",
                    status=500,
                    details={
                        "src_duration_s": src_dur,
                        "out_duration_s": out_dur,
                        "drift_s": drift_s,
                        "trim_cmd": trim_cmd,
                        "trim_returncode": int(ptrim.returncode),
                        "trim_stderr": ptrim.stderr,
                        "duration_ms": dur_ms,
                        "trim_ms": trim_ms,
                    },
                )

    public_base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    out_rel = out_path.replace("/workspace", "") if out_path.startswith("/workspace/") else out_path
    out_url = (public_base + out_rel) if public_base and out_rel.startswith("/") else (out_rel if out_rel.startswith("/") else "")

    if not ok:
        return ToolEnvelope.failure(
            "vfi_failed",
            "interpolation failed or output missing",
            status=500,
            details={
                "job_id": job,
                "input_path": abs_in,
                "output_path": out_path,
                "src_fps": src_fps,
                "target_fps": float(target_fps),
                "chosen": chosen,
                "duration_ms": dur_ms,
                "cmd": cmd,
                "stdout": p.stdout,
                "stderr": p.stderr,
                "returncode": int(p.returncode),
            },
        )

    return ToolEnvelope.success(
        {
            "job_id": job,
            "input_path": abs_in,
            "output_path": out_path,
            "output_url": out_url or None,
            "src_fps": float(src_fps),
            "src_duration_s": float(src_dur),
            "target_fps": float(target_fps),
            "out_fps": float(out_fps),
            "out_duration_s": float(out_dur),
            "duration_drift_s": float(drift_s),
            "trimmed": bool(trimmed),
            "chosen": chosen,
            "duration_ms": int(dur_ms),
            "cmd": cmd,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "returncode": int(p.returncode),
        }
    )


if __name__ == "__main__":
    import uvicorn

    # If using reload/workers, uvicorn requires an import string ("module:app") not an app object.
    workers = max(1, int(DEFAULT_WORKERS or 1))
    uvicorn.run("app:APP", host="0.0.0.0", port=RIFE_VFI_PORT, log_level="info", workers=workers)


