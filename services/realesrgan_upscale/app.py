from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request

from void_envelopes import ToolEnvelope
from void_json import JSONParser

from PIL import Image


APP = FastAPI(title="Real-ESRGAN Upscale Service", version="1.0")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads").rstrip("/")
LOG_DIR = os.getenv("LOG_DIR", "/workspace/logs").rstrip("/")

UPSCALE_PORT = int(os.getenv("UPSCALE_PORT", "8099"))
UPSCALE_UVICORN_WORKERS = int(os.getenv("UPSCALE_UVICORN_WORKERS", "1"))

REALESRGAN_CODE_ROOT = os.getenv("REALESRGAN_CODE_ROOT", "/opt/realesrgan").rstrip("/")
REALESRGAN_WEIGHTS_SRC = os.getenv("REALESRGAN_WEIGHTS_SRC", "/opt/models/realesrgan/weights").rstrip("/")

DEFAULT_MODEL_NAME = os.getenv("REALESRGAN_MODEL_NAME", "RealESRGAN_x4plus")
DEFAULT_TILE = int(os.getenv("REALESRGAN_TILE", "0"))
DEFAULT_EXT = os.getenv("REALESRGAN_EXT", "png")
DEFAULT_SUFFIX = os.getenv("REALESRGAN_SUFFIX", "up")


def _mkdirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_DIR, "upscale"), exist_ok=True)


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


def _img_size(path: str) -> Tuple[int, int]:
    im = Image.open(path)
    w, h = im.size
    im.close()
    return int(w), int(h)


def _copy_weights_into_repo() -> Dict[str, Any]:
    dst = os.path.join(REALESRGAN_CODE_ROOT, "weights")
    os.makedirs(dst, exist_ok=True)
    src_glob = os.path.join(REALESRGAN_WEIGHTS_SRC, "*.pth")
    files = sorted([p for p in glob.glob(src_glob) if os.path.isfile(p) and os.path.getsize(p) > 0])
    copied: List[str] = []
    for p in files:
        out = os.path.join(dst, os.path.basename(p))
        shutil.copy2(p, out)
        copied.append(out)
    return {"src": REALESRGAN_WEIGHTS_SRC, "dst": dst, "count": len(copied), "files": [os.path.basename(p) for p in copied]}


def _expected_output_path(out_dir: str, input_path: str, suffix: str, ext: str) -> str:
    base = os.path.basename(input_path)
    stem = base.rsplit(".", 1)[0]
    return os.path.join(out_dir, stem + "_" + suffix + "." + ext)


def _plan_passes(input_long_edge: int, target_long_edge: int) -> List[float]:
    ratio = float(target_long_edge) / float(input_long_edge) if input_long_edge > 0 else 4.0
    passes: List[float] = []
    while ratio > 4.0:
        passes.append(4.0)
        ratio = ratio / 4.0
    if ratio < 1.0:
        ratio = 1.0
    passes.append(ratio)
    return passes


def _run_one_pass(
    *,
    in_path: str,
    out_dir: str,
    model_name: str,
    outscale: float,
    tile: int,
    fp32: bool,
    ext: str,
    suffix: str,
    face_enhance: bool,
) -> Dict[str, Any]:
    cmd = [
        "python3",
        os.path.join(REALESRGAN_CODE_ROOT, "inference_realesrgan.py"),
        "-i",
        in_path,
        "-o",
        out_dir,
        "-n",
        model_name,
        "--outscale",
        str(float(outscale)),
        "--tile",
        str(int(tile)),
        "--ext",
        str(ext),
        "--suffix",
        str(suffix),
    ]
    if fp32:
        cmd.append("--fp32")
    if face_enhance:
        cmd.append("--face_enhance")

    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=REALESRGAN_CODE_ROOT, capture_output=True, text=True)
    dur_ms = int((time.monotonic() - t0) * 1000.0)

    out_path = _expected_output_path(out_dir=out_dir, input_path=in_path, suffix=suffix, ext=ext)
    ok = bool(os.path.isfile(out_path) and os.path.getsize(out_path) > 0 and int(p.returncode) == 0)

    return {
        "ok": ok,
        "cmd": cmd,
        "duration_ms": dur_ms,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "returncode": int(p.returncode),
        "output_path": out_path,
    }


@APP.on_event("startup")
def _startup() -> None:
    _mkdirs()
    _copy_weights_into_repo()


@APP.get("/healthz")
def healthz() -> Dict[str, Any]:
    _mkdirs()
    weights_ok = bool(os.path.isdir(REALESRGAN_WEIGHTS_SRC) and os.listdir(REALESRGAN_WEIGHTS_SRC))
    return {
        "ok": True,
        "port": UPSCALE_PORT,
        "upload_dir": UPLOAD_DIR,
        "weights_src": REALESRGAN_WEIGHTS_SRC,
        "weights_src_ok": weights_ok,
        "code_root": REALESRGAN_CODE_ROOT,
    }


@APP.post("/v1/upscale/image")
async def upscale(request: Request):
    _mkdirs()
    raw = await request.body()
    body_txt = (raw or b"").decode("utf-8", errors="replace")
    payload = JSONParser().parse(body_txt or "{}", {})
    if not isinstance(payload, dict):
        payload = {}

    input_path = payload.get("input_path")
    if not isinstance(input_path, str) or not input_path.strip():
        return ToolEnvelope.failure("missing_input_path", "input_path is required", status=422)

    job_id = payload.get("job_id")
    job = str(job_id).strip() if isinstance(job_id, str) and job_id.strip() else uuid.uuid4().hex

    abs_in = _safe_join(UPLOAD_DIR, input_path.strip())
    if not os.path.isfile(abs_in):
        return ToolEnvelope.failure("input_not_found", f"input not found: {input_path.strip()}", status=404, details={"input_path": input_path.strip()})

    # Ensure weights are present in repo (inference script expects ./weights).
    weights_copy = _copy_weights_into_repo()
    if int(weights_copy.get("count") or 0) <= 0:
        return ToolEnvelope.failure("weights_missing", "Real-ESRGAN weights not found", status=500, details=weights_copy)

    model_name = payload.get("model_name") if isinstance(payload.get("model_name"), str) and str(payload.get("model_name")).strip() else DEFAULT_MODEL_NAME
    tile = int(payload.get("tile")) if isinstance(payload.get("tile"), int) else DEFAULT_TILE
    fp32 = bool(payload.get("fp32")) if isinstance(payload.get("fp32"), bool) else False
    ext = payload.get("ext") if isinstance(payload.get("ext"), str) and str(payload.get("ext")).strip() else DEFAULT_EXT
    suffix = payload.get("suffix") if isinstance(payload.get("suffix"), str) and str(payload.get("suffix")).strip() else DEFAULT_SUFFIX
    face_enhance = bool(payload.get("face_enhance")) if isinstance(payload.get("face_enhance"), bool) else False

    outscale = payload.get("outscale")
    target_long_edge = payload.get("target_long_edge")

    w, h = _img_size(abs_in)
    in_long = max(int(w), int(h))

    passes: List[float]
    if isinstance(outscale, (int, float)) and float(outscale) > 0.0:
        passes = [float(outscale)]
    else:
        tgt = int(target_long_edge) if isinstance(target_long_edge, int) and int(target_long_edge) > 0 else 3840
        passes = _plan_passes(input_long_edge=in_long, target_long_edge=tgt)

    out_dir = os.path.join(UPLOAD_DIR, "upscale", job)
    os.makedirs(out_dir, exist_ok=True)

    per_pass: List[Dict[str, Any]] = []
    cur_in = abs_in
    for idx, sc in enumerate(passes):
        pass_suffix = suffix + ("_" + str(idx + 1) if len(passes) > 1 else "")
        res = _run_one_pass(
            in_path=cur_in,
            out_dir=out_dir,
            model_name=str(model_name),
            outscale=float(sc),
            tile=int(tile),
            fp32=bool(fp32),
            ext=str(ext),
            suffix=str(pass_suffix),
            face_enhance=bool(face_enhance),
        )
        per_pass.append(res)
        if not bool(res.get("ok")):
            return ToolEnvelope.failure(
                "upscale_failed",
                "Real-ESRGAN pass failed",
                status=500,
                details={"pass_index": idx, "weights_copy": weights_copy, "passes": passes, "per_pass": per_pass},
            )
        cur_in = str(res.get("output_path") or "")
        if not cur_in:
            return ToolEnvelope.failure("upscale_failed", "Real-ESRGAN produced no output_path", status=500, details={"per_pass": per_pass})

    final_out = cur_in
    public_base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    out_rel = final_out.replace("/workspace", "") if final_out.startswith("/workspace/") else final_out
    out_url = (public_base + out_rel) if public_base and out_rel.startswith("/") else (out_rel if out_rel.startswith("/") else None)

    return ToolEnvelope.success(
        {
            "job_id": job,
            "input_path": abs_in,
            "input_size": {"w": w, "h": h},
            "model_name": model_name,
            "tile": int(tile),
            "fp32": bool(fp32),
            "face_enhance": bool(face_enhance),
            "passes": passes,
            "weights_copy": weights_copy,
            "per_pass": per_pass,
            "output_path": final_out,
            "output_url": out_url,
            "artifacts": [{"kind": "image", "path": out_rel, "view_url": out_rel}],
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(APP, host="0.0.0.0", port=UPSCALE_PORT, log_level="info", workers=UPSCALE_UVICORN_WORKERS)


