from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging, sys, traceback

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s:%(name)s:%(message)s")

from .music_engine import generate_music


# Primary music model directory (generic).
MUSIC_MODEL_DIR = os.getenv("MUSIC_MODEL_DIR", "/opt/models/music")

app = FastAPI(title="Music Service", version="0.3.0")


@app.get("/healthz")
async def healthz():
    ok = os.path.isdir(MUSIC_MODEL_DIR) and bool(os.listdir(MUSIC_MODEL_DIR))
    return {"status": "ok" if ok else "missing_music_model", "model_dir": MUSIC_MODEL_DIR}


@app.post("/generate")
async def generate(body: Dict[str, Any]):
    """
    Simple synchronous music generation endpoint.

    Contract:
      - Request JSON:
          {
            "prompt": str,
            "seconds": int,
            "seed": Optional[int],
            "refs": Optional[list],
            "cid": Optional[str]
          }
      - Response JSON on success:
          {
            "wav_bytes_b64": str,
            "wav_base64": str,
            "audio_base64": str,
            "sample_rate": int,
            "artifact_id": str,
            "relative_url": str,
            "duration_s": int
          }
    The file is persisted under UPLOAD_DIR/artifacts/music/{cid}/{artifact_id}.wav
    so the orchestrator/UI can serve it via /uploads/.
    """
    prompt = body.get("prompt") or ""
    seconds = int(body.get("seconds", 8))
    if not prompt:
        # Surface validation errors as structured JSON with a synthetic stack for debugging.
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "ValidationError",
                    "message": "prompt is required for music generation",
                    "status": 400,
                    "stack": "".join(traceback.format_stack()),
                }
            },
        )
    try:
        wav = generate_music(
            prompt=prompt,
            seconds=seconds,
            seed=body.get("seed"),
            refs=body.get("refs"),
        )
        if not isinstance(wav, (bytes, bytearray)):
            # Expecting raw WAV bytes; if ndarray, encode to wav
            import soundfile as sf

            buf = io.BytesIO()
            arr = np.asarray(wav, dtype=np.float32)
            sf.write(buf, arr, 32000, format="WAV")
            wav = buf.getvalue()

        # Persist clip to disk so orchestrator/UI have a stable artifact URL.
        upload_root = os.getenv("UPLOAD_DIR", "/workspace/uploads")
        cid = body.get("cid") or body.get("trace_id") or "music"
        cid = str(cid)
        outdir = os.path.join(upload_root, "artifacts", "music", cid)
        os.makedirs(outdir, exist_ok=True)
        artifact_id = f"clip_{int(time.time())}"
        path = os.path.join(outdir, f"{artifact_id}.wav")
        with open(path, "wb") as f:
            f.write(wav)

        rel = os.path.relpath(path, upload_root).replace("\\", "/")
        relative_url = f"/uploads/{rel}"

        b64 = base64.b64encode(wav).decode("utf-8")
        # Provide multiple key aliases so different callers (RestMusicProvider,
        # legacy tools, etc.) can consume the same payload without changes.
        content = {
            "wav_bytes_b64": b64,
            "wav_base64": b64,
            "audio_base64": b64,
            "sample_rate": 32000,
            "artifact_id": artifact_id,
            "relative_url": relative_url,
            "duration_s": seconds,
            "path": path,
            "cid": cid,
        }
        return content
    except Exception as ex:  # surface full error as structured JSON
        logging.exception("music.generate error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ex.__class__.__name__ or "InternalError",
                    "message": str(ex),
                    "status": 500,
                    "stack": traceback.format_exc(),
                }
            },
        )

