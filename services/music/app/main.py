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

logger = logging.getLogger(__name__)

from .music_engine import generate_music
from orchestrator.app.routes.toolrun import ToolEnvelope  # canonical envelope


# Primary music model directory (generic).
MUSIC_MODEL_DIR = os.getenv("MUSIC_MODEL_DIR", "/opt/models/music")

app = FastAPI(title="Music Service", version="0.3.0")


@app.get("/healthz")
async def healthz():
    ok = os.path.isdir(MUSIC_MODEL_DIR) and bool(os.listdir(MUSIC_MODEL_DIR))
    logger.info("healthz: model_dir=%s ok=%s", MUSIC_MODEL_DIR, ok)
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
    cid = str(body.get("cid") or "music")
    trace_id = body.get("trace_id") if isinstance(body.get("trace_id"), str) else "music_unknown"
    seed = body.get("seed")
    logger.info(
        "music.generate.request prompt_preview=%r seconds=%s seed=%s cid=%r trace_id=%r",
        (prompt[:80] if isinstance(prompt, str) else ""),
        seconds,
        seed,
        cid,
        trace_id,
    )
    if not prompt:
        # Surface validation errors as structured JSON with a synthetic stack for debugging.
        logger.warning(
            "music.generate.validation_error missing_prompt cid=%r trace_id=%r",
            cid,
            trace_id,
        )
        return ToolEnvelope.failure(
            "ValidationError",
            "prompt is required for music generation",
            status=400,
            details={
                "cid": cid,
                "trace_id": trace_id,
                "stack": "".join(traceback.format_stack()),
            },
            request_id=trace_id,
        )
    try:
        wav = generate_music(
            prompt=prompt,
            seconds=seconds,
            seed=seed,
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
        outdir = os.path.join(upload_root, "artifacts", "music", cid)
        os.makedirs(outdir, exist_ok=True)
        artifact_id = f"clip_{int(time.time())}"
        path = os.path.join(outdir, f"{artifact_id}.wav")
        with open(path, "wb") as f:
            f.write(wav)

        rel = os.path.relpath(path, upload_root).replace("\\", "/")
        relative_url = f"/uploads/{rel}"

        logger.info(
            "music.generate.saved cid=%s artifact_id=%s path=%s url=%s bytes=%s",
            cid,
            artifact_id,
            path,
            relative_url,
            len(wav),
        )

        b64 = base64.b64encode(wav).decode("utf-8")
        # Provide multiple key aliases so different callers (RestMusicProvider,
        # legacy tools, etc.) can consume the same payload without changes.
        result = {
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
        return ToolEnvelope.success(result, request_id=trace_id)
    except Exception as ex:  # surface full error as structured JSON
        logger.exception("music.generate error cid=%r trace_id=%r", cid, trace_id)
        return ToolEnvelope.failure(
            ex.__class__.__name__ or "InternalError",
            str(ex),
            status=500,
            details={
                "cid": cid,
                "trace_id": trace_id,
                "stack": traceback.format_exc(),
            },
            request_id=trace_id,
        )

