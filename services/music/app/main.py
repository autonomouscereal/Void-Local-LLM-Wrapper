from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI
import logging, sys, traceback

logger = logging.getLogger(__name__)

from .music_engine import generate_music
from void_envelopes import ToolEnvelope  # canonical envelope (shared)


# Primary music model directory (generic).
MUSIC_MODEL_DIR = os.getenv("MUSIC_MODEL_DIR", "/opt/models/music")

app = FastAPI(title="Music Service", version="0.3.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "music.log")
    _lvl = getattr(logging, (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_lvl,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(_log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("music logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.warning("music file logging disabled: %s", _ex, exc_info=True)


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
            "conversation_id": str
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
    The file is persisted under UPLOAD_DIR/artifacts/music/{conversation_id}/{artifact_id}.wav
    so the orchestrator/UI can serve it via /uploads/.
    """
    t0 = time.time()
    prompt = body.get("prompt") or ""
    seconds = int(body.get("seconds", 8))
    trace_id = body.get("trace_id")
    conversation_id = body.get("conversation_id")
    seed = body.get("seed")
    logger.info(
        f"music.generate.request prompt_preview={prompt} "
        f"prompt_len={len(prompt)} "
        f"conversation_id={conversation_id} trace_id={trace_id}"
    )
    if not prompt:
        # Surface validation errors as structured JSON with a synthetic stack for debugging.
        logger.warning(f"music.generate.validation_error missing_prompt conversation_id={conversation_id} trace_id={trace_id}")
        return ToolEnvelope.failure(
            code="ValidationError",
            message="prompt is required for music generation",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=400,
            details={
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "stack": "".join(traceback.format_stack()),
            },
        )
    try:
        logger.info(f"music.generate.invoke_engine conversation_id={conversation_id} trace_id={trace_id}")
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
        outdir_key = conversation_id
        outdir = os.path.join(upload_root, "artifacts", "music", outdir_key)
        os.makedirs(outdir, exist_ok=True)
        artifact_id = f"clip_{int(time.time())}"
        path = os.path.join(outdir, f"{artifact_id}.wav")
        with open(path, "wb") as f:
            f.write(wav)

        rel = os.path.relpath(path, upload_root).replace("\\", "/")
        relative_url = f"/uploads/{rel}"

        logger.info(
            f"music.generate.saved conversation_id={conversation_id} trace_id={trace_id} artifact_id={artifact_id} "
            f"path={path} url={relative_url} bytes={len(wav)} total_ms={int((time.time() - t0) * 1000.0)}"
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
            "trace_id": trace_id,
            "conversation_id": conversation_id,
        }
        return ToolEnvelope.success(result=result, trace_id=trace_id, conversation_id=conversation_id)
    except Exception as ex:  # surface full error as structured JSON
        logger.exception(f"music.generate error conversation_id={conversation_id} trace_id={trace_id} total_ms={int((time.time() - t0) * 1000.0)}")
        return ToolEnvelope.failure(
            code=(ex.__class__.__name__ or "InternalError"),
            message=str(ex),
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=500,
            details={
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "stack": traceback.format_exc(),
            },
        )

