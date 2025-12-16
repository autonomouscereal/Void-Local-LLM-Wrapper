from __future__ import annotations

import base64
import io
import os
import logging
import sys
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI(title="OCR Service", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "ocr.log")
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
    logging.getLogger("ocr.logging").info("ocr logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("ocr.logging").warning("ocr file logging disabled: %s", _ex, exc_info=True)


def _b64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64)
    except Exception:
        return b""


def _image_ocr(img_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
        im = Image.open(io.BytesIO(img_bytes))
        txt = pytesseract.image_to_string(im)
        return (txt or "").strip()
    except Exception:
        return ""


def _pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            try:
                t = page.get_text() or ""
                if not t.strip():
                    # fallback OCR page render
                    pix = page.get_pixmap(dpi=180)
                    parts.append(_image_ocr(pix.tobytes("png")))
                else:
                    parts.append(t)
            except Exception:
                continue
        doc.close()
        return "\n".join([p for p in parts if p and p.strip()])
    except Exception:
        return ""


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(body: Dict[str, Any]):
    b64 = (body or {}).get("b64") or ""
    ext = (body or {}).get("ext") or ""
    if not b64:
        return JSONResponse(status_code=400, content={"error": "missing b64"})
    blob = _b64_to_bytes(b64)
    if not blob:
        return JSONResponse(status_code=400, content={"error": "invalid b64"})
    ext_l = str(ext or "").lower()
    text = ""
    if ext_l in (".pdf", "pdf"):
        text = _pdf_text(blob)
    else:
        text = _image_ocr(blob)
    return {"text": text or ""}


