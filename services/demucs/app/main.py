from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Any, Dict, List

import httpx
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI(title="Demucs v4 Service", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


def _write_temp_wav(b64: str | None, url: str | None) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    if isinstance(b64, str) and b64:
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return path
    if isinstance(url, str) and url:
        with httpx.Client() as c:
            r = c.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        return path
    raise RuntimeError("no input provided")


@app.post("/v1/audio/stems")
async def stems(body: Dict[str, Any]):
    mix_b64 = body.get("b64")
    mix_url = body.get("mix_wav") or body.get("url")
    stems = body.get("stems") or ["vocals", "drums", "bass", "other"]
    try:
        inpath = _write_temp_wav(mix_b64, mix_url)
        # Run demucs CLI (simple, robust); prefer GPU when available
        import subprocess, json as _json
        outdir = tempfile.mkdtemp(prefix="demucs_")
        import torch as _t
        device_arg = ["--device", ("cuda" if _t.cuda.is_available() else "cpu")]
        cmd = ["python", "-m", "demucs.separate", *device_arg, "--two-stems", "vocals", "-o", outdir, inpath]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            # fallback to 4-stem model
            cmd = ["python", "-m", "demucs.separate", *device_arg, "-o", outdir, inpath]
            subprocess.run(cmd, check=True)
        # Collect stems (base64 for portability)
        outs: Dict[str, str] = {}
        # Find first subdir with wavs
        subdirs = [os.path.join(outdir, d) for d in os.listdir(outdir)]
        subdirs = [d for d in subdirs if os.path.isdir(d)]
        target_dir = subdirs[0] if subdirs else outdir
        for fn in os.listdir(target_dir):
            if fn.lower().endswith(".wav"):
                stem_name = os.path.splitext(fn)[0].split(" ")[-1].lower()
                if (not stems) or (stem_name in stems):
                    p = os.path.join(target_dir, fn)
                    with open(p, "rb") as f:
                        outs[stem_name] = base64.b64encode(f.read()).decode("utf-8")
        return {"ok": True, "stems_b64": outs}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


