from __future__ import annotations

import os
import json
import subprocess
import tempfile
import textwrap
import shutil
from typing import Any, Dict, Optional

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse


WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
EXEC_TIMEOUT_SEC = int(os.getenv("EXEC_TIMEOUT_SEC", "30"))
EXEC_MEMORY_MB = int(os.getenv("EXEC_MEMORY_MB", "2048"))
ALLOW_SHELL = os.getenv("ALLOW_SHELL", "false").lower() == "true"
SHELL_WHITELIST = set([s for s in (os.getenv("SHELL_WHITELIST") or "").split(",") if s])


app = FastAPI(title="Void Executor", version="0.1.0")


def within_workspace(path: str) -> str:
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not full.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise ValueError("path escapes workspace")
    return full


def run_subprocess(cmd: list[str], cwd: Optional[str] = None, timeout: int = EXEC_TIMEOUT_SEC) -> Dict[str, Any]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or WORKSPACE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return {"returncode": proc.returncode, "stdout": out, "stderr": err}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"returncode": -1, "stdout": "", "stderr": "timeout"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/run_python")
async def run_python(body: Dict[str, Any]):
    code = body.get("code") or ""
    if not code:
        return JSONResponse(status_code=400, content={"error": "missing code"})
    with tempfile.TemporaryDirectory(dir=WORKSPACE_DIR) as tmpd:
        path = os.path.join(tmpd, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        res = run_subprocess(["python", path], cwd=tmpd)
        return res


@app.post("/run_shell")
async def run_shell(body: Dict[str, Any]):
    if not ALLOW_SHELL:
        return JSONResponse(status_code=403, content={"error": "shell disabled"})
    cmd = body.get("cmd")
    if not cmd:
        return JSONResponse(status_code=400, content={"error": "missing cmd"})
    if SHELL_WHITELIST and (cmd.split(" ")[0] not in SHELL_WHITELIST):
        return JSONResponse(status_code=403, content={"error": "command not whitelisted"})
    res = run_subprocess(["bash", "-lc", cmd])
    return res


@app.post("/write_file")
async def write_file(body: Dict[str, Any]):
    rel = body.get("path")
    content = body.get("content", "")
    if not rel:
        return JSONResponse(status_code=400, content={"error": "missing path"})
    full = within_workspace(rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True}


@app.post("/read_file")
async def read_file(body: Dict[str, Any]):
    rel = body.get("path")
    if not rel:
        return JSONResponse(status_code=400, content={"error": "missing path"})
    full = within_workspace(rel)
    if not os.path.exists(full):
        return JSONResponse(status_code=404, content={"error": "not found"})
    with open(full, "r", encoding="utf-8") as f:
        return {"content": f.read()}


