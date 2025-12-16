from __future__ import annotations
# HARD BAN (permanent): No Pydantic, no SQLAlchemy/ORM, no CSV/Parquet. JSON/NDJSON only.

import json
import os
import hashlib
import time
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
try:
    from db.core import get_pool, execute as db_execute, fetchrow as db_fetchrow
except Exception:
    async def get_pool():
        return None
    async def db_execute(*args, **kwargs):
        return ""
    async def db_fetchrow(*args, **kwargs):
        return None


TEACHER_VERSION = "1.0.0"
UPLOAD_ROOT = "/workspace/uploads"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "teacher.log")
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
    logging.getLogger("teacher.logging").info("teacher logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("teacher.logging").warning("teacher file logging disabled: %s", _ex, exc_info=True)


app = FastAPI(title="Wrapper-as-Teacher", version=TEACHER_VERSION)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _uri_from_path(path: str) -> str:
    rel = os.path.relpath(path, UPLOAD_ROOT).replace("\\", "/")
    return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if PUBLIC_BASE_URL else f"/uploads/{rel}"


def _write_text(path: str, text: str) -> Dict[str, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return {"uri": _uri_from_path(path), "hash": f"sha256:{_sha256_bytes(text.encode('utf-8'))}"}


def _write_json(path: str, obj: Any) -> Dict[str, str]:
    return _write_text(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _append_ndjson(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "ab") as f:
        f.write(line.encode("utf-8"))


def _xx64(*parts: str) -> int:
    try:
        import xxhash
        h = xxhash.xxh64()
        for p in parts:
            h.update(str(p))
        return h.intdigest() & ((1 << 64) - 1)
    except Exception:
        return int(_sha256_str("|".join(parts))[:16], 16)


_TRACE_MODE: Dict[str, Dict[str, Any]] = {"enabled": True, "capture_bodies": True, "redact_outputs": False, "label": "exp_default"}
_BUFFER: Dict[str, List[Dict[str, Any]]] = {}
_CONFIG: Dict[str, Any] = {
    "buffer_max": 1000,
    "max_text_capture_chars": 4000,
    "max_artifact_list": 64,
    "max_tools_per_trace": 64,
}


def require_keys(obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k not in obj:
            return k
    return None


def ensure_type(obj: Dict[str, Any], key: str, typ) -> bool:
    return isinstance(obj.get(key), typ)


def non_empty_str(obj: Dict[str, Any], key: str) -> bool:
    v = obj.get(key)
    return isinstance(v, str) and len(v.strip()) > 0


def uri_with_hash(obj: Dict[str, Any], uri_key: str, hash_key: str) -> bool:
    return non_empty_str(obj, uri_key) and non_empty_str(obj, hash_key)


@app.post("/teacher/trace.enable")
async def trace_enable(body: Dict[str, Any]):
    enabled = bool(body.get("enabled", True))
    capture_bodies = bool(body.get("capture_bodies", True))
    redact_outputs = bool(body.get("redact_outputs", False))
    label = str(body.get("label", "default"))
    _TRACE_MODE.update({"enabled": enabled, "capture_bodies": capture_bodies, "redact_outputs": redact_outputs, "label": label})
    _BUFFER.setdefault(label, [])
    return {"ok": True, "trace_mode": {"enabled": enabled, "label": label}}


@app.post("/teacher/trace.append")
async def trace_append(body: Dict[str, Any]):
    # Backpressure
    label = str(body.get("label") or _TRACE_MODE.get("label") or "default")
    _BUFFER.setdefault(label, [])
    if len(_BUFFER[label]) >= int(_CONFIG.get("buffer_max", 1000)):
        return JSONResponse(status_code=503, content={"error": "busy", "detail": f"buffer full; try flush; buffer_max={_CONFIG.get('buffer_max')}"})
    # Validate minimal keys
    req = body.get("request") or {}
    miss = require_keys(req, ["messages"]) if isinstance(req, dict) else "request"
    if miss:
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": f"Missing key: {miss}"})
    messages = req.get("messages") or []
    mhash = _sha256_str(json.dumps(messages, sort_keys=True, separators=(",", ":")))
    ts = _now_iso()
    trace_seed = int(body.get("seed", 0) or 0)
    workspace = str(body.get("workspace", "default"))
    trace_id = body.get("trace_id") or ("tt_" + _sha256_str("|".join([ts, workspace, str(trace_seed), mhash]))[:16])
    # Seed derivations
    seed_router = _xx64("router", trace_id, str(trace_seed))
    tools_allowed = (req.get("tools_allowed") or [])
    # Normalize tool calls order by name
    tool_calls = body.get("tool_calls") or []
    tool_calls = sorted(tool_calls, key=lambda x: x.get("name") or "")[: int(_CONFIG.get("max_tools_per_trace", 64))]
    seeds_map = {"master": trace_seed, "router": seed_router, "tools": {}}
    for tc in tool_calls:
        name = tc.get("name") or "tool"
        seeds_map["tools"][name] = _xx64("tool", name, trace_id, str(trace_seed))
        tc.setdefault("seed", seeds_map["tools"][name])
    # Trim response text
    resp = body.get("response") or {}
    max_chars = int(_CONFIG.get("max_text_capture_chars", 4000) or 4000)
    if isinstance(resp.get("text"), str) and len(resp.get("text")) > max_chars:
        resp["text"] = resp.get("text")[:max_chars]
    # Trim artifacts list
    if isinstance(resp.get("artifacts"), list):
        resp["artifacts"] = resp.get("artifacts")[: int(_CONFIG.get("max_artifact_list", 64))]
    # Stable alt models ordering
    routing = body.get("routing") or {}
    if isinstance(routing.get("alt_models"), list):
        routing["alt_models"] = sorted(routing.get("alt_models"))
    # Assemble trace line
    line = {
        "ts": ts,
        "trace_id": trace_id,
        "seed": trace_seed,
        "workspace": workspace,
        "mode": body.get("mode") or "general",
        "request": req,
        "context": body.get("context") or {},
        "routing": routing,
        "tool_calls": tool_calls,
        "response": resp,
        "metrics": body.get("metrics") or {},
        "seeds": seeds_map,
        "env": body.get("env") or {},
        "privacy": body.get("privacy") or {"vault_refs": 0, "secrets_in": 0, "secrets_out": 0},
    }
    _BUFFER[label].append(line)
    # Optional DB: attach trace to run if present (by trace_id)
    try:
        pool = await get_pool()
        if pool is not None:
            rr = await db_fetchrow("SELECT id FROM run WHERE trace_id=$1", line.get("trace_id"))
            if rr:
                await db_execute("INSERT INTO teacher_trace(run_id, trace_line) VALUES($1,$2)", int(rr[0]), line)
    except Exception as ex:
        # DB persistence is best-effort; do not fail request but surface telemetry.
        logging.warning("teacher.trace_db_write_failed: %s", ex, exc_info=True)
    return {"ok": True, "buffer_size": len(_BUFFER[label]), "label": label, "trace_id": trace_id}


def _teacher_dir(label: str) -> str:
    return os.path.join(UPLOAD_ROOT, "teacher", label)


def _teacher_paths(label: str) -> Dict[str, str]:
    base = _teacher_dir(label)
    return {
        "traces": os.path.join(base, "traces.jsonl"),
        "sft": os.path.join(base, "sft.jsonl"),
        "dpo": os.path.join(base, "dpo.jsonl"),
        "toolpolicy": os.path.join(base, "toolpolicy.jsonl"),
        "run_record": os.path.join(base, "run_record.json"),
        "cache_prompt": os.path.join(base, "cache", "prompt_index.json"),
        "cache_video": os.path.join(base, "cache", "video_policy.json"),
    }


def _prompt_hash(messages: List[Dict[str, Any]]) -> str:
    m = json.dumps(messages or [], sort_keys=True, separators=(",", ":"))
    return f"sha256:{_sha256_bytes(m.encode('utf-8'))}"


@app.post("/teacher/trace.flush")
async def trace_flush(body: Dict[str, Any]):
    label = str(body.get("label", _TRACE_MODE.get("label") or "default"))
    buf = _BUFFER.get(label) or []
    paths = _teacher_paths(label)
    counts = {"traces": 0, "sft": 0, "dpo": 0, "toolpolicy": 0}
    if not buf:
        return JSONResponse(status_code=409, content={"error": "conflict", "detail": "Trace buffer empty"})
    # Append traces (batch)
    if buf:
        chunk = "".join([json.dumps(t, ensure_ascii=False, separators=(",", ":")) + "\n" for t in buf])
        _write_text(paths["traces"], (open(paths["traces"], "r", encoding="utf-8").read() if os.path.exists(paths["traces"]) else "") + chunk)
        counts["traces"] = len(buf)
    # Derive SFT/DPO/ToolPolicy deterministically
    # SFT: text/general path
    sft_invalid = 0
    sft_lines: List[str] = []
    tp_lines: List[str] = []
    dpo_lines: List[str] = []
    for t in buf:
        mode = (t.get("mode") or "general")
        sft = None
        if mode in ("general", "research", "text"):
            sft = {"input": {"pack_hash": t.get("context", {}).get("pack_hash"), "messages": (t.get("request", {}) or {}).get("messages", [])}, "output": {"text": (t.get("response", {}) or {}).get("text", "")}, "meta": {"mode": mode, "seed": t.get("seed"), "trace_id": t.get("trace_id"), "time": t.get("ts")}}
            if not sft["output"]["text"]:
                sft = None
        elif mode == "coding":
            # Expect minimal coding structure; if missing, skip
            sft = None
        elif mode == "image":
            sft = None
        elif mode == "audio":
            sft = None
        elif mode == "video":
            sft = None
        if sft is not None:
            sft_lines.append(json.dumps(sft, ensure_ascii=False, separators=(",", ":")))
            counts["sft"] += 1
        else:
            sft_invalid += 1
        # ToolPolicy
        tp = {"context_hash": t.get("context", {}).get("pack_hash"), "tools_considered": (t.get("request", {}) or {}).get("tools_allowed", []), "decision": {"use": [c.get("name") for c in t.get("tool_calls", [])], "skip": list(set(((t.get("request", {}) or {}).get("tools_allowed", [])) ) - set([c.get("name") for c in t.get("tool_calls", [])]))}, "params": {c.get("name"): (c.get("args") or {}) for c in t.get("tool_calls", [])}, "outcome": {"success": True}, "meta": {"mode": mode, "seed": t.get("seed"), "trace_id": t.get("trace_id")}}
        tp_lines.append(json.dumps(tp, ensure_ascii=False, separators=(",", ":")))
        counts["toolpolicy"] += 1
    # DPO: pair consecutive traces with same prompt hash
    prompt_index = {}
    if os.path.exists(paths["cache_prompt"]):
        try:
            with open(paths["cache_prompt"], "r", encoding="utf-8") as f:
                prompt_index = json.load(f)
        except Exception:
            prompt_index = {}
    for t in buf:
        mhash = _prompt_hash((t.get("request") or {}).get("messages", []))
        prev = (prompt_index.get(mhash) or [])
        if prev:
            other_id = prev[-1]
            chosen = t.get("response", {}).get("text", "")
            rejected = ""
            dpo = {"prompt_hash": mhash, "chosen": {"text": chosen, "tool_seq": [c.get("name") for c in t.get("tool_calls", [])]}, "rejected": {"text": rejected, "tool_seq": []}, "meta": {"mode": (t.get("mode") or "general"), "seed": t.get("seed"), "trace_ids": [other_id, t.get("trace_id")]}}
            dpo_lines.append(json.dumps(dpo, ensure_ascii=False, separators=(",", ":")))
            counts["dpo"] += 1
        prev = (prompt_index.get(mhash) or []) + [t.get("trace_id")]
        prompt_index[mhash] = prev
    _write_json(paths["cache_prompt"], prompt_index)
    # Batch write SFT/ToolPolicy/DPO
    if sft_lines:
        _write_text(paths["sft"], (open(paths["sft"], "r", encoding="utf-8").read() if os.path.exists(paths["sft"]) else "") + "\n".join(sft_lines) + "\n")
    if tp_lines:
        _write_text(paths["toolpolicy"], (open(paths["toolpolicy"], "r", encoding="utf-8").read() if os.path.exists(paths["toolpolicy"]) else "") + "\n".join(tp_lines) + "\n")
    if dpo_lines:
        _write_text(paths["dpo"], (open(paths["dpo"], "r", encoding="utf-8").read() if os.path.exists(paths["dpo"]) else "") + "\n".join(dpo_lines) + "\n")
    # Optional DB: materialize distill rows linked to run. Failures here should not
    # break trace flushing, but we log them instead of swallowing silently.
    try:
        pool = await get_pool()
        if pool is not None:
            for t in buf:
                rr = await db_fetchrow("SELECT id FROM run WHERE trace_id=$1", t.get("trace_id"))
                if not rr:
                    continue
                rid = int(rr[0])
                t_mode = (t.get("mode") or "general")
                # Build an SFT sample directly from this trace row (avoid coupling to sft_lines order)
                if t_mode in ("general", "research", "text"):
                    sft_sample = {
                        "input": {
                            "pack_hash": (t.get("context") or {}).get("pack_hash"),
                            "messages": (t.get("request") or {}).get("messages", []),
                        },
                        "output": {"text": (t.get("response") or {}).get("text", "")},
                        "meta": {"mode": t_mode, "seed": t.get("seed"), "trace_id": t.get("trace_id"), "time": t.get("ts")},
                    }
                    if sft_sample["output"]["text"]:
                        await db_execute("INSERT INTO distill_sft(run_id, sample_json) VALUES($1,$2)", rid, sft_sample)
            # ToolPolicy/DPO are aggregate lines; insert them once per available run id
            # (safe no-op if arrays are empty)
            for t in buf:
                rr = await db_fetchrow("SELECT id FROM run WHERE trace_id=$1", t.get("trace_id"))
                if not rr:
                    continue
                rid = int(rr[0])
                from void_json.json_parser import JSONParser

                parser = JSONParser()
                for tp in tp_lines:
                    try:
                        obj_tp = parser.parse(tp, {}) or {}
                        if isinstance(obj_tp, dict):
                            await db_execute("INSERT INTO distill_toolpolicy(run_id, policy_json) VALUES($1,$2)", rid, obj_tp)
                    except Exception as ex:
                        logging.warning("teacher.distill_toolpolicy_db_write_failed: %s", ex, exc_info=True)
                        continue
                for dp in dpo_lines:
                    try:
                        obj_dp = parser.parse(dp, {}) or {}
                        if isinstance(obj_dp, dict):
                            await db_execute("INSERT INTO distill_dpo(run_id, pair_json) VALUES($1,$2)", rid, obj_dp)
                    except Exception as ex:
                        logging.warning("teacher.distill_dpo_db_write_failed: %s", ex, exc_info=True)
                        continue
    except Exception as ex:
        logging.error("teacher.distill_db_materialize_failed: %s", ex, exc_info=True)
    # run record
    inputs_hash = _sha256_str("".join([x.get("trace_id") or "" for x in buf]))
    # include wrapper config hash if present
    cfg_hash = None
    cfg_path = os.path.join("/workspace", "configs", "wrapper_config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "rb") as cf:
                cfg_hash = f"sha256:{_sha256_bytes(cf.read())}"
    except Exception:
        cfg_hash = None
    rr = {"ts": _now_iso(), "label": label, "inputs_hash": f"sha256:{inputs_hash}", "counts": counts, "timings_ms": {"derive": 0, "write": 0}, "teacher_version": TEACHER_VERSION}
    if cfg_hash:
        rr["config_hash"] = cfg_hash
    _write_json(paths["run_record"], rr)
    # index.json
    idx = {"traces": counts["traces"], "sft": counts["sft"], "dpo": counts["dpo"], "toolpolicy": counts["toolpolicy"], "last_ts": _now_iso()}
    _write_json(os.path.join(os.path.dirname(paths["traces"]), "index.json"), idx)
    batch_id = f"{_now_iso()}-{label}"
    # clear buffer
    _BUFFER[label] = []
    return {"batch_id": batch_id, "paths": {"traces_jsonl": _uri_from_path(paths["traces"]), "sft_jsonl": _uri_from_path(paths["sft"]), "dpo_jsonl": _uri_from_path(paths["dpo"]), "toolpolicy_jsonl": _uri_from_path(paths["toolpolicy"]), "run_record": _uri_from_path(paths["run_record"])}, "counts": counts}


@app.post("/teacher/gc")
async def teacher_gc(body: Dict[str, Any]):
    label = str(body.get("label", _TRACE_MODE.get("label") or "default"))
    keep_last = int(body.get("keep_last_batches", 3) or 3)
    min_age_days = int(body.get("min_age_days", 7) or 7)
    # Minimal placeholder GC: create archive dir and pointer; actual date-based move is implementation-specific
    base = _teacher_dir(label)
    arch = os.path.join(base, "archive", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    os.makedirs(arch, exist_ok=True)
    _write_json(os.path.join(arch, "POINTER.json"), {"note": "manual GC placeholder", "keep_last": keep_last, "min_age_days": min_age_days})
    return {"ok": True, "archive": _uri_from_path(arch)}


@app.post("/teacher/trainset.ingest")
async def teacher_ingest(body: Dict[str, Any]):
    seed = int(body.get("seed", 0) or 0)
    inputs = body.get("inputs") or {}
    shard_size = int(body.get("shard_size", 5000) or 5000)
    out_dir_uri = str(body.get("output_dir") or "")
    if not out_dir_uri:
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "output_dir required"})

    def _path_from_uri(u: str) -> str:
        if not u:
            return ""
        if u.startswith("/uploads/"):
            return os.path.join(UPLOAD_ROOT, u[len("/uploads/"):])
        if PUBLIC_BASE_URL and u.startswith(PUBLIC_BASE_URL.rstrip("/") + "/uploads/"):
            return os.path.join(UPLOAD_ROOT, u.split("/uploads/", 1)[-1])
        if u.startswith("uri:///uploads/"):
            return os.path.join(UPLOAD_ROOT, u.split("/uploads/", 1)[-1])
        return u

    out_dir = _path_from_uri(out_dir_uri)
    os.makedirs(out_dir, exist_ok=True)

    shards_meta: List[Dict[str, Any]] = []
    for key in ("sft", "dpo", "toolpolicy"):
        src = str((inputs or {}).get(key) or "")
        if not src:
            continue
        path = _path_from_uri(src)
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        except Exception as ex:
            return JSONResponse(status_code=400, content={"error": "bad_request", "detail": f"cannot read {key}: {ex}"})
        idx = 1
        for i in range(0, len(lines), shard_size):
            shard = lines[i: i + shard_size]
            shard_name = f"{key}_shard_{idx:05d}.jsonl"
            shard_path = os.path.join(out_dir, shard_name)
            _write_text(shard_path, "\n".join(shard) + ("\n" if shard else ""))
            _sh_text = ("\n".join(shard) + "\n").encode("utf-8") if shard else b""
            _sh_hash = _sha256_bytes(_sh_text)
            shards_meta.append({"uri": _uri_from_path(shard_path), "count": len(shard), "hash": f"sha256:{_sh_hash}"})
            idx += 1

    rr = {"ts": _now_iso(), "seed": seed, "paths": [m["uri"] for m in shards_meta]}
    rec = _write_json(os.path.join(out_dir, "run_record.json"), rr)
    return {"shards": shards_meta, "run_record": rec["uri"]}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "openai_compat": True, "teacher_enabled": bool(_TRACE_MODE.get("enabled", False))}


