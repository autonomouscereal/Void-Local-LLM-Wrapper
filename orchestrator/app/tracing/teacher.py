from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..db.pool import get_pg_pool
from ..json_parser import JSONParser

log = logging.getLogger(__name__)

try:
    import xxhash  # type: ignore
except Exception:  # pragma: no cover
    xxhash = None  # type: ignore


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _prompt_hash(messages: List[Dict[str, Any]]) -> str:
    m = json.dumps(messages or [], sort_keys=True, separators=(",", ":"))
    return f"sha256:{_sha256_bytes(m.encode('utf-8'))}"


def _xx64(*parts: str) -> int:
    """
    Deterministic 64-bit hash used for seed derivation.
    """
    if xxhash is not None:
        try:
            h = xxhash.xxh64()
            for p in parts:
                h.update(str(p))
            return h.intdigest() & ((1 << 64) - 1)
        except Exception:
            pass
    return int(_sha256_str("|".join(parts))[:16], 16)


def _uploads_root() -> str:
    root = os.getenv("UPLOAD_DIR", "/workspace/uploads")
    root = root.strip() if isinstance(root, str) else "/workspace/uploads"
    return root or "/workspace/uploads"


def _public_base_url() -> str:
    return os.getenv("PUBLIC_BASE_URL", "") or ""


def _uri_from_path(path: str) -> str:
    """
    Convert an absolute filesystem path under UPLOAD_DIR into a URL-ish `/uploads/...` URI
    matching the rest of the wrapper conventions.
    """
    up = _uploads_root()
    rel = os.path.relpath(path, up).replace("\\", "/")
    if rel.startswith("../"):
        # Not under uploads; best-effort return raw.
        return path
    base = _public_base_url()
    return f"{base.rstrip('/')}/uploads/{rel}" if base else f"/uploads/{rel}"


def _teacher_dir(label: str) -> str:
    return os.path.join(_uploads_root(), "teacher", label)


def _teacher_paths(label: str) -> Dict[str, str]:
    base = _teacher_dir(label)
    return {
        "traces": os.path.join(base, "traces.jsonl"),
        "sft": os.path.join(base, "sft.jsonl"),
        "dpo": os.path.join(base, "dpo.jsonl"),
        "toolpolicy": os.path.join(base, "toolpolicy.jsonl"),
        "run_record": os.path.join(base, "run_record.json"),
        "cache_prompt": os.path.join(base, "cache", "prompt_index.json"),
    }


def _append_ndjson(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "ab") as f:
        f.write(line.encode("utf-8"))


def _write_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    os.replace(tmp, path)


def _load_json_best_effort(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        obj = JSONParser().parse(raw or "", {})
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


async def _db_find_run_id(trace_id: str) -> Optional[int]:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM run WHERE trace_id=$1", str(trace_id))
            return int(row[0]) if row else None
    except Exception:
        log.debug("teacher.tap.db_find_run_id_failed trace_id=%r", trace_id, exc_info=True)
        return None


async def _db_insert_teacher_trace(run_id: int, trace_line: Dict[str, Any]) -> None:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO teacher_trace(run_id, trace_line) VALUES($1,$2)",
                int(run_id),
                json.dumps(trace_line, ensure_ascii=False),
            )
    except Exception:
        log.warning("teacher.tap.db_insert_teacher_trace_failed run_id=%r", run_id, exc_info=True)


async def _db_insert_distill_sft(run_id: int, sample_json: Dict[str, Any]) -> None:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO distill_sft(run_id, sample_json) VALUES($1,$2)",
                int(run_id),
                json.dumps(sample_json, ensure_ascii=False),
            )
    except Exception:
        log.debug("teacher.tap.db_insert_distill_sft_failed run_id=%r", run_id, exc_info=True)


async def _db_insert_distill_toolpolicy(run_id: int, policy_json: Dict[str, Any]) -> None:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO distill_toolpolicy(run_id, policy_json) VALUES($1,$2)",
                int(run_id),
                json.dumps(policy_json, ensure_ascii=False),
            )
    except Exception:
        log.debug("teacher.tap.db_insert_distill_toolpolicy_failed run_id=%r", run_id, exc_info=True)


async def _db_insert_distill_dpo(run_id: int, pair_json: Dict[str, Any]) -> None:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO distill_dpo(run_id, pair_json) VALUES($1,$2)",
                int(run_id),
                json.dumps(pair_json, ensure_ascii=False),
            )
    except Exception:
        log.debug("teacher.tap.db_insert_distill_dpo_failed run_id=%r", run_id, exc_info=True)


def _derive_sft_from_trace(t: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mode = (t.get("mode") or "general")
    if mode not in ("general", "research", "text"):
        return None
    req = t.get("request") if isinstance(t.get("request"), dict) else {}
    resp = t.get("response") if isinstance(t.get("response"), dict) else {}
    out_text = resp.get("text") if isinstance(resp.get("text"), str) else ""
    if not out_text.strip():
        return None
    return {
        "input": {"pack_hash": (t.get("context") or {}).get("pack_hash"), "messages": req.get("messages", [])},
        "output": {"text": out_text},
        "meta": {"mode": mode, "seed": t.get("seed"), "trace_id": t.get("trace_id"), "time": t.get("ts")},
    }


def _derive_toolpolicy_from_trace(t: Dict[str, Any]) -> Dict[str, Any]:
    req = t.get("request") if isinstance(t.get("request"), dict) else {}
    tools_allowed = req.get("tools_allowed") or []
    if not isinstance(tools_allowed, list):
        tools_allowed = []
    tool_calls = t.get("tool_calls") if isinstance(t.get("tool_calls"), list) else []
    used = [c.get("name") for c in tool_calls if isinstance(c, dict) and c.get("name")]
    used2 = [str(x) for x in used if isinstance(x, (str, int, float))]
    used_set = set(used2)
    allowed_set = set([str(x) for x in tools_allowed if isinstance(x, (str, int, float))])
    skipped = sorted(list(allowed_set - used_set))
    params: Dict[str, Any] = {}
    for c in tool_calls:
        if not isinstance(c, dict):
            continue
        nm = c.get("name")
        if not isinstance(nm, str) or not nm:
            continue
        params[nm] = (c.get("args") if isinstance(c.get("args"), dict) else (c.get("arguments") if isinstance(c.get("arguments"), dict) else {}))
    mode = (t.get("mode") or "general")
    return {
        "context_hash": (t.get("context") or {}).get("pack_hash"),
        "tools_considered": tools_allowed,
        "decision": {"use": used2, "skip": skipped},
        "params": params,
        "outcome": {"success": True},
        "meta": {"mode": mode, "seed": t.get("seed"), "trace_id": t.get("trace_id")},
    }


def _normalize_trace_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Normalize the orchestrator-built payload into the canonical trace line shape.
    Returns (trace_line, label).
    """
    label = str(payload.get("label") or "default")
    req = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    messages = req.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    mhash = _sha256_str(json.dumps(messages, sort_keys=True, separators=(",", ":")))
    ts = _now_iso()
    trace_seed = int(payload.get("seed", 0) or 0)
    workspace = str(payload.get("workspace", "default"))
    trace_id = payload.get("trace_id")

    # Deterministic seed derivations (kept for backwards-compat with the removed service)
    seed_router = _xx64("router", str(trace_id), str(trace_seed))
    tool_calls = payload.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
    tool_calls = [tc for tc in tool_calls if isinstance(tc.get("tool_name"), str)]
    ranked_tool_calls: List[tuple[str, int, Dict[str, Any]]] = []
    for tool_call_index, tool_call in enumerate(tool_calls):
        ranked_tool_calls.append((str(tool_call.get("tool_name") or ""), int(tool_call_index), tool_call))
    ranked_tool_calls.sort()
    tool_calls = [ranked[2] for ranked in ranked_tool_calls]
    seeds_map: Dict[str, Any] = {"master": trace_seed, "router": seed_router, "tools": {}}
    for tc in tool_calls:
        tool_name = tc.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        seeds_map["tools"][tool_name] = _xx64("tool", str(tool_name), str(trace_id), str(trace_seed))
        tc.setdefault("seed", seeds_map["tools"][tool_name])

    resp = payload.get("response") or {}
    if not isinstance(resp, dict):
        resp = {"_raw": resp}
    if isinstance(resp.get("text"), str) and len(resp.get("text")) > 4000:
        resp["text"] = resp.get("text")[:4000]

    routing = payload.get("routing") or {}
    if not isinstance(routing, dict):
        routing = {"_raw": routing}
    if isinstance(routing.get("alt_models"), list):
        routing["alt_models"] = sorted([str(x) for x in routing.get("alt_models") if x is not None])

    line: Dict[str, Any] = {
        "ts": ts,
        "trace_id": trace_id,
        "seed": trace_seed,
        "workspace": workspace,
        "mode": payload.get("mode") or "general",
        "request": req,
        "context": payload.get("context") or {},
        "routing": routing,
        "tool_calls": tool_calls[:64],
        "response": resp,
        "metrics": payload.get("metrics") or {},
        "seeds": seeds_map,
        "env": payload.get("env") or {},
        "privacy": payload.get("privacy") or {"vault_refs": 0, "secrets_in": 0, "secrets_out": 0},
    }
    return line, label


async def tap_trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    In-process replacement for the removed `services/teacher` container.

    Behavior:
    - Normalize payload into a canonical trace line.
    - Append to `/uploads/teacher/<label>/traces.jsonl`.
    - Derive and append SFT + ToolPolicy, and opportunistically emit a DPO pair
      when the same prompt_hash has been seen before for this label.
    - Best-effort write-through into Postgres: teacher_trace + distill_* tables.
    """
    if not isinstance(payload, dict):
        return {"ok": False, "error": {"code": "payload_not_object", "message": "payload must be an object", "status": 422}}

    t, label = _normalize_trace_payload(payload)
    paths = _teacher_paths(label)

    # File append (best-effort; never fail the request path)
    try:
        _append_ndjson(paths["traces"], t)
    except Exception:
        log.debug("teacher.tap.file_append_failed kind=traces label=%r", label, exc_info=True)

    # SFT
    sft = _derive_sft_from_trace(t)
    if isinstance(sft, dict):
        try:
            _append_ndjson(paths["sft"], sft)
        except Exception:
            log.debug("teacher.tap.file_append_failed kind=sft label=%r", label, exc_info=True)

    # ToolPolicy
    tp = _derive_toolpolicy_from_trace(t)
    try:
        _append_ndjson(paths["toolpolicy"], tp)
    except Exception:
        log.debug("teacher.tap.file_append_failed kind=toolpolicy label=%r", label, exc_info=True)

    # DPO (prompt_index cache is the minimal state we persist)
    dpo_obj: Optional[Dict[str, Any]] = None
    try:
        mhash = _prompt_hash(((t.get("request") or {}).get("messages") or []) if isinstance(t.get("request"), dict) else [])
        idx = _load_json_best_effort(paths["cache_prompt"])
        prev = idx.get(mhash) or []
        prev_ids: List[str] = []
        if isinstance(prev, list):
            prev_ids = [str(x) for x in prev if isinstance(x, (str, int, float))]
        if prev_ids:
            other_id = prev_ids[-1]
            chosen = ((t.get("response") or {}).get("text") or "") if isinstance(t.get("response"), dict) else ""
            dpo_obj = {
                "prompt_hash": mhash,
                "chosen": {"text": chosen, "tool_seq": [c.get("name") for c in (t.get("tool_calls") or []) if isinstance(c, dict)]},
                "rejected": {"text": "", "tool_seq": []},
                "meta": {"mode": (t.get("mode") or "general"), "seed": t.get("seed"), "trace_ids": [other_id, t.get("trace_id")]},
            }
            try:
                _append_ndjson(paths["dpo"], dpo_obj)
            except Exception:
                log.debug("teacher.tap.file_append_failed kind=dpo label=%r", label, exc_info=True)
        # Update prompt index (append current trace_id)
        cur_id = t.get("trace_id")
        if isinstance(cur_id, str) and cur_id:
            idx[mhash] = prev_ids + [cur_id]
            _write_json_atomic(paths["cache_prompt"], idx)
    except Exception:
        log.debug("teacher.tap.dpo_failed label=%r", label, exc_info=True)

    # Minimal run_record (overwritten each time)
    try:
        rr = {
            "ts": _now_iso(),
            "label": label,
            "teacher_version": "orchestrator.inproc.v1",
            "paths": {
                "traces_jsonl": _uri_from_path(paths["traces"]),
                "sft_jsonl": _uri_from_path(paths["sft"]),
                "dpo_jsonl": _uri_from_path(paths["dpo"]),
                "toolpolicy_jsonl": _uri_from_path(paths["toolpolicy"]),
            },
        }
        _write_json_atomic(paths["run_record"], rr)
    except Exception:
        log.debug("teacher.tap.run_record_write_failed label=%r", label, exc_info=True)

    # DB write-through (best-effort)
    run_id = await _db_find_run_id(str(t.get("trace_id") or ""))
    if isinstance(run_id, int) and run_id > 0:
        await _db_insert_teacher_trace(run_id, t)
        if isinstance(sft, dict):
            await _db_insert_distill_sft(run_id, sft)
        if isinstance(tp, dict):
            await _db_insert_distill_toolpolicy(run_id, tp)
        if isinstance(dpo_obj, dict):
            await _db_insert_distill_dpo(run_id, dpo_obj)

    return {"ok": True, "label": label, "trace_id": t.get("trace_id")}


