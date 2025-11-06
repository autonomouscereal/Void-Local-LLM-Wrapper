import json
from typing import Any, Dict, List, Optional
from .db import get_pg_pool
from .pathutil import get_in, set_in, pop_in


def _apply_ops(args: Dict[str, Any], ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(args)
    for op in (ops or []):
        t = op.get("op")
        if t == "rename":
            parent = op.get("parent")
            src = op.get("from"); dst = op.get("to")
            if parent:
                val_ok, cur = get_in(out, parent)
                if val_ok and isinstance(cur, dict) and src in cur and dst not in cur:
                    cur[dst] = cur.pop(src)
            else:
                if src in out and dst not in out:
                    out[dst] = out.pop(src)
        elif t == "wrap":
            p = op.get("path")
            if isinstance(p, str) and p and p not in out:
                out = {p: out}
        elif t == "unwrap":
            p = op.get("path")
            if p in out and isinstance(out.get(p), dict):
                inner = out.pop(p) or {}
                out.update(inner)
        elif t == "move":
            src = op.get("from"); dst = op.get("to")
            val = pop_in(out, src)
            if val is not None:
                set_in(out, dst, val, create=True)
        elif t == "enum_map":
            path = op.get("path")
            ok, current = get_in(out, path)
            if ok and current == op.get("from"):
                set_in(out, path, op.get("to"), create=True)
        elif t == "drop_extra":
            path = op.get("path"); pop_in(out, path)
        # casts/defaults are idempotent/no-op at preapply stage
    return out


async def preapply(tool_name: str, version: str, from_hash: str, args: Dict[str, Any]) -> Dict[str, Any]:
    pool = await get_pg_pool()
    if pool is None:
        return args
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                "select patch_json from tool_patches where tool_name=$1 and version=$2 and from_hash=$3 order by patch_id asc",
                tool_name, version, from_hash,
            )
        except Exception:
            return args
    out = dict(args)
    for r in rows or []:
        patch = (r.get("patch_json") or {})
        ops = patch.get("ops") if isinstance(patch, dict) else None
        if isinstance(ops, list):
            out = _apply_ops(out, ops)
    return out


async def persist_success(tool_name: str, version: str, from_hash: str, to_hash: str, ops: List[Dict[str, Any]]) -> Optional[int]:
    pool = await get_pg_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                "insert into tool_patches (tool_name, version, from_hash, to_hash, patch_json) values ($1,$2,$3,$4,$5) returning patch_id",
                tool_name, version, from_hash, to_hash, json.dumps({"ops": ops or []}),
            )
            return int(row["patch_id"]) if row and "patch_id" in row else None
        except Exception:
            return None


