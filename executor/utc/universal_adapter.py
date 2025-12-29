import re
from typing import Any, Dict, List, Tuple
import logging
import traceback


def _camel_to_snake(s: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def _norm(s: str) -> str:
    return re.sub(r'[\s_\-]+', '', str(s)).lower()


def _rename(args: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    props = schema.get("properties") or {}
    ops: List[Dict[str, Any]] = []
    out = dict(args)
    target_keys = set(props.keys())
    for k in list(args.keys()):
        if k in target_keys:
            continue
        ks = _camel_to_snake(k)
        if ks in target_keys and ks not in out:
            out[ks] = out.pop(k)
            ops.append({"op": "rename", "from": k, "to": ks})
    return out, ops


def _wrap_unwrap(args: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out = dict(args); ops: List[Dict[str, Any]] = []
    if "input" in required and "input" not in out:
        out = {"input": out}
        ops.append({"op": "wrap", "path": "input"})
    if "input" not in props and "input" in out and isinstance(out.get("input"), dict):
        inner = out.get("input") or {}
        if any(k in props for k in inner.keys()):
            base = dict(out); base.pop("input", None); base.update(inner); out = base
            ops.append({"op": "unwrap", "path": "input"})
    return out, ops


def _move(args: Dict[str, Any], errors: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out = dict(args); ops: List[Dict[str, Any]] = []
    for e in errors:
        if e.get("code") == "required_missing":
            need = (e.get("path") or "").lstrip("/")
            for key in list(out.keys()):
                if _norm(s=key) == _norm(s=need) and key != need and need not in out:
                    out[need] = out.pop(key); ops.append({"op": "move", "from": key, "to": need}); break
    return out, ops


def _cast(args: Dict[str, Any], schema: Dict[str, Any], errors: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out = dict(args); ops: List[Dict[str, Any]] = []
    for err in errors:
        if err.get("code") != "type_mismatch":
            continue
        path = (err.get("path") or "").lstrip("/")
        exp = err.get("expected")
        if not path or path not in out:
            continue
        val = out[path]
        try:
            if exp == "integer" and isinstance(val, str) and val.replace(".", "", 1).isdigit():
                out[path] = int(float(val)); ops.append({"op": "cast", "path": path, "to": "integer"})
            elif exp == "number" and isinstance(val, str):
                out[path] = float(val); ops.append({"op": "cast", "path": path, "to": "number"})
            elif exp == "boolean" and isinstance(val, str) and val.strip().lower() in ("true", "false"):
                out[path] = (val.strip().lower() == "true"); ops.append({"op": "cast", "path": path, "to": "boolean"})
            elif exp == "string" and not isinstance(val, str):
                out[path] = str(val); ops.append({"op": "cast", "path": path, "to": "string"})
        except Exception as e:
            logging.error(
                "universal_adapter._cast failed for path=%s expected=%s value=%r: %s\n%s",
                path,
                exp,
                val,
                e,
                traceback.format_exc(),
            )
            continue
    return out, ops


def _defaults(args: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    props = schema.get("properties") or {}
    out = dict(args); ops: List[Dict[str, Any]] = []
    for k, ps in props.items():
        if k not in out and "default" in ps:
            out[k] = ps.get("default"); ops.append({"op": "defaults", "path": k})
    return out, ops


def _enum_map(args: Dict[str, Any], errors: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out = dict(args); ops: List[Dict[str, Any]] = []
    for e in errors:
        if e.get("code") == "enum_mismatch":
            path = (e.get("path") or "").lstrip("/")
            allowed = e.get("allowed") or []
            val = out.get(path)
            if isinstance(val, str) and allowed:
                best_value = None
                best_score = None
                norm_val = _norm(s=val)
                for allowed_value in allowed:
                    score = (_norm(s=allowed_value) == norm_val, -abs(len(allowed_value) - len(val)))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_value = allowed_value
                if best_value is not None and best_value != val:
                    out[path] = best_value; ops.append({"op": "enum_map", "path": path, "from": val, "to": best_value})
    return out, ops


def _drop_extra(args: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out = dict(args); ops: List[Dict[str, Any]] = []
    if schema.get("additionalProperties") is False:
        props = set((schema.get("properties") or {}).keys())
        for k in list(out.keys()):
            if k not in props:
                out.pop(k, None); ops.append({"op": "drop_extra", "path": k})
    return out, ops


def repair(args: Dict[str, Any], errors: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic repair: rename → wrap/unwrap → move → cast → defaults → enum_map → drop_extra."""
    current = dict(args); ops_all: List[Dict[str, Any]] = []
    current, ops = _rename(current, schema); ops_all.extend(ops)
    current, ops = _wrap_unwrap(current, schema); ops_all.extend(ops)
    current, ops = _move(current, errors); ops_all.extend(ops)
    current, ops = _cast(current, schema, errors); ops_all.extend(ops)
    current, ops = _defaults(current, schema); ops_all.extend(ops)
    current, ops = _enum_map(current, errors); ops_all.extend(ops)
    current, ops = _drop_extra(current, schema); ops_all.extend(ops)
    return {"fixed_args": current, "ops": ops_all}


