import os
import json
from typing import Any, Dict, List


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _type_ok(val: Any, t: Any) -> bool:
    if t == "string":
        return isinstance(val, str)
    if t == "integer":
        return isinstance(val, int)
    if t == "boolean":
        return isinstance(val, bool)
    if t == "number":
        return isinstance(val, (int, float))
    if t == "object":
        return isinstance(val, dict)
    if t == "array":
        return isinstance(val, list)
    return True


def _local_validate(schema: Dict[str, Any], args: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []
    if not isinstance(schema, dict) or not isinstance(args, dict):
        return [{"path": "/", "message": "invalid", "expected": "object", "got": type(args).__name__}]
    reqs = schema.get("required") or []
    props = schema.get("properties") or {}
    for r in reqs:
        if args.get(r) is None:
            errs.append({"path": f"/{r}", "message": "required", "expected": "present", "got": None})
    for k, v in args.items():
        ps = props.get(k)
        if ps and not _type_ok(v, ps.get("type")):
            errs.append({"path": f"/{k}", "message": "type_mismatch", "expected": ps.get("type"), "got": type(v).__name__})
        if ps and isinstance(ps.get("enum"), list) and v not in ps.get("enum"):
            errs.append({"path": f"/{k}", "message": "enum", "expected": ps.get("enum"), "got": v})
    return errs


def check(name: str, args: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer server-side /tool.validate; fallback to minimal validator."""
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    url = base.rstrip("/") + "/tool.validate"
    try:
        obj = _post(url, {"name": name, "args": args})
        if obj and obj.get("ok") is True:
            return {"ok": True, "errors": []}
        if obj and obj.get("ok") is False:
            details = ((obj.get("error") or {}).get("details") or {})
            return {"ok": False, "errors": details.get("errors") or []}
    except Exception:
        pass
    # fallback
    errs = _local_validate(schema, args)
    return {"ok": len(errs) == 0, "errors": errs}


