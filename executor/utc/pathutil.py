from typing import Any, Tuple, Dict


def _split(path: str) -> list[str]:
    return [p for p in str(path).split(".") if p]


def get_in(d: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    cur: Any = d
    for k in _split(path):
        if not isinstance(cur, dict) or k not in cur:
            return False, None
        cur = cur[k]
    return True, cur


def set_in(d: Dict[str, Any], path: str, val: Any, create: bool = True) -> bool:
    cur: Any = d
    keys = _split(path)
    if not keys:
        return False
    for k in keys[:-1]:
        if k not in cur:
            if not create:
                return False
            cur[k] = {}
        cur = cur[k]
        if not isinstance(cur, dict):
            return False
    cur[keys[-1]] = val
    return True


def pop_in(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    keys = _split(path)
    if not keys:
        return None
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            return None
        cur = cur[k]
    return cur.pop(keys[-1], None)


