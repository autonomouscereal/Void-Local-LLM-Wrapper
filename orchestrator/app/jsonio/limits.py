from __future__ import annotations


MAX_FIELD_BYTES = 200_000  # per-field guard for logs/payloads


def _clip_bytes(s: str, max_b: int) -> str:
    try:
        b = (s or "").encode("utf-8")
        if len(b) <= max_b:
            return s or ""
        return b[: max_b].decode("utf-8", errors="ignore")
    except Exception:
        return (s or "")[: max_b]


def enforce_field_limits(env: dict) -> dict:
    # apply only to text-like fields; artifacts/log payloads should live in files
    if not isinstance(env, dict):
        return env
    msg = env.get("message") or {}
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        msg["content"] = _clip_bytes(msg.get("content") or "", MAX_FIELD_BYTES)
    env["message"] = msg
    return env


