from __future__ import annotations

SCHEMA_VERSION = 1


def bump_envelope(env: dict) -> dict:
    env.setdefault("meta", {})
    env["meta"].setdefault("schema_version", SCHEMA_VERSION)
    return env


def assert_envelope(env: dict) -> None:
    sv = ((env or {}).get("meta") or {}).get("schema_version")
    if sv is None:
        migrate_v0_to_v1(env)
        return
    if sv > SCHEMA_VERSION:
        meta = env.setdefault("meta", {})
        notes = meta.get("notes") or []
        notes.append(f"future_schema:{sv}")
        meta["notes"] = notes


def migrate_v0_to_v1(env: dict) -> None:
    env.setdefault("meta", {})
    env["meta"]["schema_version"] = SCHEMA_VERSION
    env.setdefault("message", {"role": "assistant", "type": "text", "content": ""})
    env.setdefault("reasoning", {"goal": "", "constraints": [], "decisions": []})
    env.setdefault("evidence", [])
    env.setdefault("tool_calls", [])
    env.setdefault("artifacts", [])
    env.setdefault(
        "telemetry",
        {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    )


