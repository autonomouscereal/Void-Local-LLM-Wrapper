from __future__ import annotations

from typing import Any

from ..json_parser import JSONParser


def parse_json_text(text: str, expected: Any) -> Any:
    try:
        return JSONParser().parse(text, expected if expected is not None else {})
    except Exception:
        if isinstance(expected, dict):
            return {}
        if isinstance(expected, list):
            return []
        return {}


def resp_json(resp, expected: Any) -> Any:
    return parse_json_text(getattr(resp, "text", "") or "", expected)


