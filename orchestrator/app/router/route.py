from __future__ import annotations

import os


def route_for_request(_: dict) -> str:
    """Return routing mode for ICW/windowed solver. Values: 'windowed' | 'off'"""
    return os.getenv("ICW_MODE", "windowed").strip().lower() or "windowed"


