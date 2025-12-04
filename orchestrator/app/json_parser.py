from __future__ import annotations

# Thin compatibility shim so existing orchestrator imports continue to work.
# The canonical implementation now lives in the shared top-level package
# `void_json.json_parser` so it can be reused by all services.
from void_json.json_parser import JSONParser  # noqa: F401