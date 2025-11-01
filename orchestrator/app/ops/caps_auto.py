from __future__ import annotations

from typing import List, Dict, Any


def caps_from_registries(models: List[Dict[str, Any]], tool_registry: Dict[str, Any], features: Dict[str, Any], edge_profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "models": [{"name": m.get("name"), "ctx_tokens": m.get("ctx_tokens"), "step_tokens": m.get("step_tokens")} for m in (models or [])],
        "tools": sorted(list(tool_registry.keys() if isinstance(tool_registry, dict) else [])),
        "features": features or {},
        "edge": edge_profile or {},
    }


