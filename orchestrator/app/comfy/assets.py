from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from ..json_parser import JSONParser


def build_view_url(base: str, filename: str, ftype: str = "output", subfolder: Optional[str] = None) -> str:
    """
    Build a ComfyUI /view URL for a single asset.
    """
    q: Dict[str, Any] = {"filename": filename, "type": ftype or "output"}
    if subfolder:
        q["subfolder"] = subfolder
    return f"{(base or '').rstrip('/')}/view?{urlencode(q)}"


def normalize_history_entry(raw: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
    """
    Normalize a /history response into a single history entry for the given prompt_id.

    Handles common ComfyUI shapes:
      1) {"history": { "<pid>": {...} }, ...}
      2) {"<pid>": {...}, ...}
      3) {"outputs": {...}, "status": {...}, ...}  (direct entry)

    Also tolerates wrapper keys such as ok/code/status/detail that some callers add.
    """
    if not isinstance(raw, dict):
        return {}

    data: Dict[str, Any] = dict(raw)
    for k in ("ok", "code", "status", "detail"):
        data.pop(k, None)

    entry: Dict[str, Any] | None = None

    # Shape 1: nested history map
    hblock = data.get("history")
    if isinstance(hblock, dict):
        entry = hblock.get(prompt_id) or next(iter(hblock.values()), None)

    # Shape 2: top-level pid key
    if entry is None and prompt_id in data:
        val = data.get(prompt_id)
        if isinstance(val, dict):
            entry = val

    # Shape 3: direct entry with outputs/status
    if entry is None and isinstance(data.get("outputs"), dict):
        entry = data  # type: ignore[assignment]

    return entry if isinstance(entry, dict) else {}


def _ensure_detail_mapping(detail: Dict[str, Any] | str) -> Dict[str, Any]:
    """
    Best-effort coercion of a detail payload into a mapping with at least
    'outputs' and 'status' keys when present.
    """
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, str):
        try:
            parsed = JSONParser().parse(detail, {"outputs": dict, "status": dict})
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def extract_comfy_asset_urls(detail: Dict[str, Any] | str, base_url: str) -> List[Dict[str, Any]]:
    """
    Extract a flat list of asset descriptors from a single ComfyUI history entry.

    Handles both common 'outputs' shapes:
      - { node_id: [ {filename, type, subfolder}, ... ] }
      - { node_id: { "images": [ {filename, type, subfolder}, ... ], ... }, ... }

    Returns a list of:
      {
        "node": <node id as str>,
        "filename": <str>,
        "type": <str>,
        "subfolder": <str | None>,
        "url": <full /view URL using base_url>,
      }
    """
    out: List[Dict[str, Any]] = []
    base = (base_url or "").rstrip("/")
    if not base:
        return out

    detail_map = _ensure_detail_mapping(detail)
    outputs = (detail_map or {}).get("outputs", {}) or {}
    if not isinstance(outputs, dict):
        return out

    for node_id, items in outputs.items():
        # Shape A: items is already a list of {filename, type, subfolder}
        if isinstance(items, list):
            candidates = items
        # Shape B: items is a mapping with "images": [...]
        elif isinstance(items, dict) and isinstance(items.get("images"), list):
            candidates = items.get("images") or []
        else:
            continue

        for it in candidates:
            if not isinstance(it, dict):
                continue
            fn = it.get("filename")
            if not isinstance(fn, str) or not fn:
                continue
            tp = it.get("type") or "output"
            sub = it.get("subfolder")
            out.append(
                {
                    "node": str(node_id),
                    "filename": fn,
                    "type": tp,
                    "subfolder": sub,
                    "url": build_view_url(base, fn, str(tp), sub if isinstance(sub, str) else None),
                }
            )

    return out



