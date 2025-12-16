from __future__ import annotations

from typing import Any, Dict, Optional

from ..datasets.trace import append_sample as _append_dataset_sample
from .runtime import trace_event


_ALLOWED_MODALITIES = {"image", "tts", "music", "video"}
_OUTPUT_KEYS = ("path", "url", "audio_ref", "image_ref", "track_ref", "video_ref", "video_path", "image_path", "audio_path")


def append_training_sample(modality: str, row: Dict[str, Any] | Any) -> Optional[str]:
    """
    Append a modality training sample into:
      <uploads>/datasets/trace/<modality>.jsonl

    This must NOT be used for generic logging events. If the input doesn't look
    like a real sample (no tool, no output path/url, no cid/trace_id), we reject
    it and instead emit a runtime trace describing the rejection.
    """
    mod = str(modality or "").strip().lower()
    if mod not in _ALLOWED_MODALITIES:
        trace_event("training.sample.reject", {"modality": mod, "reason": "unsupported_modality", "row": row})
        return None

    if not isinstance(row, dict):
        row = {"_raw": row}

    has_tool = isinstance(row.get("tool"), str) and bool(str(row.get("tool") or "").strip())
    has_id = isinstance(row.get("trace_id"), str) and bool(str(row.get("trace_id") or "").strip())
    if not has_id:
        cid = row.get("cid")
        has_id = isinstance(cid, str) and bool(cid.strip())
    has_output = any(isinstance(row.get(k), str) and bool(str(row.get(k) or "").strip()) for k in _OUTPUT_KEYS)

    if not (has_tool and has_id and has_output):
        trace_event(
            "training.sample.reject",
            {
                "modality": mod,
                "reason": "invalid_sample_shape",
                "missing": {
                    "tool": (not has_tool),
                    "cid_or_trace_id": (not has_id),
                    "output_path_or_url": (not has_output),
                },
                # Include a trimmed view of the row keys for debugging shape drift.
                "row_keys": sorted(list(row.keys()))[:64],
            },
        )
        return None

    return _append_dataset_sample(mod, row)


