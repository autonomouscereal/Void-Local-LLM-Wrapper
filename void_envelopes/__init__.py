from __future__ import annotations

from .tool_envelope import ToolEnvelope, _build_success_envelope, _build_error_envelope
from .assistant_envelope import normalize_to_envelope, normalize_envelope, coerce_to_envelope
from .assistant_envelope_versioning import bump_envelope, assert_envelope
from .openai_compat import (
    build_openai_envelope,
    stitch_openai,
    stitch_openai_response,
    merge_envelopes,
    strip_tags,
)

__all__ = [
    "ToolEnvelope",
    "_build_success_envelope",
    "_build_error_envelope",
    # assistant envelope (internal)
    "normalize_to_envelope",
    "normalize_envelope",
    "coerce_to_envelope",
    "bump_envelope",
    "assert_envelope",
    # openai compat + stitching
    "build_openai_envelope",
    "stitch_openai",
    "stitch_openai_response",
    "merge_envelopes",
    "strip_tags",
]


