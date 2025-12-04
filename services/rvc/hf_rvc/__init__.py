"""
Local shim for the legacy `hf_rvc` package.

This exists solely to satisfy old imports like:

    from hf_rvc import RVCFeatureExtractor, RVCModel

in environments that still have that code path. It does NOT implement the
full hf-rvc functionality; the canonical RVC pipeline in this project goes
through the external rvc-python engine and HTTP endpoints exposed by the
RVC service.
"""

from __future__ import annotations

from typing import Any, Dict


class RVCFeatureExtractor:
    """
    Minimal no-op shim. Included only so that imports succeed.

    Any actual usage of this class at runtime should be treated as a bug,
    because the supported pipeline is the HTTP-based RVC service.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        self.config: Dict[str, Any] = {"args": args, "kwargs": kwargs}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            "hf_rvc.RVCFeatureExtractor shim was called; this code path is deprecated. "
            "The RVC service should use the HTTP-based rvc-python engine instead."
        )


class RVCModel:
    """
    Minimal no-op shim. Included only so that imports succeed.

    Any actual usage of this class at runtime should be treated as a bug,
    because the supported pipeline is the HTTP-based RVC service.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        self.config: Dict[str, Any] = {"args": args, "kwargs": kwargs}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            "hf_rvc.RVCModel shim was called; this code path is deprecated. "
            "The RVC service should use the HTTP-based rvc-python engine instead."
        )


