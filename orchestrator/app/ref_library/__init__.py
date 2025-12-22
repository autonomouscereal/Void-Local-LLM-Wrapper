from __future__ import annotations

"""
ref_library: Persistent, user-facing reference library (images/voices/music stems).

This subsystem stores/manages reference manifests and lightweight embeddings.
It is intentionally separate from `app.locks`, which is the canonical constraint/lock-bundle system.
"""


