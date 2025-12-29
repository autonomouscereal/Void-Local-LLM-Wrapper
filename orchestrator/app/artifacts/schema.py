"""
Canonical artifact and envelope schema definitions.

This module re-exports from void_artifacts for backward compatibility.
All new code should import directly from void_artifacts.
"""
from __future__ import annotations

# Re-export from shared void_artifacts package
from void_artifacts import (
    Artifact,
    ArtifactMeta,
    ToolCall,
    build_artifact,
    generate_artifact_id,
    artifact_id_to_safe_filename,
)

__all__ = [
    "Artifact",
    "ArtifactMeta",
    "ToolCall",
    "build_artifact",
    "generate_artifact_id",
    "artifact_id_to_safe_filename",
]
