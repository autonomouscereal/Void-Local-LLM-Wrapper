"""
Canonical artifact and envelope schema definitions.

This module provides data classes and builders to ensure artifacts and envelopes
are always constructed with all required fields, preventing random degradation
across the codebase.

Shared across orchestrator, executor, and all satellite services.
"""
from __future__ import annotations

import time
import hashlib
import uuid
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _sanitize_for_artifact_id(s: str) -> str:
    """
    Sanitize a string to be safe for use in artifact_id.
    Removes or replaces unsafe characters that could cause issues in URLs or filenames.
    """
    if not isinstance(s, str):
        s = str(s)
    # Replace unsafe characters with underscores, but keep colons (used as separators)
    # Remove or replace: / \ < > | " ? * and other problematic chars
    s = re.sub(r'[<>"|?*\x00-\x1f\x7f-\x9f]', '_', s)
    # Replace backslashes and forward slashes
    s = s.replace('\\', '_').replace('/', '_')
    # Collapse multiple underscores
    s = re.sub(r'_+', '_', s)
    # Remove leading/trailing underscores
    s = s.strip('_')
    return s


def generate_artifact_id(
    *,
    trace_id: str = "",
    tool_name: str = "",
    conversation_id: str = "",
    suffix_data: Any = None,
    existing_id: Optional[str] = None,
) -> str:
    """
    Generate a unique artifact_id for a tool output.
    
    Format: {trace_id}:{tool_name}:{unique_suffix}
    If trace_id is empty, format: {tool_name}:{unique_suffix}
    
    The artifact_id is always URL-safe and filesystem-safe by sanitizing all components
    and using a UUID-based unique suffix.
    
    Args:
        trace_id: Trace identifier (optional, will be sanitized)
        tool_name: Tool name (e.g., "image.upscale", "music.mixdown", will be sanitized)
        conversation_id: Conversation identifier (optional, used in hash)
        suffix_data: Additional data to include in uniqueness hash (optional)
        existing_id: If provided, use this instead of generating new one (will be sanitized)
        
    Returns:
        Unique artifact_id string that is always safe for URLs and filenames
    """
    if existing_id and isinstance(existing_id, str) and existing_id.strip():
        # Sanitize existing_id but preserve structure
        parts = existing_id.strip().split(':')
        sanitized_parts = [_sanitize_for_artifact_id(p) for p in parts if p]
        return ':'.join(sanitized_parts) if sanitized_parts else str(uuid.uuid4())
    
    # Sanitize inputs
    trace_id_safe = _sanitize_for_artifact_id(trace_id) if trace_id else ""
    tool_name_safe = _sanitize_for_artifact_id(tool_name) if tool_name else "unknown"
    conversation_id_safe = _sanitize_for_artifact_id(conversation_id) if conversation_id else ""
    
    # Build hash input with sanitized values
    now_ms = int(time.time() * 1000)
    hash_input = f"{trace_id_safe}:{conversation_id_safe}:{tool_name_safe}:{now_ms}"
    if suffix_data is not None:
        suffix_str = _sanitize_for_artifact_id(str(suffix_data))
        hash_input += f":{suffix_str}"
    
    # Generate unique suffix using hash + UUID for extra safety
    hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    uuid_suffix = str(uuid.uuid4()).replace('-', '')[:8]
    unique_suffix = f"{hash_suffix}{uuid_suffix}"[:16]  # Combined for uniqueness
    
    if trace_id_safe:
        return f"{trace_id_safe}:{tool_name_safe}:{unique_suffix}"
    else:
        return f"{tool_name_safe}:{unique_suffix}"


def artifact_id_to_safe_filename(artifact_id: Any, extension: str = "") -> str:
    """
    Convert artifact_id to a safe filename that matches how files are saved.
    
    This is the inverse of the filename generation used when saving files.
    Files are saved with: artifact_id.replace(":", "_").replace("/", "_").replace("\\", "_")[:200] + extension
    
    Args:
        artifact_id: The artifact_id (str, int, or other)
        extension: Optional file extension (e.g., ".wav", ".png") - should include the dot
        
    Returns:
        Safe filename string that matches the actual saved file
    """
    if artifact_id is None:
        return ""
    artifact_id_str = str(artifact_id)
    safe = artifact_id_str.replace(":", "_").replace("/", "_").replace("\\", "_")[:200]
    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        return safe + extension
    return safe


@dataclass
class ArtifactMeta:
    """Required metadata for all artifacts."""
    trace_id: str
    conversation_id: Optional[str] = None
    created_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    created_at: int = field(default_factory=lambda: int(time.time()))
    tool_name: Optional[str] = None
    tool: Optional[str] = None  # backward compat
    # Optional fields
    qa: Optional[Dict[str, Any]] = None
    locks: Optional[Dict[str, Any]] = None
    segment_id: Optional[str] = None
    shot_id: Optional[str] = None
    scene_id: Optional[str] = None
    film_id: Optional[str] = None
    window_id: Optional[str] = None
    voice_id: Optional[str] = None
    voice_lock_id: Optional[str] = None
    variant_index: Optional[int] = None
    refine_mode: Optional[str] = None
    refined: Optional[bool] = None
    degraded_reasons: Optional[List[str]] = None
    # Additional tool-specific fields
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values except where needed."""
        d: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "created_ms": self.created_ms,
            "created_at": self.created_at,
        }
        if self.conversation_id is not None:
            d["conversation_id"] = self.conversation_id
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
            d["tool"] = self.tool_name  # Always set both
        elif self.tool is not None:
            d["tool"] = self.tool
        if self.qa is not None:
            d["qa"] = self.qa
        if self.locks is not None:
            d["locks"] = self.locks
        if self.segment_id is not None:
            d["segment_id"] = self.segment_id
        if self.shot_id is not None:
            d["shot_id"] = self.shot_id
        if self.scene_id is not None:
            d["scene_id"] = self.scene_id
        if self.film_id is not None:
            d["film_id"] = self.film_id
        if self.window_id is not None:
            d["window_id"] = self.window_id
        if self.voice_id is not None:
            d["voice_id"] = self.voice_id
        if self.voice_lock_id is not None:
            d["voice_lock_id"] = self.voice_lock_id
        if self.variant_index is not None:
            d["variant_index"] = self.variant_index
        if self.refine_mode is not None:
            d["refine_mode"] = self.refine_mode
        if self.refined is not None:
            d["refined"] = self.refined
        if self.degraded_reasons is not None:
            d["degraded_reasons"] = self.degraded_reasons
        if self.extra is not None:
            d.update(self.extra)
        return d


@dataclass
class Artifact:
    """Canonical artifact structure with all required fields."""
    artifact_id: Any  # str, int, or other - unified type
    kind: str  # "image", "video", "audio", "tts", "sfx", "audio-ref", etc.
    path: str
    meta: ArtifactMeta
    # Optional fields
    url: Optional[str] = None
    view_url: Optional[str] = None
    summary: Optional[str] = None
    bytes: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    rel_path: Optional[str] = None
    # Additional fields
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format used throughout the codebase."""
        d: Dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "meta": self.meta.to_dict(),
            "tags": self.tags,
        }
        if self.url is not None:
            d["url"] = self.url
        if self.view_url is not None:
            d["view_url"] = self.view_url
        elif self.path.startswith("/workspace/"):
            # Auto-derive view_url from /workspace path
            d["view_url"] = self.path.replace("/workspace", "")
            if self.url is None:
                d["url"] = d["view_url"]
        if self.summary is not None:
            d["summary"] = self.summary
        if self.bytes is not None:
            d["bytes"] = self.bytes
        if self.rel_path is not None:
            d["rel_path"] = self.rel_path
        if self.extra is not None:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, trace_id: Optional[str] = None, conversation_id: Optional[str] = None, tool_name: Optional[str] = None) -> Artifact:
        """Create Artifact from dict, normalizing and filling required fields."""
        artifact_id = d.get("artifact_id") or d.get("id")
        if artifact_id is None:
            path = d.get("path", "")
            if isinstance(path, str):
                import os
                # Generate artifact_id if missing (for backward compatibility)
                # Use generate_artifact_id to ensure consistency and no drift
                artifact_id = generate_artifact_id(
                    trace_id=(trace_id or ""),
                    tool_name=(tool_name or "unknown"),
                    conversation_id=(conversation_id or ""),
                    suffix_data=os.path.basename(path) if path else None,
                )
            else:
                artifact_id = generate_artifact_id(
                    trace_id=(trace_id or ""),
                    tool_name=(tool_name or "unknown"),
                    conversation_id=(conversation_id or ""),
                    suffix_data="unknown",
                )
        
        kind = d.get("kind") or "unknown"
        path = d.get("path") or ""
        
        # Extract meta or build new one
        meta_dict = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        meta = ArtifactMeta(
            trace_id=trace_id or meta_dict.get("trace_id") or "",
            conversation_id=conversation_id or meta_dict.get("conversation_id"),
            created_ms=meta_dict.get("created_ms") or int(time.time() * 1000),
            created_at=meta_dict.get("created_at") or int(time.time()),
            tool_name=tool_name or meta_dict.get("tool_name") or meta_dict.get("tool"),
            tool=tool_name or meta_dict.get("tool_name") or meta_dict.get("tool"),
            qa=meta_dict.get("qa"),
            locks=meta_dict.get("locks"),
            segment_id=meta_dict.get("segment_id"),
            shot_id=meta_dict.get("shot_id"),
            scene_id=meta_dict.get("scene_id"),
            film_id=meta_dict.get("film_id"),
            window_id=meta_dict.get("window_id"),
            voice_id=meta_dict.get("voice_id"),
            voice_lock_id=meta_dict.get("voice_lock_id"),
            variant_index=meta_dict.get("variant_index"),
            refine_mode=meta_dict.get("refine_mode"),
            refined=meta_dict.get("refined"),
            degraded_reasons=meta_dict.get("degraded_reasons"),
            extra={k: v for k, v in meta_dict.items() if k not in {
                "trace_id", "conversation_id", "created_ms", "created_at", "tool_name", "tool",
                "qa", "locks", "segment_id", "shot_id", "scene_id", "film_id", "window_id",
                "voice_id", "voice_lock_id", "variant_index", "refine_mode", "refined", "degraded_reasons"
            }},
        )
        
        url = d.get("url") or d.get("view_url")
        view_url = d.get("view_url") or url
        
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            path=path,
            meta=meta,
            url=url,
            view_url=view_url,
            summary=d.get("summary"),
            bytes=d.get("bytes"),
            tags=d.get("tags") if isinstance(d.get("tags"), list) else [],
            rel_path=d.get("rel_path"),
            extra={k: v for k, v in d.items() if k not in {
                "artifact_id", "id", "kind", "path", "meta", "url", "view_url",
                "summary", "bytes", "tags", "rel_path"
            }},
        )


@dataclass
class ToolCall:
    """Canonical tool call structure."""
    tool_name: str
    args: Dict[str, Any]
    arguments: Optional[Dict[str, Any]] = None  # OpenAI compat
    tool: Optional[str] = None  # backward compat
    status: Optional[str] = None
    artifact_id: Optional[str] = None
    step_id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d: Dict[str, Any] = {
            "tool_name": self.tool_name,
            "args": self.args,
            "arguments": self.arguments if self.arguments is not None else self.args,
        }
        if self.tool is not None:
            d["tool"] = self.tool
        else:
            d["tool"] = self.tool_name  # Always set both
        if self.status is not None:
            d["status"] = self.status
        if self.artifact_id is not None:
            d["artifact_id"] = self.artifact_id
        if self.step_id is not None:
            d["step_id"] = self.step_id
        if self.error is not None:
            d["error"] = self.error
        return d


def build_artifact(
    *,
    artifact_id: Any,
    kind: str,
    path: str,
    trace_id: str,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Builder function to create a properly normalized artifact dict.
    
    This ensures all artifacts have the required fields and prevents degradation.
    """
    now_ms = int(time.time() * 1000)
    now_at = int(time.time())
    
    meta = ArtifactMeta(
        trace_id=trace_id,
        conversation_id=conversation_id,
        created_ms=kwargs.pop("created_ms", now_ms),
        created_at=kwargs.pop("created_at", now_at),
        tool_name=tool_name,
        tool=tool_name,
        segment_id=kwargs.pop("segment_id", None),
        shot_id=kwargs.pop("shot_id", None),
        scene_id=kwargs.pop("scene_id", None),
        film_id=kwargs.pop("film_id", None),
        window_id=kwargs.pop("window_id", None),
        voice_id=kwargs.pop("voice_id", None),
        voice_lock_id=kwargs.pop("voice_lock_id", None),
        variant_index=kwargs.pop("variant_index", None),
        refine_mode=kwargs.pop("refine_mode", None),
        refined=kwargs.pop("refined", None),
        degraded_reasons=kwargs.pop("degraded_reasons", None),
        qa=kwargs.pop("qa", None),
        locks=kwargs.pop("locks", None),
        extra=kwargs.pop("meta_extra", None),
    )
    
    artifact = Artifact(
        artifact_id=artifact_id,
        kind=kind,
        path=path,
        meta=meta,
        url=kwargs.pop("url", None),
        view_url=kwargs.pop("view_url", None),
        summary=kwargs.pop("summary", None),
        bytes=kwargs.pop("bytes", None),
        tags=kwargs.pop("tags", []),
        rel_path=kwargs.pop("rel_path", None),
        extra=kwargs if kwargs else None,
    )
    
    return artifact.to_dict()

