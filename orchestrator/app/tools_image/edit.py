from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import json
import logging
import time
import hashlib
from .common import ensure_dir, sidecar, make_outpaths, normalize_size, stamp_env, now_ts
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_envelope, bump_envelope, assert_envelope
from ..ref_library.apply import load_refs
from ..ref_library.registry import append_provenance
from ..artifacts.index import add_artifact as _ctx_add
from ..artifacts.index import resolve_reference as _ctx_resolve, resolve_global as _glob_resolve
from ..tracing.training import append_training_sample
from ..tracing.runtime import trace_event
from void_artifacts import build_artifact, generate_artifact_id, artifact_id_to_safe_filename

log = logging.getLogger(__name__)


def run_image_edit(
    *,
    provider,
    manifest: dict,
    trace_id: str = "",
    conversation_id: str = "",
    image_ref: str | None = None,
    mask_ref: str | None = None,
    prompt: str = "",
    negative: str | None = None,
    size: str | None = None,
    seed: int | None = None,
    ref_ids: list[str] | None = None,
    refs: dict | None = None,
    edge: bool = False,
    artifact_id: str | None = None,
    **kwargs
) -> dict:
    """
    Edit an image with explicit parameters.
    """
    if trace_id:
        trace_event("tool.image.edit.start", {"trace_id": trace_id, "conversation_id": conversation_id})
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "image", conversation_id)
    ensure_dir(outdir)
    size_normalized = normalize_size(size, edge_safe=edge)
    refs_loaded = load_refs(ref_ids, refs)
    args = {"image_ref": image_ref, "mask_ref": mask_ref, "prompt": prompt, "negative": negative}
    if not args.get("image_ref"):
        try:
            rec = _ctx_resolve(conversation_id, str(prompt or ""), "image")
            if rec and isinstance(rec.get("path"), str):
                args["image_ref"] = rec.get("path")
            if not args.get("image_ref"):
                gre = _glob_resolve(str(prompt or ""), "image")
                if gre and isinstance(gre.get("path"), str):
                    args["image_ref"] = gre.get("path")
        except Exception as exc:
            log.debug(f"image.edit: failed to resolve image_ref from context (non-fatal) conversation_id={conversation_id} exc={exc}", exc_info=True)
    args.update({"size": size_normalized, "refs": refs_loaded, "seed": seed})
    args = stamp_tool_args("image.edit", args)
    res = provider.edit(args)
    img_bytes = res.get("image_bytes") or b""
    model = res.get("model", "unknown")
    # Generate unique artifact_id BEFORE creating file, then use it for filename
    artifact_id_generated = generate_artifact_id(
        trace_id=trace_id,
        tool_name="image.edit",
        conversation_id=conversation_id,
        suffix_data=len(img_bytes),
        existing_id=artifact_id,
    )
    # Create safe filename from artifact_id (artifact_id is already sanitized, but use helper for consistency)
    safe_filename = artifact_id_to_safe_filename(artifact_id_generated, ".png")
    png_path = os.path.join(outdir, safe_filename)
    with open(png_path, "wb") as f:
        f.write(img_bytes)
    stem = os.path.splitext(safe_filename)[0]
    sidecar(png_path, {"tool": "image.edit", **args, "model": model})
    try:
        for rid in (ref_ids or []):
            append_provenance(rid, {"when": now_ts(), "tool": "image.edit", "artifact": png_path, "seed": int(args.get("seed") or 0)})
    except Exception as exc:
        log.debug(f"image.edit: append_provenance failed (non-fatal) conversation_id={conversation_id} exc={exc}", exc_info=True)
    add_manifest_row(manifest, png_path, step_id="image.edit")
    # (Removed) per-artifact image_samples.jsonl writer. Canonical dataset stream is `datasets/stream.py`
    try:
        url = f"/uploads/artifacts/image/{conversation_id}/{os.path.basename(png_path)}" if conversation_id else None
        _ctx_add(conversation_id, "image", png_path, url, args.get("image_ref"), [], {"prompt": args.get("prompt"), "trace_id": trace_id, "tool": "image.edit", "model": model})
    except Exception as exc:
        log.debug(f"image.edit: context add failed (non-fatal) conversation_id={conversation_id} trace_id={trace_id} exc={exc}", exc_info=True)
    try:
        append_training_sample("image", {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "tool": "image.edit",
            "prompt": args.get("prompt"),
            "negative": args.get("negative"),
            "seed": int(args.get("seed") or 0),
            "refs": refs,
            "model": model,
            "path": png_path,
            "parent": args.get("image_ref") or None,
        })
    except Exception as exc:
        log.debug(f"image.edit: trace append failed (non-fatal) conversation_id={conversation_id} trace_id={trace_id} exc={exc}", exc_info=True)
    if trace_id:
        trace_event("tool.image.edit.complete", {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "tool": "image.edit",
            "model": model,
            "path": png_path,
            "bytes": len(img_bytes),
            "prompt": args.get("prompt"),
        })
    log.info(f"image.edit: completed conversation_id={conversation_id!r} trace_id={trace_id!r} model={model!r} path={png_path!r} bytes={len(img_bytes)}")
    env = {
        "meta": {"model": model, "ts": now_ts(), "conversation_id": conversation_id, "trace_id": trace_id, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "image edit", "constraints": ["json-only"], "decisions": ["image.edit done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "image edited"},
        "tool_calls": [{"tool_name": "image.edit", "tool": "image.edit", "args": args, "arguments": args, "status": "done", "artifact_id": artifact_id_generated}],
        "artifacts": [
            build_artifact(
                artifact_id=artifact_id_generated,
                kind="image",
                path=png_path,
                trace_id=trace_id,
                conversation_id=conversation_id,
                tool_name="image.edit",
                url=(f"/uploads/artifacts/image/{conversation_id}/{os.path.basename(png_path)}" if conversation_id else None),
                summary=stem,
                bytes=len(img_bytes),
                tags=[],
            )
        ],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_envelope(env); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "image.edit", model)
    return env


