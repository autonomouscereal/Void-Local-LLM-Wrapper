from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import json
import logging
import time
import hashlib
from .common import ensure_dir, sidecar, make_outpaths, stamp_env, now_ts
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_envelope, bump_envelope, assert_envelope
from ..artifacts.index import add_artifact as _ctx_add
from ..artifacts.index import resolve_reference as _ctx_resolve, resolve_global as _glob_resolve
from ..tracing.training import append_training_sample
from ..tracing.runtime import trace_event
from void_artifacts import build_artifact, generate_artifact_id, artifact_id_to_safe_filename

log = logging.getLogger(__name__)


def run_image_upscale(
    *,
    provider,
    manifest: dict,
    trace_id: str = "",
    conversation_id: str = "",
    image_ref: str | None = None,
    scale: int = 2,
    denoise: float | None = None,
    seed: int | None = None,
    prompt: str | None = None,
    artifact_id: str | None = None,
    **kwargs
) -> dict:
    """
    Upscale an image with explicit parameters.
    """
    if trace_id:
        trace_event("tool.image.upscale.start", {"trace_id": trace_id, "conversation_id": conversation_id})
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "image", conversation_id); ensure_dir(outdir)
    args = {"image_ref": image_ref, "scale": scale, "denoise": denoise, "seed": seed}
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
            log.debug(f"image.upscale: failed to resolve image_ref from context (non-fatal) conversation_id={conversation_id} exc={exc}", exc_info=True)
    args = stamp_tool_args("image.upscale", args)
    res = provider.upscale(args)
    img_bytes = res.get("image_bytes") or b""; model = res.get("model", "unknown")
    # Generate unique artifact_id BEFORE creating file, then use it for filename
    artifact_id_generated = generate_artifact_id(
        trace_id=trace_id,
        tool_name="image.upscale",
        conversation_id=conversation_id,
        suffix_data=len(img_bytes),
        existing_id=artifact_id,
    )
    # Create safe filename from artifact_id (artifact_id is already sanitized, but use helper for consistency)
    safe_filename = artifact_id_to_safe_filename(artifact_id_generated, ".png")
    png_path = os.path.join(outdir, safe_filename)
    with open(png_path, "wb") as f: f.write(img_bytes)
    stem = os.path.splitext(safe_filename)[0]
    sidecar(png_path, {"tool": "image.upscale", **args, "model": model})
    add_manifest_row(manifest, png_path, step_id="image.upscale")
    # (Removed) per-artifact image_samples.jsonl writer. Canonical dataset stream is `datasets/stream.py`
    try:
        url = f"/uploads/artifacts/image/{conversation_id}/{os.path.basename(png_path)}"
        _ctx_add(conversation_id, "image", png_path, url, args.get("image_ref"), ["upscale"], {"trace_id": trace_id, "tool": "image.upscale", "model": model})
    except Exception as exc:
        log.debug(f"image.upscale: context add failed (non-fatal) conversation_id={conversation_id} trace_id={trace_id} exc={exc}", exc_info=True)
    try:
        append_training_sample("image", {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "tool": "image.upscale",
            "scale": int(args.get("scale") or 0),
            "seed": int(args.get("seed") or 0),
            "model": model,
            "path": png_path,
            "parent": args.get("image_ref") or None,
        })
    except Exception as exc:
        log.debug(f"image.upscale: trace append failed (non-fatal) conversation_id={conversation_id} trace_id={trace_id} exc={exc}", exc_info=True)
    if trace_id:
        trace_event("tool.image.upscale.complete", {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "tool": "image.upscale",
            "model": model,
            "path": png_path,
            "bytes": len(img_bytes),
            "scale": int(args.get("scale") or 0),
        })
    log.info(f"image.upscale: completed conversation_id={conversation_id!r} trace_id={trace_id!r} model={model!r} path={png_path!r} bytes={len(img_bytes)} scale={args.get('scale')}")
    env = {
      "meta": {"model": model, "ts": now_ts(), "conversation_id": conversation_id, "trace_id": trace_id, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
      "reasoning": {"goal": "image upscale", "constraints": ["json-only"], "decisions": ["image.upscale done"]},
      "evidence": [],
      "message": {"role": "assistant", "type": "tool", "content": "image upscaled"},
      "tool_calls": [{"tool_name": "image.upscale", "tool": "image.upscale", "args": args, "arguments": args, "status": "done", "artifact_id": artifact_id_generated}],
      "artifacts": [
        build_artifact(
          artifact_id=artifact_id_generated,
          kind="image",
          path=png_path,
          trace_id=trace_id,
          conversation_id=conversation_id,
          tool_name="image.upscale",
          url=(f"/uploads/artifacts/image/{conversation_id}/{os.path.basename(png_path)}"),
          summary=stem,
          bytes=len(img_bytes),
          tags=[],
        )
      ],
      "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []}
    }
    env = normalize_envelope(env); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "image.upscale", model)
    return env


