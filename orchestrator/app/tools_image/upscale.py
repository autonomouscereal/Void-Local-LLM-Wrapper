from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import json
import logging
from .common import ensure_dir, sidecar, make_outpaths, stamp_env, now_ts
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from void_envelopes import normalize_to_envelope, bump_envelope, assert_envelope
from ..artifacts.index import add_artifact as _ctx_add
from ..artifacts.index import resolve_reference as _ctx_resolve, resolve_global as _glob_resolve
from ..tracing.training import append_training_sample

log = logging.getLogger(__name__)


def run_image_upscale(job: dict, provider, manifest: dict) -> dict:
    """
    job: { "image_ref": path, "scale": 2|4, "denoise": float|None, "seed": int|None, "cid": str }
    """
    cid = job.get("cid") or ("img-" + str(now_ts()))
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "image", cid); ensure_dir(outdir)
    args = {"image_ref": job.get("image_ref"), "scale": job.get("scale", 2), "denoise": job.get("denoise"), "seed": job.get("seed")}
    if not args.get("image_ref"):
        try:
            rec = _ctx_resolve(cid, str(job.get("prompt") or ""), "image")
            if rec and isinstance(rec.get("path"), str):
                args["image_ref"] = rec.get("path")
            if not args.get("image_ref"):
                gre = _glob_resolve(str(job.get("prompt") or ""), "image")
                if gre and isinstance(gre.get("path"), str):
                    args["image_ref"] = gre.get("path")
        except Exception as exc:
            log.debug("image.upscale: failed to resolve image_ref from context (non-fatal) cid=%s: %s", cid, exc, exc_info=True)
    args = stamp_tool_args("image.upscale", args)
    res = provider.upscale(args)
    img_bytes = res.get("image_bytes") or b""; model = res.get("model", "unknown")
    stem = f"up_{now_ts()}"; png_path, _ = make_outpaths(outdir, stem)
    with open(png_path, "wb") as f: f.write(img_bytes)
    sidecar(png_path, {"tool": "image.upscale", **args, "model": model})
    add_manifest_row(manifest, png_path, step_id="image.upscale")
    # (Removed) per-artifact image_samples.jsonl writer. Canonical dataset stream is `datasets/stream.py`
    try:
        _ctx_add(cid, "image", png_path, None, args.get("image_ref"), ["upscale"], {})
    except Exception as exc:
        log.debug("image.upscale: context add failed (non-fatal) cid=%s: %s", cid, exc, exc_info=True)
    try:
        append_training_sample("image", {
            "cid": cid,
            "tool": "image.upscale",
            "scale": int(args.get("scale") or 0),
            "seed": int(args.get("seed") or 0),
            "model": model,
            "path": png_path,
            "parent": args.get("image_ref") or None,
        })
    except Exception as exc:
        log.debug("image.upscale: trace append failed (non-fatal) cid=%s: %s", cid, exc, exc_info=True)
    env = {
      "meta": {"model": model, "ts": now_ts(), "cid": cid, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
      "reasoning": {"goal": "image upscale", "constraints": ["json-only"], "decisions": ["image.upscale done"]},
      "evidence": [],
      "message": {"role": "assistant", "type": "tool", "content": "image upscaled"},
      "tool_calls": [{"tool": "image.upscale", "args": args, "status": "done", "result_ref": os.path.basename(png_path)}],
      "artifacts": [{"id": os.path.basename(png_path), "kind": "image-ref", "summary": stem, "bytes": len(img_bytes)}],
      "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []}
    }
    env = normalize_to_envelope(json.dumps(env)); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "image.upscale", model)
    return env


