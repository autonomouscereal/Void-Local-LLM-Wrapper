from __future__ import annotations

import os
import json
from .common import ensure_dir, sidecar, make_outpaths, normalize_size, stamp_env, now_ts
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope
from ..refs.apply import load_refs
from ..refs.registry import append_provenance
from .export import append_image_sample
from ..context.index import add_artifact as _ctx_add
from ..context.index import resolve_reference as _ctx_resolve, resolve_global as _glob_resolve
from ..datasets.trace import append_sample as _trace_append


def run_image_edit(job: dict, provider, manifest: dict) -> dict:
    """
    job: { "image_ref": path, "mask_ref": path|None, "prompt": str, "negative": str|None, "size": "WxH"|None,
           "seed": int|None, "refs": {...}, "cid": str }
    """
    cid = job.get("cid") or ("img-" + str(now_ts()))
    outdir = os.path.join("/workspace", "uploads", "artifacts", "image", cid)
    ensure_dir(outdir)
    size = normalize_size(job.get("size"), edge_safe=bool(job.get("edge")))
    refs = load_refs(job.get("ref_ids"), job.get("refs"))
    args = {k: job.get(k) for k in ("image_ref", "mask_ref", "prompt", "negative")}
    if not args.get("image_ref"):
        try:
            rec = _ctx_resolve(cid, str(job.get("prompt") or ""), "image")
            if rec and isinstance(rec.get("path"), str):
                args["image_ref"] = rec.get("path")
            if not args.get("image_ref"):
                gre = _glob_resolve(str(job.get("prompt") or ""), "image")
                if gre and isinstance(gre.get("path"), str):
                    args["image_ref"] = gre.get("path")
        except Exception:
            pass
    args.update({"size": size, "refs": refs, "seed": job.get("seed")})
    args = stamp_tool_args("image.edit", args)
    res = provider.edit(args)
    img_bytes = res.get("image_bytes") or b""
    model = res.get("model", "unknown")
    stem = f"edit_{now_ts()}"; png_path, _ = make_outpaths(outdir, stem)
    with open(png_path, "wb") as f:
        f.write(img_bytes)
    sidecar(png_path, {"tool": "image.edit", **args, "model": model})
    try:
        for rid in (job.get("ref_ids") or []):
            append_provenance(rid, {"when": now_ts(), "tool": "image.edit", "artifact": png_path, "seed": int(args.get("seed") or 0)})
    except Exception:
        pass
    add_manifest_row(manifest, png_path, step_id="image.edit")
    try:
        append_image_sample(outdir, {"tool": "image.edit", "prompt": args.get("prompt"), "negative": args.get("negative"), "size": args.get("size"), "seed": int(args.get("seed") or 0), "model": model, "path": png_path, "ts": now_ts()})
    except Exception:
        pass
    try:
        _ctx_add(cid, "image", png_path, None, args.get("image_ref"), [], {"prompt": args.get("prompt")})
    except Exception:
        pass
    try:
        _trace_append("image", {
            "cid": cid,
            "tool": "image.edit",
            "prompt": args.get("prompt"),
            "negative": args.get("negative"),
            "seed": int(args.get("seed") or 0),
            "refs": refs,
            "model": model,
            "path": png_path,
            "parent": args.get("image_ref") or None,
        })
    except Exception:
        pass
    env = {
        "meta": {"model": model, "ts": now_ts(), "cid": cid, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "image edit", "constraints": ["json-only"], "decisions": ["image.edit done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "image edited"},
        "tool_calls": [{"tool": "image.edit", "args": args, "status": "done", "result_ref": os.path.basename(png_path)}],
        "artifacts": [{"id": os.path.basename(png_path), "kind": "image-ref", "summary": stem, "bytes": len(img_bytes)}],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_to_envelope(json.dumps(env)); env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, "image.edit", model)
    return env


