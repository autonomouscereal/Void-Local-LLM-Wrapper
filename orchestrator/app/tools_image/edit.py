from __future__ import annotations

import os
import json
from .common import ensure_dir, sidecar, make_outpaths, normalize_size, stamp_env, now_ts
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope
from ..refs.apply import load_refs
from .export import append_image_sample


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
    args.update({"size": size, "refs": refs, "seed": job.get("seed")})
    args = stamp_tool_args("image.edit", args)
    res = provider.edit(args)
    img_bytes = res.get("image_bytes") or b""
    model = res.get("model", "unknown")
    stem = f"edit_{now_ts()}"; png_path, _ = make_outpaths(outdir, stem)
    with open(png_path, "wb") as f:
        f.write(img_bytes)
    sidecar(png_path, {"tool": "image.edit", **args, "model": model})
    add_manifest_row(manifest, png_path, step_id="image.edit")
    try:
        append_image_sample(outdir, {"tool": "image.edit", "prompt": args.get("prompt"), "negative": args.get("negative"), "size": args.get("size"), "seed": int(args.get("seed") or 0), "model": model, "path": png_path, "ts": now_ts()})
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


