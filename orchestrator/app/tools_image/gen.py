from __future__ import annotations

import os
import json
import base64
from types import SimpleNamespace
from .common import ensure_dir, sidecar, make_outpaths, normalize_size, stamp_env, now_ts
from ..determinism.seeds import stamp_tool_args
from ..artifacts.manifest import add_manifest_row
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope
from ..refs.apply import load_refs
from .export import append_image_sample


def run_image_gen(job: dict, provider, manifest: dict) -> dict:
    """
    job: {
      "prompt": str, "negative": str|None, "size": "WxH"|None, "seed": int|None,
      "refs": { "images":[...], "faces":[...], "video_frames":[...], "pose":[...], "depth":[...], "char_id":"..." },
      "cid": str, "edge": bool
    }
    provider: image generation backend with .generate(args) -> {"image_bytes": b"...", "model": "..."}
    manifest: running manifest dict
    """
    cid = job.get("cid") or ("img-" + str(now_ts()))
    outdir = os.path.join("/workspace", "uploads", "artifacts", "image", cid)
    ensure_dir(outdir)
    size = normalize_size(job.get("size"), edge_safe=bool(job.get("edge")))
    refs = load_refs(job.get("ref_ids"), job.get("refs"))
    args = {
        "prompt": job.get("prompt") or "",
        "negative": job.get("negative"),
        "size": size,
        "refs": refs,
        "seed": job.get("seed"),
    }
    args = stamp_tool_args("image.gen", args)
    res = provider.generate(args)
    img_bytes = res.get("image_bytes") or b""
    model = res.get("model", "unknown")
    stem = f"gen_{now_ts()}"
    png_path, meta_path = make_outpaths(outdir, stem)
    with open(png_path, "wb") as f:
        f.write(img_bytes)
    sidecar(png_path, {"tool": "image.gen", "prompt": args.get("prompt"), "negative": args.get("negative"), "size": args.get("size"), "seed": args.get("seed"), "refs": refs, "model": model})
    add_manifest_row(manifest, png_path, step_id="image.gen")
    try:
        append_image_sample(outdir, {"tool": "image.gen", "prompt": args.get("prompt"), "negative": args.get("negative"), "size": args.get("size"), "seed": int(args.get("seed") or 0), "model": model, "path": png_path, "ts": now_ts()})
    except Exception:
        pass
    env = {
        "meta": {"model": model, "ts": now_ts(), "cid": cid, "step": 0, "state": "halt", "cont": {"present": False, "state_hash": None, "reason": None}},
        "reasoning": {"goal": "image generation", "constraints": ["json-only", "no caps"], "decisions": ["image.gen done"]},
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "image generated"},
        "tool_calls": [{"tool": "image.gen", "args": args, "status": "done", "result_ref": os.path.basename(png_path)}],
        "artifacts": [{"id": os.path.basename(png_path), "kind": "image-ref", "summary": stem, "bytes": len(img_bytes)}],
        "telemetry": {"window": {"input_bytes": 0, "output_target_tokens": 0}, "compression_passes": [], "notes": []},
    }
    env = normalize_to_envelope(json.dumps(env))
    env = bump_envelope(env); assert_envelope(env); env = stamp_env(env, tool="image.gen", model=model)
    return env


