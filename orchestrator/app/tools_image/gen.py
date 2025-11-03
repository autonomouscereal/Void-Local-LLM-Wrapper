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
from ..refs.registry import append_provenance
from .export import append_image_sample
from ..context.index import add_artifact as _ctx_add
from ..analysis.media import analyze_image
from ..datasets.trace import append_sample as _trace_append
import base64
import httpx  # type: ignore


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
    # provenance for any referenced ids
    try:
        for rid in (job.get("ref_ids") or []):
            append_provenance(rid, {"when": now_ts(), "tool": "image.gen", "artifact": png_path, "seed": int(args.get("seed") or 0)})
    except Exception:
        pass
    try:
        _ctx_add(cid, "image", png_path, None, None, ["face_lock"] if (refs.get("images") or refs.get("faces")) else [], {"prompt": args.get("prompt")})
    except Exception:
        pass
    add_manifest_row(manifest, png_path, step_id="image.gen")
    # Committee review: VLM caption and CLIP scoring with enforced thresholds and multi-pass revisions
    try:
        max_passes = 2
        base_prompt = str(args.get("prompt") or "")
        last_bytes = None
        def _caption_score(prompt_text: str) -> float:
            try:
                if not os.getenv("VLM_API_URL"):
                    return 0.0
                with open(png_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                vlm_url = os.getenv("VLM_API_URL").rstrip("/") + "/analyze"
                with httpx.Client() as client:
                    r = client.post(vlm_url, json={"b64": b64, "ext": ".png"})
                    if r.status_code == 200 and isinstance(r.json(), dict):
                        cap = (r.json().get("caption") or r.json().get("text") or "").strip()
                        at = [t for t in prompt_text.lower().split() if len(t) > 3]
                        bt = cap.lower()
                        if not at: return 0.0
                        hits = sum(1 for t in at if t in bt)
                        return hits / max(1, len(at))
            except Exception:
                return 0.0
            return 0.0
        def _clip_score(prompt_text: str) -> float:
            try:
                ai = analyze_image(png_path, prompt=prompt_text)
                return float(ai.get("clip_score") or 0.0)
            except Exception:
                return 0.0
        def _revise_args(pass_idx: int) -> dict:
            rev = dict(args)
            if pass_idx == 0:
                rev["prompt"] = f"{base_prompt}, emphasize: {base_prompt}"
            else:
                rev["prompt"] = f"{base_prompt}, literal match, centered subject, high detail, clean background"
            return rev
        # Evaluate and revise up to max_passes until both scores cross thresholds
        for i in range(max_passes + 1):
            cap_s = _caption_score(base_prompt)
            clip_s = _clip_score(base_prompt)
            # thresholds
            ok = (cap_s >= 0.35) and (clip_s >= 0.30)
            sidecar(png_path, {"tool": "image.gen.committee", "scores": {"caption": cap_s, "clip": clip_s}, "pass": i, "ok": ok})
            if ok:
                break
            if i < max_passes:
                rev_args = _revise_args(i)
                rev = provider.generate(rev_args)
                last_bytes = rev.get("image_bytes") or b""
                if last_bytes:
                    with open(png_path, "wb") as f:
                        f.write(last_bytes)
                    sidecar(png_path, {"tool": "image.gen", **rev_args, "model": rev.get("model", model), "committee": {"caption": cap_s, "clip": clip_s, "revision": True, "pass": i}})
        # Final committee record
        final_ai = analyze_image(png_path, prompt=base_prompt)
        sidecar(png_path, {"tool": "image.gen.committee.final", "clip_score": float(final_ai.get("clip_score") or 0.0), "tags": final_ai.get("tags") or []})
    except Exception:
        pass
    # Trace row for distillation
    try:
        final_ai = analyze_image(png_path, prompt=str(args.get("prompt") or ""))
        _trace_append("image", {
            "cid": cid,
            "tool": "image.gen",
            "prompt": args.get("prompt"),
            "negative": args.get("negative"),
            "seed": int(args.get("seed") or 0),
            "refs": refs,
            "model": model,
            "path": png_path,
            "clip_score": float(final_ai.get("clip_score") or 0.0),
            "tags": final_ai.get("tags") or [],
        })
    except Exception:
        pass
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


