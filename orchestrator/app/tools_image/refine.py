from __future__ import annotations

from typing import Any, Dict, Optional


def build_image_refine_dispatch_args(
    args: Dict[str, Any],
    lock_bundle: Optional[Dict[str, Any]],
    quality_profile: str,
) -> Dict[str, Any]:
    """
    Map image.refine.segment arguments into an image.dispatch call.

    This is intentionally minimal for Step 2:
    - We reuse the canonical image.dispatch Comfy path for actual rendering.
    - We pass through prompt/seed when provided.
    - We attach the resolved lock bundle and quality_profile so QA/locks behave
      the same way as for a fresh image.dispatch call.
    - We forward optional cid/trace_id for tracing.

    The source_image field is passed through untouched so downstream tooling can
    start to rely on it once the Comfy graph is upgraded to support init images.
    """
    out: Dict[str, Any] = {}
    prompt = args.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        out["prompt"] = prompt
    negative = args.get("negative")
    if isinstance(negative, str) and negative.strip():
        out["negative"] = negative
    seed = args.get("seed")
    if isinstance(seed, int):
        out["seed"] = seed
    cid = args.get("cid")
    if isinstance(cid, str) and cid.strip():
        out["cid"] = cid.strip()
    trace_id = args.get("trace_id")
    if isinstance(trace_id, str) and trace_id.strip():
        out["trace_id"] = trace_id.strip()
    source_image = args.get("source_image")
    if isinstance(source_image, str) and source_image.strip():
        out["source_image"] = source_image.strip()
    profile_name = quality_profile or "standard"
    out["quality_profile"] = profile_name
    if lock_bundle is not None:
        out["lock_bundle"] = lock_bundle
    # Let /tool.run's image.dispatch implementation infer width/height/steps/cfg
    # when they are not provided; we do not override any explicit values here.
    for key in ("width", "height", "steps", "cfg"):
        if key in args:
            out[key] = args[key]
    return out



