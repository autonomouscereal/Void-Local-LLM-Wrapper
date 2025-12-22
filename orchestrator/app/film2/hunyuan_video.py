from __future__ import annotations

"""
Film2 → HunyuanVideo tool helpers.

This module centralizes:
- service adapter for `/v1/video/generate` (used by tool handlers)
- Film2-facing helpers to run HunyuanVideo + cleanup + tracing
- trace logging for Film2 runs
- post-generation lock/hero preparation via `video.cleanup`
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx  # type: ignore

from ..json_parser import JSONParser


log = logging.getLogger(__name__)

def _derive_face_images_from_lock_bundle(lock_bundle: Dict[str, Any]) -> List[str]:
    """
    Extract stable reference face images from a Film2 lock bundle so downstream tools
    (video.interpolate face stabilization) can compute embeddings deterministically.
    """
    if not isinstance(lock_bundle, dict):
        return []
    face = lock_bundle.get("face") if isinstance(lock_bundle.get("face"), dict) else {}
    refs = face.get("ref_faces") if isinstance(face.get("ref_faces"), list) else []
    out: List[str] = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        p = r.get("image_path")
        if isinstance(p, str) and p.strip():
            out.append(p.strip())
    # Dedup while preserving order
    return list(dict.fromkeys(out))


def _abs_upload_path(upload_dir: str, maybe_path: str) -> str:
    """
    Normalize various upload path forms into an absolute /workspace/... path.
    """
    if not isinstance(maybe_path, str) or not maybe_path:
        return ""
    if maybe_path.startswith("/workspace/"):
        return maybe_path
    if maybe_path.startswith("/uploads/"):
        return "/workspace" + maybe_path
    if maybe_path.startswith("/workspace/uploads/"):
        return maybe_path
    # allow relative path under uploads
    if not maybe_path.startswith("/"):
        return os.path.join(upload_dir, maybe_path)
    return maybe_path


def hv_args_to_generate_payload(args: Dict[str, Any], *, upload_dir: str, job_prefix: str = "hv") -> Tuple[Dict[str, Any], str, int]:
    """
    Map legacy `video.hv.*` tool args to the HunyuanVideo service `/v1/video/generate` payload.

    If `args.init_image` is present, it is forwarded (i2v). Otherwise it is omitted (t2v).
    Returns (payload_for_generate, job_id, fps).
    """
    has_init = isinstance(args.get("init_image"), str) and bool(str(args.get("init_image")).strip())
    w = int(args.get("width")) if isinstance(args.get("width"), (int, float)) else 1024
    h = int(args.get("height")) if isinstance(args.get("height"), (int, float)) else (1024 if has_init else 576)
    fps = int(args.get("fps")) if isinstance(args.get("fps"), (int, float)) else 24
    sec = int(args.get("seconds")) if isinstance(args.get("seconds"), (int, float)) else (5 if has_init else 6)
    num_frames = max(1, fps * max(1, sec) + 1)

    job_id = str(args.get("job_id") or "")
    if not job_id.strip():
        job_id = f"{job_prefix}-{int(time.time())}"

    # Preserve the full hv tool args for distillation/debugging without affecting the service pipeline call.
    req_meta: Dict[str, Any] = {}
    for k in ("cid", "trace_id", "step_id", "film_id", "scene_id", "shot_id", "act_id", "locks", "post", "latent_reinit_every", "quality", "adapter"):
        v = args.get(k)
        if v is not None:
            req_meta[k] = v

    payload: Dict[str, Any] = {
        "job_id": job_id,
        "prompt": args.get("prompt"),
        "negative_prompt": args.get("negative"),
        "width": w,
        "height": h,
        "fps": fps,
        "num_frames": num_frames,
        "num_inference_steps": int(args.get("steps")) if isinstance(args.get("steps"), (int, float)) else None,
        "guidance_scale": float(args.get("guidance_scale")) if isinstance(args.get("guidance_scale"), (int, float)) else None,
        "seed": args.get("seed") if isinstance(args.get("seed"), int) else None,
        "extra": {},
        "meta": req_meta,
    }
    # Drop None values so the service uses its defaults.
    payload = {k: v for k, v in payload.items() if v is not None}

    init_image = args.get("init_image")
    if isinstance(init_image, str) and init_image.strip():
        payload["init_image"] = _abs_upload_path(upload_dir, init_image.strip())
    return payload, job_id, fps


async def hyvideo_generate(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.post(api_url.rstrip("/") + "/v1/video/generate", json=payload)
    parser = JSONParser()
    js = parser.parse(r.text or "{}", {})
    out = js if isinstance(js, dict) else {}
    log.info(
        "hyvideo.generate status=%d dur_ms=%d keys=%s",
        int(r.status_code),
        int((time.perf_counter() - t0) * 1000),
        sorted(list(out.keys())),
    )
    return out


def normalize_generate_response(resp: Dict[str, Any], *, upload_dir: str) -> Dict[str, Any]:
    """
    Normalize service response into the legacy tool shape:
      - path/view_url for simple callers
      - video.{path,view_url} for older callers
      - artifacts[] for callers expecting an artifacts list
    """
    result = resp.get("result") if isinstance(resp.get("result"), dict) else {}
    output_path = result.get("output_path") if isinstance(result.get("output_path"), str) else ""
    output_url = result.get("output_url") if isinstance(result.get("output_url"), str) else ""
    abs_path = _abs_upload_path(upload_dir, output_path) if output_path else ""
    view_url = None
    if abs_path.startswith("/workspace/uploads/"):
        view_url = abs_path.replace("/workspace", "")
    if output_url and not view_url:
        view_url = output_url
    meta = dict(result) if isinstance(result, dict) else {}
    artifacts: List[Dict[str, Any]] = []
    if abs_path:
        artifacts.append({"kind": "video", "path": abs_path, "view_url": view_url})
    return {
        "path": abs_path,
        "view_url": view_url,
        "video": {"path": abs_path, "view_url": view_url} if abs_path else {},
        "artifacts": artifacts,
        "meta": meta,
    }


def build_hv_tool_args(
    *,
    prompt: str,
    width: int,
    height: int,
    fps: int,
    seconds: int,
    locks: Dict[str, Any] | None,
    seed: Any,
    cid: str,
    trace_id: Optional[str] = None,
    film_id: Optional[str] = None,
    scene_id: Optional[str] = None,
    shot_id: Optional[str] = None,
    act_id: Optional[str] = None,
    init_image: Optional[str] = None,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "prompt": prompt or "",
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "seconds": int(seconds),
        "locks": locks or {},
        "seed": seed,
        # These are legacy tool args; the HyVideo adapter may ignore some.
        "post": {"interpolate": True, "upscale": True, "face_lock": True, "hand_fix": True},
        "latent_reinit_every": 48,
        "cid": cid,
    }
    if isinstance(trace_id, str) and trace_id.strip():
        args["trace_id"] = trace_id.strip()
    if isinstance(film_id, str) and film_id.strip():
        args["film_id"] = film_id.strip()
    if isinstance(scene_id, str) and scene_id.strip():
        args["scene_id"] = scene_id.strip()
    if isinstance(shot_id, str) and shot_id.strip():
        args["shot_id"] = shot_id.strip()
    if isinstance(act_id, str) and act_id.strip():
        args["act_id"] = act_id.strip()
    if isinstance(init_image, str) and init_image.strip():
        args["init_image"] = init_image.strip()
    return args


async def film2_generate_video(
    *,
    trace_id: Optional[str],
    shot_id: Optional[str],
    scene_id: Optional[str],
    act_id: Optional[str],
    hv_tool_args: Dict[str, Any],
    upload_dir: str,
    log_fn,
    trace_append: Callable[[str, Dict[str, Any]], None],
    http_tool_run,
    artifact_video: Optional[Callable[[Optional[str], str], None]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Runs the HunyuanVideo tool (`video.hv.t2v`) and then runs `video.cleanup`
    on the produced video to compute lock metrics/frames for hero selection.
    """
    segment_log: List[Dict[str, Any]] = []
    cid = hv_tool_args.get("cid") if isinstance(hv_tool_args.get("cid"), str) else None

    adapter = "hv.i2v" if isinstance(hv_tool_args.get("init_image"), str) and hv_tool_args.get("init_image") else "hv.t2v"
    if isinstance(trace_id, str) and trace_id and log_fn:
        # main._log signature: _log(kind, trace_id=..., **fields)
        log_fn("film2.pass_gen_start", trace_id=trace_id, adapter=adapter, shot_id=shot_id)
    trace_append(
        "film2.hv_call_start",
        {
            "trace_id": trace_id,
            "adapter": adapter,
            "shot_id": shot_id,
            "scene_id": scene_id,
            "act_id": act_id,
            "prompt": hv_tool_args.get("prompt"),
            "width": hv_tool_args.get("width"),
            "height": hv_tool_args.get("height"),
            "fps": hv_tool_args.get("fps"),
            "seconds": hv_tool_args.get("seconds"),
            "init_image": hv_tool_args.get("init_image"),
        },
    )

    # Single tool entrypoint: `init_image` inside hv_tool_args toggles i2v behavior.
    gv = await http_tool_run("video.hv.t2v", hv_tool_args)
    if isinstance(gv, dict):
        segment_log.append(gv)
    gvr = (gv.get("result") or {}) if isinstance(gv, dict) else {}
    gen_path = None
    if isinstance(gvr, dict):
        gen_path = gvr.get("path")
        if not isinstance(gen_path, str):
            video_obj = gvr.get("video") if isinstance(gvr.get("video"), dict) else {}
            gen_path = video_obj.get("path") if isinstance(video_obj.get("path"), str) else None
    if isinstance(gen_path, str) and gen_path:
        trace_append(
            "film2.hv_call_success",
            {
                "trace_id": trace_id,
                "adapter": adapter,
                "shot_id": shot_id,
                "scene_id": scene_id,
                "act_id": act_id,
                "video_path": gen_path,
            },
        )
        if artifact_video:
            artifact_video(trace_id, gen_path)
    if isinstance(trace_id, str) and trace_id and log_fn:
        log_fn("film2.pass_gen_finish", trace_id=trace_id, adapter=adapter, shot_id=shot_id)

    # Optional post-passes driven by hv_tool_args["post"].
    # NOTE: HunyuanVideo service currently does not enforce locks directly; these passes are how Film2
    # can actually apply character/consistency fixes using existing tools (Comfy/FaceID/mediapipe).
    current_path: Optional[str] = gen_path if isinstance(gen_path, str) and gen_path else None
    post = hv_tool_args.get("post") if isinstance(hv_tool_args.get("post"), dict) else {}
    if isinstance(current_path, str) and current_path:
        # Optional upscale pass: run before interpolation so interpolation never changes duration/scale unexpectedly.
        if bool(post.get("upscale")) and isinstance(current_path, str) and current_path:
            up_args: Dict[str, Any] = {"src": current_path}
            if isinstance(cid, str) and cid:
                up_args["cid"] = cid
            if isinstance(trace_id, str) and trace_id:
                up_args["trace_id"] = trace_id
            for k in ("film_id", "scene_id", "shot_id", "act_id"):
                v = hv_tool_args.get(k)
                if isinstance(v, (str, int)) and str(v).strip():
                    up_args[k] = str(v).strip()
            if log_fn and isinstance(trace_id, str) and trace_id:
                log_fn("film2.pass_upscale_start", trace_id=trace_id, shot_id=shot_id)
            trace_append("film2.fix.upscale.start", {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id})
            uc = await http_tool_run("video.upscale", up_args)
            if isinstance(uc, dict):
                segment_log.append(uc)
            ucr = (uc.get("result") or {}) if isinstance(uc, dict) else {}
            up_path = ucr.get("path") if isinstance(ucr, dict) else None
            if log_fn and isinstance(trace_id, str) and trace_id:
                log_fn("film2.pass_upscale_finish", trace_id=trace_id, shot_id=shot_id, out_path=up_path)
            trace_append("film2.fix.upscale.finish", {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id, "out_path": up_path})
            if isinstance(up_path, str) and up_path:
                current_path = up_path
                if artifact_video:
                    artifact_video(trace_id, current_path)

        # Face stabilization and/or interpolation pass.
        # We ALWAYS call the tool when requested by post flags. The tool itself will report in result.meta
        # whether model-tied stabilization actually ran (stabilize_attempted/applied).
        if bool(post.get("interpolate")) or bool(post.get("face_lock")):
            interp_args: Dict[str, Any] = {"src": current_path, "target_fps": int(hv_tool_args.get("fps") or 24)}
            if isinstance(cid, str) and cid:
                interp_args["cid"] = cid
            if isinstance(trace_id, str) and trace_id:
                interp_args["trace_id"] = trace_id
            # Provide deterministic refs when available, but do not gate the pass.
            face_images = _derive_face_images_from_lock_bundle(hv_tool_args.get("locks") if isinstance(hv_tool_args.get("locks"), dict) else {})
            if face_images:
                interp_args["locks"] = {"image": {"face_images": face_images}}
            elif isinstance(hv_tool_args.get("locks"), dict):
                interp_args["locks"] = hv_tool_args.get("locks")
            for k in ("film_id", "scene_id", "shot_id", "act_id"):
                v = hv_tool_args.get(k)
                if isinstance(v, (str, int)) and str(v).strip():
                    interp_args[k] = str(v).strip()
            if log_fn and isinstance(trace_id, str) and trace_id:
                log_fn(
                    "film2.pass_interpolate_start",
                    trace_id=trace_id,
                    shot_id=shot_id,
                    target_fps=interp_args.get("target_fps"),
                    has_face_refs=bool(face_images),
                    face_refs_n=(len(face_images) if isinstance(face_images, list) else 0),
                )
            trace_append(
                "film2.fix.interpolate.start",
                {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id, "target_fps": interp_args.get("target_fps"), "has_face_refs": bool(face_images), "face_refs_n": len(face_images or [])},
            )
            ic = await http_tool_run("video.interpolate", interp_args)
            if isinstance(ic, dict):
                segment_log.append(ic)
            icr = (ic.get("result") or {}) if isinstance(ic, dict) else {}
            ip = icr.get("path") if isinstance(icr, dict) else None
            if log_fn and isinstance(trace_id, str) and trace_id:
                meta = (icr.get("meta") if isinstance(icr, dict) else None) or {}
                log_fn(
                    "film2.pass_interpolate_finish",
                    trace_id=trace_id,
                    shot_id=shot_id,
                    out_path=ip,
                    stabilize_attempted=(meta.get("stabilize_attempted") if isinstance(meta, dict) else None),
                    stabilize_applied=(meta.get("stabilize_applied") if isinstance(meta, dict) else None),
                )
            trace_append(
                "film2.fix.interpolate.finish",
                {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id, "out_path": ip, "tool_result_meta": (icr.get("meta") if isinstance(icr, dict) else None)},
            )
            if isinstance(ip, str) and ip:
                current_path = ip
                if artifact_video:
                    artifact_video(trace_id, current_path)

        # Hand repair pass (per-frame mediapipe mask + inpaint).
        if bool(post.get("hand_fix")) and isinstance(current_path, str) and current_path:
            hf_args: Dict[str, Any] = {"src": current_path}
            if isinstance(cid, str) and cid:
                hf_args["cid"] = cid
            if isinstance(trace_id, str) and trace_id:
                hf_args["trace_id"] = trace_id
            for k in ("film_id", "scene_id", "shot_id", "act_id"):
                v = hv_tool_args.get(k)
                if isinstance(v, (str, int)) and str(v).strip():
                    hf_args[k] = str(v).strip()
            if log_fn and isinstance(trace_id, str) and trace_id:
                log_fn("film2.pass_handsfix_start", trace_id=trace_id, shot_id=shot_id)
            trace_append("film2.fix.hands.start", {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id})
            hf = await http_tool_run("video.hands.fix", hf_args)
            if isinstance(hf, dict):
                segment_log.append(hf)
            hfr = (hf.get("result") or {}) if isinstance(hf, dict) else {}
            hp = hfr.get("path") if isinstance(hfr, dict) else None
            if log_fn and isinstance(trace_id, str) and trace_id:
                log_fn("film2.pass_handsfix_finish", trace_id=trace_id, shot_id=shot_id, out_path=hp)
            trace_append("film2.fix.hands.finish", {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id, "out_path": hp})
            if isinstance(hp, str) and hp:
                current_path = hp
                if artifact_video:
                    artifact_video(trace_id, current_path)

    # Post-pass: compute lock metrics/frames using cleanup so Film2 QA can operate on the FINAL path.
    if isinstance(current_path, str) and current_path:
        cc_args: Dict[str, Any] = {"src": current_path}
        if isinstance(cid, str) and cid:
            cc_args["cid"] = cid
        if isinstance(trace_id, str) and trace_id:
            cc_args["trace_id"] = trace_id
        # Preserve Film2 identifiers for distillation joins.
        for k in ("film_id", "scene_id", "shot_id", "act_id"):
            v = hv_tool_args.get(k)
            if isinstance(v, (str, int)) and str(v).strip():
                cc_args[k] = str(v).strip()
        # Preserve lock bundle snapshot that was used to guide generation (optional).
        if isinstance(hv_tool_args.get("locks"), dict):
            cc_args["lock_bundle"] = hv_tool_args.get("locks")
        if log_fn and isinstance(trace_id, str) and trace_id:
            log_fn("film2.pass_cleanup_start", trace_id=trace_id, shot_id=shot_id, src=current_path)
        trace_append("film2.fix.cleanup.start", {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id, "src": current_path})
        cc = await http_tool_run("video.cleanup", cc_args)
        if isinstance(cc, dict):
            segment_log.append(cc)
        ccr = (cc.get("result") or {}) if isinstance(cc, dict) else {}
        outp = ccr.get("path") if isinstance(ccr, dict) else None
        if log_fn and isinstance(trace_id, str) and trace_id:
            log_fn("film2.pass_cleanup_finish", trace_id=trace_id, shot_id=shot_id, out_path=outp)
        trace_append("film2.fix.cleanup.finish", {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id, "out_path": outp})

    if log_fn and isinstance(trace_id, str) and trace_id:
        log_fn(
            "film2.pass_video_complete",
            trace_id=trace_id,
            shot_id=shot_id,
            final_path=current_path,
            segment_results=len(segment_log),
        )
    trace_append(
        "film2.video.complete",
        {"trace_id": trace_id, "shot_id": shot_id, "scene_id": scene_id, "act_id": act_id, "final_path": current_path, "segment_results": len(segment_log)},
    )
    return current_path, segment_log


