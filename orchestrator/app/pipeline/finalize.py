from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .assets import collect_urls as assets_collect_urls
from ..traces.writer import log_artifact as _trace_log_artifact


def finalize_tool_phase(
    messages: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    master_seed: int,
    trace_id: str,
    model_name: str,
    absolutize_url: Callable[[str], str],
    estimate_usage_fn: Callable[[List[Dict[str, Any]], str], Dict[str, Any]],
    envelope_builder: Callable[..., Dict[str, Any]],
    state_dir: str,
    committee_info: Optional[Dict[str, Any]] = None,
    qa_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
	Compose an OpenAI-compatible envelope from tool results with warnings and asset links.
	- Never raises
	- Always returns a dict with optional _meta.errors
	"""
    # Build asset URLs
    asset_urls = assets_collect_urls(tool_results, absolutize_url)
	# Derive prompt/params when available
    prompt_text = ""
    meta_used: Dict[str, Any] = {}
    if isinstance(tool_results, list) and tool_results:
        first = (tool_results[0] or {}).get("result") or {}
        if isinstance(first.get("meta"), dict):
            meta_used = first.get("meta") or {}
            pt = meta_used.get("prompt")
            if isinstance(pt, str) and pt.strip():
                prompt_text = pt.strip()
    # Detect tool errors and surface them
    tool_errors: List[Dict[str, Any]] = []
    for tr in (tool_results or []):
        if not isinstance(tr, dict):
            continue
        name_t = (tr.get("name") or tr.get("tool") or "tool")
        err_obj: Any = None
        if isinstance(tr.get("error"), (str, dict)):
            err_obj = tr.get("error")
        res_t = tr.get("result") if isinstance(tr.get("result"), dict) else {}
        if isinstance(res_t.get("error"), (str, dict)):
            err_obj = res_t.get("error")
        if err_obj is not None:
            code = (err_obj.get("code") if isinstance(err_obj, dict) else str(err_obj))
            status = None
            if isinstance(err_obj, dict):
                status = err_obj.get("status") or err_obj.get("_http_status") or err_obj.get("http_status")
            message = (err_obj.get("message") if isinstance(err_obj, dict) else None) or ""
            tool_errors.append({"tool": str(name_t), "code": (code or ""), "status": status, "message": message, "error": err_obj})
    warnings = tool_errors[:] if tool_errors else []
    # Compose assistant text (generalized, tool-specific, human-readable)
    summary_lines: List[str] = []
    # Committee outcome context (optional)
    comm_act = ""
    comm_rat = ""
    if isinstance(committee_info, dict):
        comm_act = str((committee_info.get("action") or "")).strip().lower()
        comm_rat = str(committee_info.get("rationale") or "")
    domain_metrics: Dict[str, Any] = {}
    if isinstance(qa_metrics, dict):
        domain_section = qa_metrics.get("domain")
        if isinstance(domain_section, dict):
            domain_metrics = domain_section
    if (not asset_urls) and tool_errors:
        # Lead with the specific failing tool when available
        first_tool = (tool_errors[0] or {}).get("tool") or "tool"
        summary_lines.append(f"The tool `{first_tool}` failed to run.")
        if comm_act == "fail":
            summary_lines.append("Post-run checks could not automatically fix the issues; returning the best available information and errors.")
        for e in tool_errors:
            line = f"- {e.get('tool')}: {e.get('code') or 'error'}"
            if e.get("status") is not None:
                line += f" (status {e.get('status')})"
            if e.get("message"):
                line += f" — {e.get('message')}"
            summary_lines.append(line)
        if prompt_text:
            summary_lines.append(f"Prompt attempted:\n“{prompt_text}”")
        # Provide effective params when known
        if isinstance(meta_used, dict) and meta_used:
            param_bits = []
            for k in ("width", "height", "steps", "cfg", "sampler", "scheduler", "model", "seed"):
                v = meta_used.get(k)
                if v is not None and v != "":
                    param_bits.append(f"{k}={v}")
            if param_bits:
                summary_lines.append("Parameters: " + ", ".join(param_bits))
        summary_lines.append("No assets were produced.")
    else:
        # Provide a short explanation of what was done, not just raw prompt/URLs
        tools_used = []
        for tr in (tool_results or []):
            if isinstance(tr, dict) and isinstance(tr.get("result"), dict) and not tr.get("error"):
                tools_used.append(str(tr.get("name") or tr.get("tool") or "tool"))
        tools_used = list(dict.fromkeys([t for t in tools_used if t]))
        if tools_used:
            summary_lines.append("I completed your request using: " + ", ".join(tools_used) + ".")
        if comm_act == "revise":
            summary_lines.append("I ran an extra refinement pass based on post-run checks.")
        elif comm_act == "fail":
            summary_lines.append("Post-run checks indicated issues; returning best available artifacts and details.")
        if domain_metrics:
            domain_notes: List[str] = []
            images_domain = domain_metrics.get("images") if isinstance(domain_metrics.get("images"), dict) else {}
            videos_domain = domain_metrics.get("videos") if isinstance(domain_metrics.get("videos"), dict) else {}
            audio_domain = domain_metrics.get("audio") if isinstance(domain_metrics.get("audio"), dict) else {}
            hands_ratio = images_domain.get("hands_ok_ratio") if isinstance(images_domain, dict) else None
            if isinstance(hands_ratio, (int, float)) and hands_ratio < 0.85:
                domain_notes.append("image limb/hands QA is still borderline; consider another targeted pass if needed")
            artifact_ratio_img = images_domain.get("artifact_ok_ratio") if isinstance(images_domain, dict) else None
            if isinstance(artifact_ratio_img, (int, float)) and artifact_ratio_img < 0.9:
                domain_notes.append("image artifact cleanup may need additional attention")
            seam_ratio_vid = videos_domain.get("seam_ok_ratio") if isinstance(videos_domain, dict) else None
            if isinstance(seam_ratio_vid, (int, float)) and seam_ratio_vid < 0.9:
                domain_notes.append("video seams remain imperfect; consider rerunning seam-fix steps")
            clip_ratio = audio_domain.get("clipping_ratio") if isinstance(audio_domain, dict) else None
            if isinstance(clip_ratio, (int, float)) and clip_ratio > 0.0:
                domain_notes.append("audio clipping was detected; review loudness/mastering settings")
            if domain_notes:
                summary_lines.append("Post-run QA notes:")
                for note in domain_notes[:3]:
                    summary_lines.append(f"- {note}")
        if prompt_text:
            summary_lines.append(f"Summary: generated content aligned to your request.\nPrompt used:\n“{prompt_text}”")
        # Include effective params when present
        for k in ("width", "height", "steps", "cfg", "sampler", "scheduler", "model", "seed"):
            v = meta_used.get(k)
            if v is not None and v != "":
                summary_lines.append(f"{k}: {v}")
        if asset_urls:
            summary_lines.append("Assets:")
            summary_lines.extend([f"- {u}" for u in asset_urls])
        if warnings:
            summary_lines.append("")
            summary_lines.append("Warnings:")
            for e in warnings[:5]:
                code = (e.get("error") or {}).get("code") if isinstance(e.get("error"), dict) else (e.get("code") or "tool_error")
                message = (e.get("error") or {}).get("message") if isinstance(e.get("error"), dict) else (e.get("message") or "")
                summary_lines.append(f"- {code}: {message}")
    final_text = "\n".join(summary_lines) if summary_lines else "Generation completed."
    usage = estimate_usage_fn(messages, final_text)
    response = envelope_builder(
        ok=bool(asset_urls) and not bool(tool_errors),
        text=final_text,
        error=None,
        usage=usage,
        model=model_name,
        seed=master_seed,
        id_="orc-1",
    )
    if warnings:
        try:
            response["_meta"] = {"errors": warnings}
        except Exception:
            pass
    # Artifact trace logging (distillation-friendly)
    try:
        for u in asset_urls or []:
            if not isinstance(u, str) or not u.strip():
                continue
            _trace_log_artifact(
                state_dir,
                trace_id,
                {
                    "artifact_id": u,
                    "kind": "asset",
                    "source_tool": None,
                    "paths": [u],
                    "meta": {
                        "quality_profile": None,
                    },
                },
            )
    except Exception:
        # Tracing must not affect user-visible responses
        pass
    return response


def compose_openai_response(
	text: str,
	usage: Dict[str, Any],
	model_name: str,
	seed: int,
	id_: str,
	*,
	envelope_builder: Callable[..., Dict[str, Any]],
	prebuilt: Optional[Dict[str, Any]] = None,
	artifacts: Optional[Dict[str, Any]] = None,
	final_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
	Compose the final OpenAI-compatible response envelope with optional artifacts and embedded final_env.
	Does not perform I/O. Returns the response dict.
	"""
	response = envelope_builder(
		ok=True,
		text=text,
		error=None,
		usage=usage,
		model=model_name,
		seed=seed,
		id_=id_ or "orc-1",
	)
	if isinstance(prebuilt, dict) and prebuilt:
		response = prebuilt
	if isinstance(artifacts, dict) and artifacts:
		response["artifacts"] = artifacts
	if isinstance(final_env, dict) and final_env:
		response["envelope"] = final_env
	return response


