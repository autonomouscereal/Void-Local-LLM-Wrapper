from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os, json, uuid, time, asyncio, urllib.request, os.path
import logging, sys, traceback
from urllib.parse import quote, urlsplit, urlparse
import base64 as _b64
from typing import Optional
from app.state.checkpoints import append_event as checkpoints_append_event
from app.trace_utils import emit_trace as _emit_trace
from app.analysis.media import analyze_image as _qa_analyze_image, analyze_image_regions as _qa_analyze_image_regions  # type: ignore
from app.json_parser import JSONParser
import httpx


router = APIRouter()
log = logging.getLogger("orchestrator.toolrun")


def _build_success_envelope(result: dict | None, rid: str) -> dict:
	"""
	Canonical success envelope for tool-ish routes.
	Always includes schema_version, request_id, ok, result, error.
	"""
	return {
		"schema_version": 1,
		"request_id": rid,
		"ok": True,
		"result": result or {},
		"error": None,
	}


def _build_error_envelope(code: str, message: str, rid: str, status: int, details: dict | None = None) -> dict:
	"""
	Canonical error envelope for tool-ish routes.
	HTTP status is always 200; semantic status lives on error.status.
	"""
	err_details = dict(details or {})
	err_details.setdefault("status", int(status))
	return {
		"schema_version": 1,
		"request_id": rid,
		"ok": False,
		"result": None,
		"error": {
			"code": code,
			"message": message,
			"status": int(status),
			"details": err_details,
		},
	}


def _ok_response(result: dict | None, rid: str) -> JSONResponse:
	return JSONResponse(_build_success_envelope(result or {}, rid), status_code=200)


def _err_response(code: str, message: str, rid: str, status: int = 200, details: dict | None = None) -> JSONResponse:
	env = _build_error_envelope(code, message, rid, status=status, details=details)
	return JSONResponse(env, status_code=200)


class ToolEnvelope:
	@staticmethod
	def success(result: dict, *, request_id: str | None = None) -> JSONResponse:
		rid = request_id or uuid.uuid4().hex
		return _ok_response(result, rid)

	@staticmethod
	def failure(
		code: str,
		message: str,
		*,
		status: int,
		details: dict | None = None,
		request_id: str | None = None,
	) -> JSONResponse:
		rid = request_id or uuid.uuid4().hex
		return _err_response(code, message, rid, status=int(status), details=details)


def _ok_env(ok: bool, **kwargs) -> dict:
	out = {"ok": bool(ok)}
	out.update(kwargs or {})
	return out

async def _post_json(url: str, obj: dict) -> dict:
	async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
		resp = await client.post(url, json=obj)
		parser = JSONParser()
		# ComfyUI /prompt returns an object with at least prompt_id/client_id,
		# but we treat the entire JSON body as a free-form mapping so callers
		# can inspect any fields they care about.
		data = parser.parse_superset(resp.text or "{}", dict)["coerced"]
		if 200 <= resp.status_code < 300 and isinstance(data, dict):
			return _ok_env(True, **data)
		return _ok_env(False, code="http_error", status=resp.status_code, detail=data)


async def _get_json(url: str) -> dict:
	async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
		resp = await client.get(url, headers={"accept": "application/json"})
		parser = JSONParser()
		# ComfyUI /history/{id} returns a history object whose exact keys are
		# workflow-dependent; keep the response as an untyped mapping.
		data = parser.parse_superset(resp.text or "{}", dict)["coerced"]
		if 200 <= resp.status_code < 300 and isinstance(data, dict):
			return _ok_env(True, **data)
		return _ok_env(False, code="http_error", status=resp.status_code, detail=data)


def _read_text(path: str) -> str:
	with open(path, "r", encoding="utf-8") as f:
		return f.read()


def _nodes_by_type(g: dict, *class_types: str) -> list[str]:
	want = set(class_types or ())
	out: list[str] = []
	for nid, node in g.items():
		if not isinstance(node, dict):
			continue
		ctype = node.get("class_type")
		if isinstance(ctype, str) and ctype in want and isinstance(node.get("inputs"), dict):
			out.append(str(nid))
	return out


def _set_inputs(g: dict, nid: str, **kwargs) -> None:
	inputs = g[str(nid)].setdefault("inputs", {})
	for k, v in kwargs.items():
		if v is not None:
			inputs[k] = v


def _ensure_api_prompt_graph(wf: dict) -> dict:
    # Accept {"prompt": {...}} or direct mapping
    if isinstance(wf, dict) and "prompt" in wf and isinstance(wf["prompt"], dict):
        return wf["prompt"]
    if isinstance(wf, dict) and "nodes" in wf:
        # UI-exported workflows are not supported here; return empty mapping so caller can surface an error.
        return {}
    if isinstance(wf, dict):
        return wf
    # Invalid shape; return empty mapping so caller can surface an error envelope.
    return {}


def _get_ref_node_id(ref) -> str | None:
    # Comfy API references are typically ["12", 0]
    if isinstance(ref, list) and len(ref) >= 1:
        return str(ref[0])
    return None


def _first_node_id_by_class(graph: dict, class_name_prefix: str) -> str | None:
    for nid, node in graph.items():
        ct = (node.get("class_type") or "").strip() if isinstance(node, dict) else ""
        if isinstance(ct, str) and ct.startswith(class_name_prefix):
            return str(nid)
    return None


def _resolve_bindings(graph: dict) -> dict:
    ks_id = (_first_node_id_by_class(graph, "KSampler")
             or _first_node_id_by_class(graph, "KSamplerAdvanced"))
    if not ks_id:
        return {}
    ks_in = graph[ks_id].get("inputs", {})
    pos_id = _get_ref_node_id(ks_in.get("positive"))
    neg_id = _get_ref_node_id(ks_in.get("negative"))
    if not pos_id or not neg_id:
        # Missing positive/negative refs: return empty bindings so callers can
        # surface structured errors instead of raising.
        return {}
    latent_id = (_first_node_id_by_class(graph, "EmptyLatentImage")
                 or _first_node_id_by_class(graph, "LatentImage"))
    ckpt_id = (_first_node_id_by_class(graph, "CheckpointLoaderSimple")
               or _first_node_id_by_class(graph, "CheckpointLoaderSimpleSDXL"))
    return {"ks": ks_id, "pos": pos_id, "neg": neg_id, "latent": latent_id, "ckpt": ckpt_id}


def _validate_api_graph(graph: dict) -> list[str]:
    problems: list[str] = []
    if not isinstance(graph, dict) or not graph:
        problems.append("graph_not_mapping_or_empty")
        return problems
    for nid, node in graph.items():
        if not isinstance(node, dict):
            problems.append(f"node_{nid}_not_object")
            continue
        if "class_type" not in node:
            problems.append(f"node_{nid}_missing_class_type")
        if "inputs" not in node or not isinstance(node.get("inputs"), dict):
            problems.append(f"node_{nid}_missing_inputs")
    return problems


def _apply_overrides(graph: dict, bind: dict, args: dict) -> None:
    # Positive / Negative
    if bind.get("pos") and ("prompt" in args):
        graph[bind["pos"]]["inputs"]["text"] = str(args.get("prompt") or "")
    if bind.get("neg") and (("negative" in args) or ("negative_prompt" in args)):
        graph[bind["neg"]]["inputs"]["text"] = str(args.get("negative") or args.get("negative_prompt") or "")
    # Sampler
    if bind.get("ks"):
        ks_in = graph[bind["ks"]]["inputs"]
        # Seed: allow explicit None to mean "leave as-is".
        if "seed" in args and args.get("seed") is not None:
            ks_in["seed"] = int(args["seed"])
        # Only override numeric fields when the arg is non-None and non-empty;
        # this avoids int/float(None) while still letting the underlying graph
        # defaults apply when values are omitted.
        if "steps" in args and args.get("steps") not in (None, ""):
            ks_in["steps"] = int(args["steps"])
        if "cfg" in args and args.get("cfg") not in (None, ""):
            ks_in["cfg"] = float(args["cfg"])
        if "sampler" in args:
            ks_in["sampler_name"] = str(args["sampler"]).strip()
        if "sampler_name" in args:
            ks_in["sampler_name"] = str(args["sampler_name"]).strip()
        if "scheduler" in args:
            ks_in["scheduler"] = str(args["scheduler"]).strip()
    # Latent size
    if bind.get("latent"):
        li = graph[bind["latent"]]["inputs"]
        # Width / height: treat None/"" as "not set".
        if "width" in args and args.get("width") not in (None, ""):
            li["width"] = int(args["width"])
        if "height" in args and args.get("height") not in (None, ""):
            li["height"] = int(args["height"])
        # If width/height were not provided but a resolution string like
        # "1920x1080" is present, parse it and use that as a best-effort size.
        width_missing = ("width" not in args) or (args.get("width") in (None, ""))
        height_missing = ("height" not in args) or (args.get("height") in (None, ""))
        if width_missing and height_missing:
            res = args.get("resolution")
            if isinstance(res, str) and "x" in res.lower():
                w_str, h_str = res.lower().split("x", 1)
                try:
                    w_val = int(w_str.strip())
                    h_val = int(h_str.strip())
                except Exception:
                    w_val = h_val = None
                if isinstance(w_val, int) and isinstance(h_val, int) and w_val > 0 and h_val > 0:
                    li["width"] = w_val
                    li["height"] = h_val
    # Model checkpoint
    if bind.get("ckpt") and ("model" in args) and args.get("model"):
        graph[bind["ckpt"]]["inputs"]["ckpt_name"] = str(args["model"]).strip()


def _extract_node_subset(raw: dict) -> dict:
	out: dict = {}
	if isinstance(raw, dict):
		for k, v in raw.items():
			if isinstance(v, dict) and "class_type" in v and "inputs" in v:
				out[str(k)] = v
	return out

def _coerce_ui_export_to_api_graph(wf: dict) -> dict:
	"""
	Best-effort coercion: if a UI export with 'nodes' array is provided, convert it into
	a simple API graph mapping of {id: {class_type, inputs}}. Otherwise, return any
	subset that already looks API-like.
	"""
	if isinstance(wf, dict) and isinstance(wf.get("nodes"), list):
		graph: dict = {}
		next_id = 1
		for node in wf["nodes"]:
			if isinstance(node, dict) and "class_type" in node and "inputs" in node:
				graph[str(next_id)] = {"class_type": node["class_type"], "inputs": node["inputs"]}
				next_id += 1
		return graph
	return _extract_node_subset(wf) if isinstance(wf, dict) else {}

def patch_workflow_in_place(g: dict, args: dict) -> None:
	# 1) Prompt / Negative (support common CLIP encoders)
	prompt = args.get("prompt")
	negative = args.get("negative") if args.get("negative") is not None else args.get("negative_prompt")
	clip_nodes = _nodes_by_type(g, "CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeAdvanced")
	if clip_nodes:
		if prompt is not None:
			_set_inputs(g, clip_nodes[0], text=str(prompt))
		if negative is not None and len(clip_nodes) > 1:
			_set_inputs(g, clip_nodes[1], text=str(negative))
	# 2) Sampler settings
	sampler_name = args.get("sampler") or args.get("sampler_name")
	ksamplers = _nodes_by_type(g, "KSampler", "KSamplerAdvanced")
	for nid in ksamplers:
		_set_inputs(
			g, nid,
			seed=args.get("seed"),
			steps=args.get("steps"),
			cfg=args.get("cfg"),
			sampler_name=(str(sampler_name) if sampler_name is not None else None),
			scheduler=args.get("scheduler"),
		)
	# 3) Latent size
	latent_nodes = _nodes_by_type(g, "EmptyLatentImage")
	if latent_nodes:
		_set_inputs(
			g, latent_nodes[0],
			width=args.get("width"),
			height=args.get("height"),
		)
	# 4) Model checkpoint
	ckpt_nodes = _nodes_by_type(g, "CheckpointLoaderSimple", "CheckpointLoaderSimpleSDXL")
	model = args.get("model") or args.get("ckpt_name")
	if ckpt_nodes and model:
		_set_inputs(g, ckpt_nodes[0], ckpt_name=str(model))


def _build_view_url(base: str, filename: str, subfolder: str, ftype: str) -> str:
	return f"{base.rstrip('/')}/view?filename={quote(filename or '')}&subfolder={quote((subfolder or ''))}&type={quote(ftype or 'output')}"


STATE_DIR_LOCAL = os.path.join(os.getenv("UPLOAD_DIR", "/workspace/uploads"), "state")

# Runtime guard: force loopback base in host networking even if a container hostname sneaks in
def _force_loopback(url: str) -> str:
	u = urlsplit(url)
	host = "127.0.0.1"
	port = u.port or (8188 if (u.scheme or "http") == "http" else 8188)
	return f"{u.scheme or 'http'}://{host}:{port}"
_raw_comfy = os.getenv("COMFYUI_API_URL") or "http://127.0.0.1:8188"
COMFY_BASE = _force_loopback(_raw_comfy)

def _append_jsonl(path: str, obj: dict) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@router.post("/tool.run")
async def tool_run(req: Request):
	raw = await req.body()
	parser = JSONParser()
	body = parser.parse_superset(
		raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw or ""),
		{"name": str, "args": dict, "trace_id": str, "cid": str},
	)["coerced"]
	trace_val = None
	if isinstance(body, dict):
		if isinstance(body.get("trace_id"), str) and body.get("trace_id"):
			trace_val = str(body.get("trace_id"))
		elif isinstance(body.get("cid"), str) and body.get("cid"):
			trace_val = str(body.get("cid"))
	rid = trace_val or "tool.run"
	if not isinstance(body, dict):
		return ToolEnvelope.failure(
			"invalid_body_type",
			"Body must be a JSON object",
			status=422,
			request_id=rid,
		)
	name = (body.get("name") or "").strip()
	args = body.get("args") or {}

	# For non-image tools, call the internal orchestrator tool implementation directly
	# instead of routing back through the executor. This avoids recursion
	# (executor → /tool.run → executor) and ensures each planned step executes once.
	if name != "image.dispatch":
		from app.main import execute_tool_call  # type: ignore
		call = {
			"name": name,
			"arguments": (args if isinstance(args, dict) else {}),
			"trace_id": rid,
		}
		# Let execute_tool_call enforce its own invariants. Any unexpected exception
		# will naturally raise and surface as a 500 from FastAPI instead of being
		# hidden behind a generic try/except wrapper here.
		res = await execute_tool_call(call)
		if isinstance(res, dict) and isinstance(res.get("result"), dict):
			return ToolEnvelope.success(res["result"], request_id=rid)
		err_obj = res.get("error") if isinstance(res, dict) else None
		if isinstance(err_obj, dict):
			# Normalize inner tool error so we always expose a clear code/message/status,
			# while still preserving full traceback/stack fields for debugging.
			code = str(err_obj.get("code") or "tool_error")
			raw_msg = err_obj.get("message")
			status_val = err_obj.get("status") or err_obj.get("_http_status") or 422
			try:
				status_int = int(status_val)
			except Exception:
				status_int = 422
			message = (
				str(raw_msg).strip()
				if isinstance(raw_msg, str) and raw_msg.strip()
				else f"{name or 'tool'} failed with status {status_int}"
			)
			normalized_details = dict(err_obj)
			normalized_details.setdefault("code", code)
			normalized_details.setdefault("message", message)
			normalized_details.setdefault("status", status_int)
			# Preserve any traceback/details from inner tool handlers so executor logs see real stack lines.
			return ToolEnvelope.failure(code, message, status=status_int, request_id=rid, details=normalized_details)
		if isinstance(err_obj, str):
			# Attach any traceback captured at the tool layer so callers can see both
			# a clean message and the underlying stack.
			trace_val = None
			if isinstance(res, dict) and isinstance(res.get("traceback"), str):
				trace_val = res.get("traceback")
			details: dict = {}
			if trace_val:
				details["traceback"] = trace_val
			return ToolEnvelope.failure("tool_error", err_obj, status=422, request_id=rid, details=details)
		# Fallback: surface the raw response for debugging instead of hiding it behind a generic message.
		log.error("[toolrun] tool=%s returned non-standard error payload: %r", name, res)
		return ToolEnvelope.failure(
			"tool_error",
			"tool returned non-standard error payload",
			status=500,
			request_id=rid,
			details={"raw": res},
		)

	# For image.dispatch, use the existing Comfy pipeline below.
	try:
		# Inline graph or path
		inline = args.get("workflow_graph")
		if inline is not None:
			wf_obj = inline
			wf_path = "(inline)"
		else:
			wf_path = (args.get("workflow_path")
			           or os.getenv("COMFY_WORKFLOW_PATH")
			           or "/workspace/services/image/workflows/stock_smoke.json")
			if os.path.exists(wf_path):
				wf_text = _read_text(wf_path)
				parser = JSONParser()
				# Workflow graphs are open-ended; parse with open schema.
				wf_obj = parser.parse_superset(wf_text, {})["coerced"]
			else:
				# Fallback: proceed without file; we'll synthesize a valid graph below
				wf_obj = {}

		# Try to ensure API prompt mapping, with optional coercion/subset
		prompt_graph = None
		if isinstance(wf_obj, dict) and "prompt" in wf_obj and isinstance(wf_obj["prompt"], dict):
			prompt_graph = wf_obj["prompt"]
		elif isinstance(wf_obj, dict) and "nodes" not in wf_obj:
			prompt_graph = wf_obj

		if not isinstance(prompt_graph, dict) and bool(args.get("autofix_422", False)):
			log.info("[comfy] attempting UI→API coercion for workflow shape")
			coerced = _coerce_ui_export_to_api_graph(wf_obj)
			if isinstance(coerced, dict):
				prompt_graph = coerced
			if not isinstance(prompt_graph, dict) and isinstance(coerced, dict):
				subset = _extract_node_subset(coerced)
				if subset:
					prompt_graph = subset

			# Lightweight auto-bind: if the KSampler lacks positive/negative refs, inject CLIPTextEncode nodes and wire them.
			if isinstance(prompt_graph, dict):
				ks_id = _first_node_id_by_class(prompt_graph, "KSampler") or _first_node_id_by_class(prompt_graph, "KSamplerAdvanced")
				if ks_id:
					ks_in = prompt_graph[str(ks_id)].setdefault("inputs", {})
					pos_id = _get_ref_node_id(ks_in.get("positive"))
					neg_id = _get_ref_node_id(ks_in.get("negative"))
					if not pos_id or not neg_id:
						ckpt_id = _first_node_id_by_class(prompt_graph, "CheckpointLoaderSimple") or _first_node_id_by_class(prompt_graph, "CheckpointLoaderSimpleSDXL")
						if ckpt_id:
							# Find next free integer id(s)
							def _next_id(g: dict) -> str:
								max_id = 0
								for k in g.keys():
									s = str(k)
									if s.isdigit():
										max_id = max(max_id, int(s))
								return str(max_id + 1)
							if not pos_id:
								new_pos_id = _next_id(prompt_graph)
								prompt_graph[new_pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [str(ckpt_id), 1], "text": str(args.get("prompt") or "")}}
								ks_in["positive"] = [new_pos_id, 0]
							if not neg_id:
								new_neg_id = _next_id(prompt_graph)
								prompt_graph[new_neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [str(ckpt_id), 1], "text": str(args.get("negative") or "")}}
								ks_in["negative"] = [new_neg_id, 0]

			if not isinstance(prompt_graph, dict):
				# Synthesize a valid graph instead of failing
				model_ckpt = str(args.get("model") or "sd_xl_base_1.0.safetensors")
				prompt_text = str(args.get("prompt") or "")
				neg_text = str(args.get("negative") or "")
				width = int(args.get("width") or 1024)
				height = int(args.get("height") or 1024)
				steps = int(args.get("steps") or 32)
				cfg = float(args.get("cfg") or 5.5)
				prompt_graph = {
					"3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model_ckpt}},
					"4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
					"8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": prompt_text}},
					"9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": neg_text}},
					"5": {"class_type": "KSampler", "inputs": {
						"seed": int(args.get("seed") or 123456789),
						"steps": steps, "cfg": cfg,
						"sampler_name": str(args.get("sampler") or args.get("sampler_name") or "euler"),
						"scheduler": str(args.get("scheduler") or "normal"),
						"denoise": 1.0,
						"model": ["3", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["4", 0]
					}},
					"6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 2]}},
					"7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
				}

		# Validate and bind
		problems = _validate_api_graph(prompt_graph)
		if problems:
			# Replace with synthesized valid graph instead of failing
			model_ckpt = str(args.get("model") or "sd_xl_base_1.0.safetensors")
			prompt_text = str(args.get("prompt") or "")
			neg_text = str(args.get("negative") or "")
			width = int(args.get("width") or 1024)
			height = int(args.get("height") or 1024)
			steps = int(args.get("steps") or 32)
			cfg = float(args.get("cfg") or 5.5)
			prompt_graph = {
				"3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model_ckpt}},
				"4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
				"8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": prompt_text}},
				"9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": neg_text}},
				"5": {"class_type": "KSampler", "inputs": {
					"seed": int(args.get("seed") or 123456789),
					"steps": steps, "cfg": cfg,
					"sampler_name": str(args.get("sampler") or args.get("sampler_name") or "euler"),
					"scheduler": str(args.get("scheduler") or "normal"),
					"denoise": 1.0,
					"model": ["3", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["4", 0]
				}},
				"6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 2]}},
				"7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
			}
		bind = _resolve_bindings(prompt_graph)
		_apply_overrides(prompt_graph, bind, args)
		# Ensure SaveImage nodes have a filename_prefix (required by newer ComfyUI)
		_client_id = uuid.uuid4().hex
		cid = str(args["cid"]).strip()
		trace = (args.get("trace_id") or "").strip() if isinstance(args.get("trace_id"), str) else ""
		step_id = (args.get("step_id") or "").strip() if isinstance(args.get("step_id"), str) else ""
		for nid, node in (prompt_graph or {}).items():
			if isinstance(node, dict) and (node.get("class_type") or "") == "SaveImage":
				ins = node.setdefault("inputs", {})
				if not isinstance(ins.get("filename_prefix"), str) or not (ins.get("filename_prefix") or "").strip():
					prefix = f"{cid}_{trace}_{step_id}" if (cid or trace or step_id) else f"void_{_client_id}"
					ins["filename_prefix"] = prefix

		client_id = _client_id
		_emit_trace(STATE_DIR_LOCAL, "global", "comfyui.submit", {"t": int(time.time()*1000), "base": COMFY_BASE, "workflow_path": wf_path})
		log.info("[comfy] POST /prompt url=%s", COMFY_BASE.rstrip("/") + "/prompt")
		submit_res = await _post_json(COMFY_BASE.rstrip("/") + "/prompt", {"prompt": prompt_graph, "client_id": client_id})
		if not bool(submit_res.get("ok")):
			return ToolEnvelope.failure(
				"http_error",
				"comfy submit failed",
				status=502,
				request_id=rid,
				details=submit_res,
			)
		prompt_id = submit_res.get("prompt_id") or submit_res.get("promptId") or ""
		log.info("[comfy] prompt_id=%s client_id=%s", prompt_id, client_id)

		images = []
		_first_hist = True
		_poll_delay = 0.25
		_POLL_MAX = 2.0
		while True:
			await asyncio.sleep(_poll_delay)
			log.info("[comfy] polling /history/%s", prompt_id)
			hist = await _get_json(f"{COMFY_BASE.rstrip('/')}/history/{prompt_id}")
			if not bool(hist.get("ok")):
				return ToolEnvelope.failure(
					"http_error",
					"comfy history failed",
					status=502,
					request_id=rid,
					details={"prompt_id": prompt_id, **hist},
				)
			if _first_hist:
				_emit_trace(STATE_DIR_LOCAL, "global", "comfyui.history", {"t": int(time.time()*1000), "prompt_id": prompt_id})
				_first_hist = False
			if not isinstance(hist, dict):
				continue
			# ComfyUI /history responses can appear in multiple shapes depending on
			# version / plugins:
			#   1) {"history": { "<pid>": {...} }, ...}
			#   2) {"<pid>": {...}, ...}
			#   3) {"outputs": {...}, "status": {...}, ...}  (direct entry)
			# Normalize to a single entry dict so downstream code doesn't spin
			# forever looking up hist[prompt_id] when the data lives under 'history'.
			entry: dict | None = None
			# Strip helper fields from _get_json (_ok_env wrapper)
			data_candidates = dict(hist)
			for k in ("ok", "code", "status", "detail"):
				data_candidates.pop(k, None)
			# Shape 1: nested history map
			hblock = data_candidates.get("history")
			if isinstance(hblock, dict):
				entry = hblock.get(prompt_id) or next(iter(hblock.values()), None)
			# Shape 2: top-level pid key
			if entry is None and prompt_id in data_candidates:
				val = data_candidates.get(prompt_id)
				if isinstance(val, dict):
					entry = val
			# Shape 3: direct entry with outputs/status
			if entry is None and isinstance(data_candidates.get("outputs"), dict):
				entry = data_candidates
			if not isinstance(entry, dict):
				entry = {}
			# Detect terminal error/success states when available
			status_obj = entry.get("status") or {}
			state = str((status_obj.get("status") or "")).lower()
			if state in ("error", "failed", "canceled", "cancelled"):
				return ToolEnvelope.failure(
					"comfy_error",
					"workflow reported error state",
					status=422,
					request_id=rid,
					details={"prompt_id": prompt_id, "status": status_obj},
				)
			outs = entry.get("outputs") or {}
			if not outs:
				# progressive backoff to avoid busy spin
				_poll_delay = _POLL_MAX if _poll_delay >= _POLL_MAX else min(_POLL_MAX, _poll_delay * 2.0)
				# If history reports a completed/executed state but no outputs, fail deterministically
				status_obj = entry.get("status") or {}
				state = str((status_obj.get("status") or "")).lower()
				if state in ("completed", "success", "executed"):
					return ToolEnvelope.failure(
						"no_outputs",
						"workflow completed without outputs",
						status=422,
						request_id=rid,
						details={"prompt_id": prompt_id, "status": status_obj},
					)
				continue
			for _, out in outs.items():
				for im in (out.get("images") or []):
					fn = im.get("filename")
					sf = im.get("subfolder") or ""
					tp = im.get("type") or "output"
					if fn:
						images.append({
							"filename": fn,
							"subfolder": sf,
							"type": tp,
							"view_url": _build_view_url(COMFY_BASE, fn, sf, tp),
						})
			if images:
				break

		# Should not reach here without images due to terminal checks above

		# Canonical ids/meta (include flattened lists)
		image_files = []
		view_urls = []
		for im in images:
			fn = (im.get("subfolder") or "").strip()
			if fn:
				image_files.append(f"{fn}/{im.get('filename')}")
			else:
				image_files.append(im.get("filename"))
			view_urls.append(im.get("view_url"))
		_emit_trace(STATE_DIR_LOCAL, "global", "comfyui.done", {"t": int(time.time()*1000), "count": len(images)})
		log.info("[comfy] images=%d", len(images))

		# Echo effective params if present
		eff = {}
		for k_src, k_dst in (("seed","seed"),("steps","steps"),("cfg","cfg"),("sampler","sampler"),("sampler_name","sampler"),
		                     ("scheduler","scheduler"),("width","width"),("height","height"),("model","model")):
			if args.get(k_src) is not None and eff.get(k_dst) is None:
				eff[k_dst] = args.get(k_src)

		result = {
			"ids": {
				"prompt_id": prompt_id,
				"client_id": client_id,
				"images": images,
				"image_files": image_files,
			},
			"meta": {
				"submitted": True,
				"workflow_path": wf_path,
				"comfy_base": COMFY_BASE,
				"view_urls": view_urls,
				"image_count": len(images),
				"prompt": str(args.get("prompt") or ""),
				"negative": str(args.get("negative") or args.get("negative_prompt") or ""),
				**eff,
			},
		}
		# Persist artifacts under orchestrator /uploads and return absolute URLs for UI
		from app.main import UPLOAD_DIR as _UPLOAD_DIR, PUBLIC_BASE_URL as _PUBLIC_BASE_URL  # type: ignore
		from app.locks.runtime import bundle_to_image_locks as _lock_to_assets  # type: ignore
		from app.services.image.analysis.locks import compute_region_scores as _qa_region_scores, compute_face_lock_score as _qa_face_lock  # type: ignore
		import os as _os
		# Use prompt_id as the canonical image cid for artifact paths
		_cid = prompt_id or client_id
		save_dir = _os.path.join(_UPLOAD_DIR, "artifacts", "image", _cid)
		_os.makedirs(save_dir, exist_ok=True)
		orch_urls: list[str] = []
		saved_paths: list[str] = []
		async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
			for im in images:
				fn = im.get("filename")
				sf = im.get("subfolder") or ""
				tp = im.get("type") or "output"
				if not fn:
					continue
				src = f"{COMFY_BASE.rstrip('/')}/view?filename={fn}&subfolder={sf}&type={tp}"
				resp = await client.get(src)
				if int(getattr(resp, "status_code", 0) or 0) != 200:
					return ToolEnvelope.failure(
						"fetch_failed",
						f"download failed for {src}",
						status=int(getattr(resp, "status_code", 0) or 0) or 500,
						request_id=rid,
						details={"status": int(getattr(resp, "status_code", 0) or 0)},
					)
				dst = _os.path.join(save_dir, fn)
				with open(dst, "wb") as _f:
					_f.write(resp.content)
				saved_paths.append(dst)
				rel = _os.path.relpath(dst, _UPLOAD_DIR).replace("\\", "/")
				orch_urls.append(f"{_PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if _PUBLIC_BASE_URL else f"/uploads/{rel}")
		if orch_urls:
			# Attach orchestrator-served URLs and artifact descriptors for downstream consumers
			result["meta"]["orch_view_urls"] = orch_urls
			result["meta"]["cid"] = _cid
			arts: list[dict[str, object]] = []
			for im, url in zip(images, orch_urls):
				fn = im.get("filename")
				if not isinstance(fn, str) or not fn:
					continue
				arts.append(
					{
						"id": fn,
						"kind": "image",
						"path": f"/uploads/artifacts/image/{_cid}/{fn}",
						"view_url": url,
					}
				)
			if arts:
				result["artifacts"] = arts
		# Global + per-region QA for the first image (full-fat analyzer)
		if saved_paths:
			if _qa_analyze_image is not None and _qa_analyze_image_regions is not None:
				first_image_path = saved_paths[0]
				prompt_str = str(args.get("prompt") or "")
				global_info = _qa_analyze_image(first_image_path, prompt_str)
				region_info = _qa_analyze_image_regions(first_image_path, prompt_str, global_info)
				# Attach global scores/semantics into result["qa"]["images"]
				img_qa = result.setdefault("qa", {}).setdefault("images", {})
				if isinstance(img_qa, dict) and isinstance(global_info, dict):
					score_block = global_info.get("score") or {}
					sem_block = global_info.get("semantics") or {}
					if isinstance(score_block, dict):
						img_qa["overall"] = float(score_block.get("overall") or 0.0)
						img_qa["semantic"] = float(score_block.get("semantic") or 0.0)
						img_qa["technical"] = float(score_block.get("technical") or 0.0)
						img_qa["aesthetic"] = float(score_block.get("aesthetic") or 0.0)
					if isinstance(sem_block, dict):
						cs = sem_block.get("clip_score")
						if isinstance(cs, (int, float)):
							img_qa["clip_score"] = float(cs)
				# Attach region aggregates into QA so compute_domain_qa can see them
				if isinstance(region_info, dict):
					agg = region_info.get("aggregates") or {}
					if isinstance(agg, dict) and isinstance(img_qa, dict):
						fl = agg.get("face_lock")
						if isinstance(fl, (int, float)):
							img_qa["face_lock"] = float(fl)
						il = agg.get("id_lock")
						if isinstance(il, (int, float)):
							img_qa["id_lock"] = float(il)
						hr = agg.get("hands_ok_ratio")
						if isinstance(hr, (int, float)):
							img_qa["hands_ok_ratio"] = float(hr)
						tr = agg.get("text_readable_lock")
						if isinstance(tr, (int, float)):
							img_qa["text_readable_lock"] = float(tr)
						bq = agg.get("background_quality")
						if isinstance(bq, (int, float)):
							img_qa["background_quality"] = float(bq)
						# Expose full regions_info to the committee for region-aware patch planning
						img_qa["regions_info"] = region_info
					# Emit a distillation trace row for global + region QA
					trc = args.get("trace_id") or args.get("cid")
					if isinstance(trc, str) and trc.strip():
						_emit_trace(STATE_DIR_LOCAL, trc, "image.region.qa", {
							"path": first_image_path,
							"prompt": prompt_str,
							"global": {
								"score": global_info.get("score") if isinstance(global_info, dict) else {},
								"semantics": global_info.get("semantics") if isinstance(global_info, dict) else {},
							},
							"aggregates": agg if isinstance(region_info, dict) else {},
						})

		# Populate basic per-entity QA metrics when a lock bundle is present, and
		# attach a global face/identity lock score using the same embedding space
		# used for lock construction.
		lock_bundle = args.get("lock_bundle") if isinstance(args.get("lock_bundle"), dict) else None
		if lock_bundle and saved_paths:
			locks = _lock_to_assets(lock_bundle)
			regions = locks.get("regions") if isinstance(locks.get("regions"), dict) else {}
			entities_qa: dict[str, dict[str, float]] = {}
			first_image_path = saved_paths[0]
			for region_id, region_data in regions.items():
				if not isinstance(region_data, dict):
					continue
				metrics = await _qa_region_scores(first_image_path, region_data)
				if not isinstance(metrics, dict):
					continue
				ent_metrics: dict[str, float] = {}
				clip_lock_val = metrics.get("clip_lock")
				texture_score_val = metrics.get("texture_score")
				shape_score_val = metrics.get("shape_score")
				if isinstance(clip_lock_val, (int, float)):
					ent_metrics["clip_lock"] = float(clip_lock_val)
				if isinstance(texture_score_val, (int, float)):
					ent_metrics["texture_lock"] = float(texture_score_val)
				if isinstance(shape_score_val, (int, float)):
					ent_metrics["shape_lock"] = float(shape_score_val)
				if ent_metrics:
					entities_qa[str(region_id)] = ent_metrics
			if entities_qa:
				img_qa = result.setdefault("qa", {}).setdefault("images", {})
				if isinstance(img_qa, dict):
					img_qa["entities"] = entities_qa
				# Attach per-call lock metadata for distillation
				locks_meta: dict[str, object] = {"bundle": lock_bundle}
				qp = args.get("quality_profile")
				if isinstance(qp, str) and qp:
					locks_meta["quality_profile"] = qp
				# Simple composite entity lock score: average of per-entity minima
				entity_scores: list[float] = []
				for ent in entities_qa.values():
					if not isinstance(ent, dict):
						continue
					vals: list[float] = []
					for k in ("clip_lock", "texture_lock", "shape_lock"):
						v = ent.get(k)
						if isinstance(v, (int, float)):
							vals.append(float(v))
					if vals:
						entity_scores.append(min(vals))
				if entity_scores:
					locks_meta["entity_lock_score"] = sum(entity_scores) / len(entity_scores)
				meta_block = result.setdefault("meta", {})
				if isinstance(meta_block, dict):
					meta_block["locks"] = locks_meta
			# Global face/identity lock score derived from the lock bundle's
			# face embedding and the generated image. This is best-effort and
			# should never break the tool, but failures must be logged.
			try:
				face_ref: Optional[list] = None
				vis = lock_bundle.get("visual")
				if isinstance(vis, dict):
					faces = vis.get("faces")
					if isinstance(faces, list) and faces:
						emb_block = faces[0].get("embeddings") or {}
						if isinstance(emb_block, dict) and isinstance(emb_block.get("id_embedding"), list):
							face_ref = emb_block.get("id_embedding")  # type: ignore[assignment]
				if face_ref is None:
					legacy_face = lock_bundle.get("face")
					if isinstance(legacy_face, dict) and isinstance(legacy_face.get("embedding"), list):
						face_ref = legacy_face.get("embedding")  # type: ignore[assignment]
				if isinstance(face_ref, list):
					face_score = await _qa_face_lock(first_image_path, face_ref)
					if isinstance(face_score, (int, float)):
						img_qa = result.setdefault("qa", {}).setdefault("images", {})
						if isinstance(img_qa, dict):
							# Use keys that compute_domain_qa already understands.
							img_qa["face_lock"] = float(face_score)
							img_qa["id_lock"] = float(face_score)
			except Exception as ex:
				# Identity scoring is best-effort and should never break the tool,
				# but we still log structured errors for observability.
				log.error(
					"[toolrun] image.dispatch face_lock QA failed: %s",
					str(ex),
					exc_info=True,
				)
			# Trace for distillation: emit chat.append with media parts for this trace
			trc = args.get("trace_id") or args.get("cid")
			if isinstance(trc, str) and trc.strip():
				parts = [{"image": u} for u in orch_urls if isinstance(u, str) and u.strip()]
				_emit_trace(STATE_DIR_LOCAL, trc, "chat.append", {
					"t": int(time.time()*1000),
					"message": {"role": "assistant", "parts": parts},
					"tool": "image.dispatch",
					"prompt_id": prompt_id,
				})
		return ToolEnvelope.success(result, request_id=rid)
	except Exception as ex:
		# Catch-all guard so any unexpected Python exception in the Comfy
		# pipeline surfaces as a structured error with a real stack trace.
		tb = traceback.format_exc()
		details = {
			"exception_type": ex.__class__.__name__ or "Exception",
			"traceback": tb,
			"tool": "image.dispatch",
			"args_keys": sorted(list((args or {}).keys())) if isinstance(args, dict) else [],
		}
		msg = str(ex) or "image.dispatch raised an unexpected exception"
		log.error("[toolrun] image.dispatch raised %s", ex, exc_info=True)
		return ToolEnvelope.failure(
			"tool_exception",
			msg,
			status=500,
			request_id=rid,
			details=details,
		)
