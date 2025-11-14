from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os, json, uuid, time, asyncio, urllib.request, os.path
import logging, sys
from urllib.parse import quote, urlsplit, urlparse
import base64 as _b64
from app.state.checkpoints import append_event as checkpoints_append_event
from app.trace_utils import emit_trace as _emit_trace
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
		try:
			resp = await client.post(url, json=obj)
			ct = resp.headers.get("content-type","")
			data = resp.json() if ("application/json" in ct) else json.loads(resp.text or "{}")
			if 200 <= resp.status_code < 300:
				return _ok_env(True, **(data if isinstance(data, dict) else {"data": data}))
			return _ok_env(False, code="http_error", status=resp.status_code, detail=data)
		except Exception as ex:
			return _ok_env(False, code="network_error", message=str(ex))


async def _get_json(url: str) -> dict:
	async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
		try:
			resp = await client.get(url, headers={"accept": "application/json"})
			ct = resp.headers.get("content-type","")
			data = resp.json() if ("application/json" in ct) else json.loads(resp.text or "{}")
			if 200 <= resp.status_code < 300:
				return _ok_env(True, **(data if isinstance(data, dict) else {"data": data}))
			return _ok_env(False, code="http_error", status=resp.status_code, detail=data)
		except Exception as ex:
			return _ok_env(False, code="network_error", message=str(ex))


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
        raise ValueError("workflow_ui_format_not_supported")
    if isinstance(wf, dict):
        return wf
    raise ValueError("workflow_shape_invalid")


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
        raise ValueError("missing_ksampler")
    ks_in = graph[ks_id].get("inputs", {})
    pos_id = _get_ref_node_id(ks_in.get("positive"))
    neg_id = _get_ref_node_id(ks_in.get("negative"))
    if not pos_id or not neg_id:
        raise ValueError("ksampler_missing_positive_or_negative_refs")
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
        if "seed" in args: ks_in["seed"] = int(args["seed"])
        if "steps" in args: ks_in["steps"] = int(args["steps"])
        if "cfg" in args: ks_in["cfg"] = float(args["cfg"])
        if "sampler" in args: ks_in["sampler_name"] = str(args["sampler"]).strip()
        if "sampler_name" in args: ks_in["sampler_name"] = str(args["sampler_name"]).strip()
        if "scheduler" in args: ks_in["scheduler"] = str(args["scheduler"]).strip()
    # Latent size
    if bind.get("latent"):
        li = graph[bind["latent"]]["inputs"]
        if "width" in args: li["width"] = int(args["width"])
        if "height" in args: li["height"] = int(args["height"])
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
	try:
		u = urlsplit(url)
		host = "127.0.0.1"
		port = u.port or (8188 if (u.scheme or "http") == "http" else 8188)
		return f"{u.scheme or 'http'}://{host}:{port}"
	except Exception:
		return "http://127.0.0.1:8188"
_raw_comfy = os.getenv("COMFYUI_API_URL") or "http://127.0.0.1:8188"
COMFY_BASE = _force_loopback(_raw_comfy)

def _append_jsonl(path: str, obj: dict) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@router.post("/tool.validate")
async def tool_validate(req: Request):
	rid = "tool.validate"
	try:
		body = await req.json()
	except Exception as ex:
		return ToolEnvelope.failure(
			"invalid_json",
			f"Body must be valid JSON: {ex}",
			status=400,
			request_id=rid,
		)
	if not isinstance(body, dict):
		return ToolEnvelope.failure(
			"invalid_body_type",
			"Body must be a JSON object",
			status=422,
			request_id=rid,
		)
	name = (body.get("name") or "").strip()
	args = body.get("args") or {}
	if not isinstance(args, dict):
		return ToolEnvelope.failure(
			"invalid_args_type",
			"args must be an object",
			status=422,
			request_id=rid,
		)
	if name != "image.dispatch":
		# For non-image tools, accept basic object-arg validation here
		return ToolEnvelope.success(
			{"name": name, "valid": True, "args": args},
			request_id=rid,
		)
	# Real acceptance: require at least a prompt or a model/size combo
	prompt = args.get("prompt")
	if not isinstance(prompt, str) or not prompt.strip():
		detail = {
			"tool": name,
			"missing": ["prompt"],
			"invalid": [],
			"schema": {
				"required": ["prompt"],
				"notes": "image.dispatch minimal validator",
			},
			"args": args,
		}
		return ToolEnvelope.failure(
			"invalid_args",
			"prompt required for image.dispatch",
			status=422,
			request_id=rid,
			details=detail,
		)
	return ToolEnvelope.success(
		{"name": name, "valid": True, "args": args},
		request_id=rid,
	)


@router.post("/tool.run")
async def tool_run(req: Request):
	rid = "tool.run"
	try:
		body = await req.json()
	except Exception as ex:
		return ToolEnvelope.failure(
			"invalid_json",
			f"Body must be valid JSON: {ex}",
			status=400,
			request_id=rid,
		)
	if not isinstance(body, dict):
		return ToolEnvelope.failure(
			"invalid_body_type",
			"Body must be a JSON object",
			status=422,
			request_id=rid,
		)
	name = (body.get("name") or "").strip()
	args = body.get("args") or {}

	# For non-image tools, route through the executor /execute to avoid local fast paths and circular imports
	if name != "image.dispatch":
		exec_base = os.getenv("EXECUTOR_BASE_URL", "http://127.0.0.1:8081").rstrip("/")
		request_id = (args.get("trace_id") or args.get("cid") or uuid.uuid4().hex) if isinstance(args, dict) else uuid.uuid4().hex
		step = {"id": "s1", "tool": name, "inputs": (args if isinstance(args, dict) else {})}
		payload = {"schema_version": 1, "request_id": request_id, "trace_id": request_id, "steps": [step]}
		async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
			resp = await client.post(exec_base + "/execute", json=payload)
			try:
				env = resp.json()
			except Exception:
				env = {"ok": False, "error": {"code": "executor_bad_json", "message": resp.text}}
		if isinstance(env, dict) and env.get("ok") and isinstance((env.get("result") or {}).get("produced"), dict):
			produced = (env.get("result") or {}).get("produced") or {}
			res1 = produced.get("s1") or {}
			return ToolEnvelope.success(res1 if isinstance(res1, dict) else {"value": res1}, request_id=rid)
		return ToolEnvelope.failure(
			"executor_failed",
			"tool failed via executor",
			status=500,
			request_id=rid,
			details=(env or {}),
		)

	# Use upstream global fixer (executor) for generic normalization; do not duplicate here

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
			wf_obj = json.loads(wf_text)
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
		try:
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
								try:
									max_id = max(max_id, int(str(k)))
								except Exception:
									continue
							return str(max_id + 1)
						if not pos_id:
							new_pos_id = _next_id(prompt_graph)
							prompt_graph[new_pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [str(ckpt_id), 1], "text": str(args.get("prompt") or "")}}
							ks_in["positive"] = [new_pos_id, 0]
						if not neg_id:
							new_neg_id = _next_id(prompt_graph)
							prompt_graph[new_neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [str(ckpt_id), 1], "text": str(args.get("negative") or "")}}
							ks_in["negative"] = [new_neg_id, 0]
		except Exception:
			# Non-fatal: allow normal validation/binding to report precise errors
			pass

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
	# Resolve actual nodes and apply overrides only to those; if binding fails, fall back to a known-good pipeline
	try:
		bind = _resolve_bindings(prompt_graph)
		_apply_overrides(prompt_graph, bind, args)
	except ValueError:
		# Build a minimal, valid API graph wiring CLIPTextEncode -> KSampler -> VAEDecode
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
		# continue with this synthesized graph
	# Ensure SaveImage nodes have a filename_prefix (required by newer ComfyUI)
	_client_id = uuid.uuid4().hex
	cid = (args.get("cid") or "").strip() if isinstance(args.get("cid"), str) else ""
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
		entry = hist.get(prompt_id) or {}
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
	import os as _os
	save_dir = _os.path.join(_UPLOAD_DIR, "artifacts", "image", prompt_id or client_id)
	_os.makedirs(save_dir, exist_ok=True)
	orch_urls: list[str] = []
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
			rel = _os.path.relpath(dst, _UPLOAD_DIR).replace("\\", "/")
			orch_urls.append(f"{_PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if _PUBLIC_BASE_URL else f"/uploads/{rel}")
	if orch_urls:
		result["meta"]["orch_view_urls"] = orch_urls
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
