from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os, json, uuid, time, asyncio, urllib.request, os.path
import logging, sys
from urllib.parse import quote, urlsplit, urlparse
import base64 as _b64
from app.main import execute_tool_call as _execute_tool_call


router = APIRouter()
log = logging.getLogger("orchestrator.toolrun")


def ok_envelope(result, rid: str) -> JSONResponse:
	return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


def err_envelope(code: str, message: str, rid: str, status: int = 422, details: dict | None = None) -> JSONResponse:
	return JSONResponse(
		{"schema_version": 1, "request_id": rid, "ok": False, "error": {"code": code, "message": message, "details": (details or {})}},
		status_code=status,
	)


def _post_json(url: str, obj: dict) -> dict:
	data = json.dumps(obj).encode("utf-8")
	req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="POST")
	with urllib.request.urlopen(req) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str) -> dict:
	req = urllib.request.Request(url, headers={"accept": "application/json"}, method="GET")
	with urllib.request.urlopen(req) as resp:
		return json.loads(resp.read().decode("utf-8"))


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
_raw_comfy = os.getenv("COMFYUI_API_URL") or "http://127.0.0.1:8188"
_u = urlsplit(_raw_comfy)
_port = _u.port or 8188
if _u.hostname not in ("127.0.0.1", "localhost"):
	COMFY_BASE = f"{_u.scheme or 'http'}://127.0.0.1:{_port}"
else:
	COMFY_BASE = _raw_comfy

def _append_jsonl(path: str, obj: dict) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@router.post("/tool.validate")
async def tool_validate(req: Request):
	body = await req.json()
	name = (body.get("name") or "").strip()
	args = body.get("args") or {}
	if name != "image.dispatch":
		# Delegate validation to main catalog when not image.dispatch
		return ok_envelope({"name": name, "valid": True, "args": args}, rid="tool.validate")
	# Real acceptance: require at least a prompt or a model/size combo
	prompt = args.get("prompt")
	if not isinstance(prompt, str) or not prompt.strip():
		return JSONResponse(
			{"schema_version": 1, "request_id": "tool.validate", "ok": False,
			 "error": {"code": "invalid_args", "message": "prompt required for image.dispatch", "details": {}}},
			status_code=422
		)
	return ok_envelope({"name": name, "valid": True, "args": args}, rid="tool.validate")


@router.post("/tool.run")
async def tool_run(req: Request):
	body = await req.json()
	name = (body.get("name") or "").strip()
	args = body.get("args") or {}

	# Delegate to the main tool runner for everything except image.dispatch,
	# which we wire directly to ComfyUI in this minimal entrypoint.
	if name != "image.dispatch":
		res = await _execute_tool_call({"name": name, "arguments": args})
		if isinstance(res, dict) and res.get("error"):
			return JSONResponse({"schema_version": 1, "request_id": "tool.run", "ok": False, "error": {"code": "tool_error", "message": str(res.get("error"))}}, status_code=422)
		result = (res.get("result") if isinstance(res, dict) else res) or {}
		return ok_envelope(result, rid="tool.run")

	# Normalize args (aliases, numerics, multiples)
	def _normalize_args(a: dict) -> dict:
		out = dict(a or {})
		if "sampler_name" in out and "sampler" not in out: out["sampler"] = out.pop("sampler_name")
		if "negative_prompt" in out and "negative" not in out: out["negative"] = out.pop("negative_prompt")
		def snap(x, mult=8, lo=64, default=512):
			try: v = int(float(x))
			except: v = default
			if v < lo: v = lo
			return (v // mult) * mult
		if "width" in out: out["width"] = snap(out.get("width"))
		if "height" in out: out["height"] = snap(out.get("height"))
		try: out["steps"] = max(1, min(150, int(float(out.get("steps", 30)))))
		except: out["steps"] = 30
		try: out["cfg"] = float(out.get("cfg", 7.0))
		except: out["cfg"] = 7.0
		if not out.get("sampler"): out["sampler"] = "euler"
		if not out.get("scheduler"): out["scheduler"] = "normal"
		seed = out.get("seed")
		if seed in (None, "", "random", -1, "-1"):
			out["seed"] = None
		else:
			try: out["seed"] = int(seed)
			except: out["seed"] = None
		out["prompt"] = str(out.get("prompt","") or "")
		out["negative"] = str(out.get("negative","") or "")
		return out
	args = _normalize_args(args)

	# Inline graph or path
	inline = args.get("workflow_graph")
	if inline is not None:
		wf_obj = inline
		wf_path = "(inline)"
	else:
		wf_path = (args.get("workflow_path")
		           or os.getenv("COMFY_WORKFLOW_PATH")
		           or "/workspace/services/image/workflows/stock_smoke.json")
		if not os.path.exists(wf_path):
			return JSONResponse(
				{"schema_version": 1, "request_id": "tool.run", "ok": False,
				 "error": {"code": "missing_workflow", "message": f"workflow path not found: {wf_path}", "details": {}}},
				status_code=422
			)
		wf_text = _read_text(wf_path)
		wf_obj = json.loads(wf_text)

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

	if not isinstance(prompt_graph, dict):
		detail = {"top_level_type": type(wf_obj).__name__}
		if isinstance(wf_obj, dict):
			detail["top_level_keys"] = list(wf_obj.keys())[:10]
			detail["has_nodes_key"] = ("nodes" in wf_obj)
		return err_envelope("workflow_invalid", "Workflow must be a dict of nodes with class_type and inputs", rid="tool.run", status=422,
		                   details=detail)

	# Validate and bind
	problems = _validate_api_graph(prompt_graph)
	if problems:
		avail = [{"id": str(nid), "class": (node.get("class_type") if isinstance(node, dict) else None)} for nid, node in prompt_graph.items() if isinstance(node, dict)]
		return err_envelope("workflow_invalid", ";".join(problems[:4]), rid="tool.run", status=422, details={"available": avail})
	# Resolve actual nodes and apply overrides only to those
	try:
		bind = _resolve_bindings(prompt_graph)
	except ValueError as ve:
		avail = [{"id": str(nid), "class": (node.get("class_type") if isinstance(node, dict) else None)} for nid, node in prompt_graph.items() if isinstance(node, dict)]
		return err_envelope("workflow_binding_missing", str(ve), rid="tool.run", status=422, details={"available": avail})
	_apply_overrides(prompt_graph, bind, args)

	client_id = uuid.uuid4().hex
	_append_jsonl(os.path.join(STATE_DIR_LOCAL, "tools.jsonl"), {"t": int(time.time()*1000), "event": "comfyui.submit", "base": COMFY_BASE, "workflow_path": wf_path})
	log.info("[comfy] POST /prompt url=%s", COMFY_BASE.rstrip("/") + "/prompt")
	submit_res = _post_json(COMFY_BASE.rstrip("/") + "/prompt", {"prompt": prompt_graph, "client_id": client_id})
	prompt_id = submit_res.get("prompt_id") or submit_res.get("promptId") or ""
	log.info("[comfy] prompt_id=%s client_id=%s", prompt_id, client_id)

	images = []
	deadline = time.time() + float(args.get("timeout_sec") or os.getenv("COMFY_TIMEOUT_SEC") or 120)
	_first_hist = True
	while time.time() < deadline:
		await asyncio.sleep(0.25)
		log.info("[comfy] polling /history/%s", prompt_id)
		hist = _get_json(f"{COMFY_BASE.rstrip('/')}/history/{prompt_id}")
		if _first_hist:
			_append_jsonl(os.path.join(STATE_DIR_LOCAL, "tools.jsonl"), {"t": int(time.time()*1000), "event": "comfyui.history", "prompt_id": prompt_id})
			_first_hist = False
		if not isinstance(hist, dict):
			continue
		entry = hist.get(prompt_id) or {}
		outs = entry.get("outputs") or {}
		if not outs:
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

	# If no outputs, fail deterministically
	if not images:
		log.info("[comfy] images=0 (timeout)")
		return JSONResponse(
			{"schema_version": 1, "request_id": "tool.run", "ok": False,
			 "error": {"code": "tool_failed", "message": "no outputs after bounded polling", "details": {"prompt_id": prompt_id}}},
			status_code=422
		)

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
	_append_jsonl(os.path.join(STATE_DIR_LOCAL, "tools.jsonl"), {"t": int(time.time()*1000), "event": "comfyui.done", "count": len(images)})
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
			**eff,
		},
	}
	return ok_envelope(result, rid="tool.run")
