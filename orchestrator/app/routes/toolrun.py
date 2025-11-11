from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os, json, uuid, time, asyncio, urllib.request
from app.main import execute_tool_call as _execute_tool_call


router = APIRouter()


def ok_envelope(result, rid: str) -> JSONResponse:
	return JSONResponse({"schema_version": 1, "request_id": rid, "ok": True, "result": result}, status_code=200)


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


def _patch_inplace_prompt_graph(prompt_graph: dict, args: dict) -> None:
	if not isinstance(prompt_graph, dict):
		return
	for _, node in prompt_graph.items():
		if not isinstance(node, dict):
			continue
		inputs = node.get("inputs")
		ctype = (node.get("class_type") or "").lower()
		if not isinstance(inputs, dict):
			continue
		# text / negative
		if "clip" in ctype or "text" in ctype:
			txt = args.get("prompt")
			neg = args.get("negative_prompt")
			if txt is not None:
				if "text" in inputs and isinstance(inputs["text"], str):
					inputs["text"] = txt
				else:
					for k in list(inputs.keys()):
						if "text" in k and isinstance(inputs[k], str):
							inputs[k] = txt
							break
			if neg is not None:
				for k in ("negative", "neg", "text_g", "negative_text"):
					if k in inputs and isinstance(inputs[k], str):
						inputs[k] = neg
						break
		# sampler params
		for k in ("seed", "steps", "cfg", "sampler_name", "scheduler"):
			if k in args and k in inputs and not isinstance(inputs[k], list):
				inputs[k] = args[k]
		# dimensions
		if "width" in inputs and "height" in inputs:
			if "width" in args:
				inputs["width"] = args["width"]
			if "height" in args:
				inputs["height"] = args["height"]
		# model (direct value)
		if "model" in args and "model" in inputs and not isinstance(inputs["model"], list):
			inputs["model"] = args["model"]


def _build_view_url(base: str, filename: str, subfolder: str, ftype: str) -> str:
	return f"{base.rstrip('/')}/view?filename={filename}&subfolder={subfolder or ''}&type={ftype or 'output'}"


@router.post("/tool.validate")
async def tool_validate(req: Request):
	body = await req.json()
	name = (body.get("name") or "").strip()
	return ok_envelope({"name": name, "valid": (name == "image.dispatch"), "args": body.get("args")}, rid="tool.validate")


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

	comfy_base = os.getenv("COMFYUI_API_URL") or "http://127.0.0.1:8188"
	wf_path = (args.get("workflow_path")
	           or os.getenv("COMFY_WORKFLOW_PATH")
	           or "/workspace/services/image/workflows/stock_smoke.json")
	wf_text = _read_text(wf_path)
	prompt_graph = json.loads(wf_text)
	if not isinstance(prompt_graph, dict):
		raise ValueError("COMFY_WORKFLOW_PATH must be a valid ComfyUI prompt graph (dict)")

	_patch_inplace_prompt_graph(prompt_graph, args)

	client_id = uuid.uuid4().hex
	submit_res = _post_json(comfy_base.rstrip("/") + "/prompt", {"prompt": prompt_graph, "client_id": client_id})
	prompt_id = submit_res.get("prompt_id") or submit_res.get("promptId") or ""

	images = []
	deadline = time.time() + float(args.get("timeout_sec") or os.getenv("COMFY_TIMEOUT_SEC") or 90)
	while time.time() < deadline:
		await asyncio.sleep(0.5)
		hist = _get_json(f"{comfy_base.rstrip('/')}/history/{prompt_id}")
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
						"view_url": _build_view_url(comfy_base, fn, sf, tp),
					})
		if images:
			break

	result = {
		"ids": {
			"prompt_id": prompt_id,
			"client_id": client_id,
			"images": images,
		},
		"meta": {
			"submitted": True,
			"workflow_path": wf_path,
			"comfy_base": comfy_base,
			"image_count": len(images),
			"timed_out": (len(images) == 0),
		},
	}
	return ok_envelope(result, rid="tool.run")
