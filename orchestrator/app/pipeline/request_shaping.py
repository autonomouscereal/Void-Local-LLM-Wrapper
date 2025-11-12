from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable

from app.ops.unicode import nfc_msgs  # type: ignore


def shape_request(
	body: Dict[str, Any],
	request_obj: Any,
	*,
	extract_attachments_fn: Callable[[List[Dict[str, Any]]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
	meta_prompt_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
	derive_seed_fn: Callable[[str, str], int],
	detect_video_intent_fn: Callable[[str], bool],
) -> Dict[str, Any]:
	"""
	Normalize and shape the incoming request into a deterministic context.
	- No network calls
	- No early exits (returns problems[] if body invalid)
	Outputs:
	  messages, normalized_msgs, attachments, last_user_text, conv_cid,
	  trace_id, master_seed, mode, problems[]
	"""
	out: Dict[str, Any] = {
		"messages": [],
		"normalized_msgs": [],
		"attachments": [],
		"last_user_text": "",
		"conv_cid": None,
		"trace_id": "",
		"master_seed": 0,
		"mode": "general",
		"problems": [],
	}
	# Validate body.messages shape (non-fatal)
	if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
		out["problems"].append({"code": "bad_request", "message": "messages must be a list"})
		# Still proceed with empty messages for single-exit discipline
	msgs_in = body.get("messages") or []
	# NFC normalize
	msgs_nfc = nfc_msgs(msgs_in)
	normalized_msgs, attachments = extract_attachments_fn(msgs_nfc)
	# Prepend attachment summary (non-invasive)
	if attachments:
		import json as _json
		attn = _json.dumps(attachments, indent=2, ensure_ascii=False)
		normalized_msgs = [{"role": "system", "content": f"Attachments available for tools:\n{attn}"}] + normalized_msgs
	# Meta prompt
	messages = meta_prompt_fn(normalized_msgs)
	# Keep CO as the first frame; any tool steering now lives in meta/system frames
	# Conversation id
	conv_cid = None
	if isinstance(body.get("cid"), (int, str)):
		conv_cid = str(body.get("cid"))
	elif isinstance(body.get("conversation_id"), (int, str)):
		conv_cid = str(body.get("conversation_id"))
	# Last user text
	last_user_text = ""
	for m in reversed(normalized_msgs):
		if (m.get("role") == "user") and isinstance(m.get("content"), str) and m.get("content").strip():
			last_user_text = m.get("content").strip()
			break
	# Mode
	mode = "film" if detect_video_intent_fn(last_user_text) else "general"
	# Seeds and trace id
	import json as _json
	msgs_for_seed = _json.dumps(
		[{"role": (m.get("role")), "content": (m.get("content"))} for m in normalized_msgs],
		ensure_ascii=False,
		separators=(",", ":"),
	)
	provided_seed = int(body.get("seed")) if isinstance(body.get("seed"), (int, float)) else None
	master_seed = provided_seed if provided_seed is not None else derive_seed_fn("chat", msgs_for_seed)
	import hashlib as _hl
	trace_id = (f"cid_{conv_cid}" if conv_cid else None) or ("tt_" + _hl.sha256(msgs_for_seed.encode("utf-8")).hexdigest()[:16])
	# Allow client-provided idempotency key to override trace_id for deduplication
	ikey = body.get("idempotency_key")
	if isinstance(ikey, str) and len(ikey) >= 8:
		trace_id = ikey.strip()
	# Output
	out.update({
		"messages": messages,
		"normalized_msgs": normalized_msgs,
		"attachments": attachments,
		"last_user_text": last_user_text,
		"conv_cid": conv_cid,
		"trace_id": trace_id,
		"master_seed": master_seed,
		"mode": mode,
	})
	return out


