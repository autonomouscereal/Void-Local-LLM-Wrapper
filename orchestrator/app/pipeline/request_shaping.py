from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable
import unicodedata

from ..state.ids import trace_id as _mint_trace_id


def unicode_normalize_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Unicode normalizer for chat messages.

    - Applies NFC normalization to any string `content` fields.
    - Leaves non-string content and unknown fields unchanged.
    - Does not hide or catch errors: if something truly pathological is in
      `content`, it will surface naturally when accessed later.
    """
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        m2 = dict(m)
        content = m2.get("content")
        if isinstance(content, str):
            m2["content"] = unicodedata.normalize("NFC", content)
        out.append(m2)
    return out


# Back-compat alias used by existing shaping code.
def nfc_msgs(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return unicode_normalize_messages(msgs)


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
      request_id, trace_id, master_seed, mode, problems[]
    """
    out: Dict[str, Any] = {
        "messages": [],
        "normalized_msgs": [],
        "attachments": [],
        "last_user_text": "",
        "conv_cid": None,
        "request_id": "",
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
    # Conversation id (always present; server-minted when client does not supply one)
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

    # Server-mint a stable conversation id when client does not provide one.
    if conv_cid is None:
        conv_cid = "chat_" + _hl.sha256(msgs_for_seed.encode("utf-8")).hexdigest()[:16]

    # Request id (rid) and trace id (trace_id) are distinct identifiers.
    # - request_id: client-visible per-request envelope id; can be supplied by clients.
    # - trace_id: server-minted tracing correlation id for internal logs/locks.
    import uuid as _uuid

    request_id = ""
    if isinstance(body.get("request_id"), (str, int)) and str(body.get("request_id")).strip():
        request_id = str(body.get("request_id")).strip()
    else:
        ikey = body.get("idempotency_key")
        if isinstance(ikey, str) and ikey.strip():
            request_id = ikey.strip()
        else:
            request_id = _uuid.uuid4().hex

    trace_id = ""
    if isinstance(body.get("trace_id"), (str, int)) and str(body.get("trace_id")).strip():
        trace_id = str(body.get("trace_id")).strip()
    else:
        trace_id = _mint_trace_id()
    # Output
    out.update(
        {
            "messages": messages,
            "normalized_msgs": normalized_msgs,
            "attachments": attachments,
            "last_user_text": last_user_text,
            "conv_cid": conv_cid,
            "request_id": request_id,
            "trace_id": trace_id,
            "master_seed": master_seed,
            "mode": mode,
        }
    )
    return out
