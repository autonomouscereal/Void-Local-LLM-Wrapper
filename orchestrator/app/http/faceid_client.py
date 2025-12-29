from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx  # type: ignore

from ..json_parser import JSONParser


def _faceid_base_url() -> str:
    return (os.environ.get("FACEID_API_URL") or "").strip()


_EXPECTED_ENVELOPE: Dict[str, Any] = {
    "schema_version": int,
    "trace_id": str,
    "ok": bool,
    "result": {
        "faces": [
            {
                "embedding": list,
                "det_score": float,
                "bbox": list,
            }
        ],
        "embeddings": list,
        "embedding": list,
        "vec": list,
        "model": str,
        "max_faces": int,
        "source": dict,
        "trace_id": str,
    },
    "error": dict,
}


def _extract_faces_from_result(result: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Return (faces, model) from faceid result payload.
    """
    if not isinstance(result, dict):
        return ([], None)
    faces = result.get("faces")
    if not isinstance(faces, list):
        return ([], result.get("model") if isinstance(result.get("model"), str) else None)
    out: List[Dict[str, Any]] = []
    for f in faces:
        if isinstance(f, dict):
            out.append(f)
    model = result.get("model") if isinstance(result.get("model"), str) else None
    return (out, model)


async def faceid_embed(
    *,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_b64: Optional[str] = None,
    model_name: Optional[str] = None,
    model_root: Optional[str] = None,
    max_faces: int = 16,
    trace_id: str,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """
    Call the faceid service and return:
      - faces (list[dict]) (sorted by det_score desc, best face at index 0)
      - model (str|None) (model used by the service)
      - raw envelope dict (post-JSONParser coercion; includes error details)

    This is intentionally "hard blocking": it awaits the HTTP request and returns
    the parsed response (no thread pool / executor usage).
    """
    base = _faceid_base_url()
    if not base:
        return ([], None, {"ok": False, "error": {"code": "missing_faceid_api_url"}})

    payload: Dict[str, Any] = {
        "image_path": image_path,
        "image_url": image_url,
        "image_b64": image_b64,
        "model_name": model_name,
        "model_root": model_root,
        "max_faces": int(max_faces) if isinstance(max_faces, int) else 16,
        "trace_id": trace_id,
    }

    url = base.rstrip("/") + "/embed"
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        r = await client.post(url, json=payload)

    parser = JSONParser()
    env_any = parser.parse(r.text or "", _EXPECTED_ENVELOPE)
    env: Dict[str, Any] = env_any if isinstance(env_any, dict) else {"ok": False, "error": {"code": "faceid_invalid_response"}}
    if not bool(env.get("ok")):
        return ([], None, env)
    faces, model = _extract_faces_from_result(env.get("result"))
    return (faces, model, env)


async def faceid_best_embedding(
    *,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_b64: Optional[str] = None,
    model_name: Optional[str] = None,
    model_root: Optional[str] = None,
    trace_id: str,
) -> Tuple[Optional[List[float]], Optional[str], Dict[str, Any]]:
    faces, model, env = await faceid_embed(
        image_path=image_path,
        image_url=image_url,
        image_b64=image_b64,
        model_name=model_name,
        model_root=model_root,
        max_faces=1,
        trace_id=trace_id,
    )
    if not faces:
        return (None, model, env)
    vec = faces[0].get("embedding") if isinstance(faces[0], dict) else None
    return (vec if isinstance(vec, list) else None, model, env)


