from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx  # type: ignore

from ..json_parser import JSONParser

log = logging.getLogger(__name__)

MUSIC_API_URL = os.getenv("MUSIC_API_URL", "http://127.0.0.1:7860").rstrip("/")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")


async def music_simple_submit(
    prompt: str, trace_id: str = "", conversation_id: str = ""
) -> Dict[str, Any]:
    """
    Simple music generation - just prompt, trace_id, and conversation_id.
    
    Uses defaults:
    - duration: 8 seconds
    - No seed (random)
    - No refs
    
    Args:
        prompt: Text prompt for music generation
        trace_id: Optional trace ID for logging
        conversation_id: Optional conversation ID for logging
    
    Returns:
        Dict with:
            - ok: bool indicating success
            - music_path: str full path to generated music file
            - music_url: str view URL for the music
            - error: Dict with error info if failed
    """
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return {
            "ok": False,
            "error": {"code": "missing_prompt", "message": "prompt is required"},
        }
    
    if not MUSIC_API_URL:
        return {
            "ok": False,
            "error": {
                "code": "music_unconfigured",
                "message": "MUSIC_API_URL is not configured",
            },
        }
    
    # Defaults: 8 seconds duration
    seconds = 8
    
    payload: Dict[str, Any] = {
        "prompt": prompt.strip(),
        "seconds": seconds,
    }
    
    if trace_id:
        payload["trace_id"] = trace_id
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    log.info(
        "[music.simple] submitting prompt_len=%d trace_id=%s conversation_id=%s",
        len(prompt),
        trace_id,
        conversation_id,
    )
    
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(f"{MUSIC_API_URL}/generate", json=payload)
        
        if r.status_code < 200 or r.status_code >= 300:
            return {
                "ok": False,
                "error": {
                    "code": "music_http_error",
                    "message": f"/generate returned {r.status_code}",
                    "status": int(r.status_code),
                    "details": {"body": r.text or ""},
                },
            }
        
        parser = JSONParser()
        response = parser.parse(r.text or "{}", {})
        
        if not isinstance(response, dict):
            return {
                "ok": False,
                "error": {
                    "code": "music_invalid_response",
                    "message": "Service returned non-dict response",
                    "details": {"raw": r.text},
                },
            }
        
        # Service returns ToolEnvelope format: {ok: bool, result: {...}, error: {...}}
        if not response.get("ok"):
            error = response.get("error") or {}
            return {
                "ok": False,
                "error": error if isinstance(error, dict) else {"code": "music_error", "message": str(error)},
            }
        
        result = response.get("result") or {}
        if not isinstance(result, dict):
            return {
                "ok": False,
                "error": {
                    "code": "music_no_result",
                    "message": "Service response missing result",
                },
            }
        
        music_path = result.get("path") or ""
        music_url = result.get("relative_url") or ""
        
        if not music_path:
            return {
                "ok": False,
                "error": {
                    "code": "music_no_path",
                    "message": "Service response missing path",
                },
            }
        
        # Normalize path to absolute
        if not music_path.startswith("/"):
            music_path = os.path.join(UPLOAD_DIR, music_path)
        elif music_path.startswith("/uploads/"):
            music_path = music_path.replace("/uploads", UPLOAD_DIR)
        # Ensure path is absolute
        if not os.path.isabs(music_path):
            music_path = os.path.abspath(music_path)
        
        # Generate view URL if not provided
        if not music_url and music_path:
            rel_path = os.path.relpath(music_path, UPLOAD_DIR).replace("\\", "/")
            music_url = f"/uploads/{rel_path}"
        
        log.info(
            "[music.simple] completed trace_id=%s conversation_id=%s path=%s",
            trace_id,
            conversation_id,
            music_path,
        )
        
        return {
            "ok": True,
            "music_path": music_path,
            "music_url": music_url,
            "trace_id": trace_id,
            "conversation_id": conversation_id,
        }
    
    except Exception as ex:
        log.warning(
            "[music.simple] error trace_id=%s conversation_id=%s: %s",
            trace_id,
            conversation_id,
            ex,
            exc_info=True,
        )
        return {
            "ok": False,
            "error": {
                "code": "music_simple_exception",
                "message": str(ex),
                "status": 500,
            },
        }
