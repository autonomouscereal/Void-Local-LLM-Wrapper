from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx  # type: ignore

from ..json_parser import JSONParser

log = logging.getLogger(__name__)

HYVIDEO_API_URL = os.getenv("HYVIDEO_API_URL", "http://127.0.0.1:8094").rstrip("/")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")


async def film2_simple_submit(
    prompt: str, trace_id: str = "", conversation_id: str = ""
) -> Dict[str, Any]:
    """
    Simple Film2 video generation - just prompt, trace_id, and conversation_id.
    
    Uses defaults:
    - fps: 24
    - resolution: 1280x720 (720p input, service upscales to 1080p)
    - duration: 6 seconds
    - No post-processing (no upscale/interpolate/hand_fix)
    
    Args:
        prompt: Text prompt for video generation
        trace_id: Optional trace ID for logging
        conversation_id: Optional conversation ID for logging
    
    Returns:
        Dict with:
            - ok: bool indicating success
            - video_path: str full path to generated video
            - video_url: str view URL for the video
            - error: Dict with error info if failed
    """
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return {
            "ok": False,
            "error": {"code": "missing_prompt", "message": "prompt is required"},
        }
    
    if not HYVIDEO_API_URL:
        return {
            "ok": False,
            "error": {
                "code": "hyvideo_unconfigured",
                "message": "HYVIDEO_API_URL is not configured",
            },
        }
    
    # Defaults: 720p input (service will upscale to 1080p), 24fps, 6 seconds
    fps = 24
    width = 1280
    height = 720
    seconds = 6
    num_frames = fps * seconds + 1
    
    # Generate job_id from trace_id or timestamp
    job_id = f"simple-{trace_id}" if trace_id else f"simple-{int(time.time())}"
    
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "prompt": prompt.strip(),
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "meta": {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
        },
    }
    
    log.info(
        "[film2.simple] submitting prompt_len=%d trace_id=%s conversation_id=%s",
        len(prompt),
        trace_id,
        conversation_id,
    )
    
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(
                f"{HYVIDEO_API_URL}/v1/video/generate", json=payload
            )
        
        if r.status_code < 200 or r.status_code >= 300:
            return {
                "ok": False,
                "error": {
                    "code": "hyvideo_http_error",
                    "message": f"/v1/video/generate returned {r.status_code}",
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
                    "code": "hyvideo_invalid_response",
                    "message": "Service returned non-dict response",
                    "details": {"raw": r.text},
                },
            }
        
        if not response.get("ok"):
            error = response.get("error") or {}
            return {
                "ok": False,
                "error": error if isinstance(error, dict) else {"code": "hyvideo_error", "message": str(error)},
            }
        
        result = response.get("result") or {}
        if not isinstance(result, dict):
            return {
                "ok": False,
                "error": {
                    "code": "hyvideo_no_result",
                    "message": "Service response missing result",
                },
            }
        
        output_path = result.get("output_path") or ""
        output_url = result.get("output_url") or ""
        
        if not output_path:
            return {
                "ok": False,
                "error": {
                    "code": "hyvideo_no_path",
                    "message": "Service response missing output_path",
                },
            }
        
        # Normalize path to absolute
        # Service returns path like /workspace/uploads/artifacts/video/{trace_id}/output.mp4
        if not output_path.startswith("/"):
            output_path = os.path.join(UPLOAD_DIR, output_path)
        elif output_path.startswith("/uploads/"):
            output_path = output_path.replace("/uploads", UPLOAD_DIR)
        # Ensure path is absolute and exists
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        
        # Generate view URL if not provided
        if not output_url and output_path:
            rel_path = os.path.relpath(output_path, UPLOAD_DIR).replace("\\", "/")
            output_url = f"/uploads/{rel_path}"
        
        log.info(
            "[film2.simple] completed trace_id=%s conversation_id=%s path=%s",
            trace_id,
            conversation_id,
            output_path,
        )
        
        return {
            "ok": True,
            "video_path": output_path,
            "video_url": output_url,
            "trace_id": trace_id,
            "conversation_id": conversation_id,
        }
    
    except Exception as ex:
        log.warning(
            "[film2.simple] error trace_id=%s conversation_id=%s: %s",
            trace_id,
            conversation_id,
            ex,
            exc_info=True,
        )
        return {
            "ok": False,
            "error": {
                "code": "film2_simple_exception",
                "message": str(ex),
                "status": 500,
            },
        }
