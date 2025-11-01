from __future__ import annotations

import time
from typing import List, Dict, Any


def stitch_openai_response(partials: List[str], model_name: str) -> Dict[str, Any]:
    content = "\n\n".join([p for p in (partials or []) if isinstance(p, str)])
    return {
        "id": "orc-windows-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


