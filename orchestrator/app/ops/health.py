from __future__ import annotations

import time
import os
from typing import Dict, Any


START_TS = time.time()


def get_capabilities() -> Dict[str, Any]:
    models = [
        {"name": os.getenv("QWEN_MODEL_ID", "qwen3:32b"), "ctx_tokens": int(os.getenv("DEFAULT_NUM_CTX", "4096")), "step_tokens": int(os.getenv("ICW_STEP_TOKENS", "900"))},
        {"name": os.getenv("GPTOSS_MODEL_ID", "gpt-oss:20b"), "ctx_tokens": int(os.getenv("DEFAULT_NUM_CTX", "4096")), "step_tokens": int(os.getenv("ICW_STEP_TOKENS", "900"))},
    ]
    tools = [
        "film.run",
        "rag_search",
        "research.run",
        "image.dispatch",
        "vlm.analyze",
        "tts.speak",
        "music.dispatch",
        "code.super_loop",
        "ocr.read",
    ]
    edge = {
        "edge_ok": os.getenv("EDGE_MODE", "off") == "on",
        "device_profile": {
            "ram_mb": int(os.getenv("EDGE_RAM_MB", "4096")),
            "max_context_bytes": int(os.getenv("EDGE_MAX_CTX_BYTES", "60000")),
            "offline": False,
        },
        "presets": {"image_default": "512x512", "tts_sr": 22050, "music_max_s": 45},
    }
    features = {
        "json_only": True,
        "no_caps": True,
        "windowed_solver": True,
        "ablation": os.getenv("ABLATE", "on") == "on",
        "rag_ttl_seconds": int(os.getenv("RAG_TTL_SECONDS", "3600")),
        "artifacts_sharding": True,
        "cancel_resume": True,
    }
    return {"models": models, "tools": tools, "features": features, "edge": edge}


def get_health() -> Dict[str, Any]:
    return {"ok": True, "ts": int(time.time()), "uptime_s": int(time.time() - START_TS)}


