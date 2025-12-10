from __future__ import annotations

from typing import Any, Dict

import aiohttp  # type: ignore

from ...comfy.client_aio import choose_sampler_name
from ...comfy.dispatcher import submit as comfy_dispatch_submit
from ...image.graph_builder import build_min_sdxl_graph


async def run_image_job(req: Dict[str, Any], client_id: str, ws=None) -> Dict[str, Any]:
    async with aiohttp.ClientSession(trust_env=False) as s:
        sampler = await choose_sampler_name(s)
    graph = build_min_sdxl_graph(req, sampler)
    # Use unified comfy dispatcher submit, which handles HTTP submit + history
    # via the shared client and waits for execution completion via websocket.
    res = await comfy_dispatch_submit({"prompt": graph})
    prompt_id = res.get("prompt_id")
    history = res.get("history") or res.get("history_error")
    return {"prompt_id": prompt_id, "history": history, "graph": graph, "sampler_used": sampler}


