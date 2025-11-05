from __future__ import annotations

from typing import Any, Dict

import aiohttp  # type: ignore

from ...comfy.client_aio import choose_sampler_name, comfy_submit, comfy_history
from ...image.graph_builder import build_min_sdxl_graph


async def run_image_job(req: Dict[str, Any], client_id: str, ws=None) -> Dict[str, Any]:
    async with aiohttp.ClientSession(trust_env=False) as s:
        sampler = await choose_sampler_name(s)
    graph = build_min_sdxl_graph(req, sampler)
    sub = await comfy_submit(graph, client_id, ws=ws)
    pid = sub.get("prompt_id")
    hist = await comfy_history(pid)
    return {"prompt_id": pid, "history": hist, "graph": graph, "sampler_used": sampler}


