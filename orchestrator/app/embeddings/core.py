from __future__ import annotations

from typing import List, Any, Dict
import time


def build_embeddings_response(emb_model, texts: List[str], model_name: str) -> Dict[str, Any]:
    vecs = emb_model.encode(texts)
    data = []
    for i, v in enumerate(vecs):
        data.append({"object": "embedding", "index": i, "embedding": [float(x) for x in list(v)]})
    return {
        "object": "list",
        "data": data,
        "model": model_name,
        "created": int(time.time()),
    }


