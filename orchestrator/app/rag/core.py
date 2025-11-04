from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import time

from sentence_transformers import SentenceTransformer  # type: ignore

from ..db.pool import get_pg_pool


_embedder: Optional[SentenceTransformer] = None
_rag_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
RAG_CACHE_TTL_SEC = 300


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # Use a small, fast default; caller can control via env in main
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


async def rag_index_dir(root: str = "/workspace", glob_exts: Optional[List[str]] = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    import glob as _glob
    import os as _os
    pool = await get_pg_pool()
    if pool is None:
        return {"error": "pgvector not configured"}
    exts = glob_exts or ["*.md", "*.py", "*.ts", "*.tsx", "*.js", "*.json", "*.txt"]
    files: List[str] = []
    for ext in exts:
        files.extend(_glob.glob(_os.path.join(root, "**", ext), recursive=True))
    embedder = get_embedder()
    total_chunks = 0
    async with pool.acquire() as conn:
        for fp in files[:5000]:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            i = 0
            length = len(content)
            while i < length:
                chunk = content[i : i + chunk_size]
                i += max(1, chunk_size - chunk_overlap)
                vec = embedder.encode([chunk])[0]
                await conn.execute("INSERT INTO rag_docs (path, chunk, embedding) VALUES ($1, $2, $3)", fp.replace(root + "/", ""), chunk, list(vec))
                total_chunks += 1
    return {"indexed_files": len(files), "chunks": total_chunks}


async def rag_search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    now = time.time()
    key = f"{query}::{k}"
    cached = _rag_cache.get(key)
    if cached and (now - cached[0] <= RAG_CACHE_TTL_SEC):
        return cached[1]
    pool = await get_pg_pool()
    if pool is None:
        return []
    vec = get_embedder().encode([query])[0]
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT path, chunk FROM rag_docs ORDER BY embedding <=> $1 LIMIT $2", list(vec), k)
        results = [{"path": r["path"], "chunk": r["chunk"]} for r in rows]
        _rag_cache[key] = (now, results)
        return results


