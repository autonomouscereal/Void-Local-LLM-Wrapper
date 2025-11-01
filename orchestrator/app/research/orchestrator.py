from __future__ import annotations

from typing import Dict, Any, List
import os
import time
import json
from ..rag.hygiene import rag_filter
from ..artifacts.shard import open_shard, append_jsonl, _finalize_shard
from ..artifacts.manifest import add_manifest_row, write_manifest_atomic
from .collect import discover_sources
from .normalize import normalize_sources
from .extract_money import extract_edges
from .graph import build_money_map
from .timeline import build_timeline
from .judge import judge_findings
from .report import make_report


RAG_TTL = int(os.getenv("RAG_TTL_SECONDS", "3600"))


def run_research(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job: {"query": str, "scope": "public|internal", "since": str|None, "until": str|None, "cid": str}
    Returns: {"phase": "done", "artifacts": [...], "cid": str}
    """
    q = (job.get("query") or "").strip()
    if not q:
        return {"phase": "done", "artifacts": [], "cid": job.get("cid")}
    cid = job.get("cid") or f"rs-{int(time.time())}"
    root = os.path.join("/workspace", "uploads", "artifacts", "research", str(cid))
    # Ensure root exists
    os.makedirs(root, exist_ok=True)
    manifest: Dict[str, Any] = {"cid": cid, "items": []}

    # Phase: discover + collect
    sources = discover_sources(q, job.get("scope", "public"), job.get("since"), job.get("until"))
    sources = rag_filter(sources, ttl_s=RAG_TTL)

    # Phase: normalize → Evidence Ledger
    ledger_rows = normalize_sources(sources)
    sh = open_shard(root, "ledger", max_bytes=int(os.getenv("ARTIFACT_SHARD_BYTES", "200000")))
    for row in ledger_rows:
        sh = append_jsonl(sh, row)
    _finalize_shard(sh)
    add_manifest_row(manifest, os.path.join(root, "ledger.index.json"), step_id="normalize")

    # Phase: analyze → Money Map
    edges = extract_edges(ledger_rows, query=q)
    money_map = build_money_map(edges)
    _write_json_atomic(os.path.join(root, "money_map.json"), money_map)
    add_manifest_row(manifest, os.path.join(root, "money_map.json"), step_id="analyze")

    # Phase: timeline
    timeline = build_timeline(ledger_rows)
    _write_json_atomic(os.path.join(root, "timeline.json"), timeline)
    add_manifest_row(manifest, os.path.join(root, "timeline.json"), step_id="timeline")

    # Phase: judge
    judge_o = judge_findings(money_map, timeline)
    _write_json_atomic(os.path.join(root, "judge.json"), judge_o)
    add_manifest_row(manifest, os.path.join(root, "judge.json"), step_id="judge")

    # Phase: report
    report = make_report(q, ledger_rows, money_map, timeline, judge_o)
    _write_json_atomic(os.path.join(root, "report.json"), report)
    add_manifest_row(manifest, os.path.join(root, "report.json"), step_id="report")

    write_manifest_atomic(root, manifest)
    return {"phase": "done", "artifacts": manifest.get("items", []), "cid": cid}


def _write_json_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


