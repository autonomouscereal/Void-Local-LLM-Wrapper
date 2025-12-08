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
from ..jobs.state import create_job, set_state, get_job
from ..jobs.progress import event as progress_event
from ..jobs.partials import emit_partial
from ..datasets.trace import append_sample as _trace_append


RAG_TTL = int(os.getenv("RAG_TTL_SECONDS", "3600"))


async def run_research(job: Dict[str, Any]) -> Dict[str, Any]:
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

    # Optionally register job in in-memory state
    jid = job.get("job_id")
    if jid:
        try:
            create_job(jid, "research.run", job)
            set_state(jid, "running", phase="discover", progress=0.0)
        except Exception:
            pass

    # Phase: discover + collect
    sources = await discover_sources(q, job.get("scope", "public"), job.get("since"), job.get("until"))
    sources = rag_filter(sources, ttl_s=RAG_TTL)
    if jid:
        try:
            set_state(jid, "running", phase="normalize", progress=0.2)
            _append_job_event(jid, progress_event("running", "discover", 0.2, artifacts=[]))
        except Exception:
            pass

    # Phase: normalize → Evidence Ledger (no explicit timeout wrapper)
    ledger_rows = normalize_sources(sources)
    sh = open_shard(root, "ledger", max_bytes=int(os.getenv("ARTIFACT_SHARD_BYTES", "200000")))
    for row in ledger_rows:
        if jid and (get_job(jid) and get_job(jid).cancel_flag):
            set_state(jid, "cancelled", phase="normalize", progress=0.0)
            _append_job_event(jid, progress_event("cancelled", "normalize", 0.0, artifacts=[]))
            write_manifest_atomic(root, manifest)
            return {"phase": "cancelled", "artifacts": manifest.get("items", []), "cid": cid}
        sh = append_jsonl(sh, row)
    _finalize_shard(sh)
    add_manifest_row(manifest, os.path.join(root, "ledger.index.json"), step_id="normalize")
    if jid:
        try:
            set_state(jid, "running", phase="analyze", progress=0.4)
            _append_job_event(jid, progress_event("running", "normalize", 0.4, artifacts=[{"path": os.path.join(root, "ledger.index.json")}]))
        except Exception:
            pass

    # Phase: analyze → Money Map (no explicit timeout wrapper)
    edges = extract_edges(ledger_rows=ledger_rows, query=q)
    money_map = build_money_map(edges)
    _write_json_atomic(os.path.join(root, "money_map.json"), money_map)
    add_manifest_row(manifest, os.path.join(root, "money_map.json"), step_id="analyze")
    if jid:
        try:
            set_state(jid, "running", phase="timeline", progress=0.6)
            _append_job_event(jid, progress_event("running", "analyze", 0.6, artifacts=[{"path": os.path.join(root, "money_map.json")}]))
        except Exception:
            pass

    # Phase: timeline
    timeline = build_timeline(ledger_rows)
    _write_json_atomic(os.path.join(root, "timeline.json"), timeline)
    add_manifest_row(manifest, os.path.join(root, "timeline.json"), step_id="timeline")
    if jid:
        try:
            set_state(jid, "running", phase="judge", progress=0.75)
            _append_job_event(jid, progress_event("running", "timeline", 0.75, artifacts=[{"path": os.path.join(root, "timeline.json")}]))
        except Exception:
            pass

    # Phase: judge
    judge_o = judge_findings(money_map, timeline)
    _write_json_atomic(os.path.join(root, "judge.json"), judge_o)
    add_manifest_row(manifest, os.path.join(root, "judge.json"), step_id="judge")
    if jid:
        try:
            set_state(jid, "running", phase="report", progress=0.9)
            _append_job_event(jid, progress_event("running", "judge", 0.9, artifacts=[{"path": os.path.join(root, "judge.json")}]))
        except Exception:
            pass

    # Phase: report
    report = make_report(q, ledger_rows, money_map, timeline, judge_o)
    _write_json_atomic(os.path.join(root, "report.json"), report)
    add_manifest_row(manifest, os.path.join(root, "report.json"), step_id="report")

    write_manifest_atomic(root, manifest)
    if jid:
        try:
            set_state(jid, "done", phase="done", progress=1.0)
            _append_job_event(jid, progress_event("done", "done", 1.0, artifacts=[{"path": os.path.join(root, "report.json")}]))
        except Exception:
            pass
    # Inline summary for chat surfaces
    try:
        # Top sources by frequency
        src_count = {}
        for r in ledger_rows:
            try:
                u = (r.get("url") or r.get("link") or "").strip()
                if u:
                    src_count[u] = src_count.get(u, 0) + 1
            except Exception:
                continue
        top_srcs = sorted(src_count.items(), key=lambda kv: kv[1], reverse=True)[:5]
        sources = [{"url": u, "count": c} for (u, c) in top_srcs]
        # Findings summary
        findings = list(report.get("findings", []) or [])
        bullets = []
        for f in findings[:5]:
            try:
                bullets.append(f"- {f.get('summary','')}")
            except Exception:
                continue
        summary_text = f"Research on: {q}\n\nTop Findings:\n" + ("\n".join(bullets) if bullets else "(no high-confidence findings)")
    except Exception:
        sources = []
        summary_text = ""
    try:
        _trace_append("research", {"cid": cid, "query": q, "summary": summary_text, "sources": sources, "artifacts": manifest.get("items", [])})
    except Exception:
        pass
    return {"phase": "done", "artifacts": manifest.get("items", []), "cid": cid, "report": report, "summary_text": summary_text, "sources": sources}


def _append_job_event(jid: str, ev: Dict[str, Any]) -> None:
    try:
        uploads = os.path.join("/workspace", "uploads")
        d = os.path.join(uploads, "jobs", str(jid))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "events.jsonl")
        line = json.dumps(ev, ensure_ascii=False) + "\n"
        with open(p, "ab") as f:
            f.write(line.encode("utf-8"))
            f.flush(); os.fsync(f.fileno())
    except Exception:
        return


def _write_json_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


