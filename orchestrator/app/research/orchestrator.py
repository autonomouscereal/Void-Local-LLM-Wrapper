from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import time
import json
import logging
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
from ..tracing.runtime import trace_event

log = logging.getLogger(__name__)

_rag_ttl_raw = os.getenv("RAG_TTL_SECONDS", "3600")
try:
    RAG_TTL = int(str(_rag_ttl_raw).strip() or "3600")
except Exception as exc:
    log.warning("research.orchestrator: bad RAG_TTL_SECONDS=%r; defaulting to 3600", _rag_ttl_raw, exc_info=True)
    RAG_TTL = 3600


async def run_research(job: Dict[str, Any], *, trace_id: Optional[str] = None, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    job: {"query": str, "scope": "public|internal", "since": str|None, "until": str|None, "conversation_id": str}
    Returns: {"phase": "done", "artifacts": [...], "conversation_id": str}
    """
    q = (job.get("query") or "").strip()
    if not q:
        conv_id = conversation_id or job.get("conversation_id") or f"rs-{int(time.time())}"
        return {"phase": "done", "artifacts": [], "conversation_id": conv_id}
    conversation_id = conversation_id or job.get("conversation_id") or f"rs-{int(time.time())}"
    trace_id = trace_id or job.get("trace_id") or ""
    root = os.path.join("/workspace", "uploads", "artifacts", "research", str(conversation_id))
    # Ensure root exists
    os.makedirs(root, exist_ok=True)
    manifest: Dict[str, Any] = {"conversation_id": conversation_id, "items": []}

    # Optionally register job in in-memory state
    jid = job.get("job_id")
    if jid:
        try:
            create_job(jid, "research.run", job)
            set_state(jid, "running", phase="discover", progress=0.0)
        except Exception as exc:
            log.debug("research.run: failed to init job state jid=%s: %s", jid, exc, exc_info=True)

    # Phase: discover + collect
    sources = await discover_sources(q, job.get("scope", "public"), job.get("since"), job.get("until"))
    sources = rag_filter(sources, ttl_s=RAG_TTL)
    if jid:
        try:
            set_state(jid, "running", phase="normalize", progress=0.2)
            _append_job_event(jid, progress_event("running", "discover", 0.2, artifacts=[]))
        except Exception as exc:
            log.debug("research.run: failed to update job state jid=%s phase=normalize: %s", jid, exc, exc_info=True)

    # Phase: normalize → Evidence Ledger (no explicit timeout wrapper)
    ledger_rows = normalize_sources(sources)
    _shard_raw = os.getenv("ARTIFACT_SHARD_BYTES", "200000")
    try:
        _max_bytes = int(str(_shard_raw).strip() or "200000")
    except Exception as exc:
        log.warning("research.orchestrator: bad ARTIFACT_SHARD_BYTES=%r; defaulting to 200000", _shard_raw, exc_info=True)
        _max_bytes = 200000
    sh = open_shard(root, "ledger", max_bytes=_max_bytes)
    for row in ledger_rows:
        if jid and (get_job(jid) and get_job(jid).cancel_flag):
            set_state(jid, "cancelled", phase="normalize", progress=0.0)
            _append_job_event(jid, progress_event("cancelled", "normalize", 0.0, artifacts=[]))
            write_manifest_atomic(root, manifest)
            return {"phase": "cancelled", "artifacts": manifest.get("items", []), "conversation_id": conversation_id}
        sh = append_jsonl(sh, row)
    _finalize_shard(sh)
    add_manifest_row(manifest, os.path.join(root, "ledger.index.json"), step_id="normalize")
    if jid:
        try:
            set_state(jid, "running", phase="analyze", progress=0.4)
            _append_job_event(jid, progress_event("running", "normalize", 0.4, artifacts=[{"artifact_id": "ledger.index.json", "kind": "json", "path": os.path.join(root, "ledger.index.json")}]))
        except Exception as exc:
            log.debug("research.run: failed to update job state jid=%s phase=analyze: %s", jid, exc, exc_info=True)

    # Phase: analyze → Money Map (no explicit timeout wrapper)
    edges = extract_edges(ledger_rows=ledger_rows, query=q)
    money_map = build_money_map(edges)
    _write_json_atomic(os.path.join(root, "money_map.json"), money_map)
    add_manifest_row(manifest, os.path.join(root, "money_map.json"), step_id="analyze")
    if jid:
        try:
            set_state(jid, "running", phase="timeline", progress=0.6)
            _append_job_event(jid, progress_event("running", "analyze", 0.6, artifacts=[{"artifact_id": "money_map.json", "kind": "json", "path": os.path.join(root, "money_map.json")}]))
        except Exception as exc:
            log.debug("research.run: failed to update job state jid=%s phase=timeline: %s", jid, exc, exc_info=True)

    # Phase: timeline
    timeline = build_timeline(ledger_rows)
    _write_json_atomic(os.path.join(root, "timeline.json"), timeline)
    add_manifest_row(manifest, os.path.join(root, "timeline.json"), step_id="timeline")
    if jid:
        try:
            set_state(jid, "running", phase="judge", progress=0.75)
            _append_job_event(jid, progress_event("running", "timeline", 0.75, artifacts=[{"artifact_id": "timeline.json", "kind": "json", "path": os.path.join(root, "timeline.json")}]))
        except Exception as exc:
            log.debug("research.run: failed to update job state jid=%s phase=judge: %s", jid, exc, exc_info=True)

    # Phase: judge
    judge_o = judge_findings(money_map, timeline)
    _write_json_atomic(os.path.join(root, "judge.json"), judge_o)
    add_manifest_row(manifest, os.path.join(root, "judge.json"), step_id="judge")
    if jid:
        try:
            set_state(jid, "running", phase="report", progress=0.9)
            _append_job_event(jid, progress_event("running", "judge", 0.9, artifacts=[{"artifact_id": "judge.json", "kind": "json", "path": os.path.join(root, "judge.json")}]))
        except Exception as exc:
            log.debug("research.run: failed to update job state jid=%s phase=report: %s", jid, exc, exc_info=True)

    # Phase: report
    report = make_report(q, ledger_rows, money_map, timeline, judge_o)
    _write_json_atomic(os.path.join(root, "report.json"), report)
    add_manifest_row(manifest, os.path.join(root, "report.json"), step_id="report")

    write_manifest_atomic(root, manifest)
    if jid:
        try:
            set_state(jid, "done", phase="done", progress=1.0)
            _append_job_event(jid, progress_event("done", "done", 1.0, artifacts=[{"artifact_id": "report.json", "kind": "json", "path": os.path.join(root, "report.json")}]))
        except Exception as exc:
            log.debug("research.run: failed to finalize job state jid=%s: %s", jid, exc, exc_info=True)
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
        ranked_sources: List[tuple[int, int, str]] = []
        for i, source_url in enumerate(src_count.keys()):
            ranked_sources.append((int(src_count.get(source_url, 0)), int(i), str(source_url)))
        ranked_sources.sort(reverse=True)
        sources = [{"url": triple[2], "count": triple[0]} for triple in ranked_sources]
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
        trace_event(
            "research.summary",
            {
                "conversation_id": conversation_id,
                "query": q,
                "summary": summary_text,
                "sources": sources,
                "artifacts": manifest.get("items", []),
            },
        )
    except Exception as exc:
        log.debug(f"research.run: runtime trace emit failed (non-fatal) conversation_id={conversation_id!r} exc={exc!r}", exc_info=True)
    return {"phase": "done", "artifacts": manifest.get("items", []), "conversation_id": conversation_id, "report": report, "summary_text": summary_text, "sources": sources}


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


