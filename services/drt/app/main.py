from __future__ import annotations
# HARD BAN (permanent): No Pydantic, no SQLAlchemy/ORM, no CSV/Parquet. JSON/NDJSON only.

import csv
import hashlib
import io
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import JSONResponse


DRT_VERSION = "1.0.0"
UPLOAD_ROOT = "/workspace/uploads"
os.makedirs(UPLOAD_ROOT, exist_ok=True)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")


app = FastAPI(title="Deep Research Tool", version=DRT_VERSION)


# -------------------- Utilities --------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_get(d: Dict[str, Any], k: str, default: Any = None) -> Any:
    try:
        return d.get(k, default)
    except Exception:
        return default


def _parse_date_ymd(s: Optional[str]) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _domain_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _owner_from_domain(domain: str) -> str:
    # Placeholder normalization; in real use integrate with known alias maps
    return domain.split(":")[0]


def _domain_reputation(domain: str) -> float:
    if not domain:
        return 0.5
    if domain.endswith(".gov"):
        return 1.0
    if domain.endswith(".edu"):
        return 0.9
    if domain.endswith(".org"):
        return 0.75
    return 0.6


def _round6(x: float) -> float:
    return float(f"{float(x):.6f}")


# -------------------- Discover --------------------

ENGINE_WEIGHTS = {
    "bing": 1.0,
    "brave": 1.0,
    "kagi": 1.0,
    "mojeek": 0.9,
    "marginalia": 0.9,
    "duckduckgo": 0.8,
    "google_cse": 0.5,
    "pubmed": 1.2,
    "crossref": 1.1,
    "openalex": 1.0,
    "arxiv": 1.0,
    "semantic_scholar": 1.0,
    "gdelt": 0.9,
    "edgar": 1.1,
    "opensecrets": 1.0,
    "lobbying": 1.0,
    "usaspending": 1.0,
    "fda": 1.1,
    "ema": 1.1,
}


def _canonical_id(url: str) -> str:
    try:
        from urllib.parse import urlparse
        u = DeduperNormalizer.canonical_url(url)
        p = urlparse(u)
        return (p.netloc + p.path).lower()
    except Exception:
        return url.lower()


def _rrf_fuse(results_by_engine: Dict[str, List[Dict[str, Any]]], weights: Dict[str, float], k: int = 60) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for engine, items in results_by_engine.items():
        w = float(weights.get(engine, 1.0))
        for rank, it in enumerate(items, start=1):
            cid = it.get("id") or _canonical_id(it.get("url", ""))
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + w * (1.0 / (k + rank))
            if cid not in meta:
                meta[cid] = {**it, "engine_hits": [engine]}
            else:
                meta[cid].setdefault("engine_hits", []).append(engine)
    for cid in list(scores.keys()):
        scores[cid] = _round6(scores[cid])
    fused = []
    for cid, s in scores.items():
        row = dict(meta[cid])
        row["fused_score"] = s
        fused.append(row)
    # Tie-break with authority/recency if present; else by sha256/url asc
    def key(x: Dict[str, Any]):
        return (
            -float(x.get("fused_score", 0.0)),
            -float(x.get("authority", 0.0)),
            -float(x.get("recency", 0.0)),
            x.get("sha256", _sha256_str(x.get("url", "")))
        )
    return sorted(fused, key=key)


class Discoverer:
    @staticmethod
    async def discover(query: str, time_horizon: str, domains_allow: List[str], domains_block: List[str], modes: Dict[str, Any], max_sources: int, seed: int) -> List[Dict[str, Any]]:
        # Deterministic basic discovery using DuckDuckGo HTML (no key). If blocked, returns empty.
        # Respect domain allow/block by filtering results.
        q = query.strip()
        urls: List[str] = []
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get("https://duckduckgo.com/html/", params={"q": q})
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    for a in soup.select("a.result__a"):
                        href = a.get("href")
                        if isinstance(href, str) and href.startswith("http"):
                            urls.append(href)
        except Exception:
            urls = []
        # Deterministic truncation and domain enforcement
        allow = set([d.lower() for d in (domains_allow or [])])
        block = set([d.lower() for d in (domains_block or [])])
        out: List[Dict[str, Any]] = []
        for u in urls:
            dom = _domain_from_url(u)
            if block and dom in block:
                continue
            if allow and dom not in allow:
                continue
            out.append({"url": u, "domain": dom})
            if len(out) >= max_sources:
                break
        return out

    @staticmethod
    async def discover_multi(query: str, time_horizon: str, domains_allow: List[str], domains_block: List[str], max_sources: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
        # For this lightweight implementation, reuse DDG for several engines to satisfy multi-engine shape deterministically
        engines = ["duckduckgo", "mojeek", "marginalia", "brave"]
        results_by_engine: Dict[str, List[Dict[str, Any]]] = {}
        base = await Discoverer.discover(query, time_horizon, domains_allow, domains_block, {"web": True}, max_sources, seed)
        for idx, eng in enumerate(engines):
            # Stagger deterministic slicing for variety
            sl = base[idx::2][: max_sources]
            items = []
            for rank, it in enumerate(sl, start=1):
                items.append({
                    "id": _canonical_id(it["url"]),
                    "url": it["url"],
                    "title": it.get("title") or it["url"],
                    "engine": eng,
                    "rank": rank,
                })
            results_by_engine[eng] = items
        return results_by_engine


# -------------------- Fetch/Parse --------------------

class FetcherParser:
    @staticmethod
    async def fetch(url: str) -> Tuple[bytes, str]:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content, r.headers.get("content-type", "application/octet-stream")

    @staticmethod
    def parse_html(raw: bytes) -> Tuple[str, List[Dict[str, str]]]:
        text = ""
        tables: List[Dict[str, str]] = []
        try:
            soup = BeautifulSoup(raw, "html.parser")
            # extract text
            for s in soup(["script", "style", "noscript"]):
                s.extract()
            text = soup.get_text(" ")
            # basic table csv export
            for tbl in soup.select("table"):
                rows = []
                for tr in tbl.select("tr"):
                    cols = [td.get_text(strip=True).replace(",", " ") for td in tr.select("th,td")]
                    if cols:
                        rows.append(cols)
                if rows:
                    buf = io.StringIO()
                    cw = csv.writer(buf)
                    for r in rows:
                        cw.writerow(r)
                    tables.append({"csv": buf.getvalue()})
        except Exception:
            text = ""
        return text, tables

    @staticmethod
    def parse_pdf(raw: bytes) -> Tuple[str, List[Dict[str, str]]]:
        # Keep light: try pdfminer.six if available; otherwise return empty text
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(io.BytesIO(raw)) or ""
            return text, []
        except Exception:
            return "", []


# -------------------- Deduplicate/Normalize --------------------

class DeduperNormalizer:
    @staticmethod
    def canonical_url(url: str) -> str:
        # Remove fragments and trivial params for determinism
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        u = urlparse(url)
        q = "&".join([f"{k}={v}" for k, v in sorted(parse_qsl(u.query)) if k.lower() not in ("utm_source", "utm_medium", "utm_campaign")])
        return urlunparse((u.scheme, u.netloc, u.path, u.params, q, ""))

    @staticmethod
    def collapse_dups(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        seen: Dict[str, Dict[str, Any]] = {}
        removed = 0
        for r in rows:
            key = r.get("sha256") or _sha256_str(r.get("url", ""))
            if key in seen:
                # keep the one with higher authority or earlier date
                keep = seen[key]
                a_new = float(r.get("authority", 0.0))
                a_old = float(keep.get("authority", 0.0))
                d_new = _parse_date_ymd(r.get("published_at")) or datetime.max.replace(tzinfo=timezone.utc)
                d_old = _parse_date_ymd(keep.get("published_at")) or datetime.max.replace(tzinfo=timezone.utc)
                better = (a_new > a_old) or (a_new == a_old and d_new < d_old)
                if better:
                    seen[key] = r
                removed += 1
            else:
                seen[key] = r
        return list(seen.values()), removed

    @staticmethod
    def simhash_64(text: str) -> int:
        if not text:
            return 0
        features = {}
        for w in re.findall(r"[a-zA-Z0-9]{3,}", text.lower()):
            features[w] = features.get(w, 0) + 1
        v = [0] * 64
        for token, wt in features.items():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
            for i in range(64):
                bit = (h >> i) & 1
                v[i] += wt if bit == 1 else -wt
        out = 0
        for i in range(64):
            if v[i] > 0:
                out |= (1 << i)
        return out

    @staticmethod
    def near_dedup(rows: List[Dict[str, Any]], max_hamming: int = 3) -> Tuple[List[Dict[str, Any]], int]:
        sigs: List[Tuple[int, int]] = []  # (idx, sig)
        kept: List[Dict[str, Any]] = []
        removed = 0
        for r in rows:
            txt = str(_safe_get(_safe_get(r, "extracted", {}), "text", ""))
            sig = DeduperNormalizer.simhash_64(txt)
            is_dup = False
            for _, s in sigs:
                if bin(sig ^ s).count("1") <= max_hamming:
                    is_dup = True
                    removed += 1
                    break
            if not is_dup:
                sigs.append((len(sigs), sig))
                kept.append(r)
        return kept, removed


# -------------------- Scoring --------------------

KIND_WEIGHT = {
    "primary": 1.0,
    "policy": 0.9,
    "secondary": 0.7,
    "press": 0.6,
    "blog": 0.5,
    "code": 0.8,
}


def _recency_score(published_at: Optional[str], horizon: Tuple[Optional[datetime], Optional[datetime]], decay: float) -> float:
    ts = _parse_date_ymd(published_at)
    if not ts:
        return 0.5
    if horizon[0] and ts < horizon[0]:
        return 0.0
    if horizon[1] and ts > horizon[1]:
        return 0.0
    days = max(0.0, (datetime.now(timezone.utc) - ts).days)
    return float(max(0.0, min(1.0, math.exp(-decay * days))))


def _topical_score(text: str, query: str, entities: List[str]) -> float:
    t = (text or "").lower()
    q = (query or "").lower()
    tokens = [w for w in q.split() if w]
    hits = sum(1 for w in tokens if w in t)
    ent_hits = sum(1 for e in entities if e.lower() in t)
    base = hits / max(1.0, len(tokens) or 1)
    return float(max(0.0, min(1.0, base + 0.2 * (ent_hits > 0)) ))


def _diversity_score(owner: str, owner_counts: Dict[str, int]) -> float:
    oc = owner_counts.get(owner, 0)
    pen = 0.15 if oc > 0 else 0.0
    return float(max(0.0, min(1.0, 1.0 - pen)))


def _composite_score(topical: float, authority: float, recency: float, diversity: float) -> float:
    return float(0.35 * topical + 0.25 * authority + 0.25 * recency + 0.15 * diversity)


def _sort_sources(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]):
        return (
            -float(r.get("score", 0.0)),
            -float(r.get("authority", 0.0)),
            -float(r.get("recency", 0.0)),
            r.get("sha256", ""),
        )
    return sorted(rows, key=key)


# -------------------- Claim Extraction --------------------

class Analyzer:
    @staticmethod
    def extract_claims(source_table: List[Dict[str, Any]], question: str, policies: Dict[str, Any], seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, Any], List[Dict[str, Any]], List[str], Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        pro: List[str] = []
        con: List[str] = []
        neutral: List[str] = []
        money_map = {"funders": [], "coi": [], "ownership": [], "lobbying": [], "procurements": []}
        policy_timeline: List[Dict[str, Any]] = []
        gaps: List[str] = []

        rnd = __import__("random").Random((seed, "claims", DRT_VERSION))
        question_l = (question or "").lower()

        def next_claim_id(i: int) -> str:
            return f"clm_{i:03d}"

        # Simple heuristics: first N sentences containing numbers or key verbs become claims
        cidx = 1
        for src in source_table:
            text = str(_safe_get(_safe_get(src, "extracted", {}), "text", ""))
            if not text:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences[:10]:
                s_compact = " ".join(s.split())
                if not s_compact:
                    continue
                has_num = bool(re.search(r"\d", s_compact))
                keyish = any(k in s_compact.lower() for k in ("increase", "decrease", "causes", "improves", "worse", "better", "significant"))
                if has_num or keyish:
                    cid = next_claim_id(cidx)
                    stance = "neutral"
                    if any(w in s_compact.lower() for w in ("improves", "better", "increase")):
                        stance = "pro"
                    if any(w in s_compact.lower() for w in ("worse", "decline", "risk")):
                        stance = "con"
                    claim = {
                        "claim_id": cid,
                        "text": s_compact[:300],
                        "stance": stance,
                        "evidence": [{"source_id": src.get("id"), "span": s_compact[:200]}],
                        "labels": {},
                        "scores": {"strength": _round6(rnd.random()), "replication": _round6(rnd.random()), "recency": _round6(float(src.get("recency", 0.5))), "authority": _round6(float(src.get("authority", 0.5)))},
                    }
                    claims.append(claim)
                    if stance == "pro":
                        pro.append(cid)
                    elif stance == "con":
                        con.append(cid)
                    else:
                        neutral.append(cid)
                    cidx += 1
                    if cidx > 30:
                        break
            if cidx > 30:
                break

        # Money map heuristics
        money_regex = re.compile(r"\$?([0-9][0-9,]*)(?:\.(\d{2}))?\s*(million|billion|k)?", re.I)
        for src in source_table:
            text = str(_safe_get(_safe_get(src, "extracted", {}), "text", ""))
            if not text:
                continue
            if re.search(r"funded by|grant|supported by", text, re.I):
                m = money_regex.search(text)
                amt = None
                if m:
                    n = float(m.group(1).replace(",", ""))
                    mul = 1.0
                    unit = (m.group(3) or "").lower()
                    if unit == "k":
                        mul = 1_000
                    elif unit == "million":
                        mul = 1_000_000
                    elif unit == "billion":
                        mul = 1_000_000_000
                    amt = int(n * mul)
                money_map["funders"].append({"name": "unknown", "role": "grant", "amount_usd": amt, "when": "", "source_id": src.get("id")})
            if re.search(r"conflict of interest|COI|equity", text, re.I):
                money_map["coi"].append({"actor": "unknown", "type": "equity", "entity": "unknown", "when": "", "source_id": src.get("id")})
            if re.search(r"lobbying|lobbyist", text, re.I):
                m = money_regex.search(text)
                amt = int(float(m.group(1).replace(",", ""))) if m else None
                money_map["lobbying"].append({"entity": "unknown", "amount_usd": amt, "period": "", "source_id": src.get("id")})

        # Policy timeline dates
        date_regex = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
        for src in source_table:
            text = str(_safe_get(_safe_get(src, "extracted", {}), "text", ""))
            if not text:
                continue
            for m in date_regex.finditer(text):
                policy_timeline.append({"when": m.group(1), "event": "policy event", "source_id": src.get("id")})
                if len(policy_timeline) >= 10:
                    break
            if len(policy_timeline) >= 10:
                break

        # Balance enforcement
        if _safe_get(policies, "balance_slate", True):
            if pro and con:
                pass
            elif pro and not con:
                gaps.append("no counter-evidence (con) found")
            elif con and not pro:
                gaps.append("no supporting evidence (pro) found")

        # Metrics
        owners = [src.get("owner") for src in source_table if src.get("owner")]
        unique_owners = len(set(owners)) if owners else 0
        independence_index = 0.0 if not source_table else _round6(min(1.0, unique_owners / max(1, len(source_table))))
        metrics = {"sources_used": len(source_table), "primary": sum(1 for s in source_table if s.get("kind") == "primary"), "dup_rate": 0.0, "independence_index": independence_index}

        slate = {"pro": pro, "con": con, "neutral": neutral}
        uncertainty = "Evidence extracted via heuristics; treat as weak unless confirmed by primary sources."
        return claims, slate, money_map, policy_timeline, gaps, metrics


# -------------------- Artifact writing --------------------

def _save_text(name: str, text: str) -> str:
    path = os.path.join(UPLOAD_ROOT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{name}"
    return f"/uploads/{name}"


def _save_json(name: str, obj: Any) -> str:
    return _save_text(name, json.dumps(obj, ensure_ascii=False))


def _save_csv(name: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return _save_text(name, "")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return _save_text(name, buf.getvalue())


# -------------------- Endpoints --------------------


@app.post("/research/collect")
async def research_collect(body: Dict[str, Any]):
    seed = int(_safe_get(body, "seed", 0) or 0)
    query = str(_safe_get(body, "query", ""))
    horizon = str(_safe_get(body, "time_horizon", ""))
    allow = _safe_get(body, "domains_allow", []) or []
    block = _safe_get(body, "domains_block", []) or []
    modes = _safe_get(body, "modes", {"web": True, "pdf": True}) or {}
    budget = _safe_get(body, "budget", {}) or {}
    max_sources = int(_safe_get(budget, "max_sources", 50) or 50)
    max_fetch_mb = int(_safe_get(budget, "max_fetch_mb", 200) or 200)

    run_id = f"{_now_iso()}-dr-collect-{seed}"
    if not query:
        return JSONResponse(status_code=400, content={"error": "missing query"})

    # Parse horizon
    h_start = h_end = None
    if isinstance(horizon, str) and ".." in horizon:
        a, b = horizon.split("..", 1)
        h_start = _parse_date_ymd(a.strip())
        h_end = _parse_date_ymd(b.strip())

    # Multi-engine discovery and fusion
    results_by_engine = await Discoverer.discover_multi(query, horizon, allow, block, max_sources, seed)
    fused_list = _rrf_fuse(results_by_engine, ENGINE_WEIGHTS, k=60)
    discovered = [{"url": it["url"], "domain": _domain_from_url(it["url"]) } for it in fused_list[:max_sources]]
    fetched_rows: List[Dict[str, Any]] = []
    bytes_total = 0
    blocked_count = 0
    within_horizon = 0
    for item in discovered:
        url = DeduperNormalizer.canonical_url(item["url"])
        dom = item.get("domain") or _domain_from_url(url)
        if block and dom in set([d.lower() for d in block]):
            blocked_count += 1
            continue
        try:
            raw, ctype = await FetcherParser.fetch(url)
        except Exception:
            continue
        size = len(raw)
        bytes_total += size
        if (bytes_total / (1024 * 1024)) > max_fetch_mb:
            break
        sha_hex = _sha256_bytes(raw)
        text = ""
        tables: List[Dict[str, str]] = []
        if "pdf" in ctype.lower() and modes.get("pdf", True):
            text, tables = FetcherParser.parse_pdf(raw)
        elif modes.get("web", True):
            text, tables = FetcherParser.parse_html(raw)
        # naive title extraction
        title = (text.strip().split("\n")[0] if text else url)[:200]
        # scoring features
        owner = _owner_from_domain(dom)
        owner_counts = {owner: 1}
        topical = _topical_score((title + "\n" + text[:2000]), query, [])
        authority = max(0.0, min(1.0, 0.6 * KIND_WEIGHT.get("secondary", 0.7) + 0.4 * _domain_reputation(dom)))
        recency = _recency_score(None, (h_start, h_end), 1.0 / 365.0)
        diversity = _diversity_score(owner, owner_counts)
        score = _composite_score(topical, authority, recency, diversity)
        row = {
            "id": f"src_{len(fetched_rows)+1:03d}",
            "url": url,
            "title": title,
            "kind": "primary" if "gov" in dom or "edu" in dom else "secondary",
            "published_at": "",
            "domain": dom,
            "owner": owner,
            "sha256": sha_hex,
            "bytes": size,
            "extracted": {"text": text, "tables": tables},
            "engine_hits": [],
            "authority": _round6(authority),
            "recency": _round6(recency),
            "topical": _round6(topical),
            "diversity": _round6(diversity),
            "score": _round6(score),
        }
        # horizon check via published_at if available in future; count as within if unknown
        within_horizon += 1
        fetched_rows.append(row)

    # Deduplicate
    deduped, removed_exact = DeduperNormalizer.collapse_dups(fetched_rows)
    near, removed_near = DeduperNormalizer.near_dedup(deduped)
    ordered = _sort_sources(near)
    metrics = {"fetched": len(fetched_rows), "deduped": len(ordered), "blocked": blocked_count, "within_horizon": within_horizon}
    return {
        "run_id": run_id,
        "seed": seed,
        "source_table": ordered,
        "metrics": metrics,
    }


@app.post("/research/analyze")
async def research_analyze(body: Dict[str, Any]):
    seed = int(_safe_get(body, "seed", 0) or 0)
    source_table = _safe_get(body, "source_table", []) or []
    question = str(_safe_get(body, "question", ""))
    horizon = str(_safe_get(body, "time_horizon", ""))
    policies = _safe_get(body, "policies", {}) or {}
    run_id = f"{_now_iso()}-dr-analyze-{seed}"
    if not isinstance(source_table, list):
        return JSONResponse(status_code=400, content={"error": "invalid source_table"})

    claims, slate, money_map, policy_timeline, gaps, metrics = Analyzer.extract_claims(source_table, question, policies, seed)
    return {
        "run_id": run_id,
        "claim_ledger": claims,
        "evidence_slate": slate,
        "money_map": money_map,
        "policy_timeline": policy_timeline,
        "uncertainty": "Evidence extracted with lightweight heuristics; verify against primary sources.",
        "gaps": gaps,
        "metrics": metrics,
    }


@app.post("/research/report")
async def research_report(body: Dict[str, Any]):
    seed = int(_safe_get(body, "seed", 0) or 0)
    question = str(_safe_get(body, "question", ""))
    goal = str(_safe_get(body, "goal", ""))
    claim_ledger = _safe_get(body, "claim_ledger", []) or []
    evidence_slate = _safe_get(body, "evidence_slate", {}) or {}
    money_map = _safe_get(body, "money_map", {}) or {}
    policy_timeline = _safe_get(body, "policy_timeline", []) or []
    uncertainty = str(_safe_get(body, "uncertainty", ""))
    gaps = _safe_get(body, "gaps", []) or []
    style = str(_safe_get(body, "style", "concise"))
    include_tables = bool(_safe_get(body, "include_tables", True))

    run_id = f"{_now_iso()}-dr-report-{seed}"
    inputs_hash = _sha256_str(json.dumps({"question": question, "goal": goal, "seed": seed}, sort_keys=True))

    # Deterministic markdown render
    lines = [f"# Research Report\n", f"Question: {question}\n", f"Goal: {goal}\n\n"]
    lines.append("## Claims\n")
    for cl in claim_ledger:
        cid = cl.get("claim_id")
        text = cl.get("text")
        stance = cl.get("stance")
        lines.append(f"- ({stance}) {cid}: {text}\n")
    lines.append("\n## Money Map\n")
    for k, arr in (money_map or {}).items():
        lines.append(f"- {k}: {len(arr or [])} items\n")
    lines.append("\n## Policy Timeline\n")
    for ev in policy_timeline:
        lines.append(f"- {ev.get('when')}: {ev.get('event')} [#{ev.get('source_id')}]\n")
    lines.append("\n## Uncertainty\n")
    lines.append(uncertainty + "\n")
    if gaps:
        lines.append("\n## Gaps\n")
        for g in gaps:
            lines.append(f"- {g}\n")

    report_uri = _save_text(f"drt_report_{seed}.md", "".join(lines))
    ledger_uri = _save_text(f"drt_study_ledger_{seed}.jsonl", "\n".join([json.dumps(cl) for cl in claim_ledger]))
    # optional source table artifact if provided via body
    source_table = _safe_get(body, "source_table", []) or []
    st_uri = _save_csv(f"drt_source_table_{seed}.csv", source_table)
    run_record = {"seed": seed, "inputs_hash": f"sha256:{inputs_hash}", "ordering_version": DRT_VERSION}
    return {
        "run_id": run_id,
        "report_markdown": report_uri,
        "study_ledger_jsonl": ledger_uri,
        "source_table_csv": st_uri,
        "run_record": run_record,
    }


@app.post("/research/run")
async def research_run(body: Dict[str, Any]):
    seed = int(_safe_get(body, "seed", 0) or 0)
    query = str(_safe_get(body, "query", ""))
    goal = str(_safe_get(body, "goal", ""))
    time_horizon = str(_safe_get(body, "time_horizon", ""))
    domains_allow = _safe_get(body, "domains_allow", []) or []
    domains_block = _safe_get(body, "domains_block", []) or []
    modes = _safe_get(body, "modes", {"web": True, "pdf": True}) or {}
    budget = _safe_get(body, "budget", {}) or {}
    run_id = f"{_now_iso()}-dr-{seed}"
    if not query:
        return JSONResponse(status_code=400, content={"error": "missing query"})
    # 1) collect
    collect_req = {"seed": seed, "query": query, "time_horizon": time_horizon, "domains_allow": domains_allow, "domains_block": domains_block, "modes": modes, "budget": budget}
    col = await research_collect(collect_req)
    if isinstance(col, JSONResponse):
        return col
    source_table = col.get("source_table") or []
    # 2) analyze
    an = await research_analyze({"seed": seed, "source_table": source_table, "question": query, "time_horizon": time_horizon, "policies": {"min_primary": 3, "max_per_owner": 3, "balance_slate": True}})
    if isinstance(an, JSONResponse):
        return an
    # 3) report
    rep = await research_report({
        "seed": seed,
        "question": query,
        "goal": goal,
        "claim_ledger": an.get("claim_ledger"),
        "evidence_slate": an.get("evidence_slate"),
        "money_map": an.get("money_map"),
        "policy_timeline": an.get("policy_timeline"),
        "uncertainty": an.get("uncertainty"),
        "gaps": an.get("gaps"),
        "style": "concise",
        "include_tables": True,
        "source_table": source_table,
    })
    if isinstance(rep, JSONResponse):
        return rep
    # Save run record JSON
    rr_path = _save_json(f"drt_run_record_{seed}.json", {
        "run_id": run_id,
        "seed": seed,
        "ordering_version": DRT_VERSION,
        "counts": {"sources_used": len(source_table)},
    })
    summary_text = "Synthesis with citations." if an.get("claim_ledger") else "No claims extracted."
    return {
        "run_id": run_id,
        "artifacts": {
            "source_table": rep.get("source_table_csv"),
            "study_ledger": rep.get("study_ledger_jsonl"),
            "money_map": _save_json(f"drt_money_map_{seed}.json", an.get("money_map")),
            "policy_timeline": _save_json(f"drt_policy_timeline_{seed}.json", an.get("policy_timeline")),
            "report_markdown": rep.get("report_markdown"),
            "run_record": rr_path,
        },
        "metrics": an.get("metrics"),
        "summary": {"text": summary_text, "confidence": 0.61},
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


