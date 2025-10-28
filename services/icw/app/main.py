from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
import time
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse


# ------------------------------------------------------------
# Infinite Context Window (ICW) — CPU‑friendly, deterministic
# ------------------------------------------------------------
# Endpoints:
#  - POST /context/pack
#  - POST /output/plan
#  - POST /output/render
#
# Determinism policy:
#  - All floating scores rounded to 1e-6
#  - Tie‑break: (score desc, authority desc, recency desc, hash asc)
#  - Inputs hash included in AuditTrail
#  - Renderer is stateless; continuation via next_cursor only


ICW_VERSION = "1.0.0"
app = FastAPI(title="ICW Core", version=ICW_VERSION)


# -------------------- Utilities --------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        v = d.get(key, default)
        return v
    except Exception:
        return default


def _parse_date_ymd(s: Optional[str]) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    try:
        # Expect YYYY-MM-DD
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _round6(x: float) -> float:
    return float(f"{x:.6f}")


class TokenEstimator:
    # Default calibration; can be overridden at runtime
    a = 0.25  # ~ 1 token per 4 chars
    b = 0.0
    c = 0.0

    @staticmethod
    def configure(a: float, b: float, c: float) -> None:
        TokenEstimator.a = float(a)
        TokenEstimator.b = float(b)
        TokenEstimator.c = float(c)

    @staticmethod
    def estimate_tokens_from_text(text: str) -> int:
        if not text:
            return 0
        lines = text.count("\n") + 1
        return max(1, int(math.ceil(TokenEstimator.a * len(text) + TokenEstimator.b * lines + TokenEstimator.c)))

    @staticmethod
    def estimate_tokens_from_items(items: List[Tuple[str, int]]) -> int:
        # items: (text, extra_overhead_tokens)
        total = 0
        for text, extra in items:
            total += TokenEstimator.estimate_tokens_from_text(text) + int(extra or 0)
        return total


# -------------------- Core scoring --------------------

KIND_WEIGHT = {
    "primary": 1.0,
    "secondary": 0.7,
    "doc": 0.5,
    "code": 0.8,
    "media": 0.6,
}


def _domain_reputation(domain: Optional[str]) -> float:
    if not isinstance(domain, str) or not domain:
        return 0.5
    dl = domain.lower()
    if dl.endswith(".gov"):
        return 1.0
    if dl.endswith(".edu"):
        return 0.9
    if dl.endswith(".org"):
        return 0.75
    return 0.6


class CandidateScorer:
    def __init__(self, goal: str, query: str, hints: Dict[str, Any], horizon: Tuple[Optional[datetime], Optional[datetime]], *, preferred_domains_delta: float = 0.05, decay: float = 1.0/365.0):
        self.goal = (goal or "").lower()
        self.query = (query or "").lower()
        self.entities = set([e.lower() for e in _safe_get(hints, "known_entities", []) or []])
        self.horizon_start, self.horizon_end = horizon
        self.preferred = set([d.lower() for d in _safe_get(hints, "preferred_domains", []) or []])
        self.pref_delta = float(preferred_domains_delta)
        self.decay = float(decay)

    def topical(self, title: str, content: str, meta_lang: Optional[str]) -> float:
        text = f"{title}\n{content}".lower()
        # simple overlap features
        q_tokens = [t for t in self.query.split() if t]
        hits = sum(1 for t in q_tokens if t in text)
        ent_hits = sum(1 for e in self.entities if e and e in text)
        # modest boost for English if unspecified
        lang_bonus = 0.05 if (meta_lang in (None, "en")) else 0.0
        base = 0.0
        if q_tokens:
            base = hits / max(1.0, len(q_tokens))
        base = _clamp01(base + 0.2 * _clamp01(ent_hits / max(1.0, len(self.entities) or 1)))
        return _clamp01(base + lang_bonus)

    def authority(self, kind: Optional[str], domain: Optional[str]) -> float:
        kw = KIND_WEIGHT.get(str(kind or "").lower(), 0.6)
        rep = _domain_reputation(domain)
        bonus = self.pref_delta if (isinstance(domain, str) and domain.lower() in self.preferred) else 0.0
        return _clamp01(0.6 * kw + 0.4 * rep + bonus)

    def recency(self, published_at: Optional[str]) -> float:
        # exp(-decay*age_days), windowed by horizon
        ts = _parse_date_ymd(published_at)
        if not ts:
            return 0.5
        if self.horizon_start and ts < self.horizon_start:
            return 0.0
        if self.horizon_end and ts > self.horizon_end:
            return 0.0
        delta_days = max(0.0, (datetime.now(timezone.utc) - ts).days)
        return _clamp01(math.exp(-self.decay * delta_days))

    @staticmethod
    def diversity(owner: Optional[str], owner_counts: Dict[str, int], dup_penalty: float) -> float:
        # More unique owners → higher diversity
        oc = owner_counts.get(owner or "", 0)
        owner_pen = 0.15 if oc > 0 else 0.0
        div = _clamp01(1.0 - owner_pen - dup_penalty)
        return div

    def score(self, c: Dict[str, Any], owner_counts: Dict[str, int], dup_penalty: float) -> Dict[str, float]:
        title = str(_safe_get(c, "title", ""))
        content = str(_safe_get(c, "content", ""))
        meta = _safe_get(c, "metadata", {}) or {}
        source = _safe_get(c, "source", {}) or {}
        domain = _safe_get(source, "domain")
        owner = _safe_get(source, "owner")
        kind = _safe_get(c, "kind")
        topical = self.topical(title, content, _safe_get(meta, "lang"))
        authority = self.authority(kind, domain)
        recency = self.recency(_safe_get(c, "published_at"))
        diversity = self.diversity(owner, owner_counts, dup_penalty)
        score = 0.40 * topical + 0.25 * authority + 0.20 * recency + 0.15 * diversity
        return {
            "topical": _round6(topical),
            "authority": _round6(authority),
            "recency": _round6(recency),
            "diversity": _round6(diversity),
            "score": _round6(score),
        }


class DeterministicSorter:
    @staticmethod
    def sort(scored: List[Tuple[Dict[str, Any], Dict[str, float]]]) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
        def key_fn(item):
            c, s = item
            # tie-break: (score desc, authority desc, recency desc, hash asc)
            h = str(_safe_get(c, "hash", ""))
            return (-s["score"], -s["authority"], -s["recency"], h)

        return sorted(scored, key=key_fn)


# -------------------- Budget + Compression --------------------

TIERS = [
    ("full", 550),      # Tier0 Full    400–700 → use mid estimate
    ("summary", 275),   # Tier1 Summary 200–350
    ("bullets", 115),   # Tier2 Bullets 80–150
    ("keylines", 50),   # Tier3 Keylines 40–60
    ("metadata", 16),   # Tier4 Metadata 12–20
]


def _compress_text(tier: str, c: Dict[str, Any]) -> str:
    cid = str(_safe_get(c, "id", ""))
    title = str(_safe_get(c, "title", ""))
    url = str(_safe_get(c, "url", ""))
    published_at = str(_safe_get(c, "published_at", ""))
    content = str(_safe_get(c, "content", ""))

    # Preserve exact numbers and dates as they appear; never fabricate
    if tier == "full":
        body = content
    elif tier == "summary":
        body = content[:2000]
    elif tier == "bullets":
        # naive sentence split → bullets
        sents = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]
        body = "\n".join([f"- {s}" for s in sents[:8]])
    elif tier == "keylines":
        # first couple of key lines
        sents = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]
        body = "\n".join(sents[:3])
    else:
        # metadata only
        body = f"title: {title}\nurl: {url}\ndate: {published_at}"
    # Append inline citation
    return f"{body} [#{cid}]"


class BudgetAllocator:
    def __init__(self, budget_tokens: int, reserve_ratio: float = 0.12):
        self.total_budget = max(1, int(budget_tokens))
        self.reserve = int(self.total_budget * reserve_ratio)
        self.available = max(1, self.total_budget - self.reserve)

    def allocate(self, scored_sorted: List[Tuple[Dict[str, Any], Dict[str, float]]], max_per_source: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Select candidates respecting per-owner cap, then fit tiers greedily degrading until under budget
        owner_used: Dict[str, int] = {}
        chosen: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        for c, s in scored_sorted:
            owner = _safe_get(_safe_get(c, "source", {}) or {}, "owner") or ""
            if max_per_source and owner_used.get(owner, 0) >= int(max_per_source):
                continue
            owner_used[owner] = owner_used.get(owner, 0) + 1
            chosen.append((c, s))

        # Initial assignment: Tier1 for all
        assign: Dict[str, str] = {}
        packed_items: List[Tuple[str, int]] = []
        texts: Dict[str, str] = {}
        for c, _ in chosen:
            tier = "summary"
            txt = _compress_text(tier, c)
            assign[c["id"]] = tier
            texts[c["id"]] = txt
            packed_items.append((txt, 2))

        est = TokenEstimator.estimate_tokens_from_items(packed_items) + self.reserve
        guard = int(math.floor(self.total_budget * 1.03))
        # Degrade to meet budget
        if est > self.total_budget:
            order_ids = [c["id"] for c, _ in chosen[::-1]]  # degrade from least important backward
            levels = ["bullets", "keylines", "metadata"]
            for level in levels:
                for cid in order_ids:
                    if est <= self.total_budget:
                        break
                    current = assign[cid]
                    # Only degrade if current is higher than target level
                    idx_cur = [t for t, _ in TIERS].index(current)
                    idx_new = [t for t, _ in TIERS].index(level)
                    if idx_new <= idx_cur:
                        assign[cid] = level
                        txt = _compress_text(level, next(c for c, _ in chosen if c["id"] == cid))
                        old_txt = texts[cid]
                        # update estimate
                        est -= TokenEstimator.estimate_tokens_from_text(old_txt)
                        est += TokenEstimator.estimate_tokens_from_text(txt)
                        texts[cid] = txt
                if est <= self.total_budget:
                    break
            # Last resort demotions by one step from the tail
            if est > self.total_budget:
                seq = [c["id"] for c, _ in chosen[::-1]]
                order = [t for t, _ in TIERS]
                for cid in seq:
                    cur = assign[cid]
                    idx = order.index(cur)
                    if idx < len(order) - 1:
                        new_level = order[idx + 1]
                        assign[cid] = new_level
                        txt = _compress_text(new_level, next(c for c, _ in chosen if c["id"] == cid))
                        old_txt = texts[cid]
                        est -= TokenEstimator.estimate_tokens_from_text(old_txt)
                        est += TokenEstimator.estimate_tokens_from_text(txt)
                        texts[cid] = txt
                    if est <= self.total_budget:
                        break

        # If budget allows, upgrade top few items to full
        if est + 400 < self.total_budget:
            for c, _ in chosen[: min(3, len(chosen))]:
                cid = c["id"]
                if assign.get(cid) == "summary":
                    txt_new = _compress_text("full", c)
                    est_delta = TokenEstimator.estimate_tokens_from_text(txt_new) - TokenEstimator.estimate_tokens_from_text(texts[cid])
                    if est + est_delta + 16 <= self.total_budget:
                        assign[cid] = "full"
                        texts[cid] = txt_new
                        est += est_delta

        # Materialize evidence index
        evidence = []
        for c, s in chosen:
            tname = assign.get(c["id"]) or "metadata"
            evidence.append({
                "id": c.get("id"),
                "url": c.get("url"),
                "hash": c.get("hash"),
                "kind": c.get("kind"),
                "authority": s.get("authority"),
                "recency": s.get("recency"),
                "diversity": s.get("diversity"),
                "compression_tier": tname,
                "char_count": len(str(c.get("content") or "")),
            })

        # Build PACK (QUERY/GOAL included in sections later)
        pack_chunks = [texts[cid] for cid in texts]
        pack_text = "\n\n".join(pack_chunks)
        final_est = TokenEstimator.estimate_tokens_from_text(pack_text) + self.reserve
        if final_est > guard:
            # Hard guard: trim last chunks until under guard
            ids_order = list(texts.keys())[::-1]
            for cid in ids_order:
                if final_est <= self.total_budget:
                    break
                old = texts.pop(cid, "")
                final_est -= TokenEstimator.estimate_tokens_from_text(old)
            pack_text = "\n\n".join([texts[k] for k in texts])
        audit = {
            "tiers": assign,
            "token_budget": self.total_budget,
            "reserved": self.reserve,
            "estimated_tokens": final_est,
        }
        return [dict(e) for e in evidence], {**audit, "pack_text": pack_text}


# -------------------- Planner/Renderer --------------------

class OutputPlanner:
    @staticmethod
    def plan(seed: int, style: str, global_budget: Dict[str, Any], outline_policy: Dict[str, Any], goal: str, pack: str) -> Dict[str, Any]:
        seg_budget = int(_safe_get(global_budget, "segment_tokens", 800) or 800)
        auto = bool(_safe_get(outline_policy, "auto_outline", True))
        max_depth = int(_safe_get(outline_policy, "max_depth", 3) or 3)
        min_sections = int(_safe_get(outline_policy, "min_sections", 3) or 3)

        # Deterministic outline using proportional allocation 15/70/15
        outline: List[Dict[str, Any]] = []
        if auto:
            total = seg_budget * 4
            intro = max(1, int(round(0.15 * total)))
            body = max(1, int(round(0.70 * total)))
            concl = max(1, int(round(0.15 * total)))
            rnd = random.Random((seed, "outline", ICW_VERSION))
            body_parts = 2 + (rnd.randint(0, 10) % 2)
            per = max(1, int(round(body / body_parts)))
            children = []
            for i in range(body_parts):
                children.append({"id": f"S2.{i+1}", "title": f"Section {i+1}", "est_tokens": per})
            outline = [
                {"id": "S1", "title": "Introduction", "est_tokens": intro},
                {"id": "S2", "title": "Core Sections", "est_tokens": body, "children": children},
                {"id": "S3", "title": "Conclusion", "est_tokens": concl},
            ]
            if len(outline) < min_sections:
                # pad with boilerplate sections deterministically
                for i in range(len(outline) + 1, min_sections + 1):
                    outline.insert(-1, {"id": f"S{i}", "title": f"Section {i}", "est_tokens": seg_budget})
        else:
            outline = [{"id": "S1", "title": "Body", "est_tokens": seg_budget}]

        plan_id = f"{_now_iso()}-io-{seed}"
        return {
            "plan_id": plan_id,
            "seed": seed,
            "outline": outline,
            "segment_budget": seg_budget,
            "cursor": {"section_id": outline[0]["id"], "offset": 0, "segment_index": 0},
        }


class OutputRenderer:
    @staticmethod
    def _section_by_id(outline: List[Dict[str, Any]], sid: str) -> Optional[Dict[str, Any]]:
        queue = list(outline)
        while queue:
            node = queue.pop(0)
            if node.get("id") == sid:
                return node
            for ch in node.get("children", []) or []:
                queue.append(ch)
        return None

    @staticmethod
    def render(seed: int, plan_id: str, segment_budget: int, cursor: Dict[str, Any], style: str, pack: str, goal: str, outline: List[Dict[str, Any]]) -> Dict[str, Any]:
        sid = str(_safe_get(cursor, "section_id", "S1"))
        offset = int(_safe_get(cursor, "offset", 0) or 0)
        seg_idx = int(_safe_get(cursor, "segment_index", 0) or 0)
        node = OutputRenderer._section_by_id(outline, sid) or {"title": "Section", "est_tokens": segment_budget}

        # Deterministic content: header + a slice of the context pack
        header = ""
        if style in ("with_headers", "markdown"):
            header = f"## {node.get('title')}\n\n"
        base_text = header + pack

        # Soft stop 92%; Cut a bounded window starting at ~ 4*offset
        soft = int(max(1, math.floor(segment_budget * 0.92)))
        char_start = max(0, offset * 4)
        char_budget = max(1, soft * 4)
        piece = base_text[char_start: char_start + char_budget]
        # End cleanly at sentence/paragraph boundary if possible
        end = max(piece.rfind("\n\n"), piece.rfind(". "))
        if end >= 0 and end >= len(header):
            piece = piece[: end + 1]

        est_tokens = TokenEstimator.estimate_tokens_from_text(piece)
        while est_tokens > segment_budget and len(piece) > 0:
            piece = piece[:-16]
            est_tokens = TokenEstimator.estimate_tokens_from_text(piece)

        next_cursor = {"section_id": sid, "offset": offset + est_tokens, "segment_index": seg_idx + 1, "done": False}

        # Mark section done if we've consumed estimated section tokens worth of content
        est_total = int(_safe_get(node, "est_tokens", segment_budget) or segment_budget)
        if offset + est_tokens >= est_total:
            # linearize outline to get next section id
            order: List[str] = []
            q = list(outline)
            while q:
                n = q.pop(0)
                order.append(n.get("id"))
                for ch in n.get("children", []) or []:
                    q.append(ch)
            try:
                idx = order.index(sid)
                nxt = order[idx + 1] if idx + 1 < len(order) else None
            except ValueError:
                nxt = None
            if nxt is None:
                next_cursor["done"] = True
            else:
                next_cursor = {"section_id": nxt, "offset": 0, "segment_index": seg_idx + 1, "done": False}

        return {
            "segment_text": piece,
            "next_cursor": next_cursor,
            "telemetry": {"estimated_tokens": est_tokens, "stop_reason": ("section_done" if next_cursor["done"] else "budget_reached")},
        }


# -------------------- API Endpoints --------------------


PACK_CACHE: Dict[str, Dict[str, Any]] = {}
PLAN_CACHE: Dict[str, Dict[str, Any]] = {}
MAX_CACHE = 128


def _lru_put(cache: Dict[str, Any], key: str, val: Any):
    cache[key] = val
    if len(cache) > MAX_CACHE:
        try:
            first_key = next(iter(cache.keys()))
            if first_key in cache and first_key != key:
                cache.pop(first_key, None)
        except Exception:
            pass


@app.post("/context/pack")
async def context_pack(body: Dict[str, Any]):
    t0 = time.perf_counter()
    seed = int(_safe_get(body, "seed", 0) or 0)
    budget_tokens = int(_safe_get(body, "budget_tokens", 3500) or 3500)
    goal = str(_safe_get(body, "goal", ""))
    query = str(_safe_get(body, "query", ""))
    candidates = _safe_get(body, "candidates", []) or []
    policies = _safe_get(body, "policies", {}) or {}
    hints = _safe_get(body, "context_hints", {}) or {}
    if budget_tokens <= 0 or not isinstance(candidates, list):
        return JSONResponse(status_code=400, content={"error": "invalid_request"})

    # Filter by domains and time horizon
    forbidden_domains = set([d.lower() for d in _safe_get(hints, "forbidden_domains", []) or []])
    preferred_domains = set([d.lower() for d in _safe_get(hints, "preferred_domains", []) or []])
    horizon_s = None
    horizon_e = None
    time_horizon = _safe_get(policies, "time_horizon")
    if isinstance(time_horizon, str) and ".." in time_horizon:
        a, b = time_horizon.split("..", 1)
        horizon_s = _parse_date_ymd(a.strip())
        horizon_e = _parse_date_ymd(b.strip())

    # Owner and duplicate statistics
    owner_counts: Dict[str, int] = {}
    hash_counts: Dict[str, int] = {}
    prefiltered: List[Dict[str, Any]] = []
    for c in candidates:
        src = _safe_get(c, "source", {}) or {}
        dom = str(_safe_get(src, "domain", "") or "").lower()
        if dom and dom in forbidden_domains:
            continue
        owner = str(_safe_get(src, "owner", "") or "")
        owner_counts[owner] = owner_counts.get(owner, 0) + 1
        h = str(_safe_get(c, "hash", "") or "")
        hash_counts[h] = hash_counts.get(h, 0) + 1
        prefiltered.append(c)

    preferred_delta = float(_safe_get(policies, "preferred_authority_delta", 0.05) or 0.05)
    decay = float(_safe_get(policies, "decay", 1.0/365.0) or (1.0/365.0))
    scorer = CandidateScorer(goal, query, hints, (horizon_s, horizon_e), preferred_domains_delta=preferred_delta, decay=decay)
    scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    t1 = time.perf_counter()
    for c in prefiltered:
        h = str(_safe_get(c, "hash", "") or "")
        dup_pen = 0.2 if (hash_counts.get(h, 0) > 1 and h) else 0.0
        s = scorer.score(c, owner_counts, dup_pen)
        scored.append((c, s))

    ordered = DeterministicSorter.sort(scored)

    max_per_source = int(_safe_get(policies, "max_per_source", 2) or 2)
    reserve_frac = float(_safe_get(policies, "reserve_frac", 0.12) or 0.12)
    allocator = BudgetAllocator(budget_tokens, reserve_ratio=reserve_frac)
    t2 = time.perf_counter()
    evidence, audit = allocator.allocate(ordered, max_per_source=max_per_source)
    t3 = time.perf_counter()

    # Independence/diversity index (owner diversity): unique owners / selected
    selected_ids = [e.get("id") for e in evidence]
    selected: List[Dict[str, Any]] = [next((c for c in prefiltered if c.get("id") == sid), {}) for sid in selected_ids]
    owners = [(_safe_get(_safe_get(c, "source", {}) or {}, "owner") or "") for c in selected]
    uniq_owners = len(set(owners)) if owners else 0
    independence_index = 0.0 if not selected else _round6(_clamp01(uniq_owners / max(1.0, len(selected))))

    # Dup rate among selected
    sel_hashes = [str(_safe_get(next((c for c in prefiltered if c.get("id") == sid), {}), "hash", "") or "") for sid in selected_ids]
    dup_rate = 0.0
    if sel_hashes:
        total_pairs = len(sel_hashes) * (len(sel_hashes) - 1) / 2.0
        if total_pairs > 0:
            dup_pairs = 0
            for i in range(len(sel_hashes)):
                for j in range(i + 1, len(sel_hashes)):
                    if sel_hashes[i] and sel_hashes[i] == sel_hashes[j]:
                        dup_pairs += 1
            dup_rate = _round6(dup_pairs / total_pairs)

    # Build sections
    sections = {
        "QUERY": query,
        "GOAL": goal,
        "PACK": audit["pack_text"],
    }

    inputs_hash = _sha256_hex(json.dumps(body, sort_keys=True, ensure_ascii=False))
    run_id = f"{_now_iso()}-icw-{seed}"
    # Validate citations subset of evidence ids
    try:
        cited = set([m.group(1) for m in re.finditer(r"\[#([^\]]+)\]", sections["PACK"])])
        ev_ids = set([e.get("id") for e in evidence])
        if not cited.issubset(ev_ids):
            missing = sorted(list(cited - ev_ids))
            return JSONResponse(status_code=422, content={"error": "citation_mismatch", "missing": missing})
    except Exception:
        pass
    if not evidence:
        return JSONResponse(status_code=409, content={
            "run_id": run_id,
            "seed": seed,
            "icw_version": ICW_VERSION,
            "budget_tokens": budget_tokens,
            "estimated_tokens": 0,
            "scores_summary": {"selected": 0, "independence_index": 0.0, "dup_rate": 0.0},
            "evidence_index": [],
            "sections": {"QUERY": query, "GOAL": goal, "PACK": "", "GAPS": "No viable candidates after policy filters."},
        })

    # Cache key
    cache_key = _sha256_hex("|".join([str(seed), str(budget_tokens), goal, query, inputs_hash, ICW_VERSION]))
    response = {
        "run_id": run_id,
        "seed": seed,
        "icw_version": ICW_VERSION,
        "budget_tokens": budget_tokens,
        "estimated_tokens": int(audit["estimated_tokens"]),
        "scores_summary": {
            "selected": len(evidence),
            "independence_index": independence_index,
            "dup_rate": dup_rate,
        },
        "evidence_index": evidence,
        "sections": sections,
        "audit": {
            "selected_ids": selected_ids,
            "tiers": audit.get("tiers"),
            "token_budget": audit.get("token_budget"),
            "reserved": audit.get("reserved"),
            "inputs_hash": inputs_hash,
            "seed": seed,
        },
        "telemetry": {
            "tokens_estimate": int(audit.get("estimated_tokens", 0)),
            "timings_ms": {
                "score": round((t2 - t1) * 1000, 2),
                "allocate": round((t3 - t2) * 1000, 2),
                "compress": round((time.perf_counter() - t3) * 1000, 2),
                "total": round((time.perf_counter() - t0) * 1000, 2),
            },
        },
    }
    _lru_put(PACK_CACHE, cache_key, response)
    return response


@app.post("/output/plan")
async def output_plan(body: Dict[str, Any]):
    seed = int(_safe_get(body, "seed", 0) or 0)
    target_style = str(_safe_get(body, "target_style", "plain"))
    global_budget = _safe_get(body, "global_budget", {}) or {}
    outline_policy = _safe_get(body, "outline_policy", {}) or {}
    goal = str(_safe_get(body, "goal", ""))
    pack = str(_safe_get(body, "context_pack", ""))

    # Optional simple cache
    pack_hash = _sha256_hex(pack)
    cache_key = _sha256_hex("|".join([str(seed), pack_hash, target_style, json.dumps(outline_policy, sort_keys=True), ICW_VERSION]))
    cached = PLAN_CACHE.get(cache_key)
    if cached:
        return cached

    plan = OutputPlanner.plan(seed, target_style, global_budget, outline_policy, goal, pack)
    _lru_put(PLAN_CACHE, cache_key, plan)
    return plan


@app.post("/output/render")
async def output_render(body: Dict[str, Any]):
    seed = int(_safe_get(body, "seed", 0) or 0)
    plan_id = str(_safe_get(body, "plan_id", ""))
    segment_budget = int(_safe_get(body, "segment_budget", 800) or 800)
    cursor = _safe_get(body, "cursor", {}) or {}
    style = str(_safe_get(body, "style", "plain"))
    pack = str(_safe_get(body, "context_pack", ""))
    goal = str(_safe_get(body, "goal", ""))
    outline = _safe_get(body, "outline", []) or _safe_get(body, "_outline", []) or []
    # If outline not provided, synthesize a trivial one matching the plan_id semantics
    if not outline:
        outline = [{"id": cursor.get("section_id", "S1"), "title": "Section", "est_tokens": segment_budget}]

    result = OutputRenderer.render(seed, plan_id, segment_budget, cursor, style, pack, goal, outline)
    return result


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


