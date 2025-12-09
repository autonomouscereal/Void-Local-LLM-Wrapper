from __future__ import annotations
# HARD BAN (permanent): No Pydantic, no SQLAlchemy/ORM, no CSV/Parquet. JSON/NDJSON only.

import json
import os
import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
try:
    from db.core import get_pool, execute as db_execute, fetchrow as db_fetchrow
except Exception:
    async def get_pool():
        return None
    async def db_execute(*args, **kwargs):
        return ""
    async def db_fetchrow(*args, **kwargs):
        return None


ABLATION_VERSION = "1.0.0"
UPLOAD_ROOT = "/workspace/uploads"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
ABLCODER_URL = os.getenv("ABLCODER_URL", "")  # optional local route to qwen3-ablation-coder
os.makedirs(UPLOAD_ROOT, exist_ok=True)


app = FastAPI(title="Ablation Layer", version=ABLATION_VERSION)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _day_folder(prefix: str) -> str:
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(UPLOAD_ROOT, prefix, d)


def _uri_from_path(path: str) -> str:
    rel = os.path.relpath(path, UPLOAD_ROOT).replace("\\", "/")
    return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if PUBLIC_BASE_URL else f"/uploads/{rel}"


def _write_text(path: str, text: str) -> Dict[str, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return {"uri": _uri_from_path(path), "hash": f"sha256:{_sha256_bytes(text.encode('utf-8'))}"}


def _write_json(path: str, obj: Any) -> Dict[str, str]:
    return _write_text(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _append_ndjson(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(tmp, "a", encoding="utf-8") as f:
        f.write(line)
    # append atomically by concatenating
    with open(tmp, "rb") as rf:
        chunk = rf.read()
    with open(path, "ab") as wf:
        wf.write(chunk)
    os.remove(tmp)


def _tokenize(text: str) -> List[str]:
    import re
    return [w for w in re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())]


def _minhash_signature(text: str, k: int = 64, seed: int = 1337) -> List[int]:
    # simple k hash functions via sha256(seed_i|token) -> int
    toks = list(set(_tokenize(text)))
    sig = [2**63 - 1] * k
    for i in range(k):
        prefix = f"{seed+i}|"
        for t in toks:
            h = int(_sha256_str(prefix + t)[:16], 16)
            if h < sig[i]:
                sig[i] = h
    return sig


def _minhash_sim(a: List[int], b: List[int]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    eq = sum(1 for i in range(len(a)) if a[i] == b[i])
    return eq / float(len(a))


async def _nli_contradiction(text_a: str, text_b: str) -> Tuple[bool, float, str]:
    # Deterministic fallback when ABLCODER_URL not configured
    if not ABLCODER_URL:
        return False, 0.0, "disabled"
    payload = {
        "model_route": "qwen3-ablation-coder",
        "deterministic": True,
        "system": "Ablation NLI Judge. Output strict JSON only.",
        "prompt": (
            "RULES: Decide if TEXT_A contradicts TEXT_B on their core claim. No speculation.\n"
            f"TEXT_A: <<< {text_a} >>>\nTEXT_B: <<< {text_b} >>>\n"
            'RETURN JSON:{"contradiction": true|false, "reason": "≤120 chars", "overlap": 0..1}'
        ),
        "seed": 0,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(ABLCODER_URL.rstrip("/") + "/nli", json=payload)
            r.raise_for_status()
            data = r.json()
            return bool(data.get("contradiction")), float(data.get("overlap", 0.0)), str(data.get("reason", ""))
    except Exception:
        return False, 0.0, "error"


def _xx_seed(master_seed: int, candidate_id: str) -> int:
    try:
        import xxhash
        h = xxhash.xxh64()
        h.update("ablate")
        h.update(str(candidate_id))
        h.update(str(master_seed))
        return h.intdigest() & ((1 << 64) - 1)
    except Exception:
        # fallback to lower 64b of sha256
        return int(_sha256_str(f"ablate|{candidate_id}|{master_seed}")[:16], 16)


def _evidence_score(cand: Dict[str, Any]) -> float:
    cites = cand.get("citations") or []
    total = len(cites)
    valid = sum(1 for z in cites if isinstance(z, dict) and z.get("url") and z.get("hash"))
    prim = sum(1 for z in cites if isinstance(z, dict) and (z.get("kind") == "primary"))
    vr = valid / max(1, total)
    pr = prim / max(1, total)
    return 0.6 * vr + 0.4 * pr


def _independence_score(cand: Dict[str, Any], evidence_index: List[Dict[str, Any]]) -> float:
    owners = []
    for z in cand.get("citations") or []:
        sid = z.get("id")
        src = next((e for e in evidence_index if e.get("id") == sid), None)
        if src and src.get("owner"):
            owners.append(src.get("owner"))
    uniq = len(set(owners))
    return min(1.0, uniq / 6.0)


def _novelty_score(cand_sig: List[int], peer_sigs: List[List[int]]) -> float:
    if not peer_sigs:
        return 1.0
    sims = [ _minhash_sim(cand_sig, s) for s in peer_sigs if s ]
    mx = max(sims) if sims else 0.0
    return 1.0 - mx


def _round6(x: float) -> float:
    return float(f"{float(x):.6f}")


@app.post("/ablate")
async def ablate(body: Dict[str, Any]):
    t0 = time.perf_counter()
    seed = int(body.get("seed", 0) or 0)
    candidates = body.get("candidates") or []
    evidence_index = body.get("evidence_index") or []
    cfg = body.get("config") or {}
    if not isinstance(candidates, list):
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "candidates must be list"})

    # Paths
    out_dir = _day_folder("datasets/ablation")
    clean_path = os.path.join(out_dir, "clean.jsonl")
    drops_path = os.path.join(out_dir, "drops.jsonl")
    report_path = os.path.join(out_dir, "ablation_report.json")
    runrec_path = os.path.join(out_dir, "run_record.json")
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    nli_index_path = os.path.join(cache_dir, "nli_index.json")
    comp_index_path = os.path.join(cache_dir, "compress_index.json")
    dedup_index: Dict[str, str] = {}
    nli_index: Dict[str, Any] = {}
    if os.path.exists(nli_index_path):
        try:
            with open(nli_index_path, "r", encoding="utf-8") as f:
                nli_index = json.load(f)
        except Exception:
            nli_index = {}

    # Precompute signatures
    sigs: Dict[str, List[int]] = {}
    for c in candidates:
        sigs[c.get("id") or "?"] = _minhash_signature(c.get("text") or "", 64, seed)

    min_citations = int(cfg.get("min_citations", 0) or 0)
    require_money = bool(cfg.get("require_money_map", False))
    dedup_threshold = float(cfg.get("dedup_threshold", 0.92) or 0.92)
    novelty_floor = float(cfg.get("novelty_floor", 0.15) or 0.15)

    # Preliminary scoring and ordering by E+I
    prelim: List[Tuple[Dict[str, Any], float, float]] = []
    for c in candidates:
        E = _evidence_score(c)
        I = _independence_score(c, evidence_index)
        prelim.append((c, E, I))
    prelim_sorted = sorted(prelim, key=lambda x: (-(x[1] + x[2]), x[0].get("id")))

    # Contradiction checks (deterministic small peer set)
    kept_items: List[Dict[str, Any]] = []
    peer_sigs: List[List[int]] = []
    kept_ids: List[str] = []
    drops = 0
    kept = 0
    near_dups = 0
    evid_owners = set([e.get("owner") for e in evidence_index if e.get("owner")])

    for c, E, I in prelim_sorted:
        cid = c.get("id")
        text = c.get("text") or ""
        sig = sigs.get(cid)
        # Dedup exact
        tnorm = " ".join((text or "").split()).lower()
        thash = _sha256_str(tnorm)
        if thash in dedup_index:
            _append_ndjson(drops_path, {"id": cid, "drop_reason": f"exact_duplicate_of:{dedup_index[thash]}"})
            drops += 1
            continue
        # Near-dup against kept
        N = _novelty_score(sig, peer_sigs)
        if (1.0 - N) >= dedup_threshold:
            near_dups += 1
            canonical = kept_ids[-1] if kept_ids else (cid or "")
            _append_ndjson(drops_path, {"id": cid, "drop_reason": f"near_duplicate_of:{canonical}"})
            drops += 1
            continue

        # Contradiction rate vs top-M peers (top by E+I, excluding current)
        n_top = int(cfg.get("nli_top_peers", 10) or 10)
        peers = []
        for pc, pE, pI in prelim_sorted:
            if pc.get("id") == cid:
                continue
            peers.append(pc)
            if len(peers) >= n_top:
                break
        contrad_count = 0
        checks = 0
        for p in peers:
            a = text; b = p.get("text") or ""
            key = "|".join(sorted([_sha256_str(a), _sha256_str(b)]))
            res = nli_index.get(key)
            if res is None:
                contradiction, overlap, reason = await _nli_contradiction(a, b)
                res = {"contradiction": bool(contradiction), "overlap": float(overlap), "reason": reason}
                nli_index[key] = res
            if res.get("contradiction"):
                contrad_count += 1
            checks += 1
        C = 1.0 - (contrad_count / float(checks) if checks else 0.0)

        # Stance balance (placeholder 1.0)
        B = 1.0

        # Evidence/constraints
        cites = c.get("citations") or []
        if min_citations and len(cites) < min_citations:
            _append_ndjson(drops_path, {"id": cid, "drop_reasons": ["weak_evidence"]})
            drops += 1
            continue
        if require_money and not (c.get("money_map") and any((c.get("money_map") or {}).values())):
            _append_ndjson(drops_path, {"id": cid, "drop_reasons": ["missing_money_map"]})
            drops += 1
            continue

        # Composite score
        S = 0.35 * E + 0.20 * I + 0.20 * C + 0.15 * N + 0.10 * B
        E = _round6(E); I = _round6(I); C = _round6(C); N = _round6(N); S = _round6(S)

        # Keep rule
        keep = False
        if (S >= 0.70 and E >= 0.65 and I >= 0.60 and (C >= 0.75 or (C >= 0.60 and E >= 0.80)) and N >= novelty_floor):
            keep = True

        if keep:
            kept_items.append({**c, "_scores": {"E": E, "I": I, "C": C, "N": N, "B": B, "S": S}})
            kept_ids.append(cid)
            peer_sigs.append(sig)
            dedup_index[thash] = cid
            kept += 1
        else:
            _append_ndjson(drops_path, {"id": cid, "drop_reasons": ["below_threshold"], "scores": {"E": E, "I": I, "C": C, "N": N}})
            drops += 1

    # Sort kept by tie-breaker
    kept_items = sorted(kept_items, key=lambda x: (-x["_scores"]["S"], -x["_scores"]["E"], -x["_scores"]["I"], -x["_scores"]["N"], x.get("id")))

    # Optional compression (deterministic stub)
    comp_policy = {"max_input_tok": 700, "max_output_tok": 600, "keep_numbers": True, "keep_units": True, "citation_tags": True, "language": "source", "style": "concise-factual"}
    for c in kept_items:
        text = c.get("text") or ""
        citations = c.get("citations") or []
        # Compression cache
        policy_hash = _sha256_str(json.dumps(comp_policy, sort_keys=True, separators=(",", ":")))
        comp_key = f"{c.get('id')}|{policy_hash}"
        comp_index = {}
        if os.path.exists(comp_index_path):
            try:
                with open(comp_index_path, "r", encoding="utf-8") as f:
                    comp_index = json.load(f)
            except Exception:
                comp_index = {}
        cached_comp = comp_index.get(comp_key)
        if cached_comp:
            c["train_payload"] = {"input": cached_comp.get("input"), "output": cached_comp.get("output"), "metadata": {"mode": c.get("mode"), "seed": seed}}
        elif ABLCODER_URL:
            try:
                async with httpx.AsyncClient() as client:
                    payload = {"model_route": "qwen3-ablation-coder", "deterministic": True, "system": "Ablation Compressor. Output strict JSON only.", "seed": 0, "text": text, "citations": citations, "policy": comp_policy}
                    r = await client.post(ABLCODER_URL.rstrip("/") + "/compress", json=payload)
                    r.raise_for_status()
                    js = r.json()
                    c["train_payload"] = {"input": js.get("input") or text[:2000], "output": js.get("output") or text[:2000], "metadata": {"mode": c.get("mode"), "seed": seed}}
                    comp_index[comp_key] = {"input": c["train_payload"]["input"], "output": c["train_payload"]["output"]}
            except Exception:
                c["train_payload"] = {"input": text[:2000], "output": text[:2000], "metadata": {"mode": c.get("mode"), "seed": seed}}
        else:
            c["train_payload"] = {"input": text[:2000], "output": text[:2000], "metadata": {"mode": c.get("mode"), "seed": seed}}
        # Persist compression cache
        _write_json(comp_index_path, comp_index)

    # Write clean.jsonl
    for c in kept_items:
        rec = {
            "id": c.get("id"),
            "keep": True,
            "score": c["_scores"]["S"],
            "signals": {"evidence": c["_scores"]["E"], "independence": c["_scores"]["I"], "consistency": c["_scores"]["C"], "novelty": c["_scores"]["N"], "stance_balance": 1.0},
            "rationale": ["kept by composite thresholds"],
            "icw_refs": [],
            "train_payload": c.get("train_payload"),
        }
        _append_ndjson(clean_path, rec)

    # Caches
    _write_json(nli_index_path, nli_index)

    # Report and run record
    kept_count = len(kept_items)
    total = len(candidates)
    avg_score = _round6(sum([c["_scores"]["S"] for c in kept_items]) / kept_count) if kept_count else 0.0
    report = {
        "ablation_version": ABLATION_VERSION,
        "config": cfg,
        "counts": {"candidates": total, "kept": kept_count, "drops": total - kept_count, "near_dups": near_dups},
        "score_hist": {"mean": avg_score},
        "balance": {"pro": 0, "con": 0, "neutral": kept_count},
        "independence_index": _round6(min(1.0, len(evid_owners) / 6.0)) if len(evid_owners) >= 1 else "N/A",
        "notes": "qwen3-ablation-coder; deterministic mode",
    }
    _write_json(report_path, report)
    inputs_hash = _sha256_str(json.dumps(body, sort_keys=True))
    config_hash = _sha256_str(json.dumps(cfg, sort_keys=True))
    run_record = {
        "ts": _now_iso(),
        "seed": seed,
        "inputs_hash": f"sha256:{inputs_hash}",
        "config_hash": f"sha256:{config_hash}",
        "ablation_version": ABLATION_VERSION,
        "model_route": "qwen3-ablation-coder",
        "timings_ms": {"score": int((time.perf_counter() - t0) * 1000)},
        "paths": {"clean": _uri_from_path(clean_path), "drops": _uri_from_path(drops_path)},
    }
    _write_json(runrec_path, run_record)

    result = {
        "run_id": f"{_now_iso()}-ablate-{seed}",
        "paths": {"clean_jsonl": _uri_from_path(clean_path), "drops_jsonl": _uri_from_path(drops_path), "report_json": _uri_from_path(report_path), "run_record": _uri_from_path(runrec_path)},
        "metrics": {"candidates": total, "kept": kept_count, "drop_rate": _round6((total - kept_count) / max(1, total)), "dup_rate": _round6(near_dups / max(1, total)), "avg_score": avg_score},
    }
    # Optional DB: persist batch + items
    try:
        pool = await get_pool()
        if pool is not None:
            batch_uid = result["run_id"]
            await db_execute("INSERT INTO ablation_run(batch_uid, config_json, report_json) VALUES($1,$2,$3)", batch_uid, cfg, report)
            row = await db_fetchrow("SELECT id FROM ablation_run WHERE batch_uid=$1", batch_uid)
            if row:
                aid = int(row[0])
                for c in kept_items:
                    await db_execute("INSERT INTO ablation_clean(ablation_id, item_json) VALUES($1,$2)", aid, c)
                # read drops.jsonl quickly
                try:
                    from void_json.json_parser import JSONParser

                    parser = JSONParser()
                    with open(drops_path, "r", encoding="utf-8") as f:
                        for ln in f:
                            if ln.strip():
                                try:
                                    sup = parser.parse_superset(ln, {})
                                    obj = sup.get("coerced") or {}
                                    if isinstance(obj, dict):
                                        await db_execute(
                                            "INSERT INTO ablation_drop(ablation_id, item_json) VALUES($1,$2)",
                                            aid,
                                            obj,
                                        )
                                except Exception:
                                    continue
                except Exception:
                    pass
    except Exception:
        pass
    return result


@app.post("/trainset/ingest")
async def trainset_ingest(body: Dict[str, Any]):
    seed = int(body.get("seed", 0) or 0)
    clean_uri = str(body.get("clean_uri") or "")
    shard_size = int(body.get("shard_size", 5000) or 5000)
    out_dir_uri = str(body.get("output_dir") or "")
    if not clean_uri or not out_dir_uri:
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "clean_uri and output_dir required"})

    # Expect URIs under /uploads
    def _path_from_uri(u: str) -> str:
        if not u:
            return ""
        if u.startswith("/uploads/"):
            return os.path.join(UPLOAD_ROOT, u[len("/uploads/"):])
        if PUBLIC_BASE_URL and u.startswith(PUBLIC_BASE_URL.rstrip("/") + "/uploads/"):
            return os.path.join(UPLOAD_ROOT, u.split("/uploads/", 1)[-1])
        if u.startswith("uri:///uploads/"):
            return os.path.join(UPLOAD_ROOT, u.split("/uploads/", 1)[-1])
        # Fallback: treat as filesystem path
        return u

    clean_path = _path_from_uri(clean_uri)
    out_dir = _path_from_uri(out_dir_uri)
    os.makedirs(out_dir, exist_ok=True)
    # Read lines and shard
    items: List[str] = []
    try:
        with open(clean_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(line.rstrip("\n"))
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": f"cannot read clean_uri: {ex}"})

    shards_meta: List[Dict[str, Any]] = []
    shard_idx = 1
    for i in range(0, len(items), shard_size):
        shard_lines = items[i: i + shard_size]
        shard_name = f"shard_{shard_idx:05d}.jsonl"
        shard_path = os.path.join(out_dir, shard_name)
        _write_text(shard_path, "\n".join(shard_lines) + ("\n" if shard_lines else ""))
        _sh_text = ("\n".join(shard_lines) + "\n").encode("utf-8") if shard_lines else b""
        _sh_hash = _sha256_bytes(_sh_text)
        shards_meta.append({"uri": _uri_from_path(shard_path), "count": len(shard_lines), "hash": f"sha256:{_sh_hash}"})
        shard_idx += 1

    runrec = {"ts": _now_iso(), "seed": seed, "paths": [m["uri"] for m in shards_meta]}
    rec_meta = _write_json(os.path.join(out_dir, "run_record.json"), runrec)
    return {"shards": shards_meta, "run_record": rec_meta["uri"]}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


