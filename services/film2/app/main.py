from __future__ import annotations
# HARD BAN (permanent): No Pydantic, no SQLAlchemy/ORM, no CSV/Parquet. JSON/NDJSON only.

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
try:
    from db.core import get_pool, execute as db_execute, fetchrow as db_fetchrow
except Exception:  # optional DB
    async def get_pool():
        return None
    async def db_execute(*args, **kwargs):
        return ""
    async def db_fetchrow(*args, **kwargs):
        return None


FILM_VERSION = "2.0.0"
UPLOAD_ROOT = "/workspace/uploads"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
PROJECTS_PREFIX = "projects"


app = FastAPI(title="Film 2.0", version=FILM_VERSION)


# -------------------- Helpers --------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _seed64(*parts: str) -> int:
    try:
        import xxhash  # optional; prefer xxhash when available
        h = xxhash.xxh64()
        for p in parts:
            h.update(str(p))
        return h.intdigest() & ((1 << 64) - 1)
    except Exception:
        # fallback: lower 64 bits of sha256
        return int(_sha256_str("::".join(parts))[:16], 16)


def _save_bytes(rel_name: str, data: bytes) -> Dict[str, str]:
    path = os.path.join(UPLOAD_ROOT, rel_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    uri = f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel_name}" if PUBLIC_BASE_URL else f"/uploads/{rel_name}"
    return {"uri": uri, "hash": f"sha256:{_sha256_bytes(data)}"}


def _det_id(prefix: str, *parts: str, take: int = 10) -> str:
    h = _sha256_str("|".join([prefix] + list(parts)))
    return f"{prefix}_{h[:take]}"


def _response_error(code: int, msg: str):
    return JSONResponse(status_code=code, content={"error": msg})


def _proj_dir(project_id: str, *parts: str) -> str:
    return os.path.join(UPLOAD_ROOT, PROJECTS_PREFIX, project_id, *parts)


def _proj_uri(project_id: str, *parts: str) -> str:
    rel = "/".join([PROJECTS_PREFIX, project_id] + list(parts))
    return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if PUBLIC_BASE_URL else f"/uploads/{rel}"


def _write_json(project_id: str, rel_name: str, obj: Any) -> Dict[str, str]:
    path = _proj_dir(project_id, rel_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)
    uri = _proj_uri(project_id, rel_name)
    return {"uri": uri, "hash": f"sha256:{_sha256_bytes(data.encode('utf-8'))}"}


def _append_ndjson(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def _log_run(project_id: str, stage: str, artifacts: List[Dict[str, str]], extra: Optional[Dict[str, Any]] = None) -> None:
    entry = {
        "ts": _now_iso(),
        "stage": stage,
        **(extra or {}),
        "artifacts": artifacts,
    }
    rr = _proj_dir(project_id, "manifests", "run_records.jsonl")
    _append_ndjson(rr, entry)


async def _get_locked_config(project_id: str) -> Dict[str, Any]:
    try:
        from db.core import get_pool
        pool = await get_pool()
        if pool is None:
            return {}
        from db.core import fetchrow as db_fetchrow
        row = await db_fetchrow("SELECT config_json FROM film_project WHERE project_uid=$1", project_id)
        if row and isinstance(row.get("config_json"), dict):
            return dict(row.get("config_json"))
    except Exception:
        pass
    # Try reading plan.json as fallback
    try:
        plan = os.path.join(_proj_dir(project_id), "plan.json")
        if os.path.exists(plan):
            with open(plan, "r", encoding="utf-8") as f:
                _ = json.load(f)
                # Not authoritative for lock; return empty
    except Exception:
        pass
    return {}


async def _get_project_row(project_id: str) -> Optional[Dict[str, Any]]:
    try:
        from db.core import get_pool, fetchrow as db_fetchrow
        pool = await get_pool()
        if pool is None:
            return None
        row = await db_fetchrow("SELECT id FROM film_project WHERE project_uid=$1", project_id)
        if row and (row.get("id") is not None):
            return {"id": int(row.get("id"))}
    except Exception:
        return None
    return None


async def _get_manifest(project_id: str, kind: str) -> Optional[Any]:
    # Prefer DB
    try:
        from db.core import get_pool, fetchrow as db_fetchrow
        pool = await get_pool()
        if pool is not None:
            prow = await _get_project_row(project_id)
            if prow:
                row = await db_fetchrow("SELECT json FROM film_manifest WHERE project_id=$1 AND kind=$2 ORDER BY id DESC LIMIT 1", int(prow["id"]), kind)
                if row and (row.get("json") is not None):
                    return row.get("json")
    except Exception:
        pass
    # Fallback to JSON files under uploads
    try:
        name = {
            "plan": "plan.json",
            "scenes": "scenes.json",
            "characters": "characters.json",
            "shots": "shots.jsonl",
            "export": "export.json",
        }.get(kind)
        if not name:
            return None
        path = _proj_dir(project_id, name)
        if not os.path.exists(path):
            return None
        if name.endswith(".jsonl"):
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        try:
                            items.append(json.loads(ln))
                        except Exception:
                            continue
            return items
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -------------------- API --------------------


@app.post("/film/plan")
async def film_plan(body: Dict[str, Any]):
    seed = int(body.get("seed", 0) or 0)
    title = str(body.get("title", "Untitled"))
    logline = str(body.get("logline", ""))
    duration_s = float(body.get("duration_s", 60) or 60)
    outputs = body.get("outputs") or {"fps": 24, "resolution": "1024x576"}
    if not isinstance(outputs, dict):
        outputs = {"fps": 24, "resolution": "1024x576"}
    style_refs = body.get("style_refs") or []
    character_images = body.get("character_images") or []
    ideas = body.get("ideas") or []
    # Project id: prefer provided, else deterministic from seed/title/logline/duration
    provided_pid = str(body.get("project_id") or "").strip()
    project_id = provided_pid if provided_pid else _det_id("film", str(seed), title, logline, str(duration_s))
    # Deterministic beats and scenes (unless already locked)
    cfg = await _get_locked_config(project_id)
    locked = bool(cfg.get("locked", False)) if isinstance(cfg, dict) else False
    if locked:
        plan_manifest = await _get_manifest(project_id, "plan") or {}
        beats = plan_manifest.get("beats") or []
        scenes = (await _get_manifest(project_id, "scenes")) or []
        # Apply locked duration_s to scenes if present
        try:
            if isinstance(cfg.get("duration_s"), (int, float)) and scenes:
                total_beats = max(1, len(beats) or len(scenes))
                per_scene = max(1.0, float(cfg.get("duration_s")) / max(1, len(scenes)))
                for i, s in enumerate(scenes):
                    s["duration_s"] = round(per_scene, 2)
        except Exception:
            pass
        # Characters locked: load from manifest if present
        locked_chars = (await _get_manifest(project_id, "characters")) or []
        characters = locked_chars if isinstance(locked_chars, list) and locked_chars else characters
    else:
        words = [w for w in (title + " " + logline).split() if w]
        # Incorporate ideas into beat generation
        if isinstance(ideas, list):
            for it in ideas:
                try:
                    if isinstance(it, str):
                        words.extend([w for w in it.split() if w])
                    elif isinstance(it, dict):
                        txt = (it.get("text") or it.get("idea") or "")
                        words.extend([w for w in str(txt).split() if w])
                except Exception:
                    continue
        beats = []
        total_beats = max(3, min(8, len(words) // 3 or 3))
        for i in range(total_beats):
            b_id = f"B{i+1}"
            desc = " ".join(words[i::total_beats][:6]) or f"Beat {i+1}"
            beats.append({"id": b_id, "desc": desc, "weight": round(1.0 / total_beats, 2)})
        scenes = []
        per_scene = max(6.0, duration_s / max(3, total_beats))
        for i, b in enumerate(beats):
            scenes.append({
                "id": f"S{i+1}",
                "beat_id": b["id"],
                "location": "interior",
                "tod": "golden hour" if (i % 2 == 0) else "night",
                "mood": "dramatic" if (i % 2 == 0) else "tense",
                "duration_s": round(per_scene, 2),
            })
    characters = [{"id": "C_A", "name": "Protagonist", "tags": ["age:29", "hair:black", "jacket:leather"]}]
    assets = {"references": style_refs, "character_images": character_images}
    shot_budget = {"target_count": int(round(duration_s / 3.2))}
    # Persist JSON manifests
    _write_json(project_id, "plan.json", {"project_id": project_id, "seed": seed, "title": title, "logline": logline, "duration_s": duration_s, "outputs": outputs, "beats": beats, "style_refs": style_refs, "character_images": character_images, "ideas": ideas})
    _write_json(project_id, "scenes.json", scenes)
    # characters with bible
    chars_json = []
    for c in characters:
        bible = {"hair": "black", "jacket": "leather"} if c.get("id") == "C_A" else {}
        if character_images:
            bible = {**bible, "images": character_images}
        chars_json.append({**c, "bible": bible})
    _write_json(project_id, "characters.json", chars_json)
    # project catalog
    _append_ndjson(os.path.join(UPLOAD_ROOT, "projects.jsonl"), {"id": project_id, "title": title, "created_at": _now_iso(), "state": "planned"})
    _log_run(project_id, "plan", [{"type": "json", "uri": _proj_uri(project_id, "plan.json") }], {"seed": seed})

    # DB upserts: enforce lock-on-first-plan semantics
    try:
        pool = await get_pool()
        if pool is not None:
            # Read current config to decide lock behavior
            existing = await db_fetchrow("SELECT id, config_json FROM film_project WHERE project_uid=$1", project_id)
            locked = False
            if existing and isinstance(existing.get("config_json"), dict):
                locked = bool(existing.get("config_json", {}).get("locked", False))
                if locked:
                    # Use locked config values; ignore new inputs
                    ex_cfg = dict(existing.get("config_json"))
                    outputs = ex_cfg.get("outputs") or outputs
                    style_refs = ex_cfg.get("style_refs") or style_refs
                    character_images = ex_cfg.get("character_images") or character_images
                    try:
                        duration_s = float(ex_cfg.get("duration_s") or duration_s)
                    except Exception:
                        pass
            cfg = {"outputs": outputs, "style_refs": style_refs, "character_images": character_images, "duration_s": int(round(duration_s)), "locked": True}
            if not existing:
                await db_execute(
                    "INSERT INTO film_project(project_uid, seed, title, duration_s, config_json, state) VALUES($1,$2,$3,$4,$5,'planning')",
                    project_id, int(seed), title, int(round(duration_s)), cfg,
                )
            else:
                # Update config to locked values, keep state
                pid_existing = int(existing["id"])
                await db_execute("UPDATE film_project SET title=$1, duration_s=$2, config_json=$3 WHERE id=$4", title, int(round(duration_s)), cfg, pid_existing)
            row = await db_fetchrow("SELECT id FROM film_project WHERE project_uid=$1", project_id)
            if row:
                pid = int(row[0])
                await db_execute("INSERT INTO film_manifest(project_id, kind, json) VALUES($1,$2,$3)", pid, "plan", {"title": title, "logline": logline, "duration_s": duration_s, "outputs": outputs, "beats": beats, "style_refs": style_refs, "character_images": character_images, "ideas": ideas})
                await db_execute("INSERT INTO film_manifest(project_id, kind, json) VALUES($1,$2,$3)", pid, "scenes", scenes)
                await db_execute("INSERT INTO film_manifest(project_id, kind, json) VALUES($1,$2,$3)", pid, "characters", chars_json)
                for sc in scenes:
                    await db_execute("INSERT INTO film_scene(project_id, scene_uid, meta_json) VALUES($1,$2,$3) ON CONFLICT (project_id, scene_uid) DO NOTHING", pid, sc.get("id"), sc)
    except Exception:
        pass

    return {
        "project_id": project_id,
        "seed": seed,
        "beats": beats,
        "scenes": scenes,
        "characters": characters,
        "assets": assets,
        "shot_budget": shot_budget,
        "outputs": outputs,
    }


@app.post("/film/breakdown")
async def film_breakdown(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    seed = int(body.get("seed", 0) or 0)
    rules = body.get("rules") or {}
    avg_len = float(rules.get("avg_shot_len_s", 3.2) or 3.2)
    max_shots = int(rules.get("max_shots", 24) or 24)
    if not project_id:
        return _response_error(400, "missing project_id")
    # Respect locked statics if present
    cfg = await _get_locked_config(project_id)
    target_duration = None
    try:
        target_duration = float(cfg.get("duration_s")) if isinstance(cfg.get("duration_s"), (int, float)) else None
    except Exception:
        target_duration = None
    # Deterministic shots for a single scene S1 as baseline
    shots: List[Dict[str, Any]] = []
    count = max(1, min(max_shots, 6))
    for i in range(count):
        sh_id = f"S1_SH{i+1:02d}"
        sh_seed = int(_sha256_str(f"{project_id}|{seed}|{sh_id}")[:8], 16) % (2**31)
        dsl = {
            "camera": {"type": "dolly", "lens_mm": 35, "fstop": 4, "shutter": "180d", "move": "in 1.2m"},
            "framing": "MS",
            "angle": "eye",
            "fps": 24,
            "duration_s": round(avg_len, 2),
            "lighting": "warm backlight, soft key",
            "palette": "amber/teal",
            "action": "C_A enters left; rain streaks; neon reflections",
            "env": "alley wet asphalt; signage kanji neon",
            "style": "neo-noir minimal grain",
            "neg": "no flicker, no warp faces, no extra limbs",
        }
        shots.append({"id": sh_id, "scene_id": "S1", "seed": sh_seed, "dsl": dsl})
    # Normalize durations to match target_duration if locked
    if target_duration:
        ssum = sum([float(s.get("dsl", {}).get("duration_s") or avg_len) for s in shots])
        scale = (target_duration / ssum) if ssum > 0 else 1.0
        acc = 0.0
        for idx, s in enumerate(shots):
            base = float(s.get("dsl", {}).get("duration_s") or avg_len)
            if idx < len(shots) - 1:
                nd = round(base * scale, 2)
                s["dsl"]["duration_s"] = nd
                acc += nd
            else:
                # Adjust last shot to exact target
                last = round(max(0.1, target_duration - acc), 2)
                s["dsl"]["duration_s"] = last
    # Append NDJSON shots
    shots_path = _proj_dir(project_id, "shots.jsonl")
    for sh in shots:
        _append_ndjson(shots_path, sh)
    _log_run(project_id, "breakdown", [{"type": "ndjson", "uri": _proj_uri(project_id, "shots.jsonl")}], {"seed": seed})
    # DB: shots manifest + rows
    try:
        pool = await get_pool()
        if pool is not None:
            row = await db_fetchrow("SELECT id FROM film_project WHERE project_uid=$1", project_id)
            if row:
                pid = int(row[0])
                await db_execute("INSERT INTO film_manifest(project_id, kind, json) VALUES($1,$2,$3)", pid, "shots", shots)
                for sh in shots:
                    seeds_json = {"shot": sh.get("seed"), "tools": {}}
                    await db_execute(
                        "INSERT INTO film_shot(project_id, shot_uid, scene_uid, dsl_json, seeds_json, status) VALUES($1,$2,$3,$4,$5,'planned') ON CONFLICT (project_id, shot_uid) DO NOTHING",
                        pid, sh.get("id"), sh.get("scene_id"), sh.get("dsl", {}), seeds_json,
                    )
    except Exception:
        pass
    return {"shots": shots}


@app.post("/film/storyboard")
async def film_storyboard(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    shots = body.get("shots") or []
    seed = int(body.get("seed", 0) or 0)
    if not project_id or not shots:
        return _response_error(400, "missing project_id or shots")
    storyboards: List[Dict[str, Any]] = []
    for sh_id in shots:
        # Deterministic placeholder images (bytes from hash)
        head = _sha256_str(f"{project_id}|{seed}|{sh_id}|kf0").encode("utf-8")[:64]
        tail = _sha256_str(f"{project_id}|{seed}|{sh_id}|kf1").encode("utf-8")[:64]
        a = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/shots/{sh_id}/kf_000.png", head)
        b = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/shots/{sh_id}/kf_100.png", tail)
        storyboards.append({
            "shot_id": sh_id,
            "keyframes": [
                {"t": 0.0, "img_uri": a["uri"], "hash": a["hash"]},
                {"t": 3.0, "img_uri": b["uri"], "hash": b["hash"]},
            ],
            "notes": ""
        })
    # Write per-shot storyboard.json
    for sb in storyboards:
        sh_id = sb["shot_id"]
        _write_json(project_id, os.path.join("shots", sh_id, "storyboard.json"), sb)
    _log_run(project_id, "storyboard", [{"type": "json", "uri": _proj_uri(project_id, "shots", shots[0], "storyboard.json")}], {"seed": seed})
    return {"storyboards": storyboards}


@app.post("/film/animatic")
async def film_animatic(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    shots = body.get("shots") or []
    seed = int(body.get("seed", 0) or 0)
    # Enforce locked outputs if present
    cfg = await _get_locked_config(project_id)
    outputs = (cfg.get("outputs") if isinstance(cfg.get("outputs"), dict) else None) or body.get("outputs") or {"fps": 12, "res": "512x288"}
    if not project_id or not shots:
        return _response_error(400, "missing project_id or shots")
    animatics: List[Dict[str, Any]] = []
    for sh_id in shots:
        data = _sha256_str(f"{project_id}|{seed}|{sh_id}|animatic").encode("utf-8")
        asset = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/shots/{sh_id}/animatic.mp4", data)
        obj = {"shot_id": sh_id, "mp4_uri": asset["uri"], "hash": asset["hash"], "fps": outputs.get("fps", 12), "res": outputs.get("res", "512x288")}
        animatics.append(obj)
        _write_json(project_id, os.path.join("shots", sh_id, "animatic.json"), obj)
    _log_run(project_id, "animatic", [{"type": "json", "uri": _proj_uri(project_id, "shots", shots[0], "animatic.json")}], {"seed": seed})
    return {"animatics": animatics}


@app.post("/film/final")
async def film_final(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    shots = body.get("shots") or []
    seed = int(body.get("seed", 0) or 0)
    # Enforce locked outputs if present
    cfg = await _get_locked_config(project_id)
    outputs = (cfg.get("outputs") if isinstance(cfg.get("outputs"), dict) else None) or body.get("outputs") or {"fps": 24, "res": "1024x576", "codec": "h264"}
    post = body.get("post") or {}
    if not project_id or not shots:
        return _response_error(400, "missing project_id or shots")
    final_shots: List[Dict[str, Any]] = []
    # Simple cache (content addressing on inputs)
    cache_path = _proj_dir(project_id, "manifests", "cache.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    cache.setdefault("by_input", {})
    cache_hits = False
    for sh_id in shots:
        input_sig = json.dumps({"project_id": project_id, "shot_id": sh_id, "outputs": outputs, "post": post, "seed": seed}, sort_keys=True)
        input_hash = f"sha256:{_sha256_bytes(input_sig.encode('utf-8'))}"
        cached = cache["by_input"].get(input_hash)
        if cached:
            cache_hits = True
            final_shots.append({"shot_id": sh_id, "mp4_uri": cached.get("artifact"), "hash": cached.get("hash"), "cache": "hit"})
            continue
        data = _sha256_str(f"{project_id}|{seed}|{sh_id}|final").encode("utf-8")
        asset = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/shots/{sh_id}/final.mp4", data)
        final_obj = {"shot_id": sh_id, "mp4_uri": asset["uri"], "hash": asset["hash"], "outputs": outputs, "post": post}
        final_shots.append(final_obj)
        _write_json(project_id, os.path.join("shots", sh_id, "final.json"), final_obj)
        cache["by_input"][input_hash] = {"artifact": asset["uri"], "hash": asset["hash"], "stage": "final", "shot_id": sh_id}
    # Persist cache
    _write_json(project_id, os.path.join("manifests", "cache.json"), cache)
    # Deterministic EDL/assembly honoring locked duration and shot durations
    # Load shot durations from shots.jsonl
    durations_map: Dict[str, float] = {}
    try:
        sp = _proj_dir(project_id, "shots.jsonl")
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        try:
                            obj = json.loads(ln)
                            sid = obj.get("id")
                            d = float(((obj.get("dsl") or {}).get("duration_s") or 3.0))
                            if sid:
                                durations_map[sid] = d
                        except Exception:
                            continue
    except Exception:
        pass
    # If a total target is locked, scale shot durations deterministically
    target_total = None
    try:
        target_total = float(cfg.get("duration_s")) if isinstance(cfg.get("duration_s"), (int, float)) else None
    except Exception:
        target_total = None
    durations: List[float] = [float(durations_map.get(s, 3.0)) for s in shots]
    if target_total and durations:
        ssum = sum(durations)
        if ssum > 0:
            scale = target_total / ssum
            durations = [round(d * scale, 2) for d in durations]
            # fix last to exact
            acc = sum(durations[:-1])
            durations[-1] = round(max(0.1, target_total - acc), 2)
    t = 0.0
    video_tracks = []
    for sid, d in zip(shots, durations):
        video_tracks.append({"shot_id": sid, "mp4": _proj_uri(project_id, "shots", sid, "final.mp4"), "in": 0.0, "out": float(round(d,2)), "at": float(round(t,2))})
        t += float(round(d,2))
    edl_obj = {"edl_version": "1.0", "project_id": project_id, "fps": outputs.get("fps", 24), "resolution": outputs.get("res", "1024x576"), "audio_sr": 48000, "tracks": {"video": video_tracks}}
    edl_meta = _write_json(project_id, "edl.json", edl_obj)
    # Nodes manifest: capture per-shot seed and a graph hash for determinism
    nodes_obj: Dict[str, Any] = {"project_id": project_id, "shots": []}
    try:
        sp = _proj_dir(project_id, "shots.jsonl")
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        try:
                            obj = json.loads(ln)
                            sid = obj.get("id")
                            sseed = obj.get("seed")
                            if sid is not None:
                                nodes_obj["shots"].append({"shot_id": sid, "seed": sseed})
                        except Exception:
                            continue
    except Exception:
        pass
    # Record post pipeline seeds (deterministic) and params if present
    post_nodes: Dict[str, Any] = {}
    try:
        # Persist post info based on export effective or last known post params
        eff = (effective if isinstance(locals().get("effective"), dict) else {}) if "effective" in locals() else {}
        post_eff = eff.get("post") if isinstance(eff.get("post"), dict) else {}
        # Derive stable seeds
        if post_eff.get("interpolate"):
            post_nodes["interpolate"] = {"seed": _seed64("post", project_id, "interpolate"), **post_eff.get("interpolate")}
        if post_eff.get("upscale"):
            post_nodes["upscale"] = {"seed": _seed64("post", project_id, "upscale"), **post_eff.get("upscale")}
    except Exception:
        post_nodes = {}
    if post_nodes:
        nodes_obj["post"] = post_nodes
    try:
        import hashlib as _hl
        ghash = _hl.sha256(json.dumps(nodes_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        nodes_obj["graph_hash"] = f"sha256:{ghash}"
    except Exception:
        nodes_obj["graph_hash"] = None
    nodes_meta = _write_json(project_id, "nodes.json", nodes_obj)
    reel = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/reel_v0.mp4", _sha256_str(f"{project_id}|{seed}|reel").encode("utf-8"))
    _log_run(project_id, "final", [{"type": "json", "uri": edl_meta["uri"]}, {"type": "json", "uri": nodes_meta["uri"]}, {"type": "mp4", "uri": reel["uri"], "hash": reel["hash"]}], {"seed": seed})
    # Optionally persist nodes manifest to DB
    try:
        pool = await get_pool()
        if pool is not None:
            row = await db_fetchrow("SELECT id FROM film_project WHERE project_uid=$1", project_id)
            if row:
                pid = int(row[0])
                await db_execute("INSERT INTO film_manifest(project_id, kind, json) VALUES($1,$2,$3)", pid, "nodes", nodes_obj)
    except Exception:
        pass
    return {"final_shots": final_shots, "assembly": {"timeline_json": edl_meta["uri"], "nodes_json": nodes_meta["uri"], "reel_mp4": reel["uri"], "hash": reel["hash"], **({"cache": "hit"} if cache_hits else {})}}


@app.post("/film/voice")
async def film_voice(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    seed = int(body.get("seed", 0) or 0)
    cast = body.get("cast") or []
    script = body.get("script") or []
    outputs = body.get("outputs") or {"format": "wav", "sr": 48000}
    if not project_id:
        return _response_error(400, "missing project_id")
    tracks: List[Dict[str, Any]] = []
    for line in script:
        shot_id = line.get("shot_id") or "S1_SH01"
        char_id = line.get("char_id") or "C_A"
        text = line.get("text") or ""
        data = _sha256_str(f"{project_id}|{seed}|{shot_id}|{char_id}|{text}").encode("utf-8")
        asset = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/shots/{shot_id}/{char_id}.wav", data)
        tracks.append({"shot_id": shot_id, "char_id": char_id, "wav_uri": asset["uri"], "hash": asset["hash"]})
    # Per-shot audio.json (append or overwrite with current set)
    for t in tracks:
        sh = t["shot_id"]
        existing = {"dialog": [], "music": [], "sfx": []}
        _write_json(project_id, os.path.join("shots", sh, "audio.json"), {"dialog": [x for x in tracks if x["shot_id"] == sh], "music": [], "sfx": []})
    _log_run(project_id, "voice", [{"type": "json", "uri": _proj_uri(project_id, "shots", tracks[0]["shot_id"], "audio.json")}], {"seed": seed})
    return {"voice_tracks": tracks}


@app.post("/film/music")
async def film_music(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    seed = int(body.get("seed", 0) or 0)
    music = body.get("music") or {}
    sfx = body.get("sfx") or []
    if not project_id:
        return _response_error(400, "missing project_id")
    stems_out: List[Dict[str, Any]] = []
    for name in (music.get("stems") or ["drums", "bass", "pad", "lead"]):
        data = _sha256_str(f"{project_id}|{seed}|stem|{name}").encode("utf-8")
        asset = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/stems/{name}.wav", data)
        stems_out.append({"name": name, "wav_uri": asset["uri"], "hash": asset["hash"]})
    sfx_out: List[Dict[str, Any]] = []
    for ev in sfx:
        name = ev.get("event") or "sfx"
        data = _sha256_str(f"{project_id}|{seed}|sfx|{name}").encode("utf-8")
        asset = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/sfx/{name}.wav", data)
        sfx_out.append({"event": name, "wav_uri": asset["uri"], "hash": asset["hash"]})
    _log_run(project_id, "music", [{"type": "json", "uri": _proj_uri(project_id, "stems") }], {"seed": seed})
    return {"stems": stems_out, "sfx": sfx_out}


@app.post("/film/qc")
async def film_qc(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    checks = body.get("checks") or []
    if not project_id:
        return _response_error(400, "missing project_id")
    # Deterministic dummy QC
    qc_report = {
        "character": {"C_A": {"match": 0.92, "notes": "ok"}},
        "color": {"deltaE_avg": 1.8},
        "audio_sync": {"ms_offset": 12},
        "artifacts": {"flicker": ["S1_SH02@t=1.2"], "warps": []},
    }
    suggested = [{"shot_id": "S1_SH02", "action": "re-render partial", "reason": "flicker", "seed_adj": "+3"}]
    _write_json(project_id, "qc_report.json", {**qc_report, "suggested_fixes": suggested})
    _log_run(project_id, "qc", [{"type": "json", "uri": _proj_uri(project_id, "qc_report.json")}])
    try:
        pool = await get_pool()
        if pool is not None:
            row = await db_fetchrow("SELECT id FROM film_project WHERE project_uid=$1", project_id)
            if row:
                pid = int(row[0])
                await db_execute("INSERT INTO film_qc(project_id, report_json) VALUES($1,$2)", pid, {**qc_report, "suggested_fixes": suggested})
                await db_execute("UPDATE film_project SET state='qc' WHERE id=$1", pid)
    except Exception:
        pass
    return {"qc_report": qc_report, "suggested_fixes": suggested}


@app.post("/film/export")
async def film_export(body: Dict[str, Any]):
    project_id = str(body.get("project_id") or "")
    if not project_id:
        return _response_error(400, "missing project_id")
    # Master
    master = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/master_v1.mp4", _sha256_str(f"{project_id}|master").encode("utf-8"))
    # Subtitles (simple SRT deterministic content)
    srt = "1\n00:00:00,000 --> 00:00:02,000\nHello world.\n\n"
    subs = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/subtitles.srt", srt.encode("utf-8"))
    # Package
    pkg = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/delivery.zip", _sha256_str(f"{project_id}|package").encode("utf-8"))
    # Capture requested/effective post pipeline if provided
    requested = body.get("requested") or {}
    effective = body.get("effective") or {}
    # Try to attach EDL/QC URIs and hashes if present
    edl_uri = _proj_uri(project_id, "edl.json")
    nodes_uri = _proj_uri(project_id, "nodes.json")
    qc_uri = _proj_uri(project_id, "qc_report.json")
    edl_hash = None
    qc_hash = None
    nodes_hash = None
    try:
        ep = _proj_dir(project_id, "edl.json")
        if os.path.exists(ep):
            with open(ep, "rb") as f:
                edl_hash = f"sha256:{_sha256_bytes(f.read())}"
    except Exception:
        edl_hash = None
    try:
        np = _proj_dir(project_id, "nodes.json")
        if os.path.exists(np):
            with open(np, "rb") as f:
                nodes_hash = f"sha256:{_sha256_bytes(f.read())}"
    except Exception:
        nodes_hash = None
    try:
        qp = _proj_dir(project_id, "qc_report.json")
        if os.path.exists(qp):
            with open(qp, "rb") as f:
                qc_hash = f"sha256:{_sha256_bytes(f.read())}"
    except Exception:
        qc_hash = None
    export = {
        "master_uri": master["uri"],
        "subs_uri": subs["uri"],
        "package_uri": pkg["uri"],
        "hash": master["hash"],
        "requested": requested,
        "effective": effective,
        "edl": {"uri": edl_uri, "hash": edl_hash},
        "nodes": {"uri": nodes_uri, "hash": nodes_hash},
        "qc_report": {"uri": qc_uri, "hash": qc_hash},
    }
    _write_json(project_id, "export.json", export)
    _log_run(project_id, "export", [{"type": "json", "uri": _proj_uri(project_id, "export.json")}])
    try:
        pool = await get_pool()
        if pool is not None:
            row = await db_fetchrow("SELECT id FROM film_project WHERE project_uid=$1", project_id)
            if row:
                pid = int(row[0])
                await db_execute("INSERT INTO film_manifest(project_id, kind, json) VALUES($1,$2,$3)", pid, "export", export)
                await db_execute("UPDATE film_project SET state='exported' WHERE id=$1", pid)
    except Exception:
        pass
    return export


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


