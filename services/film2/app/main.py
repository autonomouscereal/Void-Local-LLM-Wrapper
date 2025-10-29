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
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
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
    # Deterministic project id
    project_id = _det_id("film", str(seed), title, logline, str(duration_s))
    # Deterministic beats and scenes
    words = [w for w in (title + " " + logline).split() if w]
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
    assets = {"references": body.get("style_refs") or []}
    shot_budget = {"target_count": int(round(duration_s / 3.2))}
    # Persist JSON manifests
    _write_json(project_id, "plan.json", {"project_id": project_id, "seed": seed, "title": title, "logline": logline, "duration_s": duration_s, "outputs": outputs})
    _write_json(project_id, "scenes.json", scenes)
    # characters with bible
    chars_json = []
    for c in characters:
        bible = {"hair": "black", "jacket": "leather"} if c.get("id") == "C_A" else {}
        chars_json.append({**c, "bible": bible})
    _write_json(project_id, "characters.json", chars_json)
    # project catalog
    _append_ndjson(os.path.join(UPLOAD_ROOT, "projects.jsonl"), {"id": project_id, "title": title, "created_at": _now_iso(), "state": "planned"})
    _log_run(project_id, "plan", [{"type": "json", "uri": _proj_uri(project_id, "plan.json") }], {"seed": seed})

    # DB upserts: project, manifests, scenes
    try:
        pool = await get_pool()
        if pool is not None:
            cfg = {"outputs": outputs}
            await db_execute(
                "INSERT INTO film_project(project_uid, seed, title, duration_s, config_json, state) VALUES($1,$2,$3,$4,$5,'planning') "
                "ON CONFLICT (project_uid) DO UPDATE SET title=EXCLUDED.title, duration_s=EXCLUDED.duration_s, config_json=EXCLUDED.config_json",
                project_id, int(seed), title, int(round(duration_s)), cfg,
            )
            row = await db_fetchrow("SELECT id FROM film_project WHERE project_uid=$1", project_id)
            if row:
                pid = int(row[0])
                await db_execute("INSERT INTO film_manifest(project_id, kind, json) VALUES($1,$2,$3)", pid, "plan", {"title": title, "duration_s": duration_s, "outputs": outputs})
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
    outputs = body.get("outputs") or {"fps": 12, "res": "512x288"}
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
    outputs = body.get("outputs") or {"fps": 24, "res": "1024x576", "codec": "h264"}
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
    # Deterministic EDL/assembly
    edl_obj = {"edl_version": "1.0", "project_id": project_id, "fps": outputs.get("fps", 24), "resolution": outputs.get("res", "1024x576"), "audio_sr": 48000, "tracks": {"video": [{"shot_id": s, "mp4": _proj_uri(project_id, "shots", s, "final.mp4"), "in": 0.0, "out": 3.0, "at": 0.0} for s in shots]}}
    edl_meta = _write_json(project_id, "edl.json", edl_obj)
    reel = _save_bytes(f"{PROJECTS_PREFIX}/{project_id}/reel_v0.mp4", _sha256_str(f"{project_id}|{seed}|reel").encode("utf-8"))
    _log_run(project_id, "final", [{"type": "json", "uri": edl_meta["uri"]}, {"type": "mp4", "uri": reel["uri"], "hash": reel["hash"]}], {"seed": seed})
    return {"final_shots": final_shots, "assembly": {"timeline_json": edl_meta["uri"], "reel_mp4": reel["uri"], "hash": reel["hash"], **({"cache": "hit"} if cache_hits else {})}}


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
    export = {"master_uri": master["uri"], "subs_uri": subs["uri"], "package_uri": pkg["uri"], "hash": master["hash"]}
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


