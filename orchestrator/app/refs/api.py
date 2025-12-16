from __future__ import annotations

from typing import Dict, Any, Tuple, List
import os
import logging
from .registry import create_ref, refine_ref, list_refs, load_manifest
from .embeds import compute_face_embeddings, compute_voice_embedding, compute_music_embedding
from ..analysis.media import analyze_audio
from ..datasets.trace import append_sample as _trace_append

log = logging.getLogger(__name__)


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")


def _map_upload_path(p: str) -> str:
    if not isinstance(p, str) or not p:
        return p
    if p.startswith("/uploads/"):
        return os.path.join(UPLOAD_DIR, p.split("/")[-1])
    if p.startswith("http") and "/uploads/" in p:
        return os.path.join(UPLOAD_DIR, p.split("/")[-1])
    return p


def _normalize_files(kind: str, files: Dict[str, Any]) -> Dict[str, Any]:
    f2 = dict(files or {})
    if kind == "image":
        f2["images"] = [_map_upload_path(p) for p in (files or {}).get("images", [])]
    if kind == "voice":
        f2["voice_samples"] = [_map_upload_path(p) for p in (files or {}).get("voice_samples", [])]
    if kind == "music":
        if (files or {}).get("track"):
            f2["track"] = _map_upload_path(files["track"])  # type: ignore
        f2["stems"] = [_map_upload_path(p) for p in (files or {}).get("stems", [])]
    return f2


def post_refs_save(body: Dict[str, Any]):
    kind = body.get("kind")
    files = _normalize_files(kind, body.get("files", {}))
    ref = create_ref(kind, body.get("title", ""), files, meta=body.get("meta"))
    if body.get("compute_embeds"):
        try:
            if ref.get("kind") == "image":
                compute_face_embeddings(ref["ref_id"], [f.get("path") for f in ref.get("files", {}).get("images", [])])
            if ref.get("kind") == "voice":
                compute_voice_embedding(ref["ref_id"], [f.get("path") for f in ref.get("files", {}).get("voice_samples", [])])
            if ref.get("kind") == "music":
                tr = (ref.get("files", {}).get("track") or {}).get("path")
                st = [f.get("path") for f in ref.get("files", {}).get("stems", [])]
                compute_music_embedding(ref["ref_id"], tr, st)
        except Exception as ex:
            # Non-fatal: embed computation is best-effort
            log.warning("post_refs_save: compute_embeds failed for ref_id=%s kind=%s: %s", ref.get("ref_id"), ref.get("kind"), ex, exc_info=True)
    return {"ok": True, "ref": ref}


def post_refs_refine(body: Dict[str, Any]):
    files_delta = body.get("files_delta")
    kind = None
    try:
        man = load_manifest(body.get("parent_id"))
        kind = (man or {}).get("kind")
    except Exception as ex:
        log.warning("post_refs_refine: load_manifest failed for parent_id=%s: %s", body.get("parent_id"), ex, exc_info=True)
    if isinstance(files_delta, dict) and kind:
        files_delta = _normalize_files(kind, files_delta)
    ref = refine_ref(body.get("parent_id"), body.get("title", ""), files_delta, body.get("meta_delta"))
    if body.get("compute_embeds"):
        try:
            if ref.get("kind") == "image":
                compute_face_embeddings(ref["ref_id"], [f.get("path") for f in ref.get("files", {}).get("images", [])])
            if ref.get("kind") == "voice":
                compute_voice_embedding(ref["ref_id"], [f.get("path") for f in ref.get("files", {}).get("voice_samples", [])])
            if ref.get("kind") == "music":
                tr = (ref.get("files", {}).get("track") or {}).get("path")
                st = [f.get("path") for f in ref.get("files", {}).get("stems", [])]
                compute_music_embedding(ref["ref_id"], tr, st)
        except Exception as ex:
            # Non-fatal: embed computation is best-effort
            log.warning("post_refs_refine: compute_embeds failed for ref_id=%s kind=%s: %s", ref.get("ref_id"), ref.get("kind"), ex, exc_info=True)
    return {"ok": True, "ref": ref}


def get_refs_list(kind: str | None):
    return {"ok": True, "refs": list_refs(kind)}


def post_refs_apply(body: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    man = load_manifest(body.get("ref_id"))
    if not man:
        return ({"ok": False, "error": "not_found"}, 404)
    kind = man.get("kind")
    if kind == "image":
        return ({"ok": True, "ref_pack": {"images": [f.get("path") for f in man.get("files", {}).get("images", [])]}}, 200)
    if kind == "voice":
        return ({"ok": True, "ref_pack": {"voice_samples": [f.get("path") for f in man.get("files", {}).get("voice_samples", [])]}}, 200)
    if kind == "music":
        pack: Dict[str, Any] = {}
        tr = (man.get("files", {}).get("track") or {}).get("path")
        if tr:
            pack["track"] = tr
        pack["stems"] = [f.get("path") for f in man.get("files", {}).get("stems", [])]
        return ({"ok": True, "ref_pack": pack}, 200)
    return ({"ok": False, "error": "unsupported kind"}, 400)


def post_refs_music_profile(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a summary music profile from one or more music refs or raw audio paths.
    The profile is suitable for seeding the music.* lock bundle branch.

    Input:
      - ref_ids: [str] (optional)
      - audio_paths: [str] (optional, absolute or upload-relative paths)
    """
    ref_ids = body.get("ref_ids") if isinstance(body.get("ref_ids"), list) else []
    audio_paths = body.get("audio_paths") if isinstance(body.get("audio_paths"), list) else []
    tracks: List[str] = []
    for rid in ref_ids or []:
        try:
            man = load_manifest(rid)
        except Exception:
            man = None
        if not man:
            continue
        if (man.get("files", {}).get("track") or {}).get("path"):
            tracks.append((man.get("files", {}).get("track") or {}).get("path"))
        for f in man.get("files", {}).get("stems", []) or []:
            p = f.get("path")
            if isinstance(p, str) and p:
                tracks.append(p)
    for p in audio_paths or []:
        if isinstance(p, str) and p:
            tracks.append(_map_upload_path(p))
    metrics: List[Dict[str, Any]] = []
    for p in tracks:
        if not isinstance(p, str) or not p:
            continue
        try:
            m = analyze_audio(p)
        except Exception:
            m = {}
        if isinstance(m, dict):
            m["path"] = p
            metrics.append(m)
    if not metrics:
        return {"ok": False, "profile": {}, "reason": "no_valid_tracks"}
    tempos = [m.get("tempo_bpm") for m in metrics if isinstance(m.get("tempo_bpm"), (int, float))]
    keys = [m.get("key") for m in metrics if isinstance(m.get("key"), str) and m.get("key")]
    genres = [m.get("genre") for m in metrics if isinstance(m.get("genre"), str) and m.get("genre")]
    emotions = [m.get("emotion") for m in metrics if isinstance(m.get("emotion"), str) and m.get("emotion")]
    def _range(vals: List[float]) -> Tuple[float, float] | None:
        if not vals:
            return None
        return float(min(vals)), float(max(vals))
    bpm_range = _range([float(t) for t in tempos]) if tempos else None
    key_histogram: Dict[str, int] = {}
    for k in keys:
        key_histogram[k] = key_histogram.get(k, 0) + 1
    style_tags: List[str] = []
    # Coarse tags from genre/emotion
    for g in genres:
        if g not in style_tags:
            style_tags.append(g)
    for e in emotions:
        if e not in style_tags:
            style_tags.append(e)
    profile: Dict[str, Any] = {
        "style_tags": style_tags,
        "bpm_range": bpm_range,
        "keys": key_histogram,
        "tracks": metrics,
    }
    _trace_append(
        "music",
        {
            "event": "refs.music.build_profile",
            "ref_ids": ref_ids,
            "tracks_count": len(tracks),
            "bpm_range": bpm_range,
            "keys_histogram": key_histogram,
            "style_tags": style_tags,
        },
    )
    return {"ok": True, "profile": profile}


