from __future__ import annotations

from typing import Dict, Any, Tuple
import os
from .registry import create_ref, refine_ref, list_refs, load_manifest
from .embeds import compute_face_embeddings, compute_voice_embedding, compute_music_embedding


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
        except Exception:
            pass
    return {"ok": True, "ref": ref}


def post_refs_refine(body: Dict[str, Any]):
    files_delta = body.get("files_delta")
    kind = None
    try:
        man = load_manifest(body.get("parent_id"))
        kind = (man or {}).get("kind")
    except Exception:
        pass
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
        except Exception:
            pass
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


