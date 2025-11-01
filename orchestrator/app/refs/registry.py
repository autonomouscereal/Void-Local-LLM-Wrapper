from __future__ import annotations

import os
import time
from typing import Dict, Any, List
from .storage import new_ref_id, save_manifest, load_manifest, ref_dir, _sha_file
from ..determinism.seeds import SEEDS


def create_ref(kind: str, title: str, files: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rid = new_ref_id(kind)
    m: Dict[str, Any] = {
        "ref_id": rid,
        "kind": kind,
        "title": title,
        "created_at": int(time.time()),
        "version": 1,
        "parent": None,
        "files": (files or {}),
        "meta": (meta or {}),
        "determinism": {"seed": SEEDS.get(f"ref:{rid}")},
        "provenance": {"used_by": []},
    }
    def fhash(p: str):
        try:
            return {"path": p, "sha256": _sha_file(p)}
        except Exception:
            return {"path": p, "sha256": None}
    if kind == "image":
        m.setdefault("files", {})
        m["files"]["images"] = [fhash(p) for p in (files or {}).get("images", [])]
    if kind == "voice":
        m.setdefault("files", {})
        m["files"]["voice_samples"] = [fhash(p) for p in (files or {}).get("voice_samples", [])]
    if kind == "music":
        m.setdefault("files", {})
        if (files or {}).get("track"):
            m["files"]["track"] = fhash(files["track"])  # type: ignore
        m["files"]["stems"] = [fhash(p) for p in (files or {}).get("stems", [])]
    save_manifest(rid, m)
    return m


def refine_ref(parent_id: str, title: str, files_delta: Dict[str, Any] | None = None, meta_delta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    parent = load_manifest(parent_id)
    assert parent, "parent not found"
    rid = new_ref_id(parent["kind"])  # type: ignore
    m = dict(parent)
    m.update({"ref_id": rid, "title": title, "created_at": int(time.time()), "parent": parent_id})
    if files_delta:
        for k, v in files_delta.items():
            m.setdefault("files", {})
            m["files"][k] = v
    if meta_delta:
        m.setdefault("meta", {})
        m["meta"].update(meta_delta)
    m["version"] = int(parent.get("version", 1)) + 1
    save_manifest(rid, m)
    return m


def list_refs(kind: str | None = None) -> List[Dict[str, Any]]:
    root = ref_dir("")[:-1]
    out: List[Dict[str, Any]] = []
    if not os.path.exists(root):
        return out
    for d in os.listdir(root):
        if not d:
            continue
        man = load_manifest(d)
        if not man:
            continue
        if kind and man.get("kind") != kind:
            continue
        out.append({"ref_id": man.get("ref_id"), "kind": man.get("kind"), "title": man.get("title"), "version": man.get("version")})
    return sorted(out, key=lambda x: str(x.get("ref_id", "")), reverse=True)


