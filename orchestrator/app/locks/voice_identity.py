from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import logging
import math
import os

from ..ref_library.storage import load_manifest
from ..ref_library.registry import create_ref, refine_ref, list_refs
from ..ref_library.embeds import compute_voice_embedding, maybe_load
from .builder import _blocking_voice_embedding

log = logging.getLogger(__name__)


def resolve_voice_lock(voice_id: Optional[str], inline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Backwards-compatible lock resolver used by existing callers.
    Prefer resolve_voice_identity when you also need a canonical voice_id.
    """
    if inline and isinstance(inline, dict):
        return inline
    if voice_id:
        man = load_manifest(voice_id)
        if man and man.get("kind") == "voice":
            return {"voice_samples": [f.get("path") for f in man.get("files", {}).get("voice_samples", [])]}
        return {"voice_id": voice_id}
    return {}


def _cosine(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    num = sum(float(x) * float(y) for x, y in zip(a, b))
    den_a = math.sqrt(sum(float(x) * float(x) for x in a))
    den_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if den_a == 0.0 or den_b == 0.0:
        return None
    return num / (den_a * den_b)


def _aggregate_embed(items: List[Dict[str, Any]]) -> Optional[List[float]]:
    vecs: List[List[float]] = []
    for it in items or []:
        v = it.get("vec")
        if isinstance(v, list) and v:
            vecs.append([float(x) for x in v])
    if not vecs:
        return None
    dim = len(vecs[0])
    acc = [0.0] * dim
    count = 0
    for v in vecs:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            acc[i] += float(x)
        count += 1
    if count == 0:
        return None
    return [x / float(count) for x in acc]


def _ensure_manifest_for_voice(voice_id: str, samples: List[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Ensure a ref_library manifest exists for the given voice_id and that all sample
    paths are attached. Returns (manifest, newly_added_samples).
    """
    if not isinstance(voice_id, str) or not voice_id.strip():
        return None, []
    vid = voice_id.strip()
    man = load_manifest(vid)
    new_samples: List[str] = []
    if not man:
        files = {"voice_samples": [p for p in (samples or []) if isinstance(p, str) and p]}
        man = create_ref("voice", vid, files, meta={"source": "tts.autocreate"})
        new_samples = list(files["voice_samples"])
        return man, new_samples
    # Attach any new samples via a refine manifest so provenance is preserved.
    existing_paths = {
        f.get("path")
        for f in (man.get("files", {}) or {}).get("voice_samples", []) or []
        if isinstance(f, dict)
    }
    add_paths = [p for p in (samples or []) if isinstance(p, str) and p and p not in existing_paths]
    if add_paths:
        man = refine_ref(
            man.get("ref_id") or vid,
            f"{man.get('title') or vid}:+samples",
            {"voice_samples": [{"path": p} for p in add_paths]},
            None,
        )
        new_samples = list(add_paths)
    return man, new_samples


def _match_existing_voice(sample_paths: List[str]) -> Optional[str]:
    """
    Best-effort matching of new voice samples against existing voice refs
    using the lightweight MFCC-based embeddings on disk.
    """
    vecs: List[List[float]] = []
    for p in sample_paths or []:
        try:
            if isinstance(p, str) and p:
                v = _blocking_voice_embedding(p)
            else:
                v = None
        except Exception:
            v = None
        if isinstance(v, list) and v:
            vecs.append([float(x) for x in v])
    if not vecs:
        return None
    dim = len(vecs[0])
    acc = [0.0] * dim
    count = 0
    for v in vecs:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            acc[i] += float(x)
        count += 1
    if count == 0:
        return None
    query = [x / float(count) for x in acc]

    best_id: Optional[str] = None
    best_score: float = -1.0
    threshold_raw = os.getenv("VOICE_MATCH_THRESHOLD", "0.80")
    try:
        threshold = float(str(threshold_raw).strip() or "0.80")
    except Exception as exc:
        log.warning("voice.match: bad VOICE_MATCH_THRESHOLD=%r; defaulting to 0.80", threshold_raw, exc_info=True)
        threshold = 0.80

    for info in list_refs("voice"):
        rid = info.get("ref_id")
        if not isinstance(rid, str) or not rid:
            continue
        emb = maybe_load(rid, "voice_embed") or {}
        items = emb.get("items") if isinstance(emb.get("items"), list) else []
        if not items or not emb.get("has_vectors"):
            # Try to compute embeddings lazily if they are missing.
            try:
                man = load_manifest(rid)
            except Exception:
                man = None
            if man and man.get("kind") == "voice":
                paths = [f.get("path") for f in (man.get("files", {}) or {}).get("voice_samples", []) or []]
                compute_voice_embedding(rid, [p for p in paths if isinstance(p, str) and p])
                emb = maybe_load(rid, "voice_embed") or {}
                items = emb.get("items") if isinstance(emb.get("items"), list) else []
        agg = _aggregate_embed(items or [])
        if not isinstance(agg, list):
            continue
        sim = _cosine(query, agg)
        if sim is None:
            continue
        if sim > best_score:
            best_score = sim
            best_id = rid

    if best_id and best_score >= threshold:
        try:
            log.info("voice.match: matched samples to voice_id=%s score=%.3f", best_id, best_score)
        except Exception:
            log.debug("voice.match: failed to log match result (non-fatal)", exc_info=True)
        return best_id
    return None


def resolve_voice_identity(
    voice_id: Optional[str],
    inline: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    """
    Resolve a canonical voice_id and corresponding lock given optional
    voice_id and inline refs.

    Returns (canonical_voice_id, voice_lock, meta), where meta may contain:
      - "new_samples": [str] newly attached sample paths for this voice_id.
      - "is_new_voice": bool indicating a new voice manifest was created.
    """
    inline = inline if isinstance(inline, dict) else None
    samples: List[str] = []
    if inline:
        vs = inline.get("voice_samples")
        if isinstance(vs, list):
            samples = [p for p in vs if isinstance(p, str) and p]
    meta: Dict[str, Any] = {"new_samples": [], "is_new_voice": False}

    # Case 1: both voice_id and inline samples -> treat as "update this voice".
    if voice_id and samples:
        man, new_samples = _ensure_manifest_for_voice(str(voice_id), samples)
        if man:
            try:
                paths = [f.get("path") for f in (man.get("files", {}) or {}).get("voice_samples", []) or []]
                compute_voice_embedding(man.get("ref_id") or str(voice_id), [p for p in paths if isinstance(p, str) and p])
            except Exception as exc:
                log.warning("resolve_voice_identity: compute_voice_embedding failed voice_id=%s: %s", voice_id, exc, exc_info=True)
        meta["new_samples"] = new_samples
        meta["is_new_voice"] = bool(man and man.get("ref_id") and man.get("ref_id") != voice_id)
        # Inline lock wins to preserve exact per-call sample set.
        return str(voice_id), (inline or {}), meta

    # Case 2: voice_id only -> use existing manifest if present.
    if voice_id and not samples:
        lock = resolve_voice_lock(voice_id, None)
        return str(voice_id), lock, meta

    # Case 3: samples only -> match against existing voices or create new.
    if samples and not voice_id:
        matched = _match_existing_voice(samples)
        is_new = False
        canonical_id: Optional[str]
        if matched:
            canonical_id = matched
            man, new_samples = _ensure_manifest_for_voice(matched, samples)
            meta["new_samples"] = new_samples
        else:
            man = create_ref("voice", "voice:auto", {"voice_samples": samples}, meta={"source": "tts.autocreate"})
            canonical_id = man.get("ref_id") if isinstance(man, dict) else None
            is_new = True
            try:
                paths = [f.get("path") for f in (man.get("files", {}) or {}).get("voice_samples", []) or []] if man else []
                compute_voice_embedding(canonical_id or "", [p for p in paths if isinstance(p, str) and p])
            except Exception as exc:
                log.warning("resolve_voice_identity: compute_voice_embedding failed new_voice_id=%s: %s", canonical_id, exc, exc_info=True)
            meta["new_samples"] = list(samples)
        meta["is_new_voice"] = is_new
        # Lock is just the inline bundle (which contains the concrete sample paths).
        return canonical_id, (inline or {"voice_samples": samples}), meta

    # Case 4: neither voice_id nor samples -> no-op; caller will fall back
    # to lock-bundle/environment defaults.
    return None, {}, meta


