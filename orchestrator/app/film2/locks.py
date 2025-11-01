from __future__ import annotations

from typing import Dict, Any
from ..refs.api import post_refs_apply


def _apply_ref(ref_id: str | None) -> Dict[str, Any]:
    if not ref_id:
        return {}
    res = post_refs_apply({"ref_id": ref_id})
    if isinstance(res, tuple):
        payload, code = res
        if code != 200:
            return {}
        return payload.get("ref_pack") or {}
    return (res or {}).get("ref_pack") or {}


def build_ref_pack(char_meta: Dict[str, Any]) -> Dict[str, Any]:
    pack: Dict[str, Any] = {"image": {}, "voice": {}, "music": {}}
    if not isinstance(char_meta, dict):
        return pack
    img_id = char_meta.get("image_ref_id")
    voc_id = char_meta.get("voice_ref_id")
    mus_id = char_meta.get("music_ref_id")
    if img_id:
        pack["image"] = _apply_ref(img_id)
    if voc_id:
        pack["voice"] = _apply_ref(voc_id)
    if mus_id:
        pack["music"] = _apply_ref(mus_id)
    return pack


def apply_storyboard_locks(shot: Dict[str, Any], ref_pack: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "mode": "gen",
        "prompt": shot.get("prompt") or "",
        "size": shot.get("size", "1024x1024"),
        "refs": (ref_pack or {}).get("image") or {},
        "seed": seed,
    }


def apply_animatic_vo_locks(line: Dict[str, Any], ref_pack: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "text": line.get("text") or "",
        "voice_id": line.get("voice_ref_id"),
        "voice_refs": (ref_pack or {}).get("voice") or {},
        "seed": seed,
    }


def apply_music_cue_locks(cue: Dict[str, Any], ref_pack: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "prompt": cue.get("prompt") or "",
        "bpm": cue.get("bpm"),
        "length_s": cue.get("length_s", 30),
        "music_id": cue.get("music_ref_id"),
        "music_refs": (ref_pack or {}).get("music") or {},
        "seed": seed,
    }


