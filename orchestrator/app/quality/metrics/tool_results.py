from __future__ import annotations

"""
quality.metrics.tool_results

Pure helpers for extracting URLs, artifact counts, and aggregated domain QA metrics
from tool results. This used to live in `pipeline/assets.py` but is logically QA/metrics.
"""

from typing import Any, Callable, Dict, List, Optional
from void_artifacts import artifact_id_to_safe_filename

# Use relative imports to avoid relying on the top-level package name being `app`
# (this file is imported both in-container and in some local/dev contexts).
from ...locks.runtime import QUALITY_PRESETS as LOCK_QUALITY_PRESETS
from ...locks.runtime import quality_thresholds as _lock_quality_thresholds


def collect_urls(tool_results: List[Dict[str, Any]], absolutize_url: Callable[[str], str]) -> List[str]:
    """
    Inspect tool results and extract artifact/view URLs.
    No I/O. Dedupes and absolutizes using the provided absolutize_url.
    Relies only on structured artifact metadata and orch_view_urls to avoid stale paths.
    """
    urls: List[str] = []
    for tool_result_entry in tool_results or []:
        res = (tool_result_entry or {}).get("result") or {}
        if not isinstance(res, dict):
            continue
        # envelope-based tools: artifacts + orch_view_urls
        meta = res.get("meta")
        artifact_entries = res.get("artifacts")
        if isinstance(meta, dict) and isinstance(artifact_entries, list):
            conversation_id = meta.get("conversation_id")
            for artifact_entry in artifact_entries:
                if not isinstance(artifact_entry, dict):
                    continue
                # artifact_id accepts any type (str, int, etc.) - unified handling
                artifact_id = artifact_entry.get("artifact_id")
                kind = (artifact_entry or {}).get("kind") or ""
                # Always collect explicit URLs/paths when present; some tools
                # emit view_url/url/path without a stable (conversation_id,id) pair.
                direct = (artifact_entry or {}).get("view_url") or (artifact_entry or {}).get("url") or (artifact_entry or {}).get("path")
                if isinstance(direct, str) and direct.strip():
                    urls.append(direct.strip())
                # artifact_id accepts any type (str, int, etc.) - unified handling
                # Convert artifact_id to safe filename for URL construction
                if isinstance(conversation_id, str) and conversation_id and artifact_id:
                    safe_filename = artifact_id_to_safe_filename(artifact_id)
                    if kind.startswith("image"):
                        # Determine extension from kind or default to .png
                        ext = ".png"  # Default for images
                        urls.append(f"/uploads/artifacts/image/{conversation_id}/{safe_filename}{ext}")
                    elif kind.startswith("tts") or ("tts" in kind):
                        # TTS artifacts live under /artifacts/audio/tts/<conversation_id>/<file>
                        ext = ".wav"  # Default for TTS audio
                        urls.append(f"/uploads/artifacts/audio/tts/{conversation_id}/{safe_filename}{ext}")
                    elif kind.startswith("music") or ("music" in kind):
                        # Music artifacts (windowed + mixes) live under /artifacts/music/<conversation_id>/<file>
                        ext = ".wav"  # Default for music audio
                        urls.append(f"/uploads/artifacts/music/{conversation_id}/{safe_filename}{ext}")
                    elif kind.startswith("audio"):
                        # Generic audio artifacts: do not guess a subfolder (tts/music/sfx/etc)
                        # because it varies by tool. Prefer explicit url/path above.
                        # No-op: we already collected direct url/path/view_url if present.
                        continue
        if isinstance(meta, dict) and isinstance(meta.get("orch_view_urls"), list):
            for u in (meta.get("orch_view_urls") or []):
                if isinstance(u, str) and u.strip():
                    urls.append(u)
    # dedupe and absolutize
    urls = list(dict.fromkeys(urls))
    return [absolutize_url(u) for u in urls if isinstance(u, str)]


def count_images(tool_results: List[Dict[str, Any]]) -> int:
    """
    Count image-like artifacts in tool results.
    No I/O.
    """
    count = 0
    for tool_result_entry in tool_results or []:
        res = (tool_result_entry or {}).get("result") or {}
        if not isinstance(res, dict):
            continue
        # flat images array
        if isinstance(res.get("images"), list):
            count += len(res.get("images") or [])
        # canonical artifacts array
        artifact_entries = res.get("artifacts")
        if isinstance(artifact_entries, list):
            for artifact_entry in artifact_entries:
                kind = (artifact_entry or {}).get("kind")
                if isinstance(kind, str) and kind.startswith("image"):
                    count += 1
        # external_ids.images from Comfy bridge
        external_ids = res.get("ids") if isinstance(res, dict) else {}
        if isinstance(external_ids, dict) and isinstance(external_ids.get("image_ids"), list):
            count += len(external_ids.get("image_ids") or [])
    return count


def count_video(tool_results: List[Dict[str, Any]]) -> int:
    """
    Count video-like artifacts in tool results.
    No I/O.
    """
    count = 0
    for tool_result_entry in tool_results or []:
        res = (tool_result_entry or {}).get("result") or {}
        if not isinstance(res, dict):
            continue
        artifact_entries = res.get("artifacts")
        if isinstance(artifact_entries, list):
            for artifact_entry in artifact_entries:
                kind = (artifact_entry or {}).get("kind")
                if isinstance(kind, str) and kind.startswith("video"):
                    count += 1
        external_ids = res.get("ids") if isinstance(res, dict) else {}
        if isinstance(external_ids, dict) and isinstance(external_ids.get("video_ids"), list):
            count += len(external_ids.get("video_ids") or [])
    return count


def count_audio(tool_results: List[Dict[str, Any]]) -> int:
    """
    Count audio-like artifacts in tool results (music/tts/audio).
    No I/O.
    """
    count = 0
    for tool_result_entry in tool_results or []:
        res = (tool_result_entry or {}).get("result") or {}
        if not isinstance(res, dict):
            continue
        artifact_entries = res.get("artifacts")
        if isinstance(artifact_entries, list):
            for artifact_entry in artifact_entries:
                kind = (artifact_entry or {}).get("kind")
                if isinstance(kind, str) and (kind.startswith("audio") or kind.startswith("music") or kind.startswith("tts")):
                    count += 1
        external_ids = res.get("ids") if isinstance(res, dict) else {}
        if isinstance(external_ids, dict) and isinstance(external_ids.get("audio_ids"), list):
            count += len(external_ids.get("audio_ids") or [])
        if isinstance(external_ids, dict) and isinstance(external_ids.get("music_ids"), list):
            count += len(external_ids.get("music_ids") or [])
        if isinstance(external_ids, dict) and isinstance(external_ids.get("tts_ids"), list):
            count += len(external_ids.get("tts_ids") or [])
    return count


def _get_dict(obj: Any, key: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    val = obj.get(key)
    return val if isinstance(val, dict) else {}


def _collect_key_values(candidates: List[Dict[str, Any]], key_names: List[str]) -> List[Any]:
    values: List[Any] = []
    stack: List[Any] = list(candidates)
    key_set = set(key_names)
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        for k, v in cur.items():
            if k in key_set:
                values.append(v)
            if isinstance(v, dict):
                stack.append(v)
    return values


def compute_domain_qa(tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate domain-specific QA metrics for images, videos, and audio/music/tts.
    Borrows existing per-artifact metadata and committee fields; avoids heavy recompute.
    """
    image_face_vals: List[float] = []
    image_id_vals: List[float] = []
    image_hands_ratio_vals: List[float] = []
    image_hands_true = 0.0
    image_hands_total = 0.0
    image_art_ratio_vals: List[float] = []
    image_art_true = 0.0
    image_art_total = 0.0
    image_region_shape_vals: List[float] = []
    image_region_texture_vals: List[float] = []
    image_region_texture_min_vals: List[float] = []
    image_region_color_vals: List[float] = []
    image_region_clip_vals: List[float] = []
    image_scene_vals: List[float] = []
    image_style_vals: List[float] = []
    image_pose_vals: List[float] = []
    image_text_readable_vals: List[float] = []
    image_background_quality_vals: List[float] = []
    # Optional per-entity lock metrics (if tools emit them)
    image_entity_clip_vals: List[float] = []
    image_entity_texture_vals: List[float] = []
    image_entity_shape_vals: List[float] = []
    image_entity_lock_vals: List[float] = []
    image_quality_profile: Optional[str] = None

    video_seam_ratio_vals: List[float] = []
    video_seam_true = 0.0
    video_seam_total = 0.0
    video_art_ratio_vals: List[float] = []
    video_art_true = 0.0
    video_art_total = 0.0
    # Optional higher-level video/film metrics
    video_fvd_vals: List[float] = []
    video_fvmd_vals: List[float] = []
    video_frame_lpips_vals: List[float] = []
    video_temporal_lpips_vals: List[float] = []
    video_flow_consistency_vals: List[float] = []
    video_face_drift_vals: List[float] = []
    video_object_drift_vals: List[float] = []
    video_color_consistency_vals: List[float] = []

    audio_lufs_vals: List[float] = []
    audio_clip_true = 0.0
    audio_clip_total = 0.0
    audio_seam_ratio_vals: List[float] = []
    audio_seam_true = 0.0
    audio_seam_total = 0.0
    audio_voice_vals: List[float] = []
    audio_tempo_vals: List[float] = []
    audio_key_vals: List[float] = []
    audio_stem_vals: List[float] = []
    audio_lyrics_vals: List[float] = []
    # Optional higher-level music/tts/SFX lock metrics
    audio_motif_vals: List[float] = []
    audio_prosody_pitch_vals: List[float] = []
    audio_prosody_energy_vals: List[float] = []
    audio_prosody_duration_vals: List[float] = []
    audio_emotion_vals: List[float] = []
    audio_style_vals: List[float] = []
    audio_timing_vals: List[float] = []
    audio_sfx_timbre_vals: List[float] = []
    audio_sfx_envelope_vals: List[float] = []
    audio_sfx_spatial_vals: List[float] = []
    audio_sfx_timing_vals: List[float] = []
    audio_sfx_loudness_vals: List[float] = []
    audio_sfx_density_vals: List[float] = []

    for tr in tool_results or []:
        if not isinstance(tr, dict):
            continue
        tool_name = tr.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        tool_name = tool_name.strip().lower()
        res = tr.get("result") if isinstance(tr.get("result"), dict) else {}
        meta = _get_dict(res, "meta")
        qa_block = _get_dict(meta, "qa")
        locks_block = _get_dict(meta, "locks")
        post_block = _get_dict(meta, "post")
        committee_block = _get_dict(meta, "committee")
        root_candidates: List[Dict[str, Any]] = []
        for cand in (meta, qa_block, locks_block, post_block, committee_block, _get_dict(res, "qa"), _get_dict(res, "locks")):
            if cand:
                root_candidates.append(cand)

        # Film2 embeds hero-frame lock metrics inside its film2.run result.meta / shot_meta.meta.locks.
        # Treat film2.run as an image-metric producer so downstream QA sees those scores.
        if tool_name.startswith("image.") or tool_name == "film2.run":
            face_candidates = _collect_key_values(root_candidates, ["face_lock", "face_cos", "face_score"])
            for val in face_candidates:
                if isinstance(val, (int, float)):
                    image_face_vals.append(float(val))
            id_candidates = _collect_key_values(root_candidates, ["id_lock", "id_cos", "identity_lock", "identity_score"])
            for val in id_candidates:
                if isinstance(val, (int, float)):
                    image_id_vals.append(float(val))
            hands_dicts = _collect_key_values(root_candidates, ["hands"])
            for hd in hands_dicts:
                if isinstance(hd, dict):
                    ok = hd.get("ok")
                    total = hd.get("total")
                    if isinstance(ok, (int, float)) and isinstance(total, (int, float)) and total:
                        image_hands_true += float(ok)
                        image_hands_total += float(total)
                    bad = hd.get("bad")
                    if isinstance(bad, (int, float)):
                        # if only ok + bad provided
                        if isinstance(ok, (int, float)) and not isinstance(total, (int, float)):
                            image_hands_true += float(ok)
                            image_hands_total += float(ok) + float(bad)
            hands_ratio_values = _collect_key_values(root_candidates, ["hands_ok_ratio", "hands_ratio"])
            for val in hands_ratio_values:
                if isinstance(val, (int, float)):
                    image_hands_ratio_vals.append(float(val))
            hands_ok_values = _collect_key_values(root_candidates, ["hands_ok"])
            for val in hands_ok_values:
                if isinstance(val, bool):
                    image_hands_total += 1.0
                    if val:
                        image_hands_true += 1.0
                elif isinstance(val, (int, float)):
                    image_hands_ratio_vals.append(float(val))
            artifact_dicts = _collect_key_values(root_candidates, ["artifacts", "artifact"])
            for ad in artifact_dicts:
                if isinstance(ad, dict):
                    ok = ad.get("ok")
                    total = ad.get("total")
                    if isinstance(ok, (int, float)) and isinstance(total, (int, float)) and total:
                        image_art_true += float(ok)
                        image_art_total += float(total)
            artifact_ratio = _collect_key_values(root_candidates, ["artifact_ok_ratio", "artifacts_ok_ratio"])
            for val in artifact_ratio:
                if isinstance(val, (int, float)):
                    image_art_ratio_vals.append(float(val))
            artifact_ok_values = _collect_key_values(root_candidates, ["artifact_ok", "artifacts_ok"])
            for val in artifact_ok_values:
                if isinstance(val, bool):
                    image_art_total += 1.0
                    if val:
                        image_art_true += 1.0
                elif isinstance(val, (int, float)):
                    image_art_ratio_vals.append(float(val))
            region_scores = _collect_key_values(root_candidates, ["regions"])
            for region_entry in region_scores:
                if not isinstance(region_entry, dict):
                    continue
                for metrics in region_entry.values():
                    if not isinstance(metrics, dict):
                        continue
                    shape_score = metrics.get("shape_score")
                    texture_score = metrics.get("texture_score")
                    color_score = metrics.get("color_score")
                    if isinstance(shape_score, (int, float)):
                        image_region_shape_vals.append(float(shape_score))
                    if isinstance(texture_score, (int, float)):
                        image_region_texture_vals.append(float(texture_score))
                    if isinstance(color_score, (int, float)):
                        image_region_color_vals.append(float(color_score))
            # Film2 hero-frame scoring emits flattened region aggregates (no "regions" dict).
            # Pull them in so Film2 QA isn't silently missing region metrics.
            region_shape_flat = _collect_key_values(root_candidates, ["region_shape_min", "region_shape_mean"])
            for val in region_shape_flat:
                if isinstance(val, (int, float)):
                    image_region_shape_vals.append(float(val))
            region_texture_flat = _collect_key_values(root_candidates, ["region_texture_mean"])
            for val in region_texture_flat:
                if isinstance(val, (int, float)):
                    image_region_texture_vals.append(float(val))
            region_texture_min_flat = _collect_key_values(root_candidates, ["region_texture_min"])
            for val in region_texture_min_flat:
                if isinstance(val, (int, float)):
                    image_region_texture_min_vals.append(float(val))
            region_color_flat = _collect_key_values(root_candidates, ["region_color_mean"])
            for val in region_color_flat:
                if isinstance(val, (int, float)):
                    image_region_color_vals.append(float(val))
            region_clip_flat = _collect_key_values(root_candidates, ["region_clip_mean"])
            for val in region_clip_flat:
                if isinstance(val, (int, float)):
                    image_region_clip_vals.append(float(val))
            scene_score = _collect_key_values(root_candidates, ["scene_score"])
            for val in scene_score:
                if isinstance(val, (int, float)):
                    image_scene_vals.append(float(val))
            style_score = _collect_key_values(root_candidates, ["style_score"])
            for val in style_score:
                if isinstance(val, (int, float)):
                    image_style_vals.append(float(val))
            pose_score = _collect_key_values(root_candidates, ["pose_score"])
            for val in pose_score:
                if isinstance(val, (int, float)):
                    image_pose_vals.append(float(val))
            text_readable_vals = _collect_key_values(root_candidates, ["text_readable_lock"])
            for val in text_readable_vals:
                if isinstance(val, (int, float)):
                    image_text_readable_vals.append(float(val))
            background_quality_vals = _collect_key_values(root_candidates, ["background_quality"])
            for val in background_quality_vals:
                if isinstance(val, (int, float)):
                    image_background_quality_vals.append(float(val))
            # Optional entity-level QA: result["qa"]["images"]["entities"]
            entity_blocks = _collect_key_values(root_candidates, ["entities"])
            for ent_block in entity_blocks:
                if isinstance(ent_block, dict):
                    iterable = ent_block.values()
                elif isinstance(ent_block, list):
                    iterable = ent_block
                else:
                    continue
                for metrics in iterable:
                    if not isinstance(metrics, dict):
                        continue
                    cv = metrics.get("clip_lock")
                    tv = metrics.get("texture_lock")
                    sv = metrics.get("shape_lock")
                    if isinstance(cv, (int, float)):
                        image_entity_clip_vals.append(float(cv))
                    if isinstance(tv, (int, float)):
                        image_entity_texture_vals.append(float(tv))
                    if isinstance(sv, (int, float)):
                        image_entity_shape_vals.append(float(sv))
            # Optional aggregate entity_lock_score from locks/QA metadata
            entity_lock_scores = _collect_key_values(root_candidates, ["entity_lock_score"])
            for val in entity_lock_scores:
                if isinstance(val, (int, float)):
                    image_entity_lock_vals.append(float(val))
            # Best-effort detection of the active image quality profile
            qp = meta.get("quality_profile")
            if isinstance(qp, str) and qp.strip() and image_quality_profile is None:
                image_quality_profile = qp.strip().lower()

        if tool_name.startswith("video.") or tool_name == "film2.run":
            seam_dicts = _collect_key_values(root_candidates, ["seam", "seams"])
            for sd in seam_dicts:
                if isinstance(sd, dict):
                    ok = sd.get("ok")
                    total = sd.get("total")
                    if isinstance(ok, (int, float)) and isinstance(total, (int, float)) and total:
                        video_seam_true += float(ok)
                        video_seam_total += float(total)
            seam_ratio_values = _collect_key_values(root_candidates, ["seam_ok_ratio", "seams_ok_ratio"])
            for val in seam_ratio_values:
                if isinstance(val, (int, float)):
                    video_seam_ratio_vals.append(float(val))
            seam_ok_values = _collect_key_values(root_candidates, ["seam_ok"])
            for val in seam_ok_values:
                if isinstance(val, bool):
                    video_seam_total += 1.0
                    if val:
                        video_seam_true += 1.0
                elif isinstance(val, (int, float)):
                    video_seam_ratio_vals.append(float(val))
            video_art_dicts = _collect_key_values(root_candidates, ["artifact", "artifacts"])
            for ad in video_art_dicts:
                if isinstance(ad, dict):
                    ok = ad.get("ok")
                    total = ad.get("total")
                    if isinstance(ok, (int, float)) and isinstance(total, (int, float)) and total:
                        video_art_true += float(ok)
                        video_art_total += float(total)
            video_art_ratio_values = _collect_key_values(root_candidates, ["artifact_ok_ratio", "artifacts_ok_ratio"])
            for val in video_art_ratio_values:
                if isinstance(val, (int, float)):
                    video_art_ratio_vals.append(float(val))
            video_art_ok_values = _collect_key_values(root_candidates, ["artifact_ok", "artifacts_ok"])
            for val in video_art_ok_values:
                if isinstance(val, bool):
                    video_art_total += 1.0
                    if val:
                        video_art_true += 1.0
                elif isinstance(val, (int, float)):
                    video_art_ratio_vals.append(float(val))
            # Optional film/video-specific metrics
            fvd_values = _collect_key_values(root_candidates, ["fvd"])
            for v in fvd_values:
                if isinstance(v, (int, float)):
                    video_fvd_vals.append(float(v))
            fvmd_values = _collect_key_values(root_candidates, ["fvmd"])
            for v in fvmd_values:
                if isinstance(v, (int, float)):
                    video_fvmd_vals.append(float(v))
            frame_lpips_values = _collect_key_values(root_candidates, ["frame_lpips_mean"])
            for v in frame_lpips_values:
                if isinstance(v, (int, float)):
                    video_frame_lpips_vals.append(float(v))
            temporal_lpips_values = _collect_key_values(root_candidates, ["temporal_lpips_mean"])
            for v in temporal_lpips_values:
                if isinstance(v, (int, float)):
                    video_temporal_lpips_vals.append(float(v))
            flow_consistency_values = _collect_key_values(root_candidates, ["optical_flow_consistency"])
            for v in flow_consistency_values:
                if isinstance(v, (int, float)):
                    video_flow_consistency_vals.append(float(v))
            face_drift_values = _collect_key_values(root_candidates, ["face_track_drift_mean"])
            for v in face_drift_values:
                if isinstance(v, (int, float)):
                    video_face_drift_vals.append(float(v))
            object_drift_values = _collect_key_values(root_candidates, ["object_track_drift_mean"])
            for v in object_drift_values:
                if isinstance(v, (int, float)):
                    video_object_drift_vals.append(float(v))
            color_consistency_values = _collect_key_values(root_candidates, ["color_grade_consistency"])
            for v in color_consistency_values:
                if isinstance(v, (int, float)):
                    video_color_consistency_vals.append(float(v))

        if tool_name.startswith("music.") or tool_name.startswith("audio.") or tool_name.startswith("tts."):
            committee_values = _collect_key_values(root_candidates, ["peak_normalize", "hard_trim", "qa", "committee"])
            for cv in committee_values:
                if isinstance(cv, dict):
                    if isinstance(cv.get("lufs_before"), (int, float)):
                        audio_lufs_vals.append(float(cv.get("lufs_before")))
                    if isinstance(cv.get("lufs"), (int, float)):
                        audio_lufs_vals.append(float(cv.get("lufs")))
                    if isinstance(cv.get("lufs_gain_db"), (int, float)):
                        # if normalized, estimate final lufs
                        orig = cv.get("lufs_before")
                        gain = cv.get("lufs_gain_db")
                        if isinstance(orig, (int, float)) and isinstance(gain, (int, float)):
                            audio_lufs_vals.append(float(orig) + float(gain))
                    if isinstance(cv.get("clipping"), bool):
                        audio_clip_total += 1.0
                        if cv.get("clipping"):
                            audio_clip_true += 1.0
                    if isinstance(cv.get("clipping_ratio"), (int, float)):
                        audio_clip_true += float(cv.get("clipping_ratio"))
                        audio_clip_total += 1.0
                    if isinstance(cv.get("seam_ok_ratio"), (int, float)):
                        audio_seam_ratio_vals.append(float(cv.get("seam_ok_ratio")))
                    if isinstance(cv.get("seam_ok"), bool):
                        audio_seam_total += 1.0
                        if cv.get("seam_ok"):
                            audio_seam_true += 1.0
            # top-level meta falls back
            lufs = meta.get("lufs")
            if isinstance(lufs, (int, float)):
                audio_lufs_vals.append(float(lufs))
            clipping = meta.get("clipping")
            if isinstance(clipping, bool):
                audio_clip_total += 1.0
                if clipping:
                    audio_clip_true += 1.0
            elif isinstance(clipping, (int, float)):
                audio_clip_true += float(clipping)
                audio_clip_total += 1.0
            seam_ok = meta.get("seam_ok")
            if isinstance(seam_ok, bool):
                audio_seam_total += 1.0
                if seam_ok:
                    audio_seam_true += 1.0
            elif isinstance(seam_ok, (int, float)):
                audio_seam_ratio_vals.append(float(seam_ok))
            voice_score = meta.get("voice_score") or locks_block.get("voice_score")
            if isinstance(voice_score, (int, float)):
                audio_voice_vals.append(float(voice_score))
            tempo_score = meta.get("tempo_score") or locks_block.get("tempo_score")
            if isinstance(tempo_score, (int, float)):
                audio_tempo_vals.append(float(tempo_score))
            key_score = meta.get("key_score") or locks_block.get("key_score")
            if isinstance(key_score, (int, float)):
                audio_key_vals.append(float(key_score))
            stem_balance_score = meta.get("stem_balance_score") or locks_block.get("stem_balance_score")
            if isinstance(stem_balance_score, (int, float)):
                audio_stem_vals.append(float(stem_balance_score))
            lyrics_score = meta.get("lyrics_score") or locks_block.get("lyrics_score")
            if isinstance(lyrics_score, (int, float)):
                audio_lyrics_vals.append(float(lyrics_score))
            # Optional motif-level lock scores if tools emit them
            motif_lock = meta.get("motif_lock") or locks_block.get("motif_lock")
            if isinstance(motif_lock, (int, float)):
                audio_motif_vals.append(float(motif_lock))
            # Optional TTS prosody/emotion/style/timing locks
            prosody_pitch_lock = meta.get("prosody_pitch_lock") or locks_block.get("prosody_pitch_lock")
            if isinstance(prosody_pitch_lock, (int, float)):
                audio_prosody_pitch_vals.append(float(prosody_pitch_lock))
            prosody_energy_lock = meta.get("prosody_energy_lock") or locks_block.get("prosody_energy_lock")
            if isinstance(prosody_energy_lock, (int, float)):
                audio_prosody_energy_vals.append(float(prosody_energy_lock))
            prosody_duration_lock = meta.get("prosody_duration_lock") or locks_block.get("prosody_duration_lock")
            if isinstance(prosody_duration_lock, (int, float)):
                audio_prosody_duration_vals.append(float(prosody_duration_lock))
            emotion_lock = meta.get("emotion_lock") or locks_block.get("emotion_lock")
            if isinstance(emotion_lock, (int, float)):
                audio_emotion_vals.append(float(emotion_lock))
            style_lock = meta.get("style_lock") or locks_block.get("style_lock")
            if isinstance(style_lock, (int, float)):
                audio_style_vals.append(float(style_lock))
            timing_lock = meta.get("timing_lock") or locks_block.get("timing_lock")
            if isinstance(timing_lock, (int, float)):
                audio_timing_vals.append(float(timing_lock))
            # Optional SFX-specific locks
            sfx_timbre_lock = meta.get("sfx_timbre_lock") or locks_block.get("sfx_timbre_lock")
            if isinstance(sfx_timbre_lock, (int, float)):
                audio_sfx_timbre_vals.append(float(sfx_timbre_lock))
            sfx_envelope_lock = meta.get("sfx_envelope_lock") or locks_block.get("sfx_envelope_lock")
            if isinstance(sfx_envelope_lock, (int, float)):
                audio_sfx_envelope_vals.append(float(sfx_envelope_lock))
            sfx_spatial_lock = meta.get("sfx_spatial_lock") or locks_block.get("sfx_spatial_lock")
            if isinstance(sfx_spatial_lock, (int, float)):
                audio_sfx_spatial_vals.append(float(sfx_spatial_lock))
            sfx_timing_lock = meta.get("sfx_timing_lock") or locks_block.get("sfx_timing_lock")
            if isinstance(sfx_timing_lock, (int, float)):
                audio_sfx_timing_vals.append(float(sfx_timing_lock))
            sfx_loudness_lock = meta.get("sfx_loudness_lock") or locks_block.get("sfx_loudness_lock")
            if isinstance(sfx_loudness_lock, (int, float)):
                audio_sfx_loudness_vals.append(float(sfx_loudness_lock))
            sfx_density_lock = meta.get("sfx_density_lock") or locks_block.get("sfx_density_lock")
            if isinstance(sfx_density_lock, (int, float)):
                audio_sfx_density_vals.append(float(sfx_density_lock))

    image_face_avg = sum(image_face_vals) / len(image_face_vals) if image_face_vals else None
    image_id_avg = sum(image_id_vals) / len(image_id_vals) if image_id_vals else None

    hands_ratio = None
    if image_hands_ratio_vals:
        hands_ratio = sum(image_hands_ratio_vals) / len(image_hands_ratio_vals)
    elif image_hands_total:
        hands_ratio = image_hands_true / image_hands_total

    artifact_ratio = None
    if image_art_ratio_vals:
        artifact_ratio = sum(image_art_ratio_vals) / len(image_art_ratio_vals)
    elif image_art_total:
        artifact_ratio = image_art_true / image_art_total

    region_shape_avg = sum(image_region_shape_vals) / len(image_region_shape_vals) if image_region_shape_vals else None
    region_shape_min = min(image_region_shape_vals) if image_region_shape_vals else None
    region_texture_avg = sum(image_region_texture_vals) / len(image_region_texture_vals) if image_region_texture_vals else None
    region_texture_min = min(image_region_texture_min_vals) if image_region_texture_min_vals else None
    region_color_avg = sum(image_region_color_vals) / len(image_region_color_vals) if image_region_color_vals else None
    region_clip_avg = sum(image_region_clip_vals) / len(image_region_clip_vals) if image_region_clip_vals else None
    scene_avg = sum(image_scene_vals) / len(image_scene_vals) if image_scene_vals else None
    style_avg = sum(image_style_vals) / len(image_style_vals) if image_style_vals else None
    pose_avg = sum(image_pose_vals) / len(image_pose_vals) if image_pose_vals else None
    entity_clip_avg = sum(image_entity_clip_vals) / len(image_entity_clip_vals) if image_entity_clip_vals else None
    entity_texture_avg = sum(image_entity_texture_vals) / len(image_entity_texture_vals) if image_entity_texture_vals else None
    entity_shape_avg = sum(image_entity_shape_vals) / len(image_entity_shape_vals) if image_entity_shape_vals else None
    entity_lock_score_avg = sum(image_entity_lock_vals) / len(image_entity_lock_vals) if image_entity_lock_vals else None

    video_seam_ratio = None
    if video_seam_ratio_vals:
        video_seam_ratio = sum(video_seam_ratio_vals) / len(video_seam_ratio_vals)
    elif video_seam_total:
        video_seam_ratio = video_seam_true / video_seam_total

    video_art_ratio = None
    if video_art_ratio_vals:
        video_art_ratio = sum(video_art_ratio_vals) / len(video_art_ratio_vals)
    elif video_art_total:
        video_art_ratio = video_art_true / video_art_total
    video_fvd_avg = sum(video_fvd_vals) / len(video_fvd_vals) if video_fvd_vals else None
    video_fvmd_avg = sum(video_fvmd_vals) / len(video_fvmd_vals) if video_fvmd_vals else None
    video_frame_lpips_avg = sum(video_frame_lpips_vals) / len(video_frame_lpips_vals) if video_frame_lpips_vals else None
    video_temporal_lpips_avg = sum(video_temporal_lpips_vals) / len(video_temporal_lpips_vals) if video_temporal_lpips_vals else None
    video_flow_consistency_avg = sum(video_flow_consistency_vals) / len(video_flow_consistency_vals) if video_flow_consistency_vals else None
    video_face_drift_avg = sum(video_face_drift_vals) / len(video_face_drift_vals) if video_face_drift_vals else None
    video_object_drift_avg = sum(video_object_drift_vals) / len(video_object_drift_vals) if video_object_drift_vals else None
    video_color_consistency_avg = sum(video_color_consistency_vals) / len(video_color_consistency_vals) if video_color_consistency_vals else None

    audio_mean_lufs = sum(audio_lufs_vals) / len(audio_lufs_vals) if audio_lufs_vals else None
    audio_clip_ratio = (audio_clip_true / audio_clip_total) if audio_clip_total else None
    audio_seam_ratio = None
    if audio_seam_ratio_vals:
        audio_seam_ratio = sum(audio_seam_ratio_vals) / len(audio_seam_ratio_vals)
    elif audio_seam_total:
        audio_seam_ratio = audio_seam_true / audio_seam_total

    audio_voice_avg = sum(audio_voice_vals) / len(audio_voice_vals) if audio_voice_vals else None
    audio_tempo_avg = sum(audio_tempo_vals) / len(audio_tempo_vals) if audio_tempo_vals else None
    audio_key_avg = sum(audio_key_vals) / len(audio_key_vals) if audio_key_vals else None
    audio_stem_avg = sum(audio_stem_vals) / len(audio_stem_vals) if audio_stem_vals else None
    audio_lyrics_avg = sum(audio_lyrics_vals) / len(audio_lyrics_vals) if audio_lyrics_vals else None
    audio_motif_avg = sum(audio_motif_vals) / len(audio_motif_vals) if audio_motif_vals else None
    audio_prosody_pitch_avg = sum(audio_prosody_pitch_vals) / len(audio_prosody_pitch_vals) if audio_prosody_pitch_vals else None
    audio_prosody_energy_avg = sum(audio_prosody_energy_vals) / len(audio_prosody_energy_vals) if audio_prosody_energy_vals else None
    audio_prosody_duration_avg = sum(audio_prosody_duration_vals) / len(audio_prosody_duration_vals) if audio_prosody_duration_vals else None
    audio_emotion_avg = sum(audio_emotion_vals) / len(audio_emotion_vals) if audio_emotion_vals else None
    audio_style_avg = sum(audio_style_vals) / len(audio_style_vals) if audio_style_vals else None
    audio_timing_avg = sum(audio_timing_vals) / len(audio_timing_vals) if audio_timing_vals else None
    audio_sfx_timbre_avg = sum(audio_sfx_timbre_vals) / len(audio_sfx_timbre_vals) if audio_sfx_timbre_vals else None
    audio_sfx_envelope_avg = sum(audio_sfx_envelope_vals) / len(audio_sfx_envelope_vals) if audio_sfx_envelope_vals else None
    audio_sfx_spatial_avg = sum(audio_sfx_spatial_vals) / len(audio_sfx_spatial_vals) if audio_sfx_spatial_vals else None
    audio_sfx_timing_avg = sum(audio_sfx_timing_vals) / len(audio_sfx_timing_vals) if audio_sfx_timing_vals else None
    audio_sfx_loudness_avg = sum(audio_sfx_loudness_vals) / len(audio_sfx_loudness_vals) if audio_sfx_loudness_vals else None
    audio_sfx_density_avg = sum(audio_sfx_density_vals) / len(audio_sfx_density_vals) if audio_sfx_density_vals else None

    # Optional overall lock health scores for dashboarding / training targets
    image_lock_overall = None
    _image_lock_components: list[float] = []
    for v in (image_face_avg, image_id_avg, entity_shape_avg):
        if isinstance(v, (int, float)):
            _image_lock_components.append(float(v))
    if _image_lock_components:
        image_lock_overall = min(_image_lock_components)

    # Derive a simple categorical lock_status for images ("ok" / "weak" / "fail")
    lock_status: Optional[str] = None
    if isinstance(image_face_avg, (int, float)):
        profile_name = (image_quality_profile or "standard").lower()
        try:
            thresholds = _lock_quality_thresholds(profile_name)
            face_min = float(thresholds.get("face_min", 0.8))
        except Exception:
            try:
                preset = LOCK_QUALITY_PRESETS.get(profile_name, LOCK_QUALITY_PRESETS["standard"])
                face_min = float(preset.get("face_min", 0.8))
            except Exception:
                face_min = 0.8
        weak_margin = 0.1 * face_min
        if image_face_avg >= face_min:
            lock_status = "ok"
        elif image_face_avg >= max(0.0, face_min - weak_margin):
            lock_status = "weak"
        else:
            lock_status = "fail"
        # Optionally downgrade based on aggregate entity lock quality
        if isinstance(entity_lock_score_avg, (int, float)):
            if entity_lock_score_avg < 0.6 and lock_status == "ok":
                lock_status = "weak"
            if entity_lock_score_avg < 0.4:
                lock_status = "fail"

    video_lock_overall = None
    _video_lock_components: list[float] = []
    for v in (video_seam_ratio, video_art_ratio, video_fvmd_avg, video_flow_consistency_avg, video_color_consistency_avg):
        if isinstance(v, (int, float)):
            _video_lock_components.append(float(v))
    if _video_lock_components:
        # Higher is better for seam_ok_ratio, fvmd, flow_consistency, color_consistency; lower is better for artifact_ok_ratio.
        # Use the minimum after flipping artifact_ok_ratio so that 1.0 is best.
        if isinstance(video_art_ratio, (int, float)):
            flip_art = max(0.0, min(1.0 - float(video_art_ratio), 1.0))
            _video_lock_components.append(flip_art)
        video_lock_overall = min(_video_lock_components)

    audio_lock_overall = None
    _audio_lock_components: list[float] = []
    for v in (
        audio_voice_avg,
        audio_tempo_avg,
        audio_key_avg,
        audio_stem_avg,
        audio_lyrics_avg,
        audio_motif_avg,
        audio_prosody_pitch_avg,
        audio_prosody_energy_avg,
        audio_prosody_duration_avg,
        audio_emotion_avg,
        audio_style_avg,
        audio_timing_avg,
        audio_sfx_timing_avg,
    ):
        if isinstance(v, (int, float)):
            _audio_lock_components.append(float(v))
    if _audio_lock_components:
        audio_lock_overall = min(_audio_lock_components)

    return {
        "images": {
            "face_lock": image_face_avg,
            "id_lock": image_id_avg,
            "hands_ok_ratio": hands_ratio,
            "artifact_ok_ratio": artifact_ratio,
            "region_shape_mean": region_shape_avg,
            "region_shape_min": region_shape_min,
            "region_texture_mean": region_texture_avg,
            "region_texture_min": region_texture_min,
            "region_color_mean": region_color_avg,
            "region_clip_mean": region_clip_avg,
            "scene_lock": scene_avg,
            "style_lock": style_avg,
            "pose_lock": pose_avg,
            "text_readable_lock": (sum(image_text_readable_vals) / len(image_text_readable_vals)) if image_text_readable_vals else None,
            "background_quality": (sum(image_background_quality_vals) / len(image_background_quality_vals)) if image_background_quality_vals else None,
            # Optional aggregated per-entity lock scores
            "entity_clip_lock_mean": entity_clip_avg,
            "entity_texture_lock_mean": entity_texture_avg,
            "entity_shape_lock_mean": entity_shape_avg,
            "entity_lock_score": entity_lock_score_avg,
            "lock_overall": image_lock_overall,
            "lock_status": lock_status,
        },
        "videos": {
            "seam_ok_ratio": video_seam_ratio,
            "artifact_ok_ratio": video_art_ratio,
            "fvd": video_fvd_avg,
            "fvmd": video_fvmd_avg,
            "frame_lpips_mean": video_frame_lpips_avg,
            "temporal_lpips_mean": video_temporal_lpips_avg,
            "optical_flow_consistency": video_flow_consistency_avg,
            "face_track_drift_mean": video_face_drift_avg,
            "object_track_drift_mean": video_object_drift_avg,
            "color_grade_consistency": video_color_consistency_avg,
            "lock_overall": video_lock_overall,
        },
        "audio": {
            "mean_lufs": audio_mean_lufs,
            "clipping_ratio": audio_clip_ratio,
            "seam_ok_ratio": audio_seam_ratio,
            "voice_lock": audio_voice_avg,
            "tempo_lock": audio_tempo_avg,
            "key_lock": audio_key_avg,
            "stem_balance_lock": audio_stem_avg,
            "lyrics_lock": audio_lyrics_avg,
            "motif_lock": audio_motif_avg,
            "prosody_pitch_lock": audio_prosody_pitch_avg,
            "prosody_energy_lock": audio_prosody_energy_avg,
            "prosody_duration_lock": audio_prosody_duration_avg,
            "emotion_lock": audio_emotion_avg,
            "style_lock": audio_style_avg,
            "timing_lock": audio_timing_avg,
            "sfx_timbre_lock": audio_sfx_timbre_avg,
            "sfx_envelope_lock": audio_sfx_envelope_avg,
            "sfx_spatial_lock": audio_sfx_spatial_avg,
            "sfx_timing_lock": audio_sfx_timing_avg,
            "sfx_loudness_lock": audio_sfx_loudness_avg,
            "sfx_density_lock": audio_sfx_density_avg,
            "lock_overall": audio_lock_overall,
        },
    }


