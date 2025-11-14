from __future__ import annotations

from typing import Any, Callable, Dict, List


def collect_urls(tool_results: List[Dict[str, Any]], absolutize_url: Callable[[str], str]) -> List[str]:
    """
    Inspect tool results and extract artifact/view URLs.
    No I/O. Dedupes and absolutizes using the provided absolutize_url.
    Relies only on structured artifact metadata and orch_view_urls to avoid stale paths.
    """
    urls: List[str] = []
    for tr in tool_results or []:
        res = (tr or {}).get("result") or {}
        if not isinstance(res, dict):
            continue
        # envelope-based tools: artifacts + orch_view_urls
        meta = res.get("meta")
        arts = res.get("artifacts")
        if isinstance(meta, dict) and isinstance(arts, list):
            cid = meta.get("cid")
            for a in arts:
                aid = (a or {}).get("id")
                kind = (a or {}).get("kind") or ""
                if cid and aid:
                    if kind.startswith("image"):
                        urls.append(f"/uploads/artifacts/image/{cid}/{aid}")
                    elif kind.startswith("audio"):
                        urls.append(f"/uploads/artifacts/audio/tts/{cid}/{aid}")
                        urls.append(f"/uploads/artifacts/music/{cid}/{aid}")
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
	for tr in tool_results or []:
		res = (tr or {}).get("result") or {}
		if not isinstance(res, dict):
			continue
		# flat images array
		if isinstance(res.get("images"), list):
			count += len(res.get("images") or [])
		# ids.images from Comfy bridge
		ids_obj = res.get("ids") if isinstance(res, dict) else {}
		if isinstance(ids_obj, dict) and isinstance(ids_obj.get("images"), list):
			count += len(ids_obj.get("images") or [])
	return count


def count_video(tool_results: List[Dict[str, Any]]) -> int:
	"""
	Count video-like artifacts in tool results.
	No I/O.
	"""
	count = 0
	for tr in tool_results or []:
		res = (tr or {}).get("result") or {}
		if not isinstance(res, dict):
			continue
		arts = res.get("artifacts")
		if isinstance(arts, list):
			for a in arts:
				kind = (a or {}).get("kind")
				if isinstance(kind, str) and kind.startswith("video"):
					count += 1
		ids_obj = res.get("ids") if isinstance(res, dict) else {}
		if isinstance(ids_obj, dict) and isinstance(ids_obj.get("videos"), list):
			count += len(ids_obj.get("videos") or [])
	return count


def count_audio(tool_results: List[Dict[str, Any]]) -> int:
	"""
	Count audio-like artifacts in tool results (music/tts/audio).
	No I/O.
	"""
	count = 0
	for tr in tool_results or []:
		res = (tr or {}).get("result") or {}
		if not isinstance(res, dict):
			continue
		arts = res.get("artifacts")
		if isinstance(arts, list):
			for a in arts:
				kind = (a or {}).get("kind")
				if isinstance(kind, str) and (kind.startswith("audio") or kind.startswith("music") or kind.startswith("tts")):
					count += 1
		ids_obj = res.get("ids") if isinstance(res, dict) else {}
		if isinstance(ids_obj, dict) and isinstance(ids_obj.get("audios"), list):
			count += len(ids_obj.get("audios") or [])
		if isinstance(ids_obj, dict) and isinstance(ids_obj.get("music"), list):
			count += len(ids_obj.get("music") or [])
		if isinstance(ids_obj, dict) and isinstance(ids_obj.get("tts"), list):
			count += len(ids_obj.get("tts") or [])
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
	image_region_color_vals: List[float] = []
	image_scene_vals: List[float] = []

	video_seam_ratio_vals: List[float] = []
	video_seam_true = 0.0
	video_seam_total = 0.0
	video_art_ratio_vals: List[float] = []
	video_art_true = 0.0
	video_art_total = 0.0

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

	for tr in tool_results or []:
		if not isinstance(tr, dict):
			continue
		name_raw = tr.get("name") or tr.get("tool") or ""
		if not isinstance(name_raw, str):
			continue
		name = name_raw.strip().lower()
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

		if name.startswith("image."):
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
					ok_val = hd.get("ok")
					total_val = hd.get("total")
					if isinstance(ok_val, (int, float)) and isinstance(total_val, (int, float)) and total_val:
						image_hands_true += float(ok_val)
						image_hands_total += float(total_val)
					bad_val = hd.get("bad")
					if isinstance(bad_val, (int, float)):
						# if only ok + bad provided
						if isinstance(ok_val, (int, float)) and not isinstance(total_val, (int, float)):
							image_hands_true += float(ok_val)
							image_hands_total += float(ok_val) + float(bad_val)
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
					ok_val = ad.get("ok")
					total_val = ad.get("total")
					if isinstance(ok_val, (int, float)) and isinstance(total_val, (int, float)) and total_val:
						image_art_true += float(ok_val)
						image_art_total += float(total_val)
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
			scene_score = _collect_key_values(root_candidates, ["scene_score"])
			for val in scene_score:
				if isinstance(val, (int, float)):
					image_scene_vals.append(float(val))

		if name.startswith("video.") or name in ("film.run", "film2.run"):
			seam_dicts = _collect_key_values(root_candidates, ["seam", "seams"])
			for sd in seam_dicts:
				if isinstance(sd, dict):
					ok_val = sd.get("ok")
					total_val = sd.get("total")
					if isinstance(ok_val, (int, float)) and isinstance(total_val, (int, float)) and total_val:
						video_seam_true += float(ok_val)
						video_seam_total += float(total_val)
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
					ok_val = ad.get("ok")
					total_val = ad.get("total")
					if isinstance(ok_val, (int, float)) and isinstance(total_val, (int, float)) and total_val:
						video_art_true += float(ok_val)
						video_art_total += float(total_val)
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

		if name.startswith("music.") or name.startswith("audio.") or name.startswith("tts."):
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
			lufs_val = meta.get("lufs")
			if isinstance(lufs_val, (int, float)):
				audio_lufs_vals.append(float(lufs_val))
			clipping_val = meta.get("clipping")
			if isinstance(clipping_val, bool):
				audio_clip_total += 1.0
				if clipping_val:
					audio_clip_true += 1.0
			elif isinstance(clipping_val, (int, float)):
				audio_clip_true += float(clipping_val)
				audio_clip_total += 1.0
			seam_val = meta.get("seam_ok")
			if isinstance(seam_val, bool):
				audio_seam_total += 1.0
				if seam_val:
					audio_seam_true += 1.0
			elif isinstance(seam_val, (int, float)):
				audio_seam_ratio_vals.append(float(seam_val))
			voice_score_val = meta.get("voice_score") or locks.get("voice_score")
			if isinstance(voice_score_val, (int, float)):
				audio_voice_vals.append(float(voice_score_val))
			tempo_score_val = meta.get("tempo_score") or locks.get("tempo_score")
			if isinstance(tempo_score_val, (int, float)):
				audio_tempo_vals.append(float(tempo_score_val))
			key_score_val = meta.get("key_score") or locks.get("key_score")
			if isinstance(key_score_val, (int, float)):
				audio_key_vals.append(float(key_score_val))
			stem_score_val = meta.get("stem_balance_score") or locks.get("stem_balance_score")
			if isinstance(stem_score_val, (int, float)):
				audio_stem_vals.append(float(stem_score_val))
			lyrics_score_val = meta.get("lyrics_score") or locks.get("lyrics_score")
			if isinstance(lyrics_score_val, (int, float)):
				audio_lyrics_vals.append(float(lyrics_score_val))

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
	region_color_avg = sum(image_region_color_vals) / len(image_region_color_vals) if image_region_color_vals else None
	scene_avg = sum(image_scene_vals) / len(image_scene_vals) if image_scene_vals else None

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

	return {
		"images": {
			"face_lock": image_face_avg,
			"id_lock": image_id_avg,
			"hands_ok_ratio": hands_ratio,
			"artifact_ok_ratio": artifact_ratio,
			"region_shape_mean": region_shape_avg,
			"region_shape_min": region_shape_min,
			"region_texture_mean": region_texture_avg,
			"region_color_mean": region_color_avg,
			"scene_lock": scene_avg,
		},
		"videos": {
			"seam_ok_ratio": video_seam_ratio,
			"artifact_ok_ratio": video_art_ratio,
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
		},
	}


