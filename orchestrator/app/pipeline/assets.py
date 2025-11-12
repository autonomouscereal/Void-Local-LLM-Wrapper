from __future__ import annotations

from typing import Any, Callable, Dict, List


def collect_urls(tool_results: List[Dict[str, Any]], absolutize_url: Callable[[str], str]) -> List[str]:
	"""
	Inspect tool results and extract artifact/view URLs.
	No I/O. Dedupes and absolutizes using the provided absolutize_url.
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
		# ids.image_files fallback
		ids_obj = res.get("ids") if isinstance(res.get("ids"), dict) else {}
		if isinstance(ids_obj, dict) and isinstance(ids_obj.get("image_files"), list):
			for fp in (ids_obj.get("image_files") or []):
				if isinstance(fp, str) and fp.strip():
					urls.append(f"/uploads/{fp.replace('\\', '/')}")
		# generic path scraping
		exts = (".mp4", ".webm", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".srt")
		def _walk(v):
			if isinstance(v, str):
				s = v.strip().lower()
				if not s:
					return
				if s.startswith("http://") or s.startswith("https://"):
					urls.append(v)
					return
				if "/workspace/uploads/" in v:
					tail = v.split("/workspace", 1)[1]
					urls.append(tail)
					return
				if v.startswith("/uploads/"):
					urls.append(v)
					return
				if any(s.endswith(ext) for ext in exts) and ("/uploads/" in s or "/workspace/uploads/" in s):
					if "/workspace/uploads/" in v:
						tail = v.split("/workspace", 1)[1]
						urls.append(tail)
					else:
						urls.append(v)
			elif isinstance(v, list):
				for it in v:
					_walk(it)
			elif isinstance(v, dict):
				for it in v.values():
					_walk(it)
		_walk(res)
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


