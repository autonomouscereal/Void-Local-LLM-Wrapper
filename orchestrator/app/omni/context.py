from __future__ import annotations

from typing import Any, Dict, List


def build_omni_context(plan_text: str | None, tool_exec_meta: List[Dict[str, Any]] | None, tool_results: List[Dict[str, Any]] | None) -> str:
    sections: List[str] = []
    if isinstance(plan_text, str) and plan_text.strip():
        sections.append("### Plan\n" + plan_text.strip())

    if isinstance(tool_exec_meta, list) and tool_exec_meta:
        lines: List[str] = []
        for m in tool_exec_meta[:20]:
            nm = str((m or {}).get("name") or "tool")
            dur = int((m or {}).get("duration_ms") or 0)
            ak = ", ".join(((m or {}).get("args") or {}).keys()) if isinstance((m or {}).get("args"), dict) else ""
            lines.append(f"- {nm} ({dur} ms){' — ' + ak if ak else ''}")
        if lines:
            sections.append("### Tools\n" + "\n".join(lines))

    if isinstance(tool_results, list) and tool_results:
        urls: List[str] = []
        exts = (".mp4", ".webm", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".srt")
        for tr in tool_results:
            res = (tr or {}).get("result") or {}
            if isinstance(res, dict):
                meta = res.get("meta"); arts = res.get("artifacts")
                if isinstance(meta, dict) and isinstance(arts, list):
                    cid = meta.get("cid")
                    for a in arts:
                        aid = (a or {}).get("id"); kind = (a or {}).get("kind") or ""
                        if cid and aid:
                            if kind.startswith("image"):
                                urls.append(f"/uploads/artifacts/image/{cid}/{aid}")
                            elif kind.startswith("audio"):
                                urls.append(f"/uploads/artifacts/audio/tts/{cid}/{aid}")
                                urls.append(f"/uploads/artifacts/music/{cid}/{aid}")
                def _walk(v: Any) -> None:
                    nonlocal urls
                    if isinstance(v, str):
                        s = v.strip().lower()
                        if s.startswith("http://") or s.startswith("https://") or s.startswith("/uploads/") or "/workspace/uploads/" in s or any(s.endswith(ext) for ext in exts):
                            v2 = v
                            if "/workspace/uploads/" in v2 and "/workspace" in v2:
                                parts = v2.split("/workspace", 1)
                                v2 = parts[1] if len(parts) > 1 else v2
                            urls.append(v2)
                    elif isinstance(v, list):
                        for it in v: _walk(it)
                    elif isinstance(v, dict):
                        for it in v.values(): _walk(it)
                _walk(res)
        urls = list(dict.fromkeys(urls))
        if urls:
            sections.append("### Assets\n" + "\n".join([f"- {u}" for u in urls]))

    return ("\n\n" + "\n\n".join(sections)) if sections else ""


