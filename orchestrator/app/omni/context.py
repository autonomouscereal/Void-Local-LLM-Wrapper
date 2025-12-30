from __future__ import annotations

from typing import Any, Dict, List
from void_artifacts import artifact_id_to_safe_filename


def build_omni_context(plan_text: str | None, tool_exec_meta: List[Dict[str, Any]] | None, tool_results: List[Dict[str, Any]] | None) -> str:
    sections: List[str] = []
    if isinstance(plan_text, str) and plan_text.strip():
        sections.append("### Plan\n" + plan_text.strip())

    if isinstance(tool_exec_meta, list) and tool_exec_meta:
        lines: List[str] = []
        for m in tool_exec_meta[:20]:
            # Check tool_name (canonical) first, then name (OpenAI format), then tool as fallback
            tool_name = (m or {}).get("tool_name")
            if not isinstance(tool_name, str):
                tool_name = (m or {}).get("name")
            if not isinstance(tool_name, str):
                tool_name = (m or {}).get("tool")
            if not isinstance(tool_name, str):
                tool_name = "tool"
            dur = int((m or {}).get("duration_ms") or 0)
            ak = ", ".join(((m or {}).get("args") or {}).keys()) if isinstance((m or {}).get("args"), dict) else ""
            lines.append(f"- {tool_name} ({dur} ms){' — ' + ak if ak else ''}")
        if lines:
            sections.append("### Tools\n" + "\n".join(lines))

    if isinstance(tool_results, list) and tool_results:
        urls: List[str] = []
        exts = (".mp4", ".webm", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".srt")
        for tool_result_entry in tool_results:
            tool_payload = (tool_result_entry or {}).get("result") or {}
            if isinstance(tool_payload, dict):
                meta = tool_payload.get("meta")
                artifact_entries = tool_payload.get("artifacts")
                if isinstance(meta, dict) and isinstance(artifact_entries, list):
                    conversation_id = meta.get("conversation_id")
                    for artifact_entry in artifact_entries:
                        artifact_id = (artifact_entry or {}).get("artifact_id")
                        kind = (artifact_entry or {}).get("kind") or ""
                        if conversation_id and artifact_id:
                            if kind.startswith("image"):
                                safe_filename = artifact_id_to_safe_filename(artifact_id, ".png")
                                urls.append(f"/uploads/artifacts/image/{conversation_id}/{safe_filename}")
                            elif kind.startswith("audio"):
                                safe_filename_tts = artifact_id_to_safe_filename(artifact_id, ".wav")
                                urls.append(f"/uploads/artifacts/audio/tts/{conversation_id}/{safe_filename_tts}")
                                safe_filename_music = artifact_id_to_safe_filename(artifact_id, ".wav")
                                urls.append(f"/uploads/artifacts/music/{conversation_id}/{safe_filename_music}")
                walk_stack: List[Any] = [tool_payload]
                while walk_stack:
                    v = walk_stack.pop()
                    if isinstance(v, str):
                        s = v.strip().lower()
                        if s.startswith("http://") or s.startswith("https://") or s.startswith("/uploads/") or "/workspace/uploads/" in s or any(s.endswith(ext) for ext in exts):
                            v2 = v
                            if "/workspace/uploads/" in v2 and "/workspace" in v2:
                                parts = v2.split("/workspace", 1)
                                v2 = parts[1] if len(parts) > 1 else v2
                            urls.append(v2)
                    elif isinstance(v, list):
                        for item in v:
                            walk_stack.append(item)
                    elif isinstance(v, dict):
                        for item in v.values():
                            walk_stack.append(item)
        urls = list(dict.fromkeys(urls))
        if urls:
            sections.append("### Assets\n" + "\n".join([f"- {u}" for u in urls]))

    return ("\n\n" + "\n\n".join(sections)) if sections else ""


