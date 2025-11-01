from __future__ import annotations

from typing import List, Dict, Any


def build_timeline(shots: List[Dict[str, Any]], final_duration: int) -> Dict[str, Any]:
    t = 0
    events = []
    for s in (shots or []):
        dur = int(s.get("duration_ms", 0))
        events.append({"at_ms": t, "id": s.get("id"), "dur_ms": dur})
        t += dur
    return {"duration_ms": t, "events": events}


def _fmt_srt_time(ms: int) -> str:
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(timeline: Dict[str, Any]) -> Dict[str, Any]:
    lines: List[str] = []
    idx = 1
    cur = 0
    for ev in (timeline.get("events") or []):
        start = ev.get("at_ms", cur)
        end = start + int(ev.get("dur_ms", 0))
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}")
        lines.append(f"{ev.get('id')}")
        lines.append("")
        idx += 1
        cur = end
    return {"path": "captions.srt", "text": "\n".join(lines)}


def export_edl(timeline: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal JSON EDL containing events and total duration
    return {"path": "edl.json", "timeline": timeline}


