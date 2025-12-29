from __future__ import annotations

from typing import List, Dict, Any
import re


def build_timeline(shots: List[Dict[str, Any]], final_duration: int) -> Dict[str, Any]:
    t = 0
    events = []
    for s in (shots or []):
        dur = int(s.get("duration_ms", 0))
        shot_id = s.get("shot_id") or s.get("id")
        events.append({"at_ms": t, "shot_id": shot_id, "dur_ms": dur})
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
        lines.append(f"{ev.get('shot_id') or ev.get('id') or ''}")
        lines.append("")
        idx += 1
        cur = end
    return {"path": "captions.srt", "text": "\n".join(lines)}


def export_edl(timeline: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal JSON EDL containing events and total duration
    return {"path": "edl.json", "timeline": timeline}


def text_to_simple_srt(text: str, duration_seconds: int) -> str:
    """
    Naive SRT generator for a block of text: split into sentences and
    distribute the total duration evenly.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]
    n = max(1, len(sentences))
    dur = max(1.0, float(duration_seconds))
    per = dur / n

    def _fmt(t: float) -> str:
        h = int(t // 3600)
        t -= 3600 * h
        m = int(t // 60)
        t -= 60 * m
        s = int(t)
        ms = int((t - s) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    for i, s in enumerate(sentences, 1):
        start = per * (i - 1)
        end = per * i
        lines.append(str(i))
        lines.append(f"{_fmt(start)} --> {_fmt(end)}")
        lines.append(s)
        lines.append("")
    return "\n".join(lines)


