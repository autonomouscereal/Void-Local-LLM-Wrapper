from __future__ import annotations

from typing import Any, Dict, Tuple

from ...analysis.media import analyze_image, analyze_image_regions


def analyze_image_with_regions(*, src: str, prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global_info = analyze_image(src, prompt)
    region_info = analyze_image_regions(src, prompt, global_info)
    return (global_info if isinstance(global_info, dict) else {}), (region_info if isinstance(region_info, dict) else {})


def build_image_qa_block(*, global_info: Dict[str, Any], region_info: Dict[str, Any]) -> Dict[str, Any]:
    qa_block: Dict[str, Any] = {}
    sb = global_info.get("score") if isinstance(global_info.get("score"), dict) else {}
    sem = global_info.get("semantics") if isinstance(global_info.get("semantics"), dict) else {}
    qa_block["overall"] = float(sb.get("overall") or 0.0)
    qa_block["semantic"] = float(sb.get("semantic") or 0.0)
    qa_block["technical"] = float(sb.get("technical") or 0.0)
    qa_block["aesthetic"] = float(sb.get("aesthetic") or 0.0)
    cs = sem.get("clip_score")
    if isinstance(cs, (int, float)):
        qa_block["clip_score"] = float(cs)
    agg = region_info.get("aggregates") if isinstance(region_info.get("aggregates"), dict) else {}
    fl = agg.get("face_lock")
    if isinstance(fl, (int, float)):
        qa_block["face_lock"] = float(fl)
    il = agg.get("id_lock")
    if isinstance(il, (int, float)):
        qa_block["id_lock"] = float(il)
    hr = agg.get("hands_ok_ratio")
    if isinstance(hr, (int, float)):
        qa_block["hands_ok_ratio"] = float(hr)
    tr = agg.get("text_readable_lock")
    if isinstance(tr, (int, float)):
        qa_block["text_readable_lock"] = float(tr)
    bq = agg.get("background_quality")
    if isinstance(bq, (int, float)):
        qa_block["background_quality"] = float(bq)
    return qa_block




