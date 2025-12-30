from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Iterable, Set


def get_builtin_tools_schema() -> List[Dict[str, Any]]:
    """
    OpenAI-style tools schema exposed for planner/catalog purposes.
    This mirrors the built-in tool contracts used elsewhere in the orchestrator.
    Keep this file data-only to avoid circular imports.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "math.eval",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expr": {"type": "string"},
                        "task": {"type": "string"},
                        "var": {"type": "string"},
                        "point": {"type": "number"},
                        "order": {"type": "integer"},
                    },
                    "required": ["expr"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "api.request",
                "description": "Deprecated alias for http.request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]},
                        "headers": {"type": "object"},
                        "params": {"type": "object"},
                        "body": {"oneOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}, {"type": "null"}]},
                        "expect": {"type": "string", "enum": ["json", "text", "bytes"]},
                        "follow_redirects": {"type": "boolean"},
                        "max_bytes": {"type": "integer"},
                    },
                    "required": ["url", "method"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "film2.run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "clips": {"type": "array", "items": {"type": "string"}},
                        "images": {"type": "array", "items": {"type": "string"}},
                        "interpolate": {"type": "boolean"},
                        "scale": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
                        "quality_profile": {"type": "string"},
                        "locks": {"type": "object"},
                        "conversation_id": {"type": "string"},
                        "trace_id": {"type": "string"},
                        "film_id": {"type": "string"},
                        "duration_seconds": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
                        "fps": {"type": "integer"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "character_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "creative.alt_takes",
                "parameters": {
                    "type": "object",
                    "properties": {"tool": {"type": "string"}, "args": {"type": "object"}, "n": {"type": "integer"}},
                    "required": ["tool", "args"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "creative.pro_polish",
                "parameters": {
                    "type": "object",
                    "properties": {"kind": {"type": "string"}, "src": {"type": "string"}, "strength": {"type": "number"}},
                    "required": ["kind", "src"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "creative.repro_pack",
                "parameters": {
                    "type": "object",
                    "properties": {"tool": {"type": "string"}, "args": {"type": "object"}, "artifact_path": {"type": "string"}},
                    "required": ["tool", "args"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "music.melody.musicgen",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "melody_wav": {"type": "string"},
                        "bpm": {"type": "integer"},
                        "key": {"type": "string"},
                        "seed": {"type": "integer"},
                        "style_tags": {"type": "array", "items": {"type": "string"}},
                        "length_s": {"type": "integer"},
                        "infinite": {"type": "boolean"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "music.timed.sao",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "seconds": {"type": "integer"},
                        "bpm": {"type": "integer"},
                        "seed": {"type": "integer"},
                        "genre_tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "seconds"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "voice.sing.diffsinger.rvc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lyrics": {"type": "string"},
                        "notes_midi": {"type": "string"},
                        "melody_wav": {"type": "string"},
                        "target_voice_ref": {"type": "string"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["lyrics"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audio.stems.demucs",
                "parameters": {
                    "type": "object",
                    "properties": {"mix_wav": {"type": "string"}, "stems": {"type": "array", "items": {"type": "string"}}},
                    "required": ["mix_wav"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audio.stems.adjust",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mix_wav": {"type": "string"},
                        "stem_gains": {"type": "object"},
                        "stems": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["mix_wav"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audio.vocals.transform",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mix_wav": {"type": "string"},
                        "pitch_shift_semitones": {"type": "number"},
                        "voice_lock_id": {"type": "string"},
                    },
                    "required": ["mix_wav"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audio.vc.rvc",
                "parameters": {
                    "type": "object",
                    "properties": {"source_vocal_wav": {"type": "string"}, "target_voice_ref": {"type": "string"}},
                    "required": ["source_vocal_wav", "target_voice_ref"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "music.lyrics.align",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_path": {"type": "string"},
                        "lyrics_sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "section_id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "lines": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {"line_id": {"type": "string"}, "text": {"type": "string"}},
                                            "required": ["line_id", "text"],
                                        },
                                    },
                                },
                                "required": ["section_id", "lines"],
                            },
                        },
                        "changed_line_ids": {"type": "array", "items": {"type": "string"}},
                        "character_id": {"type": "string"},
                        "lock_bundle": {"type": "object"},
                    },
                    "required": ["audio_path", "lyrics_sections"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audio.foley.hunyuan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_ref": {"type": "string"},
                        "cue_regions": {"type": "array", "items": {"type": "object"}},
                        "style_tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["video_ref"],
                },
            },
        },
        {"type": "function", "function": {"name": "rag_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "source_fetch", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "query": {"type": "string"},
                        "k": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "metasearch.fuse",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "k": {"type": "integer"},
                    },
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "http.request",
                "description": "Call an external HTTP API and normalize the response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                        "headers": {"type": "object"},
                        "query": {"type": "object"},
                        "body": {"oneOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}]},
                        "expect_json": {"type": "boolean"},
                    },
                    "required": ["url", "method"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web.smart_get",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}, "modes": {"type": "object"}, "headers": {"type": "object"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image.dispatch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "negative": {"type": "string"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "size": {"type": "string"},
                        "steps": {"type": "integer"},
                        "cfg": {"type": "number"},
                        "seed": {"type": "integer"},
                        "mode": {"type": "string", "enum": ["gen", "edit", "upscale"]},
                        "assets": {"type": "object"},
                        "trace_id": {"type": "string"},
                        "quality_profile": {"type": "string"},
                        "lock_bundle": {"type": "object"},
                        "sampler": {"type": "string"},
                        "sampler_name": {"type": "string"},
                        "scheduler": {"type": "string"},
                        "workflow_path": {"type": "string"},
                        "workflow_graph": {"type": "object"},
                        "autofix_422": {"type": "boolean", "default": True},
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image.refine.segment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "prompt": {"type": "string"},
                        "lock_bundle": {"type": "object"},
                        "quality_profile": {"type": "string"},
                        "refine_mode": {"type": "string"},
                        "seed": {"type": "integer"},
                        "source_image": {"type": "string"},
                    },
                    "required": ["segment_id"],
                },
            },
        },
        {"type": "function", "function": {"name": "image.edit", "parameters": {"type": "object", "properties": {"image_ref": {"type": "string"}, "mask_ref": {"type": "string"}, "prompt": {"type": "string"}, "negative": {"type": "string"}, "size": {"type": "string"}, "seed": {"type": "integer"}, "refs": {"type": "object"}}, "required": ["image_ref", "prompt"]}}},
        {"type": "function", "function": {"name": "image.upscale", "parameters": {"type": "object", "properties": {"image_ref": {"type": "string"}, "scale": {"type": "integer"}, "denoise": {"type": "number"}, "seed": {"type": "integer"}}, "required": ["image_ref"]}}},
        {"type": "function", "function": {"name": "image.super_gen", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "size": {"type": "string"}, "refs": {"type": "object"}, "seed": {"type": "integer"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "music.variation", "parameters": {"type": "object", "properties": {"variation_of": {"type": "string"}, "n": {"type": "integer"}, "intensity": {"type": "number"}, "artifact_id": {"type": "string"}, "music_refs": {"type": "object"}, "seed": {"type": "integer"}}, "required": ["variation_of"]}}},
        {"type": "function", "function": {"name": "music.mixdown", "parameters": {"type": "object", "properties": {"stems": {"type": "array", "items": {"type": "object"}}, "sample_rate": {"type": "integer"}, "channels": {"type": "integer"}, "seed": {"type": "integer"}}, "required": ["stems"]}}},
        {
            "type": "function",
            "function": {
                "name": "music.infinite.windowed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "length_s": {"type": "integer"},
                        "bpm": {"type": "integer"},
                        "key": {"type": "string"},
                        "window_bars": {"type": "integer"},
                        "overlap_bars": {"type": "integer"},
                        "mode": {"type": "string"},
                        "extra_length_s": {"type": "integer"},
                        "character_id": {"type": "string"},
                        "lock_bundle": {"type": "object"},
                        "quality_profile": {"type": "string"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["prompt", "length_s"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "music.refine.window",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "window_id": {"type": "string"},
                        "lock_bundle": {"type": "object"},
                        "quality_profile": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["segment_id"],
                },
            },
        },
        {"type": "function", "function": {"name": "video.interpolate", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "target_fps": {"type": "integer"}}, "required": ["src"]}}},
        {"type": "function", "function": {"name": "video.flow.derive", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "frame_a": {"type": "string"}, "frame_b": {"type": "string"}, "step": {"type": "integer"}}, "required": []}}},
        {"type": "function", "function": {"name": "video.hv.t2v", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "negative": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "fps": {"type": "integer"}, "seconds": {"type": "integer"}, "seed": {"type": "integer"}, "locks": {"type": "object"}, "post": {"type": "object"}, "latent_reinit_every": {"type": "integer"}, "conversation_id": {"type": "string"}, "trace_id": {"type": "string"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "video.hv.i2v", "parameters": {"type": "object", "properties": {"init_image": {"type": "string"}, "prompt": {"type": "string"}, "negative": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "fps": {"type": "integer"}, "seconds": {"type": "integer"}, "seed": {"type": "integer"}, "locks": {"type": "object"}, "post": {"type": "object"}, "latent_reinit_every": {"type": "integer"}, "conversation_id": {"type": "string"}, "trace_id": {"type": "string"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "video.upscale", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "scale": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["src"]}}},
        {"type": "function", "function": {"name": "video.text.overlay", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "texts": {"type": "array", "items": {"type": "object"}}}, "required": ["src", "texts"]}}},
        {"type": "function", "function": {"name": "image.cleanup", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "denoise": {"type": "boolean"}, "sharpen": {"type": "boolean"}, "dehalo": {"type": "boolean"}, "clahe": {"type": "boolean"}}, "required": ["src"]}}},
        {"type": "function", "function": {"name": "video.cleanup", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "denoise": {"type": "boolean"}, "deband": {"type": "boolean"}, "sharpen": {"type": "boolean"}, "stabilize_faces": {"type": "boolean"}}, "required": ["src"]}}},
        {"type": "function", "function": {"name": "image.artifact_fix", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "type": {"type": "string"}, "target_time": {"type": "string"}, "region": {"type": "array", "items": {"type": "integer"}}}, "required": ["src", "type"]}}},
        {"type": "function", "function": {"name": "video.artifact_fix", "parameters": {"type": "object", "properties": {"src": {"type": "string"}, "type": {"type": "string"}, "target_time": {"type": "string"}, "region": {"type": "array", "items": {"type": "integer"}}}, "required": ["src", "type"]}}},
        {"type": "function", "function": {"name": "image.hands.fix", "parameters": {"type": "object", "properties": {"src": {"type": "string"}}, "required": ["src"]}}},
        {"type": "function", "function": {"name": "video.hands.fix", "parameters": {"type": "object", "properties": {"src": {"type": "string"}}, "required": ["src"]}}},
        {
            "type": "function",
            "function": {
                "name": "tts.speak",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "voice": {"type": "string"},
                        "voice_id": {"type": "string"},
                        "voice_refs": {"type": "object"},
                        "rate": {"oneOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}]},
                        "pitch": {"oneOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}]},
                        "sample_rate": {"type": "integer"},
                        "max_seconds": {"type": "integer"},
                        "seed": {"type": "integer"},
                        "conversation_id": {"type": "string"},
                        "edge": {"type": "boolean"},
                        "language": {"type": "string"},
                        "voice_gender": {"type": "string"},
                        "trace_id": {"type": "string"},
                        "lock_bundle": {"type": "object"},
                        "quality_profile": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audio.sfx.compose",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "length_s": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
                        "pitch": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
                        "seed": {"type": "integer"},
                        "conversation_id": {"type": "string"},
                        "trace_id": {"type": "string"},
                        "lock_bundle": {"type": "object"},
                        "sfx_event_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [],
                },
            },
        },
        {"type": "function", "function": {"name": "research.run", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "scope": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "code.super_loop", "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "repo_root": {"type": "string"}}, "required": ["task"]}}},
        {
            "type": "function",
            "function": {
                "name": "locks.build_image_bundle",
                "description": "Build an identity/style/pose lock bundle from a reference image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "character_id": {"type": "string"},
                        "image_url": {"type": "string"},
                        "options": {
                            "type": "object",
                            "properties": {
                                "detect_pose": {"type": "boolean"},
                                "extract_style": {"type": "boolean"},
                                "face_strength": {"type": "number"},
                            },
                        },
                    },
                    "required": ["character_id", "image_url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "locks.build_video_bundle",
                "description": "Build an identity lock bundle from a reference video by sampling frames and aggregating face embeddings.",
                "parameters": {
                    "type": "object",
                    "properties": {"character_id": {"type": "string"}, "video_path": {"type": "string"}, "max_frames": {"type": "integer"}},
                    "required": ["character_id", "video_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "locks.build_audio_bundle",
                "description": "Build a voice lock bundle from a reference audio clip.",
                "parameters": {"type": "object", "properties": {"character_id": {"type": "string"}, "audio_url": {"type": "string"}}, "required": ["character_id", "audio_url"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "locks.get_bundle",
                "description": "Fetch a previously stored lock bundle for a character.",
                "parameters": {"type": "object", "properties": {"character_id": {"type": "string"}}, "required": ["character_id"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "locks.build_region_locks",
                "description": "Build region-level locks such as clothing, props, textures, or scenery from a reference image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "character_id": {"type": "string"},
                        "image_url": {"type": "string"},
                        "regions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "region_id": {"type": "string"},
                                    "role": {"type": "string"},
                                    "bbox": {"type": "array", "items": {"type": "number"}},
                                    "lock_mode": {"type": "object"},
                                    "strength": {"type": "number"},
                                },
                            },
                        },
                    },
                    "required": ["character_id", "image_url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "locks.update_region_modes",
                "description": "Update lock modes or strengths for existing regions in a lock bundle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "character_id": {"type": "string"},
                        "updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "region_id": {"type": "string"},
                                    "lock_mode": {"type": "object"},
                                    "strength": {"type": "number"},
                                    "color_palette": {"type": "object"},
                                },
                                "required": ["region_id"],
                            },
                        },
                    },
                    "required": ["character_id", "updates"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "locks.update_audio_modes",
                "description": "Update audio lock targets such as tempo, key, stems, or lyric segments.",
                "parameters": {
                    "type": "object",
                    "properties": {"character_id": {"type": "string"}, "update": {"type": "object"}},
                    "required": ["character_id", "update"],
                },
            },
        },
    ]


# ---- Tool Introspection (data-only registry) ----
_INTROSPECTION_DEFAULT_NAMES: List[str] = [
    "image.dispatch",
    "film2.run",
    "music.infinite.windowed",
    "tts.speak",
    "audio.sfx.compose",
]

_INTROSPECTION_NOTES: Dict[str, str] = {
    "image.dispatch": "Comfy pipeline; requires either size or width/height. Returns ids.image_id and meta.{data_url,view_url,orch_view_url}.",
    "film2.run": (
        "Unified Film-2 front door for Void. Planner MUST treat this as the final assembly step in a film pipeline. "
        "Do NOT fabricate or hard-code clip/image URLs in args; the orchestrator automatically wires real image "
        "artifacts from prior image.dispatch (and related) tool results into args.images at execution time. "
        "If no images exist yet for this trace, film2.run will operate in prompt/story-driven mode only. "
        "Returns ids.film_id and meta with shots and view_url."
    ),
    "music.infinite.windowed": "Windowed music composition front door. Returns meta with song_graph, windows, and final mixed track artifact.",
    "tts.speak": "TTS front door. Supports voice_id + optional voice_refs for matching/training. Returns ids.audio_id and meta.{data_url,url,mime,duration_s}.",
    "audio.sfx.compose": "SFX generator (builtin). Produces a short WAV artifact based on type/length/pitch and optional lock hints.",
}

_INTROSPECTION_EXAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "api.request": [{"args": {"url": "https://httpbin.org/get", "method": "GET", "params": {"foo": "bar"}}}],
}


def _infer_tool_kind(name: str) -> str:
    nm = str(name or "")
    if nm.startswith("film2") or nm.startswith("video."):
        return "video"
    if nm.startswith("image."):
        return "image"
    if nm.startswith("music.") or nm.startswith("audio.") or nm.startswith("tts.") or nm.startswith("voice."):
        return "audio"
    return "utility"


def get_tool_introspection_registry(tool_names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Returns a stable metadata registry used by /tool.list, /tool.describe, and
    planner TSL/catalog blocks.

    IMPORTANT: This registry is derived from get_builtin_tools_schema() so the
    JSON Schema contracts remain single-sourced.
    """
    names = list(tool_names) if tool_names is not None else list(_INTROSPECTION_DEFAULT_NAMES)
    builtins = get_builtin_tools_schema() or []
    by_name: Dict[str, Dict[str, Any]] = {}
    for t in builtins:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        nm = fn.get("name") if isinstance(fn.get("name"), str) else ""
        if nm:
            by_name[nm] = t

    out: Dict[str, Dict[str, Any]] = {}
    for nm in names:
        nm_clean = str(nm or "").strip()
        if not nm_clean:
            continue
        t = by_name.get(nm_clean)
        if not t:
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        out[nm_clean] = {
            "tool_name": nm_clean,
            "name": nm_clean,
            "version": "1",
            "kind": _infer_tool_kind(nm_clean),
            "schema": params,
            "notes": _INTROSPECTION_NOTES.get(nm_clean, ""),
            "examples": _INTROSPECTION_EXAMPLES.get(nm_clean, []),
        }
    return out


# ---- Tool schema helpers (kept here so main.py doesn't duplicate logic) ----
def tool_expected_from_jsonschema(js_schema: Any) -> Dict[str, Any]:
    """
    Convert a JSON Schema parameters block into the simpler expected_schema
    format used by committee_jsonify (field -> python_type).
    """
    expected: Dict[str, Any] = {}
    if not isinstance(js_schema, dict):
        return expected
    props = js_schema.get("properties") or {}
    if not isinstance(props, dict):
        return expected
    for key, spec in props.items():
        t = spec.get("type") if isinstance(spec, dict) else None
        # Normalize union types like ["integer","null"] to a single primary type.
        if isinstance(t, list):
            # Prefer non-null, non-array/object/string if present.
            if "integer" in t:
                t = "integer"
            elif "number" in t:
                t = "number"
            elif "string" in t:
                t = "string"
            elif "boolean" in t:
                t = "boolean"
            elif "object" in t:
                t = "object"
            elif "array" in t:
                t = "array"
            else:
                t = t[0] if t else None
        if t == "integer":
            expected[key] = int
        elif t == "number":
            expected[key] = float
        elif t == "boolean":
            expected[key] = bool
        elif t == "object":
            expected[key] = dict
        elif t == "array":
            expected[key] = list
        else:
            # Default to string for unknown/nullable types.
            expected[key] = str
    return expected


def compute_tools_hash(
    body: Dict[str, Any],
    *,
    planner_visible_only: bool,
    planner_visible_tools: Optional[Set[str]] = None,
) -> str:
    """
    Compute a deterministic hash of tool names visible to the planner or executor.

    planner_visible_tools is passed in by the caller (main.py) to avoid imports
    that could create circular dependencies.
    """
    names: set[str] = set()
    client_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for t in (client_tools or []):
        if isinstance(t, dict):
            nm = (t.get("function") or {}).get("name") or t.get("name")
            if isinstance(nm, str):
                nm_clean = nm.strip()
                if not nm_clean:
                    continue
                if planner_visible_only and planner_visible_tools is not None and nm_clean not in planner_visible_tools:
                    continue
                names.add(nm_clean)
    builtins = get_builtin_tools_schema()
    for t in (builtins or []):
        fn = (t.get("function") or {})
        nm = fn.get("name")
        if isinstance(nm, str):
            nm_clean = nm.strip()
            if not nm_clean:
                continue
            if planner_visible_only and planner_visible_tools is not None and nm_clean not in planner_visible_tools:
                continue
            names.add(nm_clean)
    src = "|".join(sorted(list(names)))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def build_tools_section(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return ""
    return "Available tools (JSON schema):\n" + json.dumps(tools, indent=2, default=str)


def merge_tool_schemas(client_tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge client-provided tool schemas with builtin schemas, keyed by function name.
    """
    builtins = get_builtin_tools_schema()
    if not client_tools:
        return builtins
    seen = {((t.get("function") or {}).get("name") or t.get("name")) for t in client_tools if isinstance(t, dict)}
    out = list(client_tools)
    for t in builtins:
        name = (t.get("function") or {}).get("name") or t.get("name")
        if name not in seen:
            out.append(t)
    return out


def build_compact_tool_catalog() -> str:
    """
    Build a strict, data-only catalog summarizing tool names and required args.
    """
    builtins = get_builtin_tools_schema()
    merged: dict[str, dict] = {}
    for t in (builtins or []):
        fn = (t.get("function") or {}) if isinstance(t, dict) else {}
        tool_name = fn.get("name")
        if not tool_name:
            continue
        params = (fn.get("parameters") or {})
        reqs = list((params.get("required") or [])) if isinstance(params, dict) else []
        merged[tool_name] = {"tool_name": tool_name, "required_args": reqs}
    tools_list: list[dict] = list(merged.values())
    ranked_tools: list[tuple[str, int, dict]] = []
    for tool_index, tool_entry in enumerate(tools_list):
        ranked_tools.append((str(tool_entry.get("tool_name") or ""), int(tool_index), tool_entry))
    ranked_tools.sort()
    tools_list = [ranked[2] for ranked in ranked_tools]
    names_list = [t.get("tool_name") for t in tools_list if isinstance(t, dict) and isinstance(t.get("tool_name"), str)]
    catalog = {
        "names": names_list,
        "tools": tools_list,
        "constraints": {
            "routing": "All tools run via executor→orchestrator /tool.run (no fast paths).",
            "args": "Planner must emit all required args; snap sizes to /8.",
            "rules": ["Use only tool names present in the catalog. If none apply, return steps: []."],
        },
    }
    return "Tool catalog (strict, data-only):\n" + json.dumps(catalog, indent=2, default=str) + "\nValid tool names: " + ", ".join(names_list)
