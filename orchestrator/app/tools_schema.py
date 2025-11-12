from __future__ import annotations

from typing import Any, Dict, List


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
						"order": {"type": "integer"}
					},
					"required": ["expr"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "film.run",
				"parameters": {
					"type": "object",
					"properties": {
						"title": {"type": "string"},
						"duration_s": {"type": "integer"},
						"seed": {"type": "integer"},
						"style_refs": {"type": "array", "items": {"type": "string"}},
						"character_images": {"type": "array", "items": {"type": "object"}},
						"res": {"type": "string"},
						"refresh": {"type": "integer"},
						"base_fps": {"type": "integer"},
						"codec": {"type": "string"},
						"container": {"type": "string"},
						"post": {"type": "object"},
						"audio": {"type": "object"},
						"safety": {"type": "object"}
					},
					"required": []
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "creative.alt_takes",
				"parameters": {"type": "object", "properties": {"tool": {"type": "string"}, "args": {"type": "object"}, "n": {"type": "integer"}}, "required": ["tool", "args"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "creative.pro_polish",
				"parameters": {"type": "object", "properties": {"kind": {"type": "string"}, "src": {"type": "string"}, "strength": {"type": "number"}, "cid": {"type": "string"}}, "required": ["kind", "src"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "creative.repro_pack",
				"parameters": {"type": "object", "properties": {"tool": {"type": "string"}, "args": {"type": "object"}, "artifact_path": {"type": "string"}, "cid": {"type": "string"}}, "required": ["tool", "args"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.song.yue",
				"parameters": {"type": "object", "properties": {"lyrics": {"type": "string"}, "style_tags": {"type": "array", "items": {"type": "string"}}, "bpm": {"type": "integer"}, "key": {"type": "string"}, "seed": {"type": "integer"}, "reference_song": {"type": "string"}, "infinite": {"type": "boolean"}}, "required": ["lyrics"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.melody.musicgen",
				"parameters": {"type": "object", "properties": {"text": {"type": "string"}, "melody_wav": {"type": "string"}, "bpm": {"type": "integer"}, "key": {"type": "string"}, "seed": {"type": "integer"}, "style_tags": {"type": "array", "items": {"type": "string"}}, "length_s": {"type": "integer"}, "infinite": {"type": "boolean"}}, "required": []}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.timed.sao",
				"parameters": {"type": "object", "properties": {"text": {"type": "string"}, "seconds": {"type": "integer"}, "bpm": {"type": "integer"}, "seed": {"type": "integer"}, "genre_tags": {"type": "array", "items": {"type": "string"}}}, "required": ["text", "seconds"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "voice.sing.diffsinger.rvc",
				"parameters": {"type": "object", "properties": {"lyrics": {"type": "string"}, "notes_midi": {"type": "string"}, "melody_wav": {"type": "string"}, "target_voice_ref": {"type": "string"}, "seed": {"type": "integer"}}, "required": ["lyrics"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "audio.stems.demucs",
				"parameters": {"type": "object", "properties": {"mix_wav": {"type": "string"}, "stems": {"type": "array", "items": {"type": "string"}}}, "required": ["mix_wav"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "audio.vc.rvc",
				"parameters": {"type": "object", "properties": {"source_vocal_wav": {"type": "string"}, "target_voice_ref": {"type": "string"}}, "required": ["source_vocal_wav", "target_voice_ref"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "audio.foley.hunyuan",
				"parameters": {"type": "object", "properties": {"video_ref": {"type": "string"}, "cue_regions": {"type": "array", "items": {"type": "object"}}, "style_tags": {"type": "array", "items": {"type": "string"}}}, "required": ["video_ref"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "rag_search",
				"parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "web_search",
				"parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "source_fetch",
				"parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "web.smart_get",
				"parameters": {
					"type": "object",
					"properties": {
						"url": {"type": "string"},
						"modes": {"type": "object"},
						"headers": {"type": "object"}
					},
					"required": ["url"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "metasearch.fuse",
				"parameters": {"type": "object", "properties": {"q": {"type": "string"}, "k": {"type": "integer"}}, "required": ["q"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.dispatch",
				"parameters": {"type": "object", "properties": {"mode": {"type": "string"}, "prompt": {"type": "string"}, "scale": {"type": "integer"}}, "required": []}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.edit",
				"parameters": {"type": "object", "properties": {"image_ref": {"type": "string"}, "mask_ref": {"type": "string"}, "prompt": {"type": "string"}, "negative": {"type": "string"}, "size": {"type": "string"}, "seed": {"type": "integer"}, "refs": {"type": "object"}, "cid": {"type": "string"}}, "required": ["image_ref", "prompt"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.upscale",
				"parameters": {"type": "object", "properties": {"image_ref": {"type": "string"}, "scale": {"type": "integer"}, "denoise": {"type": "number"}, "seed": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["image_ref"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.super_gen",
				"parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "size": {"type": "string"}, "refs": {"type": "object"}, "seed": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["prompt"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.dispatch",
				"parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": []}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.compose",
				"parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "bpm": {"type": "integer"}, "length_s": {"type": "integer"}, "structure": {"type": "array", "items": {"type": "string"}}, "sample_rate": {"type": "integer"}, "channels": {"type": "integer"}, "music_id": {"type": "string"}, "music_refs": {"type": "object"}, "seed": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["prompt"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.variation",
				"parameters": {"type": "object", "properties": {"variation_of": {"type": "string"}, "n": {"type": "integer"}, "intensity": {"type": "number"}, "music_id": {"type": "string"}, "music_refs": {"type": "object"}, "seed": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["variation_of"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.mixdown",
				"parameters": {"type": "object", "properties": {"stems": {"type": "array", "items": {"type": "object"}}, "sample_rate": {"type": "integer"}, "channels": {"type": "integer"}, "seed": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["stems"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.interpolate",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "target_fps": {"type": "integer"}}, "required": ["src"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.flow.derive",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "frame_a": {"type": "string"}, "frame_b": {"type": "string"}, "step": {"type": "integer"}, "cid": {"type": "string"}}, "required": []}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.hv.t2v",
				"parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "negative": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "fps": {"type": "integer"}, "seconds": {"type": "integer"}, "seed": {"type": "integer"}, "locks": {"type": "object"}, "post": {"type": "object"}, "latent_reinit_every": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["prompt", "width", "height", "fps", "seconds"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.hv.i2v",
				"parameters": {"type": "object", "properties": {"init_image": {"type": "string"}, "prompt": {"type": "string"}, "negative": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "fps": {"type": "integer"}, "seconds": {"type": "integer"}, "seed": {"type": "integer"}, "locks": {"type": "object"}, "post": {"type": "object"}, "latent_reinit_every": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["init_image", "prompt", "width", "height", "fps", "seconds"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.svd.i2v",
				"parameters": {"type": "object", "properties": {"init_image": {"type": "string"}, "prompt": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "fps": {"type": "integer"}, "seconds": {"type": "integer"}, "seed": {"type": "integer"}, "cid": {"type": "string"}}, "required": ["init_image", "prompt", "width", "height", "fps", "seconds"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.upscale",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "scale": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["src"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.text.overlay",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "texts": {"type": "array", "items": {"type": "object"}}, "cid": {"type": "string"}}, "required": ["src", "texts"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.cleanup",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "denoise": {"type": "boolean"}, "sharpen": {"type": "boolean"}, "dehalo": {"type": "boolean"}, "clahe": {"type": "boolean"}, "cid": {"type": "string"}}, "required": ["src"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.cleanup",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "denoise": {"type": "boolean"}, "deband": {"type": "boolean"}, "sharpen": {"type": "boolean"}, "stabilize_faces": {"type": "boolean"}, "cid": {"type": "string"}}, "required": ["src"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.artifact_fix",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "type": {"type": "string"}, "target_time": {"type": "string"}, "region": {"type": "array", "items": {"type": "integer"}}, "cid": {"type": "string"}}, "required": ["src", "type"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.artifact_fix",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "type": {"type": "string"}, "target_time": {"type": "string"}, "region": {"type": "array", "items": {"type": "integer"}}, "cid": {"type": "string"}}, "required": ["src", "type"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.hands.fix",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "cid": {"type": "string"}}, "required": ["src"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "video.hands.fix",
				"parameters": {"type": "object", "properties": {"src": {"type": "string"}, "cid": {"type": "string"}}, "required": ["src"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "tts.speak",
				"parameters": {"type": "object", "properties": {"text": {"type": "string"}, "voice": {"type": "string"}}, "required": ["text"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "research.run",
				"parameters": {"type": "object", "properties": {"query": {"type": "string"}, "scope": {"type": "string"}}, "required": ["query"]}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "code.super_loop",
				"parameters": {"type": "object", "properties": {"task": {"type": "string"}, "repo_root": {"type": "string"}}, "required": ["task"]}
			}
		}
	]


