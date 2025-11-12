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
				"name": "image.dispatch",
				"parameters": {
					"type": "object",
					"properties": {
						"prompt": {"type": "string"},
						"negative": {"type": "string"},
						"width": {"type": "integer"},
						"height": {"type": "integer"},
						"steps": {"type": "integer"},
						"cfg": {"type": "number"},
						"sampler": {"type": "string"},
						"scheduler": {"type": "string"},
						"seed": {"type": "integer"},
						"model": {"type": "string"},
						"workflow_path": {"type": "string"}
					},
					"required": ["prompt"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "tts.speak",
				"parameters": {
					"type": "object",
					"properties": {
						"text": {"type": "string"},
						"voice": {"type": "string"}
					},
					"required": ["text"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "music.compose",
				"parameters": {
					"type": "object",
					"properties": {
						"prompt": {"type": "string"},
						"length_s": {"type": "integer"},
						"seed": {"type": "integer"},
						"music_lock": {"type": "object"},
						"music_refs": {"type": "object"}
					},
					"required": ["prompt"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "research.run",
				"parameters": {
					"type": "object",
					"properties": {
						"query": {"type": "string"},
						"job_id": {"type": "string"}
					},
					"required": ["query"]
				}
			}
		},
	]


