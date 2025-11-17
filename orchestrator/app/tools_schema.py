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
				"name": "api.request",
				"description": "Deprecated alias for http.request.",
				"parameters": {
					"type": "object",
					"properties": {
						"url": {"type": "string"},
						"method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
						"headers": {"type": "object"},
						"query": {"type": "object"},
						"body": {
							"oneOf": [
								{"type": "string"},
								{"type": "object"},
								{"type": "array"}
							]
						},
						"expect_json": {"type": "boolean"}
					},
					"required": ["url", "method"]
				}
			}
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
						"images": {"type": "array", "items": {"type": "object"}},
						"interpolate": {"type": "boolean"},
						"scale": {"type": "number"},
						"quality_profile": {"type": "string"},
						"locks": {"type": "object"},
						"cid": {"type": "string"},
						"trace_id": {"type": "string"}
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
				"name": "music.song.primary",
				"parameters": {"type": "object", "properties": {"lyrics": {"type": "string"}, "style_tags": {"type": "array", "items": {"type": "string"}}, "bpm": {"type": "integer"}, "key": {"type": "string"}, "seed": {"type": "integer"}, "reference_song": {"type": "string"}, "infinite": {"type": "boolean"}}, "required": ["lyrics"]}
			}
		},
		{
			# Backwards-compatible alias for older plans/specs that still
			# reference the legacy song tool name. Internally this is routed
			# through the same primary music engine as music.song.primary.
			"type": "function",
			"function": {
				"name": "music.song.legacy",
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
				"name": "audio.stems.adjust",
				"parameters": {
					"type": "object",
					"properties": {
						"mix_wav": {"type": "string"},
						"stem_gains": {"type": "object"},
						"stems": {"type": "array", "items": {"type": "string"}},
						"cid": {"type": "string"}
					},
					"required": ["mix_wav"]
				}
			}
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
						"cid": {"type": "string"}
					},
					"required": ["mix_wav"]
				}
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
											"properties": {
												"line_id": {"type": "string"},
												"text": {"type": "string"}
											},
											"required": ["line_id", "text"]
										}
									}
								},
								"required": ["section_id", "lines"]
							}
						},
						"changed_line_ids": {"type": "array", "items": {"type": "string"}},
						"cid": {"type": "string"},
						"character_id": {"type": "string"},
						"lock_bundle": {"type": "object"}
					},
					"required": ["audio_path", "lyrics_sections"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "refs.music.build_profile",
				"parameters": {
					"type": "object",
					"properties": {
						"ref_ids": {"type": "array", "items": {"type": "string"}},
						"audio_paths": {"type": "array", "items": {"type": "string"}}
					},
					"required": []
				}
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
				"name": "source_fetch",
				"parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
			}
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
						"body": {
							"oneOf": [
								{"type": "string"},
								{"type": "object"},
								{"type": "array"}
							]
						},
						"expect_json": {"type": "boolean"}
					},
					"required": ["url", "method"]
				}
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
						"seed": {"type": "integer"},
						"sampler": {"type": "string"},
						"sampler_name": {"type": "string"},
						"scheduler": {"type": "string"},
						"workflow_path": {"type": "string"},
						"workflow_graph": {"type": "object"},
						"autofix_422": {"type": "boolean", "default": True}
					},
					"required": ["prompt"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "image.refine.segment",
				"parameters": {
					"type": "object",
					"properties": {
						"segment_id": {"type": "string"},
						"cid": {"type": "string"},
						"prompt": {"type": "string"},
						"lock_bundle": {"type": "object"},
						"quality_profile": {"type": "string"},
						"refine_mode": {"type": "string"},
						"seed": {"type": "integer"},
						"source_image": {"type": "string"}
					},
					"required": ["segment_id"]
				}
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
						"cid": {"type": "string"}
					},
					"required": ["prompt", "length_s"]
				}
			}
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
						"cid": {"type": "string"},
						"lock_bundle": {"type": "object"},
						"quality_profile": {"type": "string"},
						"reason": {"type": "string"}
					},
					"required": ["segment_id"]
				}
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
		},
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
								"face_strength": {"type": "number"}
							}
						}
					},
					"required": ["character_id", "image_url"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "locks.build_audio_bundle",
				"description": "Build a voice lock bundle from a reference audio clip.",
				"parameters": {
					"type": "object",
					"properties": {
						"character_id": {"type": "string"},
						"audio_url": {"type": "string"}
					},
					"required": ["character_id", "audio_url"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "locks.get_bundle",
				"description": "Fetch a previously stored lock bundle for a character.",
				"parameters": {
					"type": "object",
					"properties": {
						"character_id": {"type": "string"}
					},
					"required": ["character_id"]
				}
			}
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
									"strength": {"type": "number"}
								}
							}
						}
					},
					"required": ["character_id", "image_url"]
				}
			}
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
									"color_palette": {"type": "object"}
								},
								"required": ["region_id"]
							}
						}
					},
					"required": ["character_id", "updates"]
				}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "locks.update_audio_modes",
				"description": "Update audio lock targets such as tempo, key, stems, or lyric segments.",
				"parameters": {
					"type": "object",
					"properties": {
						"character_id": {"type": "string"},
						"update": {"type": "object"}
					},
					"required": ["character_id", "update"]
				}
			}
		}
	]


