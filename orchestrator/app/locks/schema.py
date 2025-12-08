from __future__ import annotations

LOCK_BUNDLE_SCHEMA = {
    # Schema version 3: extends the original v2 layout with a nested "visual"
    # branch while keeping legacy top-level keys for backwards compatibility.
    "schema_version": 3,
    "character_id": "string",

    # Legacy top-level fields (still supported and migrated into visual.* when present)
    "face": {
        "embedding": "base64 or list[float]",
        "mask": "optional path or base64",
        "image_path": "optional stored path",
        "strength": 0.75,
    },
    "pose": {
        "skeleton": "optional path/json",
        "strength": 0.7,
    },
    "style": {
        "prompt_tags": ["example tag"],
        "palette": {"primary": "#D00000", "accent": "#111111"},
    },
    "audio": {
        "voice_embedding": "optional base64 or list[float]",
        "timbre_tags": ["raspy", "female", "aggressive"],
        "tempo_bpm": 0.0,
        "tempo_lock_mode": "hard|soft|off",
        "key": "C minor",
        "key_lock_mode": "hard|soft|off",
        "stem_profile": {},
        "stem_lock_mode": "hard|soft|off",
        "lyrics_segments": [],
    },
    "regions": {},
    "scene": {
        "background_embedding": None,
        "camera_style_tags": [],
        "lighting_tags": [],
        "lock_mode": "soft",
    },

    # Visual branch (schema_version >= 3)
    #
    # This is a superset that can express:
    # - global style / scene
    # - faces and pose
    # - arbitrary entities (clothing, plants, objects, FX)
    # - per-entity constraints and references
    "visual": {
        "style": {
            # Global image style tags (brush, rendering style, etc)
            "style_tags": ["cel-shaded", "high-contrast", "neon"],
            # Optional style embeddings (e.g. CLIP/image encoder)
            "clip_style_embedding": "list[float]",
            # Color palette information
            "palette": {
                "primary": ["#FF0044", "#00E5FF"],
                "secondary": ["#101010", "#303030"],
                "accent": ["#FFFF00"],
            },
            "lighting": "rim-lit, backlight blue ambient",
            "noise_grain_level": 0.2,
            "constraints": {
                "lock_mode": "off|guide|hard",
                "lock_palette": True,
                "lock_lighting": True,
            },
        },
        "scene": {
            # Semantic / layout tags (city, night, rain, etc)
            "scene_tags": ["city", "night", "rain"],
            # Optional dense scene/layout embedding (e.g. DINOv2 / diffusion encoder)
            "layout_embedding": "list[float]",
            "depth_stats": {
                "mean_depth": 0.45,
                "variance": 0.12,
            },
            "background_entity_id": "bg_entity_id",
            "constraints": {
                "lock_mode": "off|guide|hard",
                "lock_layout": True,
                "lock_depth": True,
            },
        },
        "faces": [
            {
                "entity_id": "char_shadow_face",
                "role": "main_character_face",
                "priority": 10,
                "region": {
                    "mask_path": "/locks/.../shadow_face_mask.png",
                    # Normalized x, y, w, h
                    "bbox": [0.40, 0.20, 0.20, 0.20],
                    "z_index": 10,
                },
                "embeddings": {
                    "id_embedding": "list[float]",
                    "clip_embedding": "list[float]",
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_identity": True,
                    "lock_expression": False,
                    "lock_head_pose": False,
                    "lock_hair": True,
                    "allow_movement_px": 0,
                    "allow_scale_delta": 0.0,
                },
                "refs": [
                    {"image_path": "/locks/.../shadow_face_ref.png"},
                ],
            }
        ],
        "pose": {
            "skeleton": {
                "format": "COCO|custom",
                "keypoints": {
                    "nose": [0.0, 0.0],
                },
            },
            "constraints": {
                "lock_mode": "off|guide|hard",
                "max_joint_deviation_deg": 25.0,
            },
        },
        "entities": [
            {
                "entity_id": "entity_1",
                "entity_type": "clothing|object|plant|fx|background",
                "role": "boots|snail|flower|prop",
                "priority": 5,
                "region": {
                    "mask_path": "/locks/.../entity_mask.png",
                    "bbox": [0.0, 0.0, 0.1, 0.1],
                    "z_index": 0,
                },
                "embeddings": {
                    "clip": "list[float]",
                    "dino": "list[float]",
                    "texture": "list[float]",
                    "shape": "list[float]",
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_color": True,
                    "lock_texture": True,
                    "lock_shape": True,
                    "lock_style": False,
                    "allow_movement_px": 3,
                    "allow_scale_delta": 0.03,
                },
                "refs": [
                    {"image_path": "/locks/.../entity_ref.png"},
                ],
            }
        ],
        "extras": {
            "fx_tags": ["rain", "motion_trails", "glow"],
            "fx_strength": 0.6,
            "constraints": {
                "lock_mode": "off|guide|hard",
            },
        },
    },

    # Music branch (schema_version >= 3)
    #
    # This branch is a superset schema for controllable music generation and
    # editing. It is internal to music tools (film music, mixdown, etc.)
    # and is not exposed as a planner-visible tool surface.
    "music": {
        "global": {
            "music_id": "main_theme",
            "reference_tracks": [
                {
                    "role": "primary",
                    "description": "High-energy reference track",
                    "audio_path": "/refs/primary_track.wav",
                    "time_range": [0.0, 120.0],
                }
            ],
            "text_prompt": "High-energy hybrid electronic/rock song.",
            "genre_tags": ["electronic", "rock"],
            "mood_tags": ["aggressive", "epic"],
            "embeddings": {
                "mulan": "list[float]",
                "openl3": "list[float]",
                "musicgen_style": "list[float]",
            },
            "tempo_bpm": 120,
            "time_signature": "4/4",
            "key": "C minor",
            "scale": "natural_minor",
            "swing": 0.0,
            "structure_summary": {
                "bars_total": 64,
                "sections_order": ["intro", "verse_1", "chorus_1", "outro"],
            },
            "energy_envelope": [
                {"time": 0.0, "energy": 0.2},
                {"time": 32.0, "energy": 0.8},
            ],
            "loudness_envelope_lufs": [
                {"time": 0.0, "lufs": -18.0},
                {"time": 32.0, "lufs": -9.0},
            ],
            "constraints": {
                "lock_mode": "off|guide|hard",
                "lock_genre": True,
                "lock_tempo": True,
                "lock_key": True,
                "lock_energy_shape": True,
                "lock_loudness_shape": True,
            },
        },
        "voices": [
            {
                "voice_id": "voice_1",
                "role": "lead_vocal",
                "style_tags": ["female", "rock", "powerful"],
                "embeddings": {
                    "speaker": "list[float]",
                    "timbre": "list[float]",
                    "flow": "list[float]",
                },
                "range_midi": {"min": 55, "max": 81},
                "language": "en",
                "vocal_type": "sung|rap|spoken",
                "pitch_shift_allowed_semitones": 2,
                "formant_preservation": True,
                "flow_profile": {
                    "bpm_source": 120,
                    "syllables_per_second": 5.0,
                    "syncopation_index": 0.3,
                    "rhythm_pattern_embedding": "list[float]",
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_timbre": True,
                    "lock_flow": False,
                    "lock_range": True,
                    "lock_language": True,
                },
                "refs": [
                    {
                        "audio_path": "/refs/voice_1.wav",
                        "time_range": [0.0, 20.0],
                        "notes": "Lead vocal reference",
                    }
                ],
            }
        ],
        "instruments": [
            {
                "instrument_id": "drums_main",
                "role": "drum_kit",
                "tags": ["drums", "kit"],
                "source": {
                    "stem_path": "/refs/stems/drums_main.wav",
                    "separation_model": "demucs_v4",
                    "midi_path": None,
                },
                "pattern_features": {
                    "bars_analyzed": 64,
                    "kick_density_per_bar": [0.8],
                    "snare_backbeat_ratio": 0.95,
                    "hihat_division": "16th",
                    "fill_probability_per_8bars": 0.4,
                },
                "spectral_features": {
                    "mean_spectral_centroid": 3500.0,
                    "brightness": 0.9,
                    "transient_sharpness": 0.85,
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_presence": True,
                    "lock_density": False,
                    "lock_timbre": True,
                },
            }
        ],
        "motifs": [
            {
                "motif_id": "motif_1",
                "role": "vocal_hook|riff|drop",
                "source_section_id": "chorus_1",
                "time_range": [32.0, 40.0],
                "bar_range": [17, 24],
                "embeddings": {
                    "melody": "list[float]",
                    "chroma": "list[float]",
                    "rhythm": "list[float]",
                    "texture": "list[float]",
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "allow_transposition_semitones": 0,
                    "allow_time_stretch": 0.0,
                },
                "refs": [
                    {"audio_path": "/refs/motifs/motif_1.wav"},
                ],
            }
        ],
        "sections": [
            {
                "section_id": "intro",
                "type": "intro|verse|chorus|bridge|drop|build|breakdown|outro",
                "order_index": 0,
                "bar_start": 1,
                "bar_end": 8,
                "time_start": 0.0,
                "time_end": 16.0,
                "tempo_bpm": 120,
                "key": "C minor",
                "energy_target": 0.3,
                "loudness_target_lufs": -18.0,
                "tension_target": 0.4,
                "target_voices": [],
                "target_instruments": ["pad_atmo"],
                "motif_ids": [],
                "lyrics": {
                    "text": "",
                    "lock_mode": "off|guide|hard",
                    "syllables_per_bar": [],
                },
                "instrument_overrides": {
                    "drums_main": {
                        "density_factor": 1.0,
                        "fill_probability_factor": 1.0,
                        "mute_kick": False,
                        "mute_snare": False,
                        "mute_cymbals": False,
                    }
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_energy": False,
                    "lock_loudness": False,
                    "lock_section_type": True,
                },
            }
        ],
        "events": [
            {
                "event_id": "event_1",
                "section_id": "chorus_1",
                "instrument_id": "guitar_solo_1",
                "time_start": 40.0,
                "time_end": 56.0,
                "bar_start": 21,
                "bar_end": 28,
                "bar": None,
                "beat": None,
                "pitch_register_hint": "higher|lower|same",
                "melodic_role": "solo|fill|accent",
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_presence": True,
                    "lock_timing": True,
                },
            }
        ],
    },

    # TTS branch (schema_version >= 3)
    #
    # This branch describes voice/prosody/emotion locks for text-to-speech. It is
    # internal to TTS tools (tts.speak, film VO, etc.) and is not exposed as a
    # planner-visible tool surface.
    "tts": {
        "global": {
            "default_voice_id": "narrator_01",
            "default_style_id": "neutral_mid",
            "default_speaking_rate": 1.0,
            "default_pitch_shift_semitones": 0,
            "default_energy": 0.5,
            "language": "en",
            "locale": "en-US",
            "constraints": {
                "lock_mode": "off|guide|hard",
                "lock_voice_for_narration": True,
                "lock_language": True,
                "lock_rate_range": False,
                "lock_pitch_range": False,
            },
        },
        "voices": [
            {
                "voice_id": "narrator_01",
                "role": "narrator|character|system|announcer|singer|rapper",
                "character_id": "char_main_01",
                "style_tags": ["neutral", "warm", "calm"],
                "gender": "unspecified",
                "age_hint": "adult",
                "embeddings": {
                    "speaker": "list[float]",
                    "timbre": "list[float]",
                    "gst_style": "list[float]",
                },
                "range_midi": {"min": 55, "max": 81},
                "language": "en",
                "vocal_type": "spoken|sung|rap|whisper",
                "baseline_prosody": {
                    "mean_pitch_hz": 180.0,
                    "pitch_std_hz": 30.0,
                    "mean_energy_db": -20.0,
                    "speech_rate_syll_per_s": 4.0,
                },
                "emotion_profile": {
                    "neutral": 0.8,
                    "happy": 0.2,
                    "sad": 0.0,
                    "angry": 0.0,
                    "fear": 0.0,
                    "valence_arousal": [0.1, 0.3],
                },
                "pitch_shift_allowed_semitones": 2,
                "rate_scaling_allowed": [0.8, 1.2],
                "formant_preservation": True,
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_timbre": True,
                    "lock_language": True,
                    "lock_baseline_prosody": True,
                    "lock_emotion_profile": False,
                },
                "refs": [
                    {
                        "audio_path": "/refs/narrator_01_neutral.wav",
                        "time_range": [0.0, 20.0],
                        "notes": "Neutral narration reference",
                    }
                ],
                "linked_music_voice_id": None,
            }
        ],
        "styles": [
            {
                "style_id": "neutral_mid",
                "name": "Neutral (mid energy)",
                "tags": ["neutral", "newsreader", "mid_energy"],
                "prosody_delta": {
                    "pitch_scale": 1.0,
                    "pitch_shift_semitones": 0,
                    "pitch_range_scale": 1.0,
                    "energy_scale": 1.0,
                    "rate_scale": 1.0,
                    "pause_scale": 1.0,
                },
                "emotion_delta": {
                    "happy": 0.0,
                    "sad": 0.0,
                    "angry": 0.0,
                    "fear": 0.0,
                    "valence_arousal_delta": [0.0, 0.0],
                },
                "style_embedding": {
                    "gst_token_weights": "list[float]",
                    "prosody_embedding": "list[float]",
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_prosody_shape": False,
                    "lock_emotion_target": False,
                },
            }
        ],
        "segments": [
            {
                "segment_id": "scene1_line3",
                "scene_id": "scene_intro",
                "context_id": "dialog_001",
                "line_index": 3,
                "text_ref": {
                    "text": "We are the broken ones that still believe.",
                    "hash": "sha256:...",
                    "lang": "en",
                },
                "voice_id": "narrator_01",
                "style_id": "neutral_mid",
                "prosody_targets": {
                    "pitch_contour_hint": "falling",
                    "pitch_shift_semitones": 0,
                    "pitch_range_scale": 1.0,
                    "energy_target": 0.5,
                    "speech_rate_relative": 0.95,
                    "pause_pattern": [],
                },
                "emotion_targets": {
                    "profile": {
                        "happy": 0.2,
                        "sad": 0.3,
                        "angry": 0.0,
                        "fear": 0.0,
                        "valence_arousal": [0.0, 0.2],
                    },
                    "lock_mode": "off|guide|hard",
                },
                "emphasis_spans": [],
                "timing_targets": {
                    "expected_duration_s": 3.2,
                    "max_deviation_s": 0.4,
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_voice": "hard|guide|off",
                    "lock_style": "hard|guide|off",
                    "lock_emphasis": "hard|guide|off",
                    "lock_timing": "hard|guide|off",
                },
            }
        ],
        "events": [
            {
                "event_id": "scene1_line3_breath_mid",
                "segment_id": "scene1_line3",
                "type": "breath|pause|emotion_shift|pronunciation_hint",
                "time_offset_s": 1.4,
                "token_index": 5,
                "params": {},
            }
        ],
    },

    # SFX branch (schema_version >= 3)
    #
    # This branch describes sound-effects / Foley locks. It is internal to SFX
    # tooling (film/post/game-style SFX) and not exposed as planner-visible tools.
    "sfx": {
        "global": {
            "default_bus": "SFX",
            "default_lufs_target": -18.0,
            "spatial_mode": "object|bed|stereo|ambisonic",
            "renderer_hint": "adm_object|bed",
            "constraints": {
                "lock_mode": "off|guide|hard",
                "lock_loudness_norm": True,
                "lock_bus_routing": True,
            },
        },
        "assets": [
            {
                "asset_id": "gun_9mm_pop_urban_close_01",
                "file_path": "/sfx/guns/9mm/gun_9mm_pop_urban_close_01.wav",
                "ucs": {
                    "category": "WEAPON",
                    "subcategory": "Gun_Small",
                    "details": ["9mm", "SemiAuto"],
                },
                "tags": ["gun", "9mm", "close", "urban", "dry", "sharp"],
                "recording_meta": {
                    "perspective": "close|medium|far|room|distant",
                    "mic_setup": "mono_shotgun",
                    "environment": "urban_rooftop",
                    "sample_rate": 48000,
                    "bit_depth": 24,
                    "channels": 1,
                },
                "spectral_features": {
                    "lufs_integrated": -10.0,
                    "peak_dbfs": -1.0,
                    "spectral_centroid_hz": 4500.0,
                    "bass_ratio": 0.2,
                    "brightness": 0.9,
                },
                "temporal_features": {
                    "attack_ms": 5.0,
                    "decay_ms": 250.0,
                    "tail_ms": 400.0,
                    "transient_sharpness": 0.95,
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_timbre": True,
                    "lock_envelope": True,
                    "lock_metadata": True,
                },
            }
        ],
        "layers": [
            {
                "layer_id": "gun_9mm_body",
                "role": "weapon_body",
                "tags": ["gun", "9mm", "body"],
                "components": [
                    {"asset_id": "gun_9mm_mech_01", "weight": 0.4, "offset_ms": 0},
                    {"asset_id": "gun_9mm_pop_urban_close_01", "weight": 0.5, "offset_ms": 0},
                    {"asset_id": "gun_9mm_tail_city_far_01", "weight": 0.8, "offset_ms": 20},
                ],
                "mixer_defaults": {
                    "bus": "SFX_Weapons",
                    "eq_profile": "gun_9mm_standard",
                    "compressor_profile": "gun_9mm_peak_taming",
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_components": True,
                    "allow_component_swap": False,
                    "allow_weight_variation": False,
                },
            }
        ],
        "events": [
            {
                "event_id": "shot_005_door_slam_01",
                "scene_id": "scene_hallway",
                "shot_id": "shot_005",
                "time_start": 43.250,
                "time_end": 43.750,
                "layer_id": "door_slam_heavy_01",
                "asset_id_override": None,
                "role": "door_slam",
                "ucs_category": "DOORS",
                "ucs_subcategory": "Door_Heavy",
                "spatial": {
                    "mode": "object",
                    "position": {"x": -2.5, "y": 0.0, "z": 4.0},
                    "spread_deg": 30.0,
                    "height_mode": "listener_plane",
                    "occlusion": 0.2,
                },
                "parameters": {
                    "intensity": 0.9,
                    "reverb_send": 0.7,
                    "lfe_send": 0.2,
                    "duck_dialogue_db": -4.0,
                },
                "randomization": {
                    "enabled": False,
                    "asset_pool_ids": [],
                    "pitch_semitones_jitter": 0.0,
                    "time_jitter_ms": 0.0,
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_timing": True,
                    "lock_layer": True,
                    "lock_spatial": True,
                    "lock_intensity": True,
                },
            }
        ],
        "ambiences": [
            {
                "ambience_id": "amb_city_night_rooftop",
                "scene_id": "scene_rooftop",
                "asset_ids": ["amb_city_traffic_far_01", "amb_wind_rooftop_01"],
                "role": "background_bed",
                "ucs_category": "AMB",
                "tags": ["city", "night", "wind", "traffic"],
                "spatial": {
                    "mode": "bed",
                    "channels": "5.1",
                    "height_mode": "low",
                    "spread_deg": 120.0,
                },
                "looping": {
                    "enabled": True,
                    "crossfade_ms": 500,
                    "random_seek_ms": 2500,
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_presence": True,
                    "lock_loudness": True,
                    "lock_tags": True,
                },
            }
        ],
    },

    # Film2 branch (schema_version >= 3)
    #
    # This branch describes cross-modal film/temporal structure: project, scenes,
    # shots, segments, and timelines. It is internal to film2 tools and not
    # exposed as planner-visible tools.
    "film2": {
        "project": {
            "project_id": "film_project_01",
            "title": "Untitled Film",
            "version_tag": "v1",
            "frame_rate": 24.0,
            "resolution": [1920, 1080],
            "aspect_ratio": "16:9",
            "color_space": "Rec.709",
            "audio_layout": "5.1",
            "edl_refs": [
                {
                    "role": "picture_lock",
                    "format": "cmx3600|aaf|xml|custom",
                    "uri": "/projects/film_project_01/edl/picture_lock.edl",
                }
            ],
            "constraints": {
                "lock_mode": "off|guide|hard",
                "lock_frame_rate": True,
                "lock_resolution": True,
                "lock_color_space": True,
            },
        },
        "sequences": [
            {
                "sequence_id": "seq_main",
                "name": "Main Picture Lock",
                "description": "Primary cut for final delivery",
                "edl_ref": {
                    "format": "cmx3600",
                    "uri": "/projects/film_project_01/edl/seq_main.edl",
                },
                "scene_order": ["scene_intro", "scene_heist", "scene_epilogue"],
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_scene_order": True,
                },
            }
        ],
        "scenes": [
            {
                "scene_id": "scene_heist",
                "sequence_id": "seq_main",
                "index": 1,
                "script_ref": {
                    "script_id": "scn_12",
                    "page_start": 35,
                    "page_end": 38,
                    "slugline": "INT. VAULT – NIGHT",
                },
                "timecode": {
                    "record_start": "01:02:15:00",
                    "record_end": "01:05:40:00",
                },
                "shots": ["shot_012a", "shot_012b", "shot_013a"],
                "tags": ["heist", "vault", "tension", "team"],
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_scene_order": True,
                    "lock_scene_boundaries": True,
                },
            }
        ],
        "shots": [
            {
                "shot_id": "shot_012a",
                "scene_id": "scene_heist",
                "index_in_scene": 0,
                "source": {
                    "clip_name": "A001_C003_0102AB",
                    "reel_id": "A001",
                    "src_tc_in": "00:45:10:08",
                    "src_tc_out": "00:45:18:20",
                    "rec_tc_in": "01:02:15:00",
                    "rec_tc_out": "01:02:23:12",
                },
                "visual_lock_bundle_id": "char_team_vault_01",
                "dominant_characters": ["char_leader", "char_hacker"],
                "hero_frames": [
                    {
                        "frame_index": 12,
                        "time_offset_s": 0.5,
                        "image_path": "/projects/.../frames/shot_012a_0012.png",
                        "visual_entity_ids": ["char_leader_face", "vault_door"],
                    }
                ],
                "music_section_ids": ["chorus_1"],
                "tts_segment_ids": ["scene_heist_line_1", "scene_heist_line_2"],
                "sfx_event_ids": ["shot_012a_gunshot_01", "shot_012a_alarm_start"],
                "ambience_ids": ["amb_vault_interior"],
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "lock_hero_frames": True,
                    "lock_visual_identity": True,
                    "lock_audio_structure": True,
                },
                "qa_targets": {
                    "min_seam_ok_ratio": 0.9,
                    "max_artifact_ratio": 0.05,
                    "min_face_id_lock": 0.9,
                    "min_layout_lock": 0.85,
                },
            }
        ],
        "segments": [
            {
                "segment_id": "seg_shot_012a_01",
                "shot_id": "shot_012a",
                "scene_id": "scene_heist",
                "time_local": {
                    "start_s": 0.0,
                    "end_s": 2.0,
                },
                "time_record": {
                    "start_tc": "01:02:15:00",
                    "end_tc": "01:02:17:00",
                },
                "frame_range": {
                    "start_frame": 0,
                    "end_frame": 48,
                },
                "visual": {
                    "hero_frame_indices": [0, 24],
                    "visual_lock_bundle_id": "char_team_vault_01",
                    "tracked_entities": [
                        {
                            "entity_id": "char_leader_face",
                            "track_id": "track_face_leader_012a",
                            "mask_sequence_path": "/projects/.../masks/shot_012a/leader_face_track.exr",
                            "constraints": {
                                "lock_presence": "hard|guide|off",
                                "max_position_drift_px": 10,
                                "max_scale_drift": 0.05,
                            },
                        }
                    ],
                },
                "music": {
                    "music_section_ids": ["chorus_1"],
                    "motif_ids": ["main_chorus_hook"],
                    "constraints": {
                        "lock_section_alignment": "off|guide|hard",
                        "lock_motif_presence": "off|guide|hard",
                    },
                },
                "tts": {
                    "tts_segment_ids": ["scene_heist_line_1"],
                    "constraints": {
                        "lock_dialog_sync": "off|guide|hard",
                    },
                },
                "sfx": {
                    "event_ids": ["shot_012a_gunshot_01"],
                    "constraints": {
                        "lock_sfx_timing": "off|guide|hard",
                        "lock_sfx_spatial": "off|guide|hard",
                    },
                },
                "qa_targets": {
                    "video": {
                        "fvd_max": 200.0,
                        "fvmd_min": 0.7,
                        "frame_lpips_mean_max": 0.2,
                        "temporal_lpips_mean_max": 0.2,
                        "optical_flow_consistency_min": 0.8,
                    },
                    "visual": {
                        "face_id_lock_min": 0.9,
                        "entity_layout_lock_min": 0.85,
                    },
                    "audio": {
                        "voice_lock_min": 0.9,
                        "music_section_lock_min": 0.85,
                        "sfx_timing_lock_min": 0.95,
                    },
                },
                "constraints": {
                    "lock_mode": "off|guide|hard",
                    "patch_granularity": "segment",
                    "allow_retime": False,
                    "allow_resample_audio": "off|guide|hard",
                },
            }
        ],
        "timeline": {
            "sequence_id": "seq_main",
            "fps": 24.0,
            "tracks": [
                {
                    "track_id": "V1",
                    "type": "video",
                    "clips": [
                        {
                            "clip_id": "clip_shot_012a",
                            "shot_id": "shot_012a",
                            "rec_tc_in": "01:02:15:00",
                            "rec_tc_out": "01:02:23:12",
                            "src_tc_in": "00:45:10:08",
                            "src_tc_out": "00:45:18:20",
                        }
                    ],
                },
                {
                    "track_id": "A1",
                    "type": "audio",
                    "clips": [
                        {
                            "clip_id": "music_main_theme",
                            "music_section_ids": ["chorus_1"],
                            "time_in_s": 32.0,
                            "time_out_s": 64.0,
                        }
                    ],
                },
            ],
        },
    },
}

