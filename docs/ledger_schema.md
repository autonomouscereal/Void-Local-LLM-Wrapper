# Ledger Schema (Audio Pipeline)

One JSON object per line (JSONL). Schema versioned; content-addressed artifacts and engine provenance included.

Example:

```json
{
  "schema_version": 2,
  "ts": "2025-11-04T22:10:31Z",
  "job_id": "J_abc123",
  "phase": "audio.window#3.refine#2",
  "window_idx": 3,
  "icw_capsule_id": "AUDIO_W003",
  "request": {
    "prompt": "EDM rock hybrid ...",
    "lyrics_sha256": "sha256:...",
    "markers": [{"t":64.0,"label":"drop1"}]
  },
  "locks": {
    "voice": ["flyleaf_v1","eminem_v1"],
    "music_style": "lp_vibe_2025_10",
    "sfx": ["skrillex_fx_v2"]
  },
  "inputs": {
    "score_json_sha256": "sha256:...",
    "mfa_alignment_sha256": "sha256:...",
    "structure_prev_sha256": "sha256:..."
  },
  "engines": {
    "musicgen": {"version":"2.0.0","repo":"facebook/musicgen-large","image":"sha256:..."},
    "dsinger":  {"version":"OpenVPI-EN-0.5","image":"sha256:..."},
    "rvc":      {"index_id":"rvc_flyleaf_v2","image":"sha256:..."},
    "audioldm2":{"repo":"cvssp/audioldm2-large"},
    "xtts":     {"repo":"tts_models/.../xtts_v2"}
  },
  "env": {
    "torch":"2.7.1+cu118","cuda":"11.8","driver":"535.146",
    "arch_list":"6.1;7.5;8.0;8.6;8.9","gpu":"RTX8000","p40_present":true
  },
  "scores_before": {"clap":0.29,"fad":2.4,"cer":0.11,"spk_cos":0.81,"pitch_dev_cents":42,"lufs":-13.1,"sync_ms":96},
  "deltas": [
    {"engine":"dsinger","notes":[120,146],"op":"stretch","ratio":1.05},
    {"engine":"rvc","frames":[187,196],"op":"strength","+0.1":true},
    {"engine":"mix","op":"lufs","target":-14.0}
  ],
  "scores_after":  {"clap":0.36,"fad":1.6,"cer":0.05,"spk_cos":0.89,"pitch_dev_cents":28,"lufs":-14.0,"sync_ms":64},
  "artifacts": {
    "backing_cid": "cid:Qm...a1",
    "dry_vocal_cid": "cid:Qm...b2",
    "vocal_cid":   "cid:Qm...c3",
    "mix_cid":       "cid:Qm...d4",
    "structure_cid":"cid:Qm...e5",
    "capsule_cid":   "cid:Qm...f6"
  },
  "seed_bundle": {"scene":9031,"window":3,"router":"v2"},
  "sign": {"sha256":"sha256:...", "sig":"ed25519:..."}
}
```

Notes:
- All referenced files must be content-addressed (sha256 or CID).
- Record container image digests and model repo IDs.
- Include hardware/env snapshot.
- Optional: sign each line (ed25519) and store a WORM copy.


