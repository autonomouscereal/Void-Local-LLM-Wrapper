SCHEMAS = {
  "facts": {"fields": ["claim","support[]","kind","scope","tags[]","confidence","run_id","t"]},
  "image_samples": {"fields": ["prompt","negative?","size","seed","image_ref","model","refs?","created_at"]},
  "tts_samples": {"fields": ["text","voice","rate","pitch","sample_rate","seed","voice_lock?","audio_ref","duration_s","model","created_at"]},
  "music_samples": {"fields": ["prompt","bpm","length_s","structure[]","seed","music_lock?","track_ref","stems[]?","model","created_at"]},
  "code_patches": {"fields": ["task","repo","patch","notes?","created_at"]},
  "research_ledger": {"fields": ["parts_index","parts[]"]}
}


