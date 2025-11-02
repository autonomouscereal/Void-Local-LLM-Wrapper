from __future__ import annotations

import os
from typing import Any, Dict, Optional, List, Tuple


def analyze_image(path: str, prompt: str | None = None) -> Dict[str, Any]:
    """
    Best-effort semantic analysis of an image.
    Returns { clip_score: float|None, tags: [str], emb: {clip: [float]|None} }
    Uses open_clip if available; falls back to trivial tags from prompt.
    """
    out: Dict[str, Any] = {"clip_score": None, "tags": [], "emb": {}}
    if not isinstance(path, str) or not os.path.exists(path):
        return out
    # Optional: open_clip scoring against prompt
    try:
        if prompt and isinstance(prompt, str) and prompt.strip():
            import open_clip  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            image = preprocess(Image.open(path)).unsqueeze(0)
            text = tokenizer([prompt])
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                sim = (image_features @ text_features.T).squeeze().cpu().item()
                out["clip_score"] = float(max(0.0, min(1.0, (sim + 1) / 2)))  # map from [-1,1]→[0,1]
                out["emb"]["clip"] = image_features.squeeze().cpu().tolist()
    except Exception:
        pass
    # Optional: BLIP tags (very light heuristic if transformers missing)
    try:
        from transformers import AutoProcessor, BlipForConditionalGeneration  # type: ignore
        from PIL import Image  # type: ignore
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        raw_image = Image.open(path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")
        out_ids = model.generate(**inputs, max_length=30)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        tags = [t for t in caption.replace(",", " ").split() if len(t) >= 3][:10]
        out["tags"] = tags
    except Exception:
        # fallback: prompt tokens
        if prompt:
            toks = [t for t in prompt.replace(",", " ").split() if len(t) >= 3]
            out["tags"] = list(dict.fromkeys(toks[:10]))
    return out


def _load_audio(path: str) -> Tuple[Optional[List[float]], int]:
    try:
        import soundfile as sf  # type: ignore
        y, sr = sf.read(path)
        if getattr(y, "ndim", 1) > 1:
            # mixdown to mono
            try:
                import numpy as np  # type: ignore
                y = y.mean(axis=1) if hasattr(y, "mean") else [float(sum(frames)/len(frames)) for frames in y]
            except Exception:
                y = y[:, 0]
        if hasattr(y, "tolist"):
            y = y.tolist()
        return y, int(sr)
    except Exception:
        return None, 0


def analyze_audio(path: str) -> Dict[str, Any]:
    """
    Returns { lufs: float|None, tempo_bpm: float|None, key: str|None, emotion: str|None }.
    Uses pyloudnorm/librosa if available. Graceful fallbacks otherwise.
    """
    out: Dict[str, Any] = {"lufs": None, "tempo_bpm": None, "key": None, "emotion": None, "genre": None, "pitch_mean_hz": None}
    if not isinstance(path, str) or not os.path.exists(path):
        return out
    y, sr = _load_audio(path)
    # LUFS
    try:
        import pyloudnorm as pyln  # type: ignore
        if y and sr:
            meter = pyln.Meter(sr)
            out["lufs"] = float(meter.integrated_loudness(y))
    except Exception:
        pass
    # Tempo + Key + Emotion (heuristic)
    try:
        if y and sr:
            import librosa  # type: ignore
            import numpy as np  # type: ignore
            y_np = np.asarray(y, dtype=float)
            # tempo
            tempo, _ = librosa.beat.beat_track(y=y_np, sr=sr)
            out["tempo_bpm"] = float(tempo)
            # key via chroma
            chroma = librosa.feature.chroma_cqt(y=y_np, sr=sr)
            prof_maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            prof_min = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            chroma_mean = chroma.mean(axis=1)
            maj_scores = [np.correlate(np.roll(prof_maj, i), chroma_mean)[0] for i in range(12)]
            min_scores = [np.correlate(np.roll(prof_min, i), chroma_mean)[0] for i in range(12)]
            pitch_class = int(np.argmax(maj_scores + min_scores) % 12)
            mode = "maj" if max(maj_scores) >= max(min_scores) else "min"
            key_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            out["key"] = f"{key_names[pitch_class]} {mode}"
            # pitch mean
            f0 = librosa.yin(y_np, fmin=60, fmax=400, sr=sr)
            out["pitch_mean_hz"] = float(float(f0.mean()) if f0 is not None and hasattr(f0, "mean") else 0.0)
            # emotion heuristic: energy and spectral centroid
            sc = librosa.feature.spectral_centroid(y=y_np, sr=sr).mean()
            rmse = librosa.feature.rms(y=y_np).mean()
            if float(tempo or 0) > 140 or sc > 3000:
                out["emotion"] = "excited"
            elif float(tempo or 0) < 80 and rmse < 0.02:
                out["emotion"] = "calm"
            else:
                out["emotion"] = "neutral"
            # genre heuristic: coarse rule-based using tempo and spectral features
            zcr = librosa.feature.zero_crossing_rate(y_np).mean()
            if float(tempo or 0) > 150 and sc > 3000:
                out["genre"] = "electronic"
            elif float(tempo or 0) > 100 and zcr > 0.1:
                out["genre"] = "rock"
            elif float(tempo or 0) < 80 and sc < 2000 and rmse < 0.02:
                out["genre"] = "ambient"
            else:
                out["genre"] = "pop"
    except Exception:
        pass
    return out


def normalize_lufs(path: str, target_lufs: float) -> Optional[float]:
    """
    Loudness normalize in-place to target LUFS if pyloudnorm/soundfile available.
    Returns applied gain in dB or None.
    """
    try:
        import pyloudnorm as pyln  # type: ignore
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore
        y, sr = sf.read(path)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(y)
        if loud is None:
            return None
        loud_norm = pyln.normalize.loudness(y, loud, target_lufs)
        sf.write(path, loud_norm, sr)
        return float(target_lufs - float(loud))
    except Exception:
        return None


