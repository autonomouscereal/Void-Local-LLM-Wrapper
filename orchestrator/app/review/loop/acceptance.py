from __future__ import annotations

ACCEPT_THRESHOLDS = {
    "image": {"clip_min": 0.30, "id_cos_min": 0.82},
    "audio": {
        "clap_min": 0.30,
        "overall_quality_min": 0.70,
        "fit_score_min": 0.70,
    },
    "global": {"max_loops": 6},
}


