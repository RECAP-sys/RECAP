from __future__ import annotations

import numpy as np


def summarize_detection_confidence(scores: list[float] | None) -> float:
    if not scores:
        return 0.0
    top_scores = sorted((float(score) for score in scores), reverse=True)[:5]
    if not top_scores:
        return 0.0
    return float(np.clip(np.mean(top_scores), 0.0, 1.0))
