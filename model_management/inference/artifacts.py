from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceArtifacts:
    intermediate: object | None
    final_detection_boxes: list
    final_detection_labels: list
    final_detection_scores: list
    low_threshold_boxes: list
    low_threshold_labels: list
    low_threshold_scores: list
    confidence: float
    input_tensor_shape: list[int] | None = None
    input_resize_mode: str | None = None
    proposal_count: int = 0
    retained_count: int = 0
    feature_spectral_entropy: float | None = None
    logit_entropy: float | None = None
    logit_margin: float | None = None
    logit_energy: float | None = None
    timing_ms: dict[str, float] | None = None

    def to_inference_result(self) -> dict[str, list]:
        return {
            "boxes": self.final_detection_boxes,
            "labels": self.final_detection_labels,
            "scores": self.final_detection_scores,
            "low_threshold_boxes": self.low_threshold_boxes,
            "low_threshold_labels": self.low_threshold_labels,
            "low_threshold_scores": self.low_threshold_scores,
        }
