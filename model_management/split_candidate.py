from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitCandidate:
    candidate_id: str
    edge_nodes: list[str]
    cloud_nodes: list[str]
    boundary_edges: list[tuple[str, str]]
    boundary_tensor_labels: list[str]
    edge_input_labels: list[str]
    cloud_input_labels: list[str]
    cloud_output_labels: list[str]
    estimated_edge_flops: float
    estimated_cloud_flops: float
    estimated_payload_bytes: int
    estimated_privacy_risk: float
    estimated_latency: float
    is_trainable_tail: bool
    is_validated: bool = False
    validation_error: str | None = None
    legacy_layer_index: int | None = None
    boundary_count: int = 0
    edge_parameter_count: int = 0
    total_parameter_count: int = 0
    edge_parameter_ratio: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def payload_labels(self) -> list[str]:
        return self.boundary_tensor_labels


@dataclass
class CandidateProfile:
    candidate_id: str
    edge_flops: float
    cloud_flops: float
    payload_bytes: int
    boundary_tensor_count: int
    boundary_shape_summary: list[tuple[str, tuple[int, ...] | None]]
    estimated_privacy_leakage: float
    measured_edge_latency: float
    measured_cloud_latency: float
    measured_end_to_end_latency: float
    replay_success_rate: float
    tail_trainability: bool
    stability_score: float
    validation_passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)
