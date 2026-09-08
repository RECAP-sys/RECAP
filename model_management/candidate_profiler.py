from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from model_management.split_candidate import CandidateProfile, SplitCandidate


def _candidate_boundary_shape_summary(
    runtime, candidate: SplitCandidate
) -> list[tuple[str, object]]:
    schema = dict(getattr(candidate, "metadata", {}) or {}).get("boundary_schema")
    if isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        summary: list[tuple[str, object]] = []
        for item in schema:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("canonical_id") or item.get("torchlens_label") or "")
            summary.append((label, item.get("symbolic_shape")))
        if summary:
            return summary

    graph = getattr(runtime, "trace_graph", None)
    if graph is None:
        graph = getattr(getattr(runtime, "runtime", None), "trace_graph", None)
    nodes = dict(getattr(graph, "nodes", {}) or {})
    summary = []
    for label in candidate.boundary_tensor_labels:
        node = nodes.get(str(label))
        shape = getattr(node, "shape", None) or getattr(node, "tensor_shape", None)
        summary.append((str(label), shape))
    return summary


def profile_candidates(
    runtime,
    candidates: Sequence[SplitCandidate],
    *,
    validate: bool = True,
    validation_runs: int = 1,
) -> list[CandidateProfile]:
    profiles: list[CandidateProfile] = []
    for candidate in candidates:
        edge_latency = 0.0
        cloud_latency = 0.0
        end_to_end_latency = 0.0
        successes = 0
        stability = 0.0
        trainable = candidate.is_trainable_tail
        error: str | None = None

        if validate:
            trainable = True
            for _ in range(max(1, validation_runs)):
                start = time.perf_counter()
                if hasattr(runtime, "validate_candidate"):
                    report = runtime.validate_candidate(candidate)
                else:
                    report = {"success": True, "tail_trainability": candidate.is_trainable_tail}
                elapsed = time.perf_counter() - start
                end_to_end_latency += elapsed
                edge_latency += float(report.get("edge_latency", 0.0))
                cloud_latency += float(report.get("cloud_latency", 0.0))
                successes += int(report.get("success", False))
                stability += float(report.get("stability_score", 0.0))
                trainable = trainable and bool(
                    report.get("tail_trainability", candidate.is_trainable_tail)
                )
                error = report.get("error", error)
            runs = float(max(1, validation_runs))
            replay_success_rate = successes / runs
            stability_score = stability / runs if stability else replay_success_rate
            edge_latency /= runs
            cloud_latency /= runs
            end_to_end_latency /= runs
        else:
            replay_success_rate = 0.0
            stability_score = 0.0

        profile = CandidateProfile(
            candidate_id=candidate.candidate_id,
            edge_flops=candidate.estimated_edge_flops,
            cloud_flops=candidate.estimated_cloud_flops,
            payload_bytes=candidate.estimated_payload_bytes,
            boundary_tensor_count=candidate.boundary_count,
            boundary_shape_summary=_candidate_boundary_shape_summary(runtime, candidate),
            estimated_privacy_leakage=candidate.estimated_privacy_risk,
            measured_edge_latency=edge_latency,
            measured_cloud_latency=cloud_latency,
            measured_end_to_end_latency=end_to_end_latency,
            replay_success_rate=replay_success_rate,
            tail_trainability=trainable,
            stability_score=stability_score,
            validation_passed=error is None and replay_success_rate >= 1.0,
            metadata={"error": error} if error else {},
        )
        profiles.append(profile)
    return profiles
