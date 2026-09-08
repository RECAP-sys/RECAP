from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch
from loguru import logger

from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_contract import build_runtime_contract
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_candidate_descriptor,
)

PRIVACY_LEAKAGE_EPSILON = 1e-12
FIXED_SPLIT_DYNAMIC_BATCH_MAX = 64
EligibleCandidate = tuple[SplitCandidate, float, float]
ValidatedCandidate = tuple[CandidateProfile, SplitCandidate, float, float]


def estimate_privacy_leakage_from_edge_params(
    edge_parameter_count: int | float,
    *,
    epsilon: float = PRIVACY_LEAKAGE_EPSILON,
) -> float:
    denominator = max(0.0, float(edge_parameter_count)) + max(0.0, float(epsilon))
    if denominator <= 0.0:
        return float("inf")
    return 1.0 / denominator


def min_edge_parameters_for_privacy(
    privacy_leakage_upper_bound: float,
    *,
    epsilon: float = PRIVACY_LEAKAGE_EPSILON,
) -> int:
    upper_bound = float(privacy_leakage_upper_bound)
    if upper_bound <= 0.0 or math.isinf(upper_bound):
        return 0
    theta_min = (1.0 / upper_bound) - max(0.0, float(epsilon))
    return max(0, int(math.ceil(theta_min - 1e-9)))


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False)
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        handle.close()
        os.replace(handle.name, path)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        if os.path.exists(handle.name):
            try:
                os.remove(handle.name)
            except OSError:
                pass


def _load_json_artifact(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return dict(loaded) if isinstance(loaded, Mapping) else None


@dataclass(frozen=True)
class CandidateEnumerationStats:
    total_candidates: int
    eligible_candidates: int
    rejected_not_trainable_tail: int
    rejected_privacy: int
    rejected_freezing: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_candidates": int(self.total_candidates),
            "eligible_candidates": int(self.eligible_candidates),
            "rejected_not_trainable_tail": int(self.rejected_not_trainable_tail),
            "rejected_privacy": int(self.rejected_privacy),
            "rejected_freezing": int(self.rejected_freezing),
        }


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


@dataclass(frozen=True)
class SplitConstraints:
    privacy_leakage_upper_bound: float = 0.0
    max_layer_freezing_ratio: float = 1.0
    validate_candidates: bool = True
    configured_training_batch: int | None = field(default=None, compare=False)
    validation_batches: tuple[int, ...] = field(default_factory=tuple, compare=False)
    max_boundary_count: int = 8
    max_payload_bytes: int = 32 * 1024 * 1024
    privacy_leakage_epsilon: float = PRIVACY_LEAKAGE_EPSILON

    def __post_init__(self) -> None:
        if self.configured_training_batch is not None:
            object.__setattr__(
                self,
                "configured_training_batch",
                max(1, int(self.configured_training_batch)),
            )
        batches: list[int] = []
        for batch_size in self.validation_batches:
            batch = max(1, int(batch_size))
            if batch not in batches:
                batches.append(batch)
        object.__setattr__(self, "validation_batches", tuple(batches))

    @classmethod
    def from_config(cls, config: Any | None) -> "SplitConstraints":
        if config is None:
            return cls()
        configured_training_batch = _positive_int_or_none(
            getattr(config, "configured_training_batch", None)
        )
        raw_validation_batches = getattr(config, "validation_batches", None)
        validation_batches: list[int] = []
        if isinstance(raw_validation_batches, (list, tuple)):
            for value in raw_validation_batches:
                batch = _positive_int_or_none(value)
                if batch is not None and batch not in validation_batches:
                    validation_batches.append(batch)
        return cls(
            privacy_leakage_upper_bound=float(config.privacy_leakage_upper_bound),
            max_layer_freezing_ratio=float(getattr(config, "max_layer_freezing_ratio", 1.0)),
            validate_candidates=bool(getattr(config, "validate_candidates", True)),
            configured_training_batch=configured_training_batch,
            validation_batches=tuple(validation_batches),
            max_boundary_count=int(getattr(config, "max_boundary_count", 8)),
            max_payload_bytes=int(getattr(config, "max_payload_bytes", 32 * 1024 * 1024)),
            privacy_leakage_epsilon=float(
                getattr(config, "privacy_leakage_epsilon", PRIVACY_LEAKAGE_EPSILON)
            ),
        )


def _privacy_min_edge_parameter_count(constraints: SplitConstraints) -> int:
    return min_edge_parameters_for_privacy(
        constraints.privacy_leakage_upper_bound,
        epsilon=constraints.privacy_leakage_epsilon,
    )


def _constraints_payload(constraints: SplitConstraints) -> dict[str, Any]:
    return {
        "privacy_leakage_upper_bound": float(constraints.privacy_leakage_upper_bound),
        "privacy_leakage_epsilon": float(constraints.privacy_leakage_epsilon),
        "privacy_min_edge_parameter_count": _privacy_min_edge_parameter_count(constraints),
        "max_layer_freezing_ratio": float(constraints.max_layer_freezing_ratio),
        "validate_candidates": bool(constraints.validate_candidates),
        "max_boundary_count": int(constraints.max_boundary_count),
        "max_payload_bytes": int(constraints.max_payload_bytes),
    }


@dataclass
class SplitPlan:
    split_config_id: str
    model_name: str
    candidate_id: str | None
    split_index: int | None
    split_label: str | None
    boundary_tensor_labels: list[str]
    payload_bytes: int
    privacy_metric: float
    privacy_risk: float
    layer_freezing_ratio: float
    privacy_leakage: float = 0.0
    edge_parameter_count: int = 0
    total_parameter_count: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    candidate_descriptor: dict[str, Any] = field(default_factory=dict)
    split_granularity: str = "operation"
    trace_signature: str | None = None
    trace_batch_mode: str = ""
    dynamic_batch: list[int] | None = None
    trace_batch_size: int | None = None
    canonical_split_key: str = ""
    edge_split_id: str = ""
    input_tensor_shape: list[int] = field(default_factory=list)
    input_resize_mode: str = "direct_resize"
    front_version: str = "0"
    runtime_contract: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.canonical_split_key:
            raw = self.edge_split_id or self.candidate_id or self.split_label or "auto"
            self.canonical_split_key = _normalise_after_key(raw)
        if not self.edge_split_id:
            self.edge_split_id = str(self.candidate_id or self.canonical_split_key)
        self.split_granularity = str(self.split_granularity or "operation")
        if self.dynamic_batch is not None:
            self.dynamic_batch = [int(dim) for dim in list(self.dynamic_batch)[:2]]
            if len(self.dynamic_batch) != 2:
                self.dynamic_batch = None
        if self.trace_batch_size is not None:
            self.trace_batch_size = int(self.trace_batch_size)
        self.trace_batch_mode = str(self.trace_batch_mode or "")
        self.input_tensor_shape = [int(dim) for dim in list(self.input_tensor_shape or [])]
        self.input_resize_mode = str(self.input_resize_mode or "direct_resize")
        self.front_version = str(self.front_version or "0")
        self.runtime_contract = dict(self.runtime_contract or {})

    @property
    def boundary_count(self) -> int:
        return len(self.boundary_tensor_labels)

    @property
    def logical_split_id(self) -> str:
        return str(self.runtime_contract.get("logical_split_id") or self.canonical_split_key)

    @property
    def feature_layout_id(self) -> str:
        return str(self.runtime_contract.get("feature_layout_id") or "")

    def describe(self, *, max_boundary_labels: int = 4) -> str:
        labels = [str(label) for label in self.boundary_tensor_labels]
        if len(labels) > max_boundary_labels:
            label_text = (
                f"[{', '.join(labels[:max_boundary_labels])}, "
                f"... (+{len(labels) - max_boundary_labels} more)]"
            )
        else:
            label_text = "[" + ", ".join(labels) + "]"
        return (
            f"canonical_split_key={self.canonical_split_key}, "
            f"logical_split_id={self.logical_split_id}, "
            f"split_granularity={self.split_granularity}, boundary_count={self.boundary_count}, "
            f"boundary_tensor_labels={label_text}, feature_layout_id={self.feature_layout_id}, "
            f"payload_bytes={self.payload_bytes}, privacy_leakage={self.privacy_leakage:.6g}, "
            f"edge_parameters={self.edge_parameter_count}/{self.total_parameter_count}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitPlan":
        canonical_split_key = str(
            payload.get("canonical_split_key")
            or payload.get("edge_split_id")
            or payload.get("candidate_id")
            or payload.get("split_label")
            or ""
        )
        edge_split_id = str(
            payload.get("edge_split_id") or payload.get("candidate_id") or canonical_split_key
        )
        return cls(
            split_config_id=str(payload["split_config_id"]),
            canonical_split_key=canonical_split_key,
            edge_split_id=edge_split_id,
            model_name=str(payload["model_name"]),
            candidate_id=payload.get("candidate_id"),
            split_index=payload.get("split_index"),
            split_label=payload.get("split_label"),
            boundary_tensor_labels=list(payload.get("boundary_tensor_labels", [])),
            runtime_contract=dict(payload.get("runtime_contract") or {}),
            input_tensor_shape=[
                int(dim) for dim in list(payload.get("input_tensor_shape", []) or [])
            ],
            input_resize_mode=str(payload.get("input_resize_mode") or "direct_resize"),
            front_version=str(payload.get("front_version") or "0"),
            payload_bytes=int(payload.get("payload_bytes", 0)),
            privacy_metric=float(payload.get("privacy_metric", 0.0)),
            privacy_risk=float(payload.get("privacy_risk", 0.0)),
            layer_freezing_ratio=float(payload.get("layer_freezing_ratio", 0.0)),
            privacy_leakage=float(
                payload.get(
                    "privacy_leakage",
                    payload.get("privacy_risk", payload.get("privacy_metric", 0.0)),
                )
            ),
            edge_parameter_count=int(payload.get("edge_parameter_count", 0)),
            total_parameter_count=int(payload.get("total_parameter_count", 0)),
            validation=dict(payload.get("validation", {})),
            constraints=dict(payload.get("constraints", {})),
            candidate_descriptor=dict(payload.get("candidate_descriptor", {})),
            split_granularity=str(payload.get("split_granularity") or "operation"),
            trace_signature=payload.get("trace_signature"),
            trace_batch_mode=str(payload.get("trace_batch_mode") or ""),
            dynamic_batch=(
                [int(dim) for dim in list(payload.get("dynamic_batch") or [])[:2]]
                if payload.get("dynamic_batch") is not None
                else None
            ),
            trace_batch_size=(
                int(payload["trace_batch_size"])
                if payload.get("trace_batch_size") is not None
                else None
            ),
        )

    def matches(
        self,
        *,
        model_name: str,
        constraints: SplitConstraints,
        trace_signature: str | None,
        input_tensor_shape: Sequence[int] | None = None,
        input_resize_mode: str = "direct_resize",
        front_version: str = "0",
        model_version: str = "0",
    ) -> bool:
        expected_shape = (
            [int(dim) for dim in list(input_tensor_shape)]
            if input_tensor_shape is not None
            else None
        )
        cached_model_version = str(dict(self.runtime_contract or {}).get("model_version") or "0")
        return (
            self.model_name == model_name
            and cached_model_version == str(model_version or "0")
            and self.constraints == _constraints_payload(constraints)
            and self.trace_signature == trace_signature
            and (expected_shape is None or self.input_tensor_shape == expected_shape)
            and self.input_resize_mode == str(input_resize_mode or "direct_resize")
            and self.front_version == str(front_version or "0")
        )


def _trace_signature(splitter: UniversalModelSplitter) -> str:
    runtime = getattr(splitter, "runtime", None)
    graph = getattr(runtime, "trace_graph", None)
    graph_signature = getattr(graph, "graph_shape_hash", None)
    if graph_signature:
        return str(graph_signature)
    graph_value = getattr(splitter, "graph", None)
    if isinstance(graph_value, str) and graph_value:
        return graph_value
    return "unavailable"


def _first_tensor_device_type(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.device.type)
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_device_type(item)
            if found:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_device_type(item)
            if found:
                return found
    return ""


def _candidate_boundary_schema(candidate: SplitCandidate) -> dict[str, Any]:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    boundary_schema = metadata.get("boundary_schema")
    return dict(boundary_schema) if isinstance(boundary_schema, Mapping) else {}


def _build_plan_runtime_contract(
    *,
    model_name: str,
    model_version: str,
    candidate: SplitCandidate,
    runtime: UniversalModelSplitter,
    sample_input: Any,
    input_resize_mode: str,
) -> dict[str, Any]:
    return build_runtime_contract(
        logical_split_id=_candidate_split_key(candidate),
        trace_signature=_trace_signature(runtime),
        trace_device_type=_first_tensor_device_type(sample_input),
        runtime_backend="torchlens_native",
        boundary_tensor_labels=list(candidate.boundary_tensor_labels),
        boundary_schema=_candidate_boundary_schema(candidate),
        model_id=str(model_name),
        model_version=str(model_version or "0"),
        input_tensor_shape=_input_tensor_shape_from_sample(sample_input),
        input_resize_mode=str(input_resize_mode or "direct_resize"),
    )


def _layer_freezing_ratio(splitter: UniversalModelSplitter, candidate: SplitCandidate) -> float:
    del splitter
    if int(getattr(candidate, "total_parameter_count", 0)) > 0:
        return max(0.0, min(1.0, float(getattr(candidate, "edge_parameter_ratio", 0.0))))
    return 0.0


def _privacy_leakage(candidate: SplitCandidate, constraints: SplitConstraints) -> float:
    if (
        int(getattr(candidate, "total_parameter_count", 0)) > 0
        or int(getattr(candidate, "edge_parameter_count", 0)) > 0
    ):
        return estimate_privacy_leakage_from_edge_params(
            int(getattr(candidate, "edge_parameter_count", 0)),
            epsilon=constraints.privacy_leakage_epsilon,
        )
    return float(getattr(candidate, "estimated_privacy_risk", 0.0))


def _satisfies_privacy_constraint(
    candidate: SplitCandidate,
    constraints: SplitConstraints,
    privacy_leakage: float,
) -> bool:
    if float(constraints.privacy_leakage_upper_bound) <= 0.0:
        return True
    total_parameter_count = int(getattr(candidate, "total_parameter_count", 0))
    if total_parameter_count <= 0:
        return privacy_leakage <= float(constraints.privacy_leakage_upper_bound)
    return int(getattr(candidate, "edge_parameter_count", 0)) >= _privacy_min_edge_parameter_count(
        constraints
    )


def _make_plan_id(
    *,
    model_name: str,
    candidate: SplitCandidate,
    constraints: SplitConstraints,
    runtime_contract: Mapping[str, Any] | None = None,
) -> str:
    raw = json.dumps(
        {
            "model_name": model_name,
            "logical_split_id": _candidate_split_key(candidate),
            "feature_layout_id": str(dict(runtime_contract or {}).get("feature_layout_id") or ""),
            "constraints": _constraints_payload(constraints),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _normalise_after_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError("Fixed split candidates must expose an exact split key.")
    return text if text.startswith("after:") else f"after:{text}"


def _candidate_split_key(candidate: SplitCandidate) -> str:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    raw_key = metadata.get("canonical_split_key") or getattr(candidate, "candidate_id", None)
    return _normalise_after_key(raw_key)


def _input_tensor_shape_from_sample(sample_input: Any) -> list[int]:
    def _single_sample_shape(tensor: torch.Tensor) -> list[int]:
        shape = [int(dim) for dim in tensor.shape]
        if shape:
            shape[0] = 1
        return shape

    if isinstance(sample_input, torch.Tensor):
        return _single_sample_shape(sample_input)
    if isinstance(sample_input, (list, tuple)):
        for value in sample_input:
            if isinstance(value, torch.Tensor):
                return _single_sample_shape(value)
    return []


def _first_tensor_batch_size(value: Any) -> int | None:
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    return None


def _unique_validation_batches(
    validation_batches: Sequence[int] | None,
    *,
    default_training_batch: int,
) -> list[int]:
    batches: list[int] = []
    raw_batches = list(validation_batches or [1, int(default_training_batch)])
    for batch_size in raw_batches:
        batch = max(1, int(batch_size))
        if batch not in batches:
            batches.append(batch)
    if 1 not in batches:
        batches.insert(0, 1)
    return batches


def _resolve_validation_batches(
    constraints: SplitConstraints,
    validation_batches: Sequence[int] | None,
) -> Sequence[int] | None:
    if validation_batches is not None:
        return validation_batches
    configured_batches = tuple(getattr(constraints, "validation_batches", ()) or ())
    if configured_batches:
        return configured_batches
    configured_training_batch = getattr(constraints, "configured_training_batch", None)
    if configured_training_batch is not None:
        return (1, int(configured_training_batch))
    return None


def _resize_batch(value: Any, current_batch_size: int, target_batch_size: int) -> Any:
    current = max(1, int(current_batch_size))
    target = max(1, int(target_batch_size))
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or int(value.shape[0]) != current:
            return value
        if target <= current:
            return value.narrow(0, 0, target)
        pad = value[-1:].expand(target - current, *value.shape[1:]).clone()
        return torch.cat([value, pad], dim=0)
    if isinstance(value, Mapping):
        return {
            key: _resize_batch(item, current_batch_size, target_batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_resize_batch(item, current_batch_size, target_batch_size) for item in value)
    if isinstance(value, list):
        return [_resize_batch(item, current_batch_size, target_batch_size) for item in value]
    return value


def _validation_sample_inputs(
    sample_input: Any,
    validation_batches: Sequence[int] | None,
) -> list[Any]:
    current_batch = _first_tensor_batch_size(sample_input) or 1
    batches = _unique_validation_batches(
        validation_batches,
        default_training_batch=current_batch,
    )
    return [
        sample_input
        if int(batch_size) == int(current_batch)
        else _resize_batch(sample_input, current_batch, int(batch_size))
        for batch_size in batches
    ]


def _splitter_dynamic_batch(splitter: UniversalModelSplitter) -> list[int] | None:
    split_spec = getattr(splitter, "split_spec", None) or getattr(
        getattr(splitter, "runtime", None), "split_spec", None
    )
    dynamic_batch = getattr(split_spec, "dynamic_batch", None)
    if dynamic_batch is None:
        return None
    try:
        lower, upper = list(dynamic_batch)[:2]
    except (TypeError, ValueError):
        return None
    return [int(lower), int(upper)]


def _splitter_trace_batch_mode(splitter: UniversalModelSplitter) -> str:
    split_spec = getattr(splitter, "split_spec", None) or getattr(
        getattr(splitter, "runtime", None), "split_spec", None
    )
    return str(getattr(split_spec, "trace_batch_mode", "") or "")


def _format_candidate_enumeration_stats(
    stats: CandidateEnumerationStats,
    constraints: SplitConstraints,
) -> str:
    return (
        f"total_candidates={stats.total_candidates}, "
        f"eligible_candidates={stats.eligible_candidates}, "
        f"rejected_not_trainable_tail={stats.rejected_not_trainable_tail}, "
        f"rejected_privacy={stats.rejected_privacy}, "
        f"privacy_min_edge_parameter_count={_privacy_min_edge_parameter_count(constraints)}, "
        f"rejected_freezing={stats.rejected_freezing}, "
        f"max_layer_freezing_ratio={constraints.max_layer_freezing_ratio}, "
        f"max_boundary_count={constraints.max_boundary_count}, "
        f"max_payload_bytes={constraints.max_payload_bytes}"
    )


def _enumerate_feasible_candidates(
    runtime: UniversalModelSplitter,
    constraints: SplitConstraints,
) -> tuple[list[EligibleCandidate], CandidateEnumerationStats]:
    candidates = list(
        runtime.enumerate_candidates(
            max_boundary_count=constraints.max_boundary_count,
            max_payload_bytes=constraints.max_payload_bytes,
        )
    )
    eligible: list[EligibleCandidate] = []
    rejected_not_trainable_tail = 0
    rejected_privacy = 0
    rejected_freezing = 0
    for candidate in candidates:
        if not candidate.is_trainable_tail:
            rejected_not_trainable_tail += 1
            continue
        privacy_leakage = _privacy_leakage(candidate, constraints)
        freezing_ratio = _layer_freezing_ratio(runtime, candidate)
        if not _satisfies_privacy_constraint(candidate, constraints, privacy_leakage):
            rejected_privacy += 1
            continue
        if freezing_ratio > constraints.max_layer_freezing_ratio:
            rejected_freezing += 1
            continue
        eligible.append((candidate, privacy_leakage, freezing_ratio))
    stats = CandidateEnumerationStats(
        total_candidates=len(candidates),
        eligible_candidates=len(eligible),
        rejected_not_trainable_tail=rejected_not_trainable_tail,
        rejected_privacy=rejected_privacy,
        rejected_freezing=rejected_freezing,
    )
    logger.info(
        "[FixedSplit] Candidate enumeration summary: {}.",
        _format_candidate_enumeration_stats(stats, constraints),
    )
    return eligible, stats


def _candidate_runtime_key(candidate: SplitCandidate) -> tuple[int, float, int, str]:
    return (
        int(candidate.boundary_count),
        float(candidate.estimated_latency),
        candidate.legacy_layer_index if candidate.legacy_layer_index is not None else 10**9,
        candidate.candidate_id,
    )


def _eligible_candidate_key(item: EligibleCandidate) -> tuple[int, int, float, int, str]:
    candidate = item[0]
    return (int(candidate.estimated_payload_bytes), *_candidate_runtime_key(candidate))


def _profile_from_report(
    runtime: UniversalModelSplitter,
    candidate: SplitCandidate,
    report: Mapping[str, Any],
) -> CandidateProfile:
    del runtime
    error = report.get("error")
    boundary_shape_summary = [
        (str(label), shape)
        for label, shape in list((candidate.metadata or {}).get("boundary_shape_summary", []))
    ] or [(label, None) for label in candidate.boundary_tensor_labels]
    return CandidateProfile(
        candidate_id=candidate.candidate_id,
        edge_flops=candidate.estimated_edge_flops,
        cloud_flops=candidate.estimated_cloud_flops,
        payload_bytes=candidate.estimated_payload_bytes,
        boundary_tensor_count=candidate.boundary_count,
        boundary_shape_summary=boundary_shape_summary,
        estimated_privacy_leakage=candidate.estimated_privacy_risk,
        measured_edge_latency=float(report.get("edge_latency", 0.0)),
        measured_cloud_latency=float(report.get("cloud_latency", 0.0)),
        measured_end_to_end_latency=float(report.get("end_to_end_latency", 0.0)),
        replay_success_rate=1.0 if bool(report.get("success", False)) else 0.0,
        tail_trainability=bool(report.get("tail_trainability", candidate.is_trainable_tail)),
        stability_score=float(report.get("stability_score", 0.0)),
        validation_passed=error is None and bool(report.get("success", False)),
        metadata={"error": error} if error else {},
    )


def _build_validation_payload(
    chosen: SplitCandidate, profile: CandidateProfile | None
) -> dict[str, Any]:
    validation = {
        "validation_passed": bool(profile.validation_passed) if profile is not None else True,
        "tail_trainability": bool(profile.tail_trainability)
        if profile is not None
        else bool(chosen.is_trainable_tail),
        "replay_success_rate": float(profile.replay_success_rate) if profile is not None else 1.0,
        "stability_score": float(profile.stability_score) if profile is not None else 1.0,
    }
    if profile is not None:
        validation["measured_end_to_end_latency"] = float(profile.measured_end_to_end_latency)
        validation["measured_edge_latency"] = float(profile.measured_edge_latency)
        validation["measured_cloud_latency"] = float(profile.measured_cloud_latency)
        if profile.metadata:
            validation["profile_metadata"] = dict(profile.metadata)
    return validation


def _select_candidate(
    runtime: UniversalModelSplitter,
    eligible: list[EligibleCandidate],
    constraints: SplitConstraints,
    stats: CandidateEnumerationStats | None = None,
    *,
    validation_sample_inputs: Sequence[Any] | None = None,
    blacklisted_candidate_ids: set[str] | None = None,
) -> tuple[SplitCandidate, float, float, CandidateProfile | None, Mapping[str, Any] | None]:
    if not eligible:
        stats_suffix = (
            f", {_format_candidate_enumeration_stats(stats, constraints)}"
            if stats is not None
            else ""
        )
        raise RuntimeError(
            "No split candidate satisfies the fixed split constraints. "
            f"privacy_leakage_upper_bound={constraints.privacy_leakage_upper_bound}, "
            f"privacy_min_edge_parameter_count={_privacy_min_edge_parameter_count(constraints)}, "
            f"max_layer_freezing_ratio={constraints.max_layer_freezing_ratio}"
            f"{stats_suffix}"
        )
    validation_errors: dict[str, int] = defaultdict(int)
    blacklist = {
        _normalise_after_key(candidate_id)
        for candidate_id in set(blacklisted_candidate_ids or set())
        if str(candidate_id or "").strip()
    }
    for candidate, privacy_leakage, freezing_ratio in sorted(eligible, key=_eligible_candidate_key):
        candidate_key = _candidate_split_key(candidate)
        if candidate_key in blacklist:
            validation_errors[f"candidate blacklisted this attempt: {candidate_key}"] += 1
            continue
        if not constraints.validate_candidates and not validation_sample_inputs:
            runtime.split(candidate=candidate)
            return candidate, privacy_leakage, freezing_ratio, None, None
        try:
            bound = runtime.split(candidate=candidate)
            report = runtime.validate_candidate(
                bound,
                validation_sample_inputs=validation_sample_inputs,
            )
        except Exception as exc:
            validation_errors[str(exc) or type(exc).__name__] += 1
            blacklist.add(candidate_key)
            continue
        if not bool(report.get("success", False)):
            validation_errors[str(report.get("error") or "unknown")] += 1
            blacklist.add(candidate_key)
            continue
        if not bool(report.get("tail_trainability", bound.is_trainable_tail)):
            validation_errors["selected split does not have trainable suffix parameters"] += 1
            blacklist.add(candidate_key)
            continue
        return (
            bound,
            privacy_leakage,
            freezing_ratio,
            _profile_from_report(runtime, bound, report),
            report,
        )
    top_errors = "; ".join(
        f"{error} x{count}" for error, count in sorted(validation_errors.items())[:3]
    )
    raise RuntimeError(
        "No replayable TorchLens split candidate satisfies the fixed split constraints. "
        f"eligible_candidates={len(eligible)}, validation_errors={top_errors}"
    )


def apply_split_plan(splitter: UniversalModelSplitter, plan: SplitPlan) -> SplitCandidate:
    raw_candidate_id = plan.candidate_id or plan.edge_split_id or plan.canonical_split_key
    if raw_candidate_id is None:
        raise RuntimeError(
            "Fixed split plans must include candidate_id "
            f"(split_config_id={plan.split_config_id!r})."
        )
    return splitter.split(candidate_id=_normalise_after_key(raw_candidate_id))


def validate_split_plan(splitter: UniversalModelSplitter, plan: SplitPlan) -> dict[str, Any]:
    candidate = apply_split_plan(splitter, plan)
    report = splitter.validate_candidate(candidate)
    if not bool(report.get("success", False)):
        raise RuntimeError(
            "Persisted split plan is no longer replayable. "
            f"candidate_id={candidate.candidate_id}, error={report.get('error')}"
        )
    return report


def _ensure_fixed_split_runtime_traced(
    runtime: UniversalModelSplitter,
    model: torch.nn.Module,
    sample_input: Any,
    *,
    sample_kwargs: Mapping[str, Any] | None = None,
    model_name: str | None = None,
) -> None:
    if sample_kwargs:
        raise RuntimeError("TorchLens fixed split planning expects positional example inputs.")
    if runtime.runtime is not None and runtime.model is not None:
        return
    runtime.trace(
        model,
        sample_input,
        boundary="auto",
        model_name=model_name,
        enable_dynamic_batch=True,
        dynamic_batch_min=1,
        dynamic_batch_max=FIXED_SPLIT_DYNAMIC_BATCH_MAX,
    )


def compute_fixed_split_for_model(
    model: torch.nn.Module,
    constraints: SplitConstraints,
    *,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    model_name: str | None = None,
    splitter: UniversalModelSplitter | None = None,
    cache_path: str | None = None,
    input_resize_mode: str = "direct_resize",
    front_version: str = "0",
    model_version: str = "0",
    validation_batches: Sequence[int] | None = None,
    blacklisted_candidate_ids: set[str] | None = None,
) -> SplitPlan:
    del cache_path
    runtime = splitter or UniversalModelSplitter(device=device)
    _ensure_fixed_split_runtime_traced(
        runtime,
        model,
        sample_input,
        sample_kwargs=sample_kwargs,
        model_name=model_name,
    )
    eligible, enumeration_stats = _enumerate_feasible_candidates(runtime, constraints)
    resolved_validation_batches = _resolve_validation_batches(
        constraints,
        validation_batches,
    )
    validation_inputs = _validation_sample_inputs(sample_input, resolved_validation_batches)
    chosen, privacy_leakage, freezing_ratio, profile, report = _select_candidate(
        runtime,
        eligible,
        constraints,
        enumeration_stats,
        validation_sample_inputs=validation_inputs,
        blacklisted_candidate_ids=blacklisted_candidate_ids,
    )
    validation = _build_validation_payload(chosen, profile)
    if report is not None:
        validation.update(dict(report))
    validation.update(
        {
            "runtime": "torchlens_native",
            "selection": "constraints",
            "split_id": chosen.candidate_id,
            "candidate_pool_size": enumeration_stats.total_candidates,
            "eligible_candidate_count": enumeration_stats.eligible_candidates,
            "candidate_rejection_counts": {
                "not_trainable_tail": enumeration_stats.rejected_not_trainable_tail,
                "privacy": enumeration_stats.rejected_privacy,
                "freezing": enumeration_stats.rejected_freezing,
            },
        }
    )
    canonical_split_key = _candidate_split_key(chosen)
    candidate_descriptor = build_candidate_descriptor(chosen)
    runtime_contract = _build_plan_runtime_contract(
        model_name=model_name or model.__class__.__name__,
        model_version=model_version,
        candidate=chosen,
        runtime=runtime,
        sample_input=sample_input,
        input_resize_mode=input_resize_mode,
    )
    return SplitPlan(
        split_config_id=_make_plan_id(
            model_name=model_name or model.__class__.__name__,
            candidate=chosen,
            constraints=constraints,
            runtime_contract=runtime_contract,
        ),
        canonical_split_key=canonical_split_key,
        edge_split_id=canonical_split_key,
        model_name=model_name or model.__class__.__name__,
        candidate_id=chosen.candidate_id,
        split_index=chosen.legacy_layer_index,
        split_label=chosen.candidate_id,
        boundary_tensor_labels=list(chosen.boundary_tensor_labels),
        runtime_contract=runtime_contract,
        input_tensor_shape=_input_tensor_shape_from_sample(sample_input),
        input_resize_mode=str(input_resize_mode or "direct_resize"),
        front_version=str(front_version or "0"),
        payload_bytes=int(chosen.estimated_payload_bytes),
        privacy_metric=float(privacy_leakage),
        privacy_risk=float(privacy_leakage),
        layer_freezing_ratio=float(freezing_ratio),
        privacy_leakage=float(privacy_leakage),
        edge_parameter_count=int(getattr(chosen, "edge_parameter_count", 0)),
        total_parameter_count=int(getattr(chosen, "total_parameter_count", 0)),
        validation=validation,
        constraints=_constraints_payload(constraints),
        candidate_descriptor=candidate_descriptor,
        split_granularity=str((chosen.metadata or {}).get("split_granularity") or "operation"),
        trace_signature=_trace_signature(runtime),
        trace_batch_mode=_splitter_trace_batch_mode(runtime),
        dynamic_batch=_splitter_dynamic_batch(runtime),
        trace_batch_size=_first_tensor_batch_size(sample_input),
    )


def load_split_plan(path: str) -> SplitPlan | None:
    payload = _load_json_artifact(path)
    if payload is None:
        return None
    return SplitPlan.from_dict(payload)


def persist_split_plan(path: str, plan: SplitPlan) -> None:
    _atomic_write_json(path, plan.to_dict())


def load_or_compute_fixed_split_plan(
    model: torch.nn.Module,
    constraints: SplitConstraints,
    *,
    sample_input: Any,
    sample_kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    model_name: str | None = None,
    cache_path: str | None = None,
    splitter: UniversalModelSplitter | None = None,
    validate_cached_plan: bool = True,
    input_resize_mode: str = "direct_resize",
    front_version: str = "0",
    model_version: str = "0",
    validation_batches: Sequence[int] | None = None,
) -> SplitPlan:
    runtime = splitter or UniversalModelSplitter(device=device)
    sample_input_shape = _input_tensor_shape_from_sample(sample_input)
    model_key = model_name or model.__class__.__name__
    resolved_validation_batches = _resolve_validation_batches(
        constraints,
        validation_batches,
    )
    validation_inputs = _validation_sample_inputs(sample_input, resolved_validation_batches)
    blacklisted_candidate_ids: set[str] = set()
    cached = load_split_plan(cache_path) if cache_path else None
    cached_invalidated = False
    if cached is not None:
        _ensure_fixed_split_runtime_traced(
            runtime,
            model,
            sample_input,
            sample_kwargs=sample_kwargs,
            model_name=model_key,
        )
        cache_matches = cached.matches(
            model_name=model_key,
            constraints=constraints,
            trace_signature=_trace_signature(runtime),
            input_tensor_shape=sample_input_shape,
            input_resize_mode=input_resize_mode,
            front_version=front_version,
            model_version=model_version,
        )
        if cache_matches:
            try:
                cached_candidate = apply_split_plan(runtime, cached)
                if validate_cached_plan or validation_inputs:
                    started = time.perf_counter()
                    report = runtime.validate_candidate(
                        cached_candidate,
                        validation_sample_inputs=validation_inputs,
                    )
                    if not bool(report.get("success", False)):
                        blacklisted_candidate_ids.add(_candidate_split_key(cached_candidate))
                        raise RuntimeError(
                            "Persisted split plan is no longer replayable. "
                            f"candidate_id={cached_candidate.candidate_id}, "
                            f"error={report.get('error')}"
                        )
                    cached.validation = {
                        **cached.validation,
                        **report,
                        "cached_validation_time_sec": float(time.perf_counter() - started),
                    }
                return cached
            except (KeyError, RuntimeError, ValueError) as exc:
                cached_invalidated = True
                logger.info("Cached fixed split plan invalidated; recomputing. {}", exc)
        else:
            cached_invalidated = True
            logger.info("Cached fixed split plan metadata is stale; recomputing.")
    plan = compute_fixed_split_for_model(
        model,
        constraints,
        sample_input=sample_input,
        sample_kwargs=sample_kwargs,
        device=device,
        model_name=model_key,
        splitter=runtime,
        cache_path=cache_path,
        input_resize_mode=input_resize_mode,
        front_version=front_version,
        model_version=model_version,
        validation_batches=resolved_validation_batches,
        blacklisted_candidate_ids=blacklisted_candidate_ids,
    )
    if cache_path and (cached is None or cached_invalidated):
        persist_split_plan(cache_path, plan)
    elif cache_path and cached is not None:
        logger.info(
            "Existing fixed split plan cache is stale for the current TorchLens trace; "
            "using an in-memory plan without overwriting {}.",
            cache_path,
        )
    return plan


__all__ = [
    "FIXED_SPLIT_DYNAMIC_BATCH_MAX",
    "PRIVACY_LEAKAGE_EPSILON",
    "SplitConstraints",
    "SplitPlan",
    "apply_split_plan",
    "compute_fixed_split_for_model",
    "estimate_privacy_leakage_from_edge_params",
    "load_or_compute_fixed_split_plan",
    "load_split_plan",
    "min_edge_parameters_for_privacy",
    "persist_split_plan",
    "validate_split_plan",
]
