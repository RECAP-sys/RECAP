"""
Split Payload/Privacy Motivation Figure Experiment
==================================================

Visualizes intermediate feature size and privacy leakage by split depth
for a detection model under different TorchLens split candidates.

This script performs split candidate profiling only to generate the
split_payload_privacy_by_depth figure. It does not participate in training
or modify the fixed_split/split_runtime/retrain pipelines.

Usage:
    python tools/run_split_tradeoff_motivation_experiment.py \\
        --model tinynext \\
        --device cpu \\
        --input-size 1080 1920 \\
        --initial-input-size 1080 1920 \\
        --max-candidates 64 \\
        --output-dir results/split_tradeoff/tinynext

Output:
    - split_payload_privacy_by_depth.svg/pdf/tiff/png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import numpy as np

try:
    from loguru import logger
except ModuleNotFoundError:

    class _FallbackLogger:
        def _log(self, level: str, message: str) -> None:
            print(f"{level}: {message}", file=sys.stderr)

        def debug(self, message: str) -> None:
            self._log("DEBUG", message)

        def info(self, message: str) -> None:
            self._log("INFO", message)

        def warning(self, message: str) -> None:
            self._log("WARNING", message)

        def error(self, message: str) -> None:
            self._log("ERROR", message)

    logger = _FallbackLogger()

try:
    import torch
except ModuleNotFoundError as exc:
    _TORCH_IMPORT_ERROR = exc

    class _MissingDevice:
        def __init__(self, spec: str):
            self.spec = str(spec)
            self.type = self.spec.split(":", 1)[0]

        def __str__(self) -> str:
            return self.spec

    class _MissingCuda:
        @staticmethod
        def empty_cache() -> None:
            return None

    class _MissingTorch:
        cuda = _MissingCuda()

        class nn:
            class Module:
                pass

        @staticmethod
        def manual_seed(seed: int) -> None:
            return None

        @staticmethod
        def device(spec: str) -> _MissingDevice:
            return _MissingDevice(spec)

        @staticmethod
        def zeros(*args, **kwargs):
            raise RuntimeError("torch is required to create model inputs") from _TORCH_IMPORT_ERROR

    torch = _MissingTorch()  # type: ignore[assignment]

try:
    from torchlens.split import SplitRuntime
except Exception as exc:
    _TORCHLENS_IMPORT_ERROR = exc

    class SplitRuntime:
        pass


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import load_runtime_config
except Exception as exc:
    _CONFIG_IMPORT_ERROR = exc

    def load_runtime_config(*args, **kwargs):
        raise RuntimeError("config runtime dependencies are required") from _CONFIG_IMPORT_ERROR


try:
    from model_management.model_zoo import build_detection_model, list_available_models
except Exception as exc:
    _MODEL_ZOO_IMPORT_ERROR = exc

    def build_detection_model(*args, **kwargs):
        raise RuntimeError(
            "model_management.model_zoo is required to build models"
        ) from _MODEL_ZOO_IMPORT_ERROR

    def list_available_models() -> list[str]:
        return []


try:
    from model_management.split_candidate import SplitCandidate
except Exception as exc:
    _SPLIT_CANDIDATE_IMPORT_ERROR = exc

    @dataclass
    class SplitCandidate:
        candidate_id: str
        edge_nodes: list[str]
        cloud_nodes: list[str]
        boundary_edges: list[Any]
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
        legacy_layer_index: int | None
        boundary_count: int
        edge_parameter_count: int
        total_parameter_count: int
        edge_parameter_ratio: float
        metadata: dict[str, Any] = field(default_factory=dict)


try:
    from model_management.universal_model_split import UniversalModelSplitter
except Exception as exc:
    _UNIVERSAL_SPLITTER_IMPORT_ERROR = exc

    class UniversalModelSplitter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "model_management.universal_model_split is required to trace models"
            ) from _UNIVERSAL_SPLITTER_IMPORT_ERROR


try:
    from model_management.split_model_adapters import (
        build_split_runtime_sample_input,
        get_split_runtime_model,
    )
except Exception as exc:
    _SPLIT_ADAPTER_IMPORT_ERROR = exc

    def build_split_runtime_sample_input(*args, **kwargs):
        raise RuntimeError(
            "model_management.split_model_adapters is required to prepare trace inputs"
        ) from _SPLIT_ADAPTER_IMPORT_ERROR

    def get_split_runtime_model(model):
        return model


# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

PRIVACY_LEAKAGE_EPSILON = 1e-12
DEFAULT_CONFIG_PATH = "config/config.yaml"
MODEL_ALIAS_BUILD_CANDIDATES = {
    "yolo26": ("yolo26n", "yolo26s", "yolo26"),
    "tinynext": ("tinynext_s", "tinynext_m", "tinynext"),
    "rfdetr": ("rfdetr_nano", "rfdetr_small", "rfdetr"),
}


# ───────────────────────────────────────────────────────────────────────
# Data Structures
# ───────────────────────────────────────────────────────────────────────


@dataclass
class CandidateRecord:
    """Records profiling metrics for a single split candidate."""

    candidate_id: str
    legacy_layer_index: int | None
    canonical_split_key: str
    boundary_tensor_count: int
    boundary_tensor_labels: str  # JSON array string
    boundary_shape_summary: str  # JSON string
    payload_bytes: int
    payload_mb: float
    input_tensor_bytes: int
    payload_ratio_to_input: float
    edge_parameter_count: int
    total_parameter_count: int
    edge_parameter_ratio: float
    privacy_leakage_official: float
    privacy_leakage_log10: float
    privacy_leakage_score: float
    estimated_edge_flops: float
    estimated_cloud_flops: float
    estimated_latency: float
    is_trainable_tail: bool
    validation_passed: bool
    replay_success_rate: float
    tail_trainability: bool
    measured_edge_latency: float | None = None
    measured_cloud_latency: float | None = None
    measured_end_to_end_latency: float | None = None
    stability_score: float | None = None
    validation_error: str | None = None


# ───────────────────────────────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────────────────────────────


def safe_estimate_privacy_leakage(
    edge_parameter_count: int | float,
    *,
    epsilon: float = PRIVACY_LEAKAGE_EPSILON,
) -> float:
    """Safely compute privacy leakage score from edge parameter count."""
    denominator = max(0.0, float(edge_parameter_count)) + max(0.0, float(epsilon))
    if denominator <= 0.0:
        return float("inf")
    return 1.0 / denominator


def safe_log10(value: float) -> float:
    """Safely compute log10, handling inf values."""
    if not math.isfinite(value):
        return 0.0  # Represent inf as 0 in log scale for plotting
    if value <= 0:
        return 0.0
    return math.log10(value)


def normalize_candidate_limit(max_candidates: int | None) -> int | None:
    """Return None when candidate enumeration should be uncapped."""
    if max_candidates is None:
        return None
    max_candidates = int(max_candidates)
    return max_candidates if max_candidates > 0 else None


def format_candidate_limit(max_candidates: int | None) -> str:
    """Format candidate limit for logs and reports."""
    limit = normalize_candidate_limit(max_candidates)
    return str(limit) if limit is not None else "all"


def normalize_model_name(model_name: str) -> str:
    """Normalize user-facing model names consistently with model_zoo."""
    return str(model_name).strip().lower().replace("-", "_")


def resolve_model_build_name(
    model_name: str,
    available_models: Sequence[str] | None = None,
) -> str:
    """Resolve experiment aliases such as yolo26/tinynext/rfdetr to buildable names."""
    normalized = normalize_model_name(model_name)
    available = {normalize_model_name(name) for name in available_models or []}

    if not available:
        try:
            available = {normalize_model_name(name) for name in list_available_models()}
        except Exception as exc:
            logger.warning(f"Failed to inspect model_zoo registry: {exc}")

    if normalized in available:
        return normalized

    for candidate in MODEL_ALIAS_BUILD_CANDIDATES.get(normalized, ()):
        candidate_normalized = normalize_model_name(candidate)
        if not available or candidate_normalized in available:
            return candidate_normalized

    return normalized


def safe_model_dir_name(model_name: str) -> str:
    """Return a filesystem-safe directory name for a model label."""
    normalized = normalize_model_name(model_name)
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in normalized)


def create_deterministic_sample_input(
    model_name: str,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> Any:
    """Create a deterministic sample input compatible with the model.

    Returns batched tensor [1,3,H,W] as default format.
    """
    logger.info(f"Creating sample input for {model_name} ({input_height}x{input_width})")

    # Use batched tensor format [1,3,H,W] as default
    try:
        sample_input = torch.zeros(1, 3, input_height, input_width, device=device)
        logger.debug("Created Tensor[1,3,H,W] format")
        return sample_input
    except Exception as e:
        logger.error(f"Failed to create input: {e}")
        raise RuntimeError(f"Cannot create deterministic sample input for {model_name}") from e


def create_split_runtime_sample_input(
    model: torch.nn.Module,
    model_name: str,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> Any:
    """Create a sample input for the model object actually used by split runtime."""
    logger.info(
        f"Creating split runtime sample input for {model_name} ({input_height}x{input_width})"
    )
    try:
        sample_input = build_split_runtime_sample_input(
            model,
            image_size=(input_height, input_width),
            device=device,
        )
        return sample_input
    except Exception as exc:
        logger.warning(
            f"Split adapter sample input failed for {model_name}: {exc}; "
            "falling back to Tensor[1,3,H,W]"
        )
        return create_deterministic_sample_input(
            model_name,
            input_height,
            input_width,
            device,
        )


def get_default_model_name() -> str:
    """Read default model name from config or use fallback."""
    try:
        config = load_runtime_config(DEFAULT_CONFIG_PATH)
        model_name = getattr(config.client, "lightweight", None)
        if model_name:
            logger.info(f"Using default model from config: {model_name}")
            return model_name
    except Exception as e:
        logger.warning(f"Failed to read config: {e}")

    logger.info("Using fallback default model: yolov8s")
    return "yolov8s"


def build_model_safe(model_name: str, device: torch.device) -> torch.nn.Module:
    """Build a detection model with proper error handling."""
    logger.info(f"Building model: {model_name} on {device}")
    try:
        model = build_detection_model(model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        logger.info(f"Model built successfully: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to build model {model_name}: {e}")
        raise RuntimeError(f"Cannot build model '{model_name}': {e}") from e


# ───────────────────────────────────────────────────────────────────────
# Tracing and Candidate Enumeration
# ───────────────────────────────────────────────────────────────────────


def trace_model_with_splitter(
    model: torch.nn.Module,
    sample_input: Any,
    model_name: str,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> tuple[UniversalModelSplitter, SplitRuntime]:
    """Trace model using UniversalModelSplitter with fallback input formats."""
    logger.info("Tracing model with UniversalModelSplitter...")

    splitter = UniversalModelSplitter(device=device)

    # Try primary input format (batched tensor)
    try:
        logger.debug("Attempting trace with batched tensor [1,3,H,W]...")
        traced = splitter.trace(model, sample_input, model_name=model_name)
        runtime = getattr(traced, "runtime", None)
        if runtime is None:
            raise RuntimeError("UniversalModelSplitter.trace() did not produce a runtime")
        plan = getattr(runtime, "plan", None)
        logger.info(
            f"Trace successful with batched tensor format. "
            f"Prefix nodes: {len(getattr(plan, 'prefix_nodes', ()) or ())}, "
            f"Suffix nodes: {len(getattr(plan, 'suffix_nodes', ()) or ())}"
        )
        return splitter, runtime
    except Exception as e:
        logger.warning(f"Trace failed with batched tensor: {e}")

    # Try fallback format (list of single tensors)
    try:
        logger.debug("Attempting trace with list[Tensor[3,H,W]]...")
        fallback_input = [torch.zeros(3, input_height, input_width, device=device)]
        traced = splitter.trace(model, fallback_input, model_name=model_name)
        runtime = getattr(traced, "runtime", None)
        if runtime is None:
            raise RuntimeError("UniversalModelSplitter.trace() did not produce a runtime")
        plan = getattr(runtime, "plan", None)
        logger.info(
            f"Trace successful with list format. "
            f"Prefix nodes: {len(getattr(plan, 'prefix_nodes', ()) or ())}, "
            f"Suffix nodes: {len(getattr(plan, 'suffix_nodes', ()) or ())}"
        )
        return splitter, runtime
    except Exception as e:
        logger.warning(f"Trace failed with list format: {e}")

    # Both formats failed
    logger.error("Trace failed with both input formats")
    raise RuntimeError(
        f"Failed to trace model {model_name} with any input format. "
        f"Tried: Tensor[1,3,H,W] and list[Tensor[3,H,W]]"
    )


def enumerate_candidates(
    splitter: UniversalModelSplitter,
    runtime: SplitRuntime,
    max_candidates: int | None = None,
    max_boundary_count: int = 8,
    max_payload_bytes: int = 128 * 1024 * 1024,
) -> list[SplitCandidate]:
    """Enumerate split candidates from traced runtime."""
    max_candidates = normalize_candidate_limit(max_candidates)
    logger.info(
        f"Enumerating split candidates (max={format_candidate_limit(max_candidates)}, "
        f"max_boundary={max_boundary_count}, "
        f"max_payload={max_payload_bytes / (1024 * 1024):.1f} MB)..."
    )

    try:
        del runtime
        candidates = splitter.enumerate_candidates(
            max_candidates=max_candidates,
            max_boundary_count=max_boundary_count,
            max_payload_bytes=max_payload_bytes,
        )
        logger.info(f"Enumerated {len(candidates)} valid candidates")
        return list(candidates)
    except Exception as e:
        logger.error(f"Candidate enumeration failed: {e}")
        raise RuntimeError(f"Failed to enumerate candidates: {e}") from e


# ───────────────────────────────────────────────────────────────────────
# Candidate Profiling
# ───────────────────────────────────────────────────────────────────────


def profile_candidates(
    candidates: list[SplitCandidate],
    sample_input: Any,
    runtime: SplitRuntime,
    splitter: UniversalModelSplitter,
    input_size_bytes: int,
    privacy_epsilon: float = PRIVACY_LEAKAGE_EPSILON,
    validate: bool = False,
    initial_input_shape: Sequence[int] | None = None,
) -> list[CandidateRecord]:
    """Profile all candidates and create records."""
    logger.info(f"Profiling {len(candidates)} candidates...")

    records: list[CandidateRecord] = []

    for idx, candidate in enumerate(candidates):
        logger.debug(f"Profiling candidate {idx + 1}/{len(candidates)}: {candidate.candidate_id}")

        try:
            record = _profile_single_candidate(
                candidate,
                idx,
                sample_input,
                runtime,
                splitter,
                input_size_bytes,
                privacy_epsilon,
                validate,
                initial_input_shape,
            )
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to profile candidate {candidate.candidate_id}: {e}")
            # Still create a record with error information
            record = _create_error_record(candidate, str(e), input_size_bytes)
            records.append(record)

    if records and not any(record.legacy_layer_index == 0 for record in records):
        records.insert(
            0,
            _create_initial_input_record(
                sample_input,
                input_size_bytes,
                records,
                initial_input_shape=initial_input_shape,
            ),
        )

    logger.info(f"Profiled {len(records)} candidates")
    return records


def _profile_single_candidate(
    candidate: SplitCandidate,
    index: int,
    sample_input: Any,
    runtime: SplitRuntime,
    splitter: UniversalModelSplitter,
    input_size_bytes: int,
    privacy_epsilon: float,
    validate: bool,
    initial_input_shape: Sequence[int] | None = None,
) -> CandidateRecord:
    """Profile a single candidate."""

    # Compute privacy metrics
    privacy_leakage_official = safe_estimate_privacy_leakage(
        candidate.edge_parameter_count,
        epsilon=privacy_epsilon,
    )
    privacy_leakage_score = 1.0 - candidate.edge_parameter_ratio
    privacy_leakage_score = max(0.0, min(1.0, privacy_leakage_score))

    # Compute payload metrics
    payload_bytes = _display_payload_bytes(candidate, input_size_bytes)
    payload_mb = payload_bytes / (1024 * 1024)
    payload_ratio = float(payload_bytes) / float(input_size_bytes) if input_size_bytes > 0 else 0.0

    # Boundary tensor summary
    boundary_labels_json = json.dumps(candidate.boundary_tensor_labels)

    # Boundary shape summary
    if candidate.legacy_layer_index == 0 and initial_input_shape is not None:
        boundary_shape_json = json.dumps(_initial_input_shape_summary(initial_input_shape))
    else:
        boundary_shape = _get_boundary_shape_summary(candidate, runtime)
        boundary_shape_json = json.dumps(boundary_shape)

    # Validation
    validation_passed = True
    validation_error = None
    if validate:
        try:
            validation_passed = _validate_candidate(candidate, splitter, sample_input)
        except Exception as e:
            validation_passed = False
            validation_error = str(e)

    # Canonical split key
    canonical_split_key = candidate.metadata.get("canonical_split_key", candidate.candidate_id)

    return CandidateRecord(
        candidate_id=candidate.candidate_id,
        legacy_layer_index=candidate.legacy_layer_index,
        canonical_split_key=canonical_split_key,
        boundary_tensor_count=candidate.boundary_count,
        boundary_tensor_labels=boundary_labels_json,
        boundary_shape_summary=boundary_shape_json,
        payload_bytes=payload_bytes,
        payload_mb=payload_mb,
        input_tensor_bytes=input_size_bytes,
        payload_ratio_to_input=payload_ratio,
        edge_parameter_count=candidate.edge_parameter_count,
        total_parameter_count=candidate.total_parameter_count,
        edge_parameter_ratio=candidate.edge_parameter_ratio,
        privacy_leakage_official=privacy_leakage_official,
        privacy_leakage_log10=safe_log10(privacy_leakage_official),
        privacy_leakage_score=privacy_leakage_score,
        estimated_edge_flops=candidate.estimated_edge_flops,
        estimated_cloud_flops=candidate.estimated_cloud_flops,
        estimated_latency=candidate.estimated_latency,
        is_trainable_tail=candidate.is_trainable_tail,
        validation_passed=validation_passed,
        replay_success_rate=1.0 if validation_passed else 0.0,
        tail_trainability=candidate.is_trainable_tail,
        validation_error=validation_error,
    )


def _display_payload_bytes(candidate: SplitCandidate, input_size_bytes: int) -> int:
    """Return the payload size to show for split trade-off outputs."""
    if candidate.legacy_layer_index == 0 and input_size_bytes > 0:
        return int(input_size_bytes)
    return int(candidate.estimated_payload_bytes)


def _sample_input_shape_summary(sample_input: Any) -> list[list[Any]]:
    """Return a compact shape summary for the initial model input."""
    if isinstance(sample_input, list):
        return [
            [f"input_{idx}", list(item.shape)]
            for idx, item in enumerate(sample_input)
            if hasattr(item, "shape")
        ]
    if hasattr(sample_input, "shape"):
        return [["input", list(sample_input.shape)]]
    return [["input", None]]


def _initial_input_shape_summary(initial_input_shape: Sequence[int]) -> list[list[Any]]:
    return [["input", [int(dim) for dim in initial_input_shape]]]


def _create_initial_input_record(
    sample_input: Any,
    input_size_bytes: int,
    profiled_records: Sequence[CandidateRecord],
    *,
    initial_input_shape: Sequence[int] | None = None,
) -> CandidateRecord:
    """Create the layer-0 baseline that represents sending the raw input."""
    total_parameter_count = max(
        (record.total_parameter_count for record in profiled_records),
        default=0,
    )
    privacy_leakage_official = safe_estimate_privacy_leakage(0)
    payload_mb = input_size_bytes / (1024 * 1024)
    return CandidateRecord(
        candidate_id="initial_input",
        legacy_layer_index=0,
        canonical_split_key="initial_input",
        boundary_tensor_count=1,
        boundary_tensor_labels=json.dumps(["input"]),
        boundary_shape_summary=json.dumps(
            _initial_input_shape_summary(initial_input_shape)
            if initial_input_shape is not None
            else _sample_input_shape_summary(sample_input)
        ),
        payload_bytes=int(input_size_bytes),
        payload_mb=payload_mb,
        input_tensor_bytes=int(input_size_bytes),
        payload_ratio_to_input=1.0 if input_size_bytes > 0 else 0.0,
        edge_parameter_count=0,
        total_parameter_count=total_parameter_count,
        edge_parameter_ratio=0.0,
        privacy_leakage_official=privacy_leakage_official,
        privacy_leakage_log10=safe_log10(privacy_leakage_official),
        privacy_leakage_score=1.0,
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
        validation_passed=True,
        replay_success_rate=1.0,
        tail_trainability=True,
    )


def _get_boundary_shape_summary(candidate: SplitCandidate, runtime: SplitRuntime) -> list:
    """Extract boundary tensor shape information."""
    shapes = []

    # Try to get from metadata first
    if "boundary_shape_summary" in candidate.metadata:
        return candidate.metadata["boundary_shape_summary"]
    schema = candidate.metadata.get("boundary_schema")
    if isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        return [
            [
                str(item.get("canonical_id") or item.get("torchlens_label") or ""),
                list(item.get("symbolic_shape") or []),
            ]
            for item in schema
            if isinstance(item, Mapping)
        ]

    # Try to get from TraceGraph if available.
    try:
        graph = getattr(runtime, "trace_graph", None)
        if graph is not None and hasattr(graph, "nodes"):
            for label in candidate.boundary_tensor_labels:
                node = dict(getattr(graph, "nodes", {}) or {}).get(str(label))
                shape = getattr(node, "shape", None) or getattr(node, "tensor_shape", None)
                shapes.append([label, list(shape) if shape is not None else None])
    except Exception as e:
        logger.debug(f"Failed to extract boundary shape: {e}")

    return shapes


def _validate_candidate(
    candidate: SplitCandidate,
    splitter: UniversalModelSplitter,
    sample_input: Any,
) -> bool:
    """Validate a candidate through the RECAP split facade."""
    del sample_input
    if not hasattr(splitter, "validate_candidate"):
        return True

    try:
        result = splitter.validate_candidate(candidate)
        if isinstance(result, Mapping):
            return bool(result.get("success", result.get("valid", False)))
        return bool(result)
    except Exception as e:
        logger.debug(f"Validation error for {candidate.candidate_id}: {e}")
        return False


def _create_error_record(
    candidate: SplitCandidate,
    error_msg: str,
    input_size_bytes: int = 0,
) -> CandidateRecord:
    """Create an error record for a failed candidate."""
    privacy_score = 1.0 - candidate.edge_parameter_ratio
    privacy_score = max(0.0, min(1.0, privacy_score))
    payload_bytes = _display_payload_bytes(candidate, input_size_bytes)

    return CandidateRecord(
        candidate_id=candidate.candidate_id,
        legacy_layer_index=candidate.legacy_layer_index,
        canonical_split_key=candidate.candidate_id,
        boundary_tensor_count=candidate.boundary_count,
        boundary_tensor_labels=json.dumps(candidate.boundary_tensor_labels),
        boundary_shape_summary="null",
        payload_bytes=payload_bytes,
        payload_mb=payload_bytes / (1024 * 1024),
        input_tensor_bytes=input_size_bytes,
        payload_ratio_to_input=(
            float(payload_bytes) / float(input_size_bytes) if input_size_bytes > 0 else 0.0
        ),
        edge_parameter_count=candidate.edge_parameter_count,
        total_parameter_count=candidate.total_parameter_count,
        edge_parameter_ratio=candidate.edge_parameter_ratio,
        privacy_leakage_official=float("inf"),
        privacy_leakage_log10=0.0,
        privacy_leakage_score=privacy_score,
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_latency=0.0,
        is_trainable_tail=candidate.is_trainable_tail,
        validation_passed=False,
        replay_success_rate=0.0,
        tail_trainability=candidate.is_trainable_tail,
        validation_error=error_msg,
    )


# ───────────────────────────────────────────────────────────────────────
# Plotting Functions
# ───────────────────────────────────────────────────────────────────────


def _is_initial_input_record(record: CandidateRecord) -> bool:
    return record.legacy_layer_index == 0 or record.candidate_id == "initial_input"


def _initial_input_plot_label(record: CandidateRecord) -> str:
    return f"layer 0 input\n{record.payload_mb:.2f} MB"


def _apply_motivation_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def _style_motivation_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#D8D8D8", linewidth=0.45, alpha=0.65)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#4D4D4D")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="#4D4D4D", length=2.5, width=0.7)


def _save_motivation_figure(fig: Any, stem: Path) -> None:
    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix in {".tiff", ".png"}:
            kwargs["dpi"] = 600
        fig.savefig(stem.with_suffix(suffix), **kwargs)


def _write_candidate_records(records: list[CandidateRecord], output_dir: Path) -> None:
    payload = [asdict(record) for record in records]
    (output_dir / "candidate_records.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def plot_payload_privacy_by_depth(
    records: list[CandidateRecord],
    output_dir: Path,
) -> None:
    """Create payload and privacy leakage plot by split depth."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not available, skipping plotting")
        return

    if not records:
        logger.warning("No records to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_motivation_plot_style(plt)

    def layer_index(record: CandidateRecord, fallback: int) -> int:
        if record.legacy_layer_index is not None:
            return int(record.legacy_layer_index)
        return fallback

    sorted_items = sorted(
        enumerate(records),
        key=lambda item: (layer_index(item[1], item[0]), item[1].candidate_id),
    )
    sorted_records = [record for _idx, record in sorted_items]
    x_values: list[int] = []
    split_combination_index = 0
    for record in sorted_records:
        if _is_initial_input_record(record):
            x_values.append(0)
        else:
            split_combination_index += 1
            x_values.append(split_combination_index)
    x = np.array(x_values, dtype=float)
    payload_mb = [r.payload_mb for r in sorted_records]
    privacy_score = [r.privacy_leakage_score for r in sorted_records]
    max_split_combination_count = split_combination_index
    initial_indices = [
        idx for idx, record in enumerate(sorted_records) if _is_initial_input_record(record)
    ]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.1, 4.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.05, 1]},
    )

    payload_color = "#7884B4"
    payload_edge = "#484878"
    input_color = "#B64342"
    privacy_color = "#B64342"
    bar_colors = [
        input_color if _is_initial_input_record(record) else payload_color
        for record in sorted_records
    ]
    bar_edges = [
        "#7A2A2A" if _is_initial_input_record(record) else payload_edge
        for record in sorted_records
    ]
    ax1.bar(x, payload_mb, color=bar_colors, alpha=0.82, edgecolor=bar_edges, linewidth=0.25)
    ax1.set_ylabel("Payload (MB)")
    ax1.set_title("Boundary payload size")
    ax1.set_ylim([0, max(payload_mb) * 1.2 if payload_mb else 1])
    _style_motivation_axis(ax1)

    ax2.plot(
        x,
        privacy_score,
        marker="o",
        color=privacy_color,
        linewidth=1.15,
        markersize=2.2,
        markeredgewidth=0,
        alpha=0.9,
    )
    ax2.fill_between(x, privacy_score, alpha=0.16, color=privacy_color)
    ax2.set_ylabel("Privacy leakage score")
    ax2.set_xlabel("Split candidate index")
    ax2.set_title("Privacy leakage proxy")
    ax2.set_ylim([0, 1.05])
    _style_motivation_axis(ax2)

    for idx in initial_indices:
        record = sorted_records[idx]
        layer_x = x[idx]
        ax1.axvline(layer_x, color=input_color, linestyle=":", linewidth=0.85, alpha=0.75)
        ax2.axvline(layer_x, color=input_color, linestyle=":", linewidth=0.85, alpha=0.75)
        ax1.annotate(
            _initial_input_plot_label(record),
            (layer_x, payload_mb[idx]),
            xytext=(4, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6,
            color=input_color,
        )
        ax2.scatter(
            [layer_x],
            [privacy_score[idx]],
            s=45,
            marker="*",
            c=input_color,
            edgecolors="white",
            linewidth=0.4,
            zorder=5,
        )
        ax2.annotate(
            "layer 0",
            (layer_x, privacy_score[idx]),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=6,
            color=input_color,
        )
    if x.size:
        from matplotlib.ticker import MaxNLocator

        min_x = float(np.min(x))
        max_x = float(np.max(x))
        x_range = max(max_x - min_x, 1.0)
        ax2.set_xlim(min_x - 0.5, max_x + max(0.5, x_range * 0.015))
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        if max_split_combination_count > 0:
            max_tick = int(max_split_combination_count)
            min_gap_to_max = max(2, int(math.ceil(max_tick * 0.025)))
            ticks = {int(round(tick)) for tick in ax2.get_xticks() if min_x <= float(tick) <= max_x}
            ticks = {
                tick
                for tick in ticks
                if tick in {0, max_tick} or (max_tick - tick) >= min_gap_to_max
            }
            ticks.update({0, max_tick})
            ax2.set_xticks(sorted(ticks))

    ax1.legend(
        handles=[
            Patch(facecolor=payload_color, edgecolor=payload_edge, label="Intermediate feature"),
            Line2D([0], [0], color=input_color, linestyle=":", label="Layer-0 input"),
        ],
        loc="upper right",
        ncol=2,
    )

    stem = output_dir / "split_payload_privacy_by_depth"
    _save_motivation_figure(fig, stem)
    plt.close()

    logger.info(f"Saved payload/privacy plot to {stem}.svg/.pdf/.tiff/.png")


# ───────────────────────────────────────────────────────────────────────
# Main Experiment
# ───────────────────────────────────────────────────────────────────────


def compute_raw_input_size_bytes(
    input_hw: Sequence[int],
    *,
    channels: int = 3,
    bytes_per_channel: int = 1,
) -> int:
    """Compute the shared raw RGB frame size used as the layer-0 baseline."""
    if len(input_hw) != 2:
        raise ValueError("input_hw must contain height and width")
    height, width = (int(input_hw[0]), int(input_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("input height and width must be positive")
    return height * width * int(channels) * int(bytes_per_channel)


def run_single_model_experiment(
    args: argparse.Namespace,
    model_name: str,
    output_dir: Path,
    device: torch.device,
) -> list[CandidateRecord]:
    """Run trace, enumeration, profiling, validation, and target plotting."""
    display_model_name = normalize_model_name(model_name)
    build_model_name = resolve_model_build_name(display_model_name)

    logger.info("-" * 70)
    logger.info(f"Running split tradeoff experiment for {display_model_name}")
    logger.info(f"Build target: {build_model_name}")
    logger.info("-" * 70)

    model = build_model_safe(build_model_name, device)
    trace_model = get_split_runtime_model(model)
    trace_model = trace_model.to(device)
    trace_model.eval()
    sample_input = create_split_runtime_sample_input(
        model,
        build_model_name,
        args.input_size[0],
        args.input_size[1],
        device,
    )
    splitter, runtime = trace_model_with_splitter(
        trace_model,
        sample_input,
        build_model_name,
        args.input_size[0],
        args.input_size[1],
        device,
    )

    candidates = enumerate_candidates(
        splitter,
        runtime,
        max_candidates=args.max_candidates,
        max_boundary_count=args.max_boundary_count,
        max_payload_bytes=args.max_payload_mb * 1024 * 1024,
    )

    if not candidates:
        raise RuntimeError(f"No split candidates enumerated for {display_model_name}")

    records = profile_candidates(
        candidates,
        sample_input,
        runtime,
        splitter,
        args.initial_input_bytes,
        privacy_epsilon=args.privacy_epsilon,
        validate=args.validate_candidates,
        initial_input_shape=args.initial_input_shape,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_candidate_records(records, output_dir)
    plot_payload_privacy_by_depth(records, output_dir)
    return records


def run_experiment(args: argparse.Namespace) -> None:
    """Run the single-figure split tradeoff motivation experiment."""
    logger.info("=" * 70)
    logger.info("Split Payload/Privacy Motivation Figure Experiment")
    logger.info("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    initial_input_shape = list(getattr(args, "initial_input_size", None) or args.input_size)
    args.initial_input_shape = initial_input_shape
    args.initial_input_bytes = compute_raw_input_size_bytes(initial_input_shape)
    output_dir = Path(args.output_dir)
    logger.info(
        "Shared raw input baseline: "
        f"{initial_input_shape[0]}x{initial_input_shape[1]}x3 uint8 = "
        f"{args.initial_input_bytes / (1024 * 1024):.2f} MB"
    )
    logger.info(f"Output directory: {output_dir}")

    records = run_single_model_experiment(args, args.model, output_dir, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info(
        "Experiment completed. "
        f"Generated split_payload_privacy_by_depth for {len(records)} candidates."
    )
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate split_payload_privacy_by_depth for one model"
    )

    # Model and device
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Detection model name (default: read from config or yolov8s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (default: cpu)",
    )

    # Input
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[1080, 1920],
        help="Input size H W (default: 1080 1920)",
    )
    parser.add_argument(
        "--initial-input-size",
        type=int,
        nargs=2,
        default=None,
        help=(
            "Raw input frame size H W for the shared layer-0 baseline "
            "(default: same as --input-size)"
        ),
    )

    # Candidate enumeration
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "Maximum number of candidates to enumerate. Omit, 0, or negative "
            "means enumerate all candidates that satisfy boundary/payload filters."
        ),
    )
    parser.add_argument(
        "--max-boundary-count",
        type=int,
        default=8,
        help="Maximum number of boundary tensors (default: 8)",
    )
    parser.add_argument(
        "--max-payload-mb",
        type=int,
        default=128,
        help="Maximum payload size in MB (default: 128)",
    )

    # Privacy
    parser.add_argument(
        "--privacy-epsilon",
        type=float,
        default=PRIVACY_LEAKAGE_EPSILON,
        help=f"Privacy leakage epsilon (default: {PRIVACY_LEAKAGE_EPSILON})",
    )

    # Validation and output
    parser.add_argument(
        "--validate-candidates",
        action="store_true",
        default=False,
        help="Validate candidates using UniversalModelSplitter.validate_candidate()",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=("Output directory (default: results/split_tradeoff/{model_name})"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    args.model = args.model or get_default_model_name()

    # Set default output dir if not provided
    if args.output_dir is None:
        args.output_dir = f"results/split_tradeoff/{safe_model_dir_name(args.model)}"

    # Run experiment
    try:
        run_experiment(args)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
