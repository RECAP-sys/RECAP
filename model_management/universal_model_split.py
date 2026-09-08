from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

from cloud.feature_cache.shard_reader import ShardFeatureBatchReader
from cloud.feature_cache.types import FeatureShardRef
from model_management.payload import (
    BoundaryPayload,
    deserialize_boundary_payload,
    serialize_boundary_payload,
)
from model_management.split_candidate import CandidateProfile, SplitCandidate
from model_management.split_runtime import (
    BoundaryPayloadCacheCodec,
    SplitRuntime,
    SplitSpec,
    compare_outputs,
    make_split_spec,
    prepare_boundary_for_runtime,
    prepare_split_runtime,
)
from model_management.torchlens_optimized_replay import (
    TorchScriptSplitReplay,
    build_torchscript_split_replay,
)

AUTO_TRACE_PROBE_BOUNDARY = "50%"


def _runtime_args(sample_input: Any) -> tuple[Any, ...]:
    if isinstance(sample_input, tuple):
        return sample_input
    return (sample_input,)


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


def _batch_dimension_multiplier(value: Any, *, batch_symbol: str = "B") -> int | None:
    if isinstance(value, str):
        if value == batch_symbol:
            return 1
        prefix = f"{batch_symbol}*"
        if value.startswith(prefix):
            try:
                multiplier = int(value[len(prefix) :])
            except ValueError:
                return None
            return multiplier if multiplier > 0 else None
        return None
    expression = getattr(value, "expression", None)
    if expression != batch_symbol:
        return None
    offset = int(getattr(value, "offset", 0) or 0)
    if offset != 0:
        return None
    multiplier = int(getattr(value, "multiplier", 1) or 1)
    return multiplier if multiplier > 0 else None


def _boundary_matches_input_batch(payload: BoundaryPayload, batch_size: int) -> bool:
    spec = dict(getattr(payload, "spec", {}) or {})
    for label, tensor in dict(getattr(payload, "tensors", {}) or {}).items():
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor_spec = spec.get(str(label))
        symbolic_shape = tuple(
            getattr(tensor_spec, "shape", None)
            or getattr(tensor_spec, "symbolic_shape", None)
            or ()
        )
        if tensor.ndim != len(symbolic_shape):
            continue
        for axis, dim in enumerate(symbolic_shape):
            multiplier = _batch_dimension_multiplier(dim)
            if multiplier is None:
                continue
            if int(tensor.shape[axis]) == int(batch_size) * int(multiplier):
                return True
    return False


def _normalise_boundary_batch_metadata(
    payload: BoundaryPayload,
    inputs: tuple[Any, ...],
) -> BoundaryPayload:
    input_batch_size = _first_tensor_batch_size(inputs)
    if input_batch_size is None or input_batch_size <= 0:
        return payload
    if not _boundary_matches_input_batch(payload, input_batch_size):
        return payload
    metadata = dict(getattr(payload, "metadata", {}) or {})
    if metadata.get("batch_size") == int(input_batch_size):
        return payload
    metadata["batch_size"] = int(input_batch_size)
    return replace(payload, metadata=metadata)


def _run_suffix_segment(
    runtime: Any,
    boundary: BoundaryPayload,
    *,
    trusted: bool = False,
) -> Any:
    segments = getattr(runtime, "segments", None)
    suffix = getattr(segments, "suffix", None)
    if trusted and callable(suffix):
        return suffix(boundary)
    if not trusted:
        if callable(suffix):
            boundary = prepare_boundary_for_runtime(runtime, boundary, validate=True)
            return suffix(boundary)
        boundary = prepare_boundary_for_runtime(runtime, boundary, validate=False)
    return runtime.run_suffix(boundary)


def _clone_tensor_tree_for_training(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tensor_tree_for_training(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree_for_training(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree_for_training(item) for item in value]
    return value


def _clone_boundary_for_training(boundary: BoundaryPayload) -> BoundaryPayload:
    tensors = getattr(boundary, "tensors", None)
    if not isinstance(tensors, Mapping):
        return boundary
    return replace(
        boundary,
        tensors={
            str(label): _clone_tensor_tree_for_training(tensor)
            for label, tensor in dict(tensors).items()
        },
        metadata=dict(getattr(boundary, "metadata", {}) or {}),
    )


def _default_split_suffix_loss(outputs: Any, targets: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
        if targets.dtype == torch.long and outputs.ndim >= 2:
            return F.cross_entropy(outputs, targets)
        return F.mse_loss(outputs, targets)
    raise TypeError("A loss_fn is required for non-tensor outputs or targets.")


def _trusted_train_suffix_batch(
    runtime: Any,
    boundary: BoundaryPayload,
    targets: Any,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer | None,
) -> torch.Tensor:
    segments = getattr(runtime, "segments", None)
    suffix = getattr(segments, "suffix", None)
    if not callable(suffix):
        raise RuntimeError("Trusted split suffix training requires runtime.segments.suffix.")

    tensors = getattr(boundary, "tensors", None)
    if not isinstance(tensors, Mapping):
        raise TypeError("Trusted split suffix training requires mapping boundary tensors.")
    detached_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Trusted split suffix training expected tensor boundary {key!r}, "
                f"got {type(tensor).__name__}."
            )
        root = tensor.detach().clone(memory_format=torch.contiguous_format)
        detached_tensors[str(key)] = root.requires_grad_(
            root.is_floating_point() or root.is_complex()
        )

    split_spec = getattr(runtime, "split_spec", None)
    metadata = dict(getattr(boundary, "metadata", {}) or {})
    metadata["use_live_param_sources"] = getattr(
        split_spec,
        "effective_use_live_param_sources",
        metadata.get("use_live_param_sources"),
    )
    detached_boundary = replace(
        boundary,
        tensors=detached_tensors,
        metadata=metadata,
    )

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    outputs = suffix(detached_boundary)
    loss = _default_split_suffix_loss(outputs, targets) if loss_fn is None else loss_fn(
        outputs,
        targets,
    )
    loss.backward()
    _boundary_grads = {key: tensor.grad for key, tensor in detached_tensors.items()}
    if optimizer is not None:
        optimizer.step()
    return loss.detach()


def train_split_suffix_batch(
    runtime: Any,
    boundary: BoundaryPayload,
    targets: Any,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer | None,
    *,
    trusted_runtime_boundary: bool = False,
) -> torch.Tensor:
    """Train one split-suffix batch through TorchLens.

    The default path delegates to SplitRuntime.train_suffix after adapting the
    boundary to the runtime device. The trusted path is only for boundaries that
    were already prepared and validated for the same runtime.
    """

    runtime_obj = _runtime_from_splitter(runtime)
    if trusted_runtime_boundary:
        return _trusted_train_suffix_batch(runtime_obj, boundary, targets, loss_fn, optimizer)
    boundary = prepare_boundary_for_runtime(runtime_obj, boundary, validate=False)
    boundary = _clone_boundary_for_training(boundary)
    loss, _boundary_grads = runtime_obj.train_suffix(
        boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    return loss


def _runtime_trace_signature(runtime: Any) -> str:
    graph = getattr(runtime, "trace_graph", None)
    return str(getattr(graph, "graph_shape_hash", "") or "")


def _normalise_after_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if text.startswith("after:") else f"after:{text}"


def _normalise_after_key(value: object) -> str:
    key = _normalise_after_id(value)
    if not key:
        raise RuntimeError("Fixed split candidates must expose an exact split key.")
    return key


def _torch_dtype_size(dtype: torch.dtype | None) -> int:
    if dtype is None:
        return 4
    try:
        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:
        return 4


def _symbolic_dim_size(dim: Any) -> int:
    if isinstance(dim, int):
        return max(1, int(dim))
    text = str(dim)
    if text == "B":
        return 1
    if text.startswith("B*"):
        try:
            return max(1, int(text[2:]))
        except ValueError:
            return 1
    try:
        return max(1, int(text))
    except ValueError:
        return 1


def _shape_numel(shape: Sequence[Any] | Any) -> int:
    total = 1
    for dim in list(shape or ()):
        total *= _symbolic_dim_size(dim)
    return int(total)


def _payload_bytes_from_specs(specs: Mapping[str, Any]) -> int:
    total = 0
    for spec in dict(specs or {}).values():
        total += _shape_numel(getattr(spec, "shape", None) or ()) * _torch_dtype_size(
            getattr(spec, "dtype", None)
        )
    return int(total)


def _boundary_schema_summary(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for label, spec in dict(specs or {}).items():
        summary[str(label)] = {
            "canonical_id": str(getattr(spec, "canonical_id", "")),
            "torchlens_label": str(getattr(spec, "torchlens_label", label)),
            "module_path": str(getattr(spec, "module_path", "")),
            "op_type": str(getattr(spec, "op_type", "")),
            "symbolic_shape": [str(dim) for dim in list(getattr(spec, "shape", ()) or ())],
            "dtype": str(getattr(spec, "dtype", "") or ""),
            "requires_grad": bool(getattr(spec, "requires_grad", False)),
            "role": str(getattr(spec, "role", "")),
            "output_index": getattr(spec, "output_index", None),
            "device_policy": str(getattr(spec, "device_policy", "runtime") or "runtime"),
        }
    return summary


def _boundary_shape_summary(specs: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    return [
        (str(label), [str(dim) for dim in list(getattr(spec, "shape", ()) or ())])
        for label, spec in dict(specs or {}).items()
    ]


def _parameter_logs_for_node(node: Any) -> list[Any]:
    return list(getattr(getattr(node, "layer", None), "parent_param_logs", []) or [])


def _parameter_from_log(
    log: Any,
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> torch.nn.Parameter | None:
    param = getattr(log, "_param_ref", None)
    if isinstance(param, torch.nn.Parameter):
        return param
    address = str(getattr(log, "address", "") or "")
    if address and address in named_parameters:
        return named_parameters[address]
    return None


def _parameter_count_for_nodes(
    runtime: Any,
    node_names: Iterable[str],
) -> int:
    graph = getattr(runtime, "trace_graph", None)
    model = getattr(runtime, "model", None)
    named_parameters = dict(model.named_parameters()) if isinstance(model, torch.nn.Module) else {}
    selected = {str(name) for name in node_names}
    seen: set[int | str] = set()
    total = 0
    for node in getattr(graph, "ordered_nodes", lambda: ())():
        if str(getattr(node, "torchlens_label", "")) not in selected:
            continue
        for log in _parameter_logs_for_node(node):
            param = _parameter_from_log(log, named_parameters)
            if param is not None:
                key: int | str = id(param)
                numel = int(param.numel())
            else:
                key = str(getattr(log, "address", "") or id(log))
                shape = getattr(log, "shape", None) or getattr(log, "tensor_shape", None) or ()
                numel = _shape_numel(shape)
            if key in seen:
                continue
            seen.add(key)
            total += numel
    return int(total)


def _parameter_ids_for_nodes(
    *,
    graph: Any,
    named_parameters: Mapping[str, torch.nn.Parameter],
    node_names: Iterable[str],
) -> set[int]:
    selected = {str(name) for name in node_names}
    parameter_ids: set[int] = set()
    for node in getattr(graph, "ordered_nodes", lambda: ())():
        if str(getattr(node, "torchlens_label", "")) not in selected:
            continue
        for log in _parameter_logs_for_node(node):
            param = _parameter_from_log(log, named_parameters)
            if param is not None:
                parameter_ids.add(id(param))
    return parameter_ids


def _candidate_boundary_edges(runtime: Any, plan: Any) -> list[tuple[str, str]]:
    boundary = set(getattr(plan, "boundary_nodes", ()) or ())
    suffix = set(getattr(plan, "suffix_nodes", ()) or ())
    graph = getattr(runtime, "trace_graph", None)
    edges: list[tuple[str, str]] = []
    if graph is None:
        return edges
    for child_label in suffix:
        try:
            child = graph.get(child_label)
        except Exception:
            continue
        for parent in tuple(getattr(child, "parents", ()) or ()):
            if str(parent) in boundary:
                edges.append((str(parent), str(child_label)))
    return edges


def _candidate_legacy_index(runtime: Any, plan: Any) -> int | None:
    prefix = set(getattr(plan, "prefix_nodes", ()) or ())
    graph = getattr(runtime, "trace_graph", None)
    if graph is None:
        return None
    indexes = [
        index
        for index, node in enumerate(graph.ordered_nodes())
        if str(getattr(node, "torchlens_label", "")) in prefix
    ]
    return max(indexes) if indexes else None


def _candidate_from_plan(runtime: SplitRuntime, split_spec: SplitSpec, plan: Any) -> SplitCandidate:
    graph = getattr(runtime, "trace_graph", None)
    prefix_nodes = list(getattr(plan, "prefix_nodes", ()) or ())
    suffix_nodes = list(getattr(plan, "suffix_nodes", ()) or ())
    boundary_labels = list(getattr(plan, "boundary_nodes", ()) or ())
    specs = dict(getattr(plan, "boundary_specs", {}) or {})
    payload_bytes = _payload_bytes_from_specs(specs)
    total_nodes = (
        [str(getattr(node, "torchlens_label", "")) for node in graph.ordered_nodes()]
        if graph
        else []
    )
    edge_parameter_count = _parameter_count_for_nodes(runtime, prefix_nodes)
    suffix_parameter_count = _parameter_count_for_nodes(runtime, suffix_nodes)
    total_parameter_count = _parameter_count_for_nodes(runtime, total_nodes)
    edge_parameter_ratio = (
        float(edge_parameter_count) / float(total_parameter_count)
        if total_parameter_count > 0
        else 0.0
    )
    privacy_risk = 1.0 / float(edge_parameter_count) if edge_parameter_count > 0 else float("inf")
    split_id = _normalise_after_id(
        getattr(plan, "split_label", None) or getattr(plan, "split_id", None)
    )
    split_id = split_id or str(getattr(plan, "split_id", split_spec.boundary))
    return SplitCandidate(
        candidate_id=split_id,
        edge_nodes=prefix_nodes,
        cloud_nodes=suffix_nodes,
        boundary_edges=_candidate_boundary_edges(runtime, plan),
        boundary_tensor_labels=boundary_labels,
        edge_input_labels=list(getattr(graph, "input_nodes", ()) or ()),
        cloud_input_labels=boundary_labels,
        cloud_output_labels=list(getattr(graph, "output_nodes", ()) or ()),
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=payload_bytes,
        estimated_privacy_risk=privacy_risk,
        estimated_latency=float(payload_bytes),
        is_trainable_tail=(
            suffix_parameter_count > 0 or bool(getattr(split_spec, "trainable", True))
        ),
        is_validated=True,
        legacy_layer_index=_candidate_legacy_index(runtime, plan),
        boundary_count=len(boundary_labels),
        edge_parameter_count=edge_parameter_count,
        total_parameter_count=total_parameter_count,
        edge_parameter_ratio=edge_parameter_ratio,
        metadata={
            "runtime": "torchlens_native",
            "runtime_backend": "torchlens_native",
            "graph_signature": _runtime_trace_signature(runtime),
            "canonical_split_key": split_id,
            "split_granularity": "operation",
            "torchlens_split_id": getattr(plan, "split_id", None),
            "torchlens_split_label": getattr(plan, "split_label", None),
            "torchlens_prefix_node_count": len(prefix_nodes),
            "torchlens_suffix_node_count": len(suffix_nodes),
            "suffix_parameter_count": suffix_parameter_count,
            "boundary_shape_summary": _boundary_shape_summary(specs),
            "boundary_schema": _boundary_schema_summary(specs),
            "split_spec": {
                "boundary": split_spec.boundary,
                "dynamic_batch": split_spec.dynamic_batch,
                "trace_batch_mode": split_spec.trace_batch_mode,
                "mode": split_spec.mode,
            },
        },
    )


def _candidate_from_runtime(runtime: SplitRuntime, split_spec: SplitSpec) -> SplitCandidate:
    return _candidate_from_plan(runtime, split_spec, runtime.plan)


def build_candidate_descriptor(candidate: SplitCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "boundary_tensor_labels": list(candidate.boundary_tensor_labels),
        "legacy_layer_index": candidate.legacy_layer_index,
        "split_granularity": (candidate.metadata or {}).get("split_granularity"),
        "metadata": dict(candidate.metadata),
    }


def reconstruct_candidate_from_descriptor(
    graph: Any,
    descriptor: Mapping[str, Any] | None,
    *,
    source: str | None = None,
) -> SplitCandidate | None:
    del graph
    if not descriptor:
        return None
    labels = list(descriptor.get("boundary_tensor_labels", []))
    candidate_id = str(descriptor.get("candidate_id") or (labels[-1] if labels else "after:auto"))
    metadata = dict(descriptor.get("metadata", {}))
    metadata["source"] = source or "descriptor"
    return SplitCandidate(
        candidate_id=candidate_id,
        edge_nodes=[],
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=labels,
        edge_input_labels=[],
        cloud_input_labels=labels,
        cloud_output_labels=[],
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_payload_bytes=0,
        estimated_privacy_risk=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
        is_validated=True,
        legacy_layer_index=descriptor.get("legacy_layer_index"),
        boundary_count=len(labels),
        metadata=metadata,
    )


def prepare_exact_split_runtime(
    model: torch.nn.Module,
    sample_input: Any,
    split_spec: SplitSpec,
    *,
    mode: str | None = None,
    expected_boundary_tensor_labels: Sequence[str] | None = None,
) -> SplitRuntime:
    spec = split_spec if mode is None else replace(split_spec, mode=mode)
    runtime = prepare_split_runtime(model, sample_input, spec, mode=mode)
    expected = [str(label) for label in list(expected_boundary_tensor_labels or [])]
    if expected and list(getattr(runtime.plan, "boundary_nodes", ()) or ()) != expected:
        raise ValueError(
            "Fixed split runtime contract mismatch for requested boundary tensors "
            f"(boundary={spec.boundary!r}, expected={expected!r}, "
            f"actual={list(getattr(runtime.plan, 'boundary_nodes', ()) or ())!r})."
        )
    return runtime


class UniversalModelSplitter:
    """Thin RECAP facade over TorchLens native SplitRuntime."""

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.runtime: SplitRuntime | None = None
        self.split_spec: SplitSpec | None = None
        self.current_candidate: SplitCandidate | None = None
        self.candidates: list[SplitCandidate] = []
        self.graph: str | None = None
        self.history = None
        self.trace_timings: dict[str, float] = {}
        self.trainability_loss_fn = None
        self.model_name: str | None = None
        self.model_family: str | None = None
        self._trace_sample_input: Any = None
        self._last_replay_validation: dict[str, Any] | None = None
        self._inference_replay_runner: TorchScriptSplitReplay | None = None

    def _clear_inference_replay(self) -> None:
        self._inference_replay_runner = None

    def prepare_inference_replay(self, sample_input: Any) -> None:
        self._clear_inference_replay()
        runtime = self._ensure_runtime()
        inputs = _runtime_args(sample_input)
        runner = build_torchscript_split_replay(runtime, inputs)
        with torch.inference_mode():
            reference_payload = self.edge_forward(sample_input)
            reference_output = _run_suffix_segment(
                runtime,
                reference_payload,
                trusted=True,
            )
            replay_payload = runner.run_prefix(*inputs)
            replay_output = runner.run_suffix(replay_payload)
        boundary_equivalent, boundary_max_diff = compare_outputs(
            reference_payload.tensors,
            replay_payload.tensors,
        )
        output_equivalent, output_max_diff = compare_outputs(
            reference_output,
            replay_output,
        )
        if not boundary_equivalent or not output_equivalent:
            raise RuntimeError(
                "Optimized split replay validation failed "
                f"(boundary_max_diff={boundary_max_diff:.6g}, "
                f"output_max_diff={output_max_diff:.6g})."
            )
        self._inference_replay_runner = runner
        logger.info(
            "[EdgeCL] TorchScript split replay ready "
            "boundary_max_diff={:.6g} output_max_diff={:.6g}.",
            float(boundary_max_diff),
            float(output_max_diff),
        )

    def trace(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        sample_kwargs: Mapping[str, Any] | None = None,
        *,
        split_spec: SplitSpec | None = None,
        boundary: str = "auto",
        mode: str = "generated_eager",
        model_name: str | None = None,
        model_family: str | None = None,
        enable_dynamic_batch: bool = True,
        dynamic_batch_min: int | None = None,
        dynamic_batch_max: int = 64,
        **_: Any,
    ) -> "UniversalModelSplitter":
        if sample_kwargs:
            raise RuntimeError("TorchLens native split runtime expects positional example inputs.")
        self.model = model
        self.model_name = model_name
        self.model_family = model_family
        trace_batch_size = _first_tensor_batch_size(sample_input) or 1
        trace_batch_mode = "batch_gt1" if trace_batch_size > 1 else "batch_1"
        if enable_dynamic_batch:
            lower = (
                int(dynamic_batch_min)
                if dynamic_batch_min is not None
                else (2 if trace_batch_size > 1 else 1)
            )
            lower = max(1, lower)
            upper = max(lower, int(dynamic_batch_max))
            dynamic_batch = (lower, upper)
        else:
            dynamic_batch = None
        requested_boundary = str(boundary)
        resolved_boundary = (
            AUTO_TRACE_PROBE_BOUNDARY if requested_boundary == "auto" else requested_boundary
        )
        self.split_spec = split_spec or make_split_spec(
            resolved_boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
            mode=mode,
        )
        prepare_started = time.perf_counter()
        logger.info(
            "[FixedSplit] Preparing TorchLens native split runtime "
            "(model_name={}, batch_size={}, dynamic_batch={}, mode={}, "
            "requested_boundary={}, trace_probe_boundary={}).",
            model_name or type(model).__name__,
            trace_batch_size,
            dynamic_batch,
            self.split_spec.mode,
            requested_boundary,
            resolved_boundary,
        )
        self.runtime = prepare_split_runtime(
            model,
            sample_input,
            self.split_spec,
            mode=self.split_spec.mode,
        )
        self._clear_inference_replay()
        if requested_boundary == "auto":
            logger.info(
                "[FixedSplit] TorchLens trace probe runtime completed in {:.3f}s "
                "(trace_probe_split_id={}; final split will be selected from "
                "enumerated candidates).",
                time.perf_counter() - prepare_started,
                getattr(self.runtime, "split_id", None),
            )
        else:
            logger.info(
                "[FixedSplit] TorchLens prepare_split_runtime completed in {:.3f}s (split_id={}).",
                time.perf_counter() - prepare_started,
                getattr(self.runtime, "split_id", None),
            )
        self.graph = _runtime_trace_signature(self.runtime)
        self.current_candidate = _candidate_from_runtime(self.runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        self._trace_sample_input = sample_input
        self._last_replay_validation = self.validate_candidate(self.current_candidate)
        return self

    def bind_runtime(
        self,
        runtime: SplitRuntime,
        *,
        model: torch.nn.Module | None = None,
        split_spec: SplitSpec | None = None,
    ) -> "UniversalModelSplitter":
        self.runtime = runtime
        self.model = model or getattr(runtime, "model", None)
        self.split_spec = split_spec or getattr(runtime, "split_spec", None)
        if self.split_spec is None:
            self.split_spec = make_split_spec(getattr(runtime, "split_id", "50%"))
        self.graph = _runtime_trace_signature(runtime)
        self.current_candidate = _candidate_from_runtime(runtime, self.split_spec)
        self.candidates = [self.current_candidate]
        self._trace_sample_input = None
        self._last_replay_validation = None
        self._clear_inference_replay()
        return self

    def bind_graph(
        self,
        model: torch.nn.Module,
        graph: Any,
        **_: Any,
    ) -> "UniversalModelSplitter":
        if isinstance(graph, SplitRuntime):
            return self.bind_runtime(graph, model=model)
        raise RuntimeError(
            "Graph templates are no longer supported; bind a TorchLens SplitRuntime."
        )

    def _ensure_runtime(self) -> SplitRuntime:
        if self.runtime is None:
            raise RuntimeError(
                "prepare_split_runtime() or trace() must be called before split execution."
            )
        return self.runtime

    def _bind_candidate_id(self, candidate_id: str) -> SplitCandidate:
        runtime = self._ensure_runtime()
        if self.model is None or self._trace_sample_input is None:
            if self.current_candidate and self.current_candidate.candidate_id == candidate_id:
                return self.current_candidate
            raise KeyError(
                f"TorchLens split candidate {candidate_id!r} is not available for rebinding."
            )
        base_spec = (
            self.split_spec or getattr(runtime, "split_spec", None) or make_split_spec(candidate_id)
        )
        exact_spec = replace(base_spec, boundary=_normalise_after_key(candidate_id))
        self.runtime = prepare_split_runtime(
            self.model,
            self._trace_sample_input,
            exact_spec,
            mode=exact_spec.mode,
        )
        self._clear_inference_replay()
        self.split_spec = exact_spec
        self.graph = _runtime_trace_signature(self.runtime)
        self.current_candidate = _candidate_from_runtime(self.runtime, exact_spec)
        self.candidates = [self.current_candidate]
        return self.current_candidate

    def split(
        self,
        *,
        candidate: SplitCandidate | None = None,
        candidate_id: str | None = None,
        layer_index: int | None = None,
        layer_label: str | None = None,
        boundary_tensor_labels: list[str] | None = None,
    ) -> SplitCandidate:
        del layer_index, boundary_tensor_labels
        if candidate is not None:
            if (
                self.current_candidate
                and self.current_candidate.candidate_id == candidate.candidate_id
            ):
                return self.current_candidate
            if self.model is not None and self._trace_sample_input is not None:
                return self._bind_candidate_id(candidate.candidate_id)
            self.current_candidate = candidate
            return candidate
        if candidate_id is not None:
            return self._bind_candidate_id(candidate_id)
        if layer_label is not None:
            return self._bind_candidate_id(layer_label)
        if self.current_candidate is None:
            runtime = self._ensure_runtime()
            self.current_candidate = _candidate_from_runtime(
                runtime,
                self.split_spec
                or getattr(runtime, "split_spec", None)
                or make_split_spec(runtime.split_id),
            )
        return self.current_candidate

    def enumerate_candidates(self, **kwargs: Any) -> list[SplitCandidate]:
        runtime = self._ensure_runtime()
        graph = getattr(runtime, "trace_graph", None)
        if graph is None:
            return list(self.candidates or [self.split()])
        max_boundary_count = kwargs.get("max_boundary_count")
        max_payload_bytes = kwargs.get("max_payload_bytes")
        max_candidates = kwargs.get("max_candidates")
        base_spec = (
            self.split_spec or getattr(runtime, "split_spec", None) or make_split_spec("50%")
        )
        candidates: list[SplitCandidate] = []
        # Metadata-only candidate enumeration. Final selected runtime construction
        # remains centralized in prepare_split_runtime/tl.prepare_split.
        from torchlens.split.planner import plan_split

        for node in graph.ordered_nodes():
            if bool(getattr(node, "is_input", False)) or bool(getattr(node, "is_output", False)):
                continue
            boundary = f"after:{node.torchlens_label}"
            try:
                spec = replace(base_spec, boundary=boundary)
                plan = plan_split(graph, spec)
            except Exception:
                continue
            candidate = _candidate_from_plan(runtime, spec, plan)
            if max_boundary_count is not None and candidate.boundary_count > int(
                max_boundary_count
            ):
                continue
            if max_payload_bytes is not None and candidate.estimated_payload_bytes > int(
                max_payload_bytes
            ):
                continue
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                int(item.estimated_payload_bytes),
                int(item.boundary_count),
                item.legacy_layer_index if item.legacy_layer_index is not None else 10**9,
                item.candidate_id,
            )
        )
        if max_candidates is not None:
            candidates = candidates[: max(0, int(max_candidates))]
        self.candidates = list(candidates)
        return list(candidates)

    def bind_candidate_descriptor(
        self,
        descriptor: Mapping[str, Any],
        *,
        include_in_candidates: bool = False,
    ) -> SplitCandidate:
        candidate = reconstruct_candidate_from_descriptor(None, descriptor)
        if candidate is None:
            raise RuntimeError("Could not bind empty split candidate descriptor.")
        self.current_candidate = candidate
        if include_in_candidates:
            self.candidates = [candidate]
        return candidate

    def edge_forward(
        self,
        *args: Any,
        candidate: SplitCandidate | None = None,
        **kwargs: Any,
    ) -> BoundaryPayload:
        del candidate
        if kwargs:
            raise RuntimeError("TorchLens prefix execution expects positional runtime inputs.")
        boundary = self._ensure_runtime().run_prefix(*args)
        return _normalise_boundary_batch_metadata(boundary, args)

    run_prefix = edge_forward

    def cloud_forward(
        self,
        payload: BoundaryPayload,
        *args: Any,
        candidate: SplitCandidate | None = None,
        **kwargs: Any,
    ) -> Any:
        del args, candidate, kwargs
        runtime = self._ensure_runtime()
        return _run_suffix_segment(runtime, payload)

    run_suffix = cloud_forward

    def cloud_train_step(
        self,
        payload: BoundaryPayload,
        targets: Any = None,
        *,
        loss_fn=None,
        optimizer: torch.optim.Optimizer | None = None,
        candidate: SplitCandidate | None = None,
        **_: Any,
    ) -> tuple[None, torch.Tensor]:
        del candidate
        runtime = self._ensure_runtime()
        loss, _grads = runtime.train_suffix(
            prepare_boundary_for_runtime(runtime, payload, validate=False),
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
        )
        return None, loss

    def train_suffix(
        self,
        boundary: BoundaryPayload,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
    ):
        runtime = self._ensure_runtime()
        return runtime.train_suffix(
            prepare_boundary_for_runtime(runtime, boundary, validate=False),
            targets,
            loss_fn=loss_fn or self.trainability_loss_fn,
            optimizer=optimizer,
        )

    def replay_inference(
        self,
        sample_input: Any,
        *,
        return_split_output: bool = False,
        profile: dict[str, float] | None = None,
    ):
        inputs = _runtime_args(sample_input)
        runner = self._inference_replay_runner
        if runner is None:
            raise RuntimeError("TorchScript inference replay has not been prepared.")
        started = time.perf_counter()
        payload = runner.run_prefix(*inputs)
        if profile is not None:
            profile["split_prefix"] = profile.get("split_prefix", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0

        started = time.perf_counter()
        outputs = runner.run_suffix(payload)
        if profile is not None:
            profile["split_suffix"] = profile.get("split_suffix", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
        return (outputs, payload) if return_split_output else outputs

    def validate_candidate(
        self,
        candidate: SplitCandidate | None = None,
        validation_sample_inputs: Sequence[Any] | None = None,
        trainability_smoke: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        chosen = candidate or self.current_candidate
        if self.runtime is None:
            return {
                "success": False,
                "candidate_id": getattr(chosen, "candidate_id", None),
                "runtime": "torchlens_native",
                "error": "runtime is not prepared",
            }
        if (
            candidate is not None
            and self.current_candidate
            and candidate.candidate_id != self.current_candidate.candidate_id
        ):
            try:
                chosen = self.split(candidate=candidate)
            except Exception as exc:
                return {
                    "success": False,
                    "tail_trainability": bool(getattr(chosen, "is_trainable_tail", False)),
                    "candidate_id": getattr(chosen, "candidate_id", None),
                    "runtime": "torchlens_native",
                    "error": str(exc),
                }
        report = dict(self._last_replay_validation or {"success": True})
        if self.model is not None and (
            self._trace_sample_input is not None or validation_sample_inputs
        ):
            sample_inputs = (
                list(validation_sample_inputs)
                if validation_sample_inputs
                else [self._trace_sample_input]
            )
            batch_reports: list[dict[str, Any]] = []
            for sample_input in sample_inputs:
                batch_size = _first_tensor_batch_size(sample_input)
                batch_report = _runtime_replay_report(
                    self.runtime,
                    self.model,
                    sample_input,
                    require_trainable=bool(getattr(chosen, "is_trainable_tail", False)),
                    trainability_smoke=bool(trainability_smoke),
                )
                batch_report["batch_size"] = batch_size
                batch_reports.append(batch_report)
                if not bool(batch_report.get("success", False)):
                    report = {
                        **batch_report,
                        "validation_batches": [item.get("batch_size") for item in batch_reports],
                        "batch_reports": batch_reports,
                    }
                    break
            else:
                max_diff = max(float(item.get("max_diff", 0.0) or 0.0) for item in batch_reports)
                report = {
                    "success": True,
                    "tail_trainability": all(
                        bool(item.get("tail_trainability", False)) for item in batch_reports
                    ),
                    "max_diff": max_diff,
                    "error": None,
                    "validation_batches": [item.get("batch_size") for item in batch_reports],
                    "batch_reports": batch_reports,
                }
            self._last_replay_validation = report
        return {
            **report,
            "candidate_id": getattr(chosen, "candidate_id", None),
            "runtime": "torchlens_native",
        }

    def freeze_head(self, chosen: SplitCandidate | None = None) -> None:
        del chosen

    def unfreeze_tail(self, chosen: SplitCandidate | None = None) -> None:
        del chosen
        collect_suffix_trainable_parameters(self)

    def get_tail_trainable_params(
        self,
        chosen: SplitCandidate | None = None,
    ) -> Iterable[torch.nn.Parameter]:
        del chosen
        return collect_suffix_trainable_parameters(self)


def _runtime_replay_report(
    runtime: SplitRuntime,
    model: torch.nn.Module,
    sample_input: Any,
    *,
    require_trainable: bool,
    trainability_smoke: bool = False,
) -> dict[str, Any]:
    del model, trainability_smoke
    try:
        tail_trainability = bool(
            collect_suffix_trainable_parameters(runtime, update_requires_grad=False)
        )
    except RuntimeError:
        tail_trainability = False
    if require_trainable and not tail_trainability:
        return {
            "success": False,
            "tail_trainability": False,
            "error": "selected split does not have trainable suffix parameters",
        }
    inputs = _runtime_args(sample_input)
    try:
        with torch.no_grad():
            boundary = runtime.run_prefix(*inputs)
            runtime.run_suffix(boundary)
    except Exception as exc:
        return {
            "success": False,
            "tail_trainability": tail_trainability,
            "error": str(exc),
        }
    return {
        "success": True,
        "tail_trainability": tail_trainability,
        "max_diff": 0.0,
        "error": None,
    }


def extract_split_features(splitter: UniversalModelSplitter, sample_input: Any) -> BoundaryPayload:
    return splitter.edge_forward(sample_input)


def slice_boundary_payload_batch(
    payload: BoundaryPayload,
    *,
    start: int = 0,
    length: int = 1,
) -> BoundaryPayload:
    start = max(0, int(start))
    length = max(1, int(length))
    codec = BoundaryPayloadCacheCodec(None)
    samples = codec.split_batch(payload, actual_batch_size=start + length)
    selected = samples[start : start + length]
    if len(selected) == 1:
        return selected[0]
    return codec.collate(selected)


_SPLIT_TARGET_METADATA_FIELDS = (
    "input_image_size",
    "input_tensor_shape",
    "input_resize_mode",
)


def _target_with_split_meta(target: Any, record: Mapping[str, Any]) -> Any:
    if not isinstance(target, Mapping):
        return target

    enriched = dict(target)
    existing_meta = enriched.get("_split_meta")
    split_meta = dict(existing_meta) if isinstance(existing_meta, Mapping) else {}
    for field_name in _SPLIT_TARGET_METADATA_FIELDS:
        if split_meta.get(field_name) is not None:
            continue
        value = record.get(field_name)
        if value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        split_meta[field_name] = value
    if split_meta:
        enriched["_split_meta"] = split_meta
    return enriched


def _runtime_from_splitter(splitter: Any) -> Any:
    ensure_runtime = getattr(splitter, "_ensure_runtime", None)
    if callable(ensure_runtime):
        return ensure_runtime()
    return getattr(splitter, "runtime", splitter)


def load_cached_split_batches(
    *,
    cache_path: str,
    all_indices: list[Any],
    annotations: Mapping[Any, Any],
    batch_size: int,
    runtime: Any,
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None = None,
    profile: Any | None = None,
) -> list[tuple[list[Any], BoundaryPayload, list[Any]]]:
    batches: list[tuple[list[Any], BoundaryPayload, list[Any]]] = []
    metadata_index_path = os.path.join(cache_path, "metadata_index.json")
    metadata_samples: dict[str, Any] = {}
    if os.path.exists(metadata_index_path):
        with open(metadata_index_path, "r", encoding="utf-8") as handle:
            metadata_index = json.load(handle)
        if isinstance(metadata_index, Mapping):
            metadata_samples = dict(metadata_index.get("samples") or {})
    shard_reader = ShardFeatureBatchReader()

    def _record_for_index(index: Any) -> dict[str, Any]:
        if preloaded_records is not None:
            record = preloaded_records.get(index) or preloaded_records.get(str(index))
            if isinstance(record, Mapping):
                return dict(record)
        key = str(index)
        record = metadata_samples.get(key)
        if isinstance(record, Mapping):
            return dict(record)
        raise FileNotFoundError(
            f"TrainingCacheView metadata for sample {key!r} is missing feature_ref."
        )

    for offset in range(0, len(all_indices), max(1, int(batch_size))):
        batch_indices = list(all_indices[offset : offset + max(1, int(batch_size))])
        prepare_started = time.perf_counter()
        records = [_record_for_index(index) for index in batch_indices]
        refs = [
            FeatureShardRef.from_dict(dict(record.get("feature_ref") or {})) for record in records
        ]
        targets = []
        for index, record in zip(batch_indices, records, strict=True):
            target = annotations.get(index)
            if target is None:
                target = annotations.get(str(index))
            if target is None and ("pseudo_boxes" in record or "pseudo_labels" in record):
                target = {
                    "boxes": list(record.get("pseudo_boxes") or []),
                    "labels": list(record.get("pseudo_labels") or []),
                }
            targets.append(_target_with_split_meta(target, record))
        if profile is not None:
            profile.add("target_construction_time", time.perf_counter() - prepare_started)
        batch_started = time.perf_counter()
        boundary = shard_reader.read_batch(refs, runtime=runtime)
        if profile is not None:
            profile.add("boundary_payload_batching_time", time.perf_counter() - batch_started)
            profile.add("training_batch_preparation_time", time.perf_counter() - prepare_started)
        batches.append((batch_indices, boundary, targets))
    return batches


class _GradClippingOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: list[torch.nn.Parameter],
        max_norm: float,
    ) -> None:
        self._optimizer = optimizer
        self._params = list(params)
        self._max_norm = float(max_norm)

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self._optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        torch.nn.utils.clip_grad_norm_(self._params, self._max_norm)
        return self._optimizer.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        unique.append(parameter)
    return unique


def _runtime_model(runtime: Any) -> torch.nn.Module | None:
    runtime_obj = _runtime_from_splitter(runtime)
    model = getattr(runtime_obj, "model", None)
    return model if isinstance(model, torch.nn.Module) else None


def _set_suffix_training_state(runtime: Any, *, update_requires_grad: bool) -> None:
    runtime_obj = _runtime_from_splitter(runtime)
    model = _runtime_model(runtime_obj)
    if model is None:
        return
    model.train()
    if update_requires_grad:
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def _suffix_parameter_entries(runtime: Any) -> list[tuple[str, torch.nn.Parameter]]:
    runtime_obj = _runtime_from_splitter(runtime)
    graph = getattr(runtime_obj, "trace_graph", None)
    plan = getattr(runtime_obj, "plan", None)
    model = _runtime_model(runtime_obj)
    if graph is None or plan is None:
        raise RuntimeError(
            "TorchLens suffix optimizer requires runtime.trace_graph and runtime.plan."
        )
    if model is None:
        raise RuntimeError("TorchLens suffix optimizer requires runtime.model.")
    named_parameters = dict(model.named_parameters())
    parameter_names_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    suffix_nodes = set(getattr(plan, "suffix_nodes", ()) or ())
    if not suffix_nodes:
        raise RuntimeError("TorchLens suffix optimizer found no suffix nodes.")
    frozen_source_nodes = set(getattr(plan, "prefix_nodes", ()) or ()) | set(
        getattr(plan, "boundary_nodes", ()) or ()
    )
    frozen_source_parameter_ids = _parameter_ids_for_nodes(
        graph=graph,
        named_parameters=named_parameters,
        node_names=frozen_source_nodes,
    )
    entries: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for node in graph.ordered_nodes():
        if str(getattr(node, "torchlens_label", "")) not in suffix_nodes:
            continue
        for log in _parameter_logs_for_node(node):
            param = _parameter_from_log(log, named_parameters)
            if param is None:
                continue
            param_id = id(param)
            if param_id in seen or param_id in frozen_source_parameter_ids:
                continue
            name = parameter_names_by_id.get(param_id)
            if name is None:
                continue
            seen.add(param_id)
            entries.append((str(name), param))
    return entries


def _suffix_parameters(runtime: Any) -> list[torch.nn.Parameter]:
    return [parameter for _name, parameter in _suffix_parameter_entries(runtime)]


def _suffix_parameter_names(runtime: Any) -> list[str]:
    return [name for name, _parameter in _suffix_parameter_entries(runtime)]


def collect_suffix_trainable_parameters(
    runtime: Any,
    *,
    update_requires_grad: bool = True,
) -> list[torch.nn.Parameter]:
    params = _unique_parameters(_suffix_parameters(runtime))
    if update_requires_grad:
        _set_suffix_training_state(runtime, update_requires_grad=True)
        for parameter in params:
            parameter.requires_grad_(True)
    if not params:
        raise RuntimeError("TorchLens suffix optimizer found no trainable suffix parameters.")
    return params


def build_split_retrain_optimizer(
    model: torch.nn.Module,
    *,
    runtime: Any = None,
    learning_rate: float = 1e-4,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
) -> torch.optim.Optimizer | _GradClippingOptimizer | None:
    del model
    params = collect_suffix_trainable_parameters(runtime)
    if not params:
        return None
    normalized_name = str(optimizer_name or "adam").strip().lower()
    if normalized_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    elif normalized_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    else:
        optimizer = torch.optim.Adam(
            params,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
        return _GradClippingOptimizer(optimizer, params, float(grad_clip_norm))
    return optimizer


@dataclass
class SplitRetrainProfile:
    training_batch_preparation_time: float = 0.0
    target_construction_time: float = 0.0
    boundary_payload_batching_time: float = 0.0
    device_transfer_time: float = 0.0
    validation_time: float = 0.0
    suffix_forward_backward_time: float = 0.0
    optimizer_step_time: float = 0.0
    total_retraining_time: float = 0.0

    def add(self, field_name: str, elapsed: float) -> None:
        setattr(self, field_name, float(getattr(self, field_name, 0.0)) + max(0.0, float(elapsed)))


def log_split_retrain_profile(profile: SplitRetrainProfile) -> None:
    logger.info(
        "[FixedSplitCL][RetrainProfile] training_batch_preparation_time={:.3f}s "
        "suffix_forward_backward_time={:.3f}s total_retraining_time={:.3f}s.",
        profile.training_batch_preparation_time,
        profile.suffix_forward_backward_time,
        profile.total_retraining_time,
    )


def universal_split_retrain(
    *,
    model: torch.nn.Module,
    sample_input: Any,
    cache_path: str,
    all_indices: list[Any],
    gt_annotations: Mapping[Any, Any] | None = None,
    device: str | torch.device = "cpu",
    num_epoch: int = 1,
    learning_rate: float = 1e-4,
    loss_fn=None,
    splitter: UniversalModelSplitter | None = None,
    batch_size: int = 2,
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
    shuffle_samples: bool = False,
    epoch_log_context: str | None = None,
    log_every_n_batches: int = 1,
    log_batches: bool = True,
    log_every_n_epochs: int = 1,
    log_first_epoch: bool = True,
    epoch_log_start: int = 0,
    epoch_log_total: int | None = None,
    retrain_profile: SplitRetrainProfile | None = None,
    **_: Any,
) -> list[float]:
    del (
        device,
        log_every_n_batches,
        log_batches,
        log_every_n_epochs,
        log_first_epoch,
        epoch_log_start,
        epoch_log_total,
    )
    retrain_started = time.perf_counter()
    if loss_fn is None:
        raise RuntimeError("Split-tail training requires an explicit loss function.")
    runtime = splitter or UniversalModelSplitter().trace(model, sample_input)
    if optimizer is None:
        optimizer = build_split_retrain_optimizer(
            model,
            runtime=runtime,
            learning_rate=float(learning_rate),
            optimizer_name=optimizer_name,
            weight_decay=float(weight_decay),
            grad_clip_norm=grad_clip_norm,
        )
    annotations = dict(gt_annotations or {})
    prepared_batches = load_cached_split_batches(
        cache_path=cache_path,
        all_indices=list(all_indices),
        annotations=annotations,
        batch_size=max(1, int(batch_size)),
        # Keep the full training window on CPU. Preparing every cached batch
        # for the runtime here moves all boundary tensors to CUDA at once and
        # makes peak memory scale with the number of training samples instead
        # of the configured batch size.
        runtime=None,
        preloaded_records=preloaded_records,
        profile=retrain_profile,
    )
    if not prepared_batches:
        raise RuntimeError("Split retraining did not prepare any batches.")
    losses: list[float] = []
    try:
        for epoch in range(int(num_epoch)):
            epoch_batches = list(prepared_batches) if shuffle_samples else prepared_batches
            if shuffle_samples and len(epoch_batches) > 1:
                random.shuffle(epoch_batches)
            epoch_loss_total: torch.Tensor | None = None
            epoch_loss_count = 0
            for _batch_indices, boundary, targets in epoch_batches:
                started = time.perf_counter()
                runtime_boundary = prepare_boundary_for_runtime(
                    _runtime_from_splitter(runtime),
                    boundary,
                    validate=True,
                )
                try:
                    loss = train_split_suffix_batch(
                        runtime,
                        runtime_boundary,
                        targets,
                        loss_fn,
                        optimizer,
                        trusted_runtime_boundary=True,
                    )
                finally:
                    del runtime_boundary
                if retrain_profile is not None:
                    retrain_profile.add(
                        "suffix_forward_backward_time",
                        time.perf_counter() - started,
                    )
                batch_loss = loss.detach()
                if batch_loss.is_floating_point() and batch_loss.dtype in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    batch_loss = batch_loss.float()
                epoch_loss_total = (
                    batch_loss if epoch_loss_total is None else epoch_loss_total + batch_loss
                )
                epoch_loss_count += 1
            if epoch_loss_total is None or epoch_loss_count <= 0:
                raise RuntimeError("Split retraining did not produce any finite batch loss.")
            losses.append(float((epoch_loss_total / epoch_loss_count).detach().cpu().item()))
            if epoch_log_context:
                logger.info(
                    "[FixedSplitCL] {} epoch {}/{} avg_loss={:.6f}.",
                    epoch_log_context,
                    epoch + 1,
                    int(num_epoch),
                    losses[-1],
                )
        return losses
    finally:
        if retrain_profile is not None:
            retrain_profile.add("total_retraining_time", time.perf_counter() - retrain_started)


__all__ = [
    "BoundaryPayload",
    "CandidateProfile",
    "SplitCandidate",
    "SplitRetrainProfile",
    "SplitRuntime",
    "SplitSpec",
    "UniversalModelSplitter",
    "build_candidate_descriptor",
    "build_split_retrain_optimizer",
    "collect_suffix_trainable_parameters",
    "compare_outputs",
    "deserialize_boundary_payload",
    "extract_split_features",
    "log_split_retrain_profile",
    "load_cached_split_batches",
    "prepare_exact_split_runtime",
    "reconstruct_candidate_from_descriptor",
    "serialize_boundary_payload",
    "slice_boundary_payload_batch",
    "train_split_suffix_batch",
    "universal_split_retrain",
]
