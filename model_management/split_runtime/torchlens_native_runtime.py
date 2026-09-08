from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
import torchlens as tl

from .torchlens_forward_guard import torchlens_forward_guard

DEFAULT_SPLIT_MODE = "generated_eager"
BoundaryPayload = tl.ReplayBoundary
ReplayBoundary = tl.ReplayBoundary
SplitRuntime = tl.SplitRuntime
SplitSpec = tl.SplitSpec


def require_torchlens_native_split_api() -> None:
    required = [
        "SplitSpec",
        "ReplayBoundary",
        "SplitRuntime",
        "prepare_split",
        "prepare_split_replay",
    ]
    missing = [name for name in required if not hasattr(tl, name)]
    if missing:
        raise RuntimeError(
            "Installed torchlens wheel does not expose native split API: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class SplitRuntimeConfig:
    boundary: str
    dynamic_batch: tuple[int, int] = (1, 64)
    trace_batch_size: int = 1
    mode: Literal["generated_eager", "compiled"] = DEFAULT_SPLIT_MODE
    trainable: bool = True


@dataclass(frozen=True)
class SplitCandidateMetadata:
    """Metadata-only split resolution used by fixed candidate enumeration."""

    requested_boundary: str
    actual_split_id: str
    split_label: str
    graph_signature: str
    boundary_nodes: tuple[str, ...]
    prefix_nodes: tuple[str, ...]
    suffix_nodes: tuple[str, ...]


def normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    return (example_inputs,)


def _normalize_mode(mode: str | None) -> Literal["generated_eager", "compiled"]:
    normalized = str(mode or DEFAULT_SPLIT_MODE).strip()
    if normalized not in {"generated_eager", "compiled"}:
        raise ValueError(f"Unsupported TorchLens split mode: {mode!r}.")
    return normalized  # type: ignore[return-value]


def make_split_spec(
    boundary: str | SplitRuntimeConfig,
    *,
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = (1, 64),
    trainable: bool = True,
    trace_batch_mode: str = "batch_1",
    model_family: str | None = None,
    mode: str = DEFAULT_SPLIT_MODE,
) -> SplitSpec:
    """Build a TorchLens native SplitSpec for RECAP split runtimes."""

    del model_family
    if isinstance(boundary, SplitRuntimeConfig):
        config = boundary
        return SplitSpec(
            boundary=str(config.boundary),
            batch_symbol=batch_symbol,
            dynamic_batch=config.dynamic_batch,
            trainable=bool(config.trainable),
            trace_batch_mode=("batch_gt1" if int(config.trace_batch_size) > 1 else "batch_1"),
            device_policy="runtime",
            mode=_normalize_mode(config.mode),
        )
    return SplitSpec(
        boundary=str(boundary),
        batch_symbol=batch_symbol,
        dynamic_batch=dynamic_batch,
        trainable=bool(trainable),
        trace_batch_mode=str(trace_batch_mode or "batch_gt1"),
        device_policy="runtime",
        mode=_normalize_mode(mode),
    )


def _spec_with_mode(split_spec: SplitSpec | str, mode: str | None) -> SplitSpec:
    spec = (
        make_split_spec(split_spec, mode=mode or DEFAULT_SPLIT_MODE)
        if isinstance(split_spec, str)
        else split_spec
    )
    if mode is None:
        return spec
    normalized_mode = _normalize_mode(mode)
    if getattr(spec, "mode", None) == normalized_mode:
        return spec
    return replace(spec, mode=normalized_mode)


def prepare_split_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    """Prepare a split runtime using the TorchLens native split API."""

    spec = _spec_with_mode(split_spec, mode)
    with torchlens_forward_guard():
        require_torchlens_native_split_api()
        try:
            return tl.prepare_split(model, normalize_example_inputs(example_inputs), spec)
        except Exception as exc:
            raise RuntimeError(
                "TorchLens native split runtime construction failed during tl.prepare_split. "
                "This does not necessarily mean the installed torchlens wheel is missing "
                "the native split API. The cause may be runtime tracing failure, model "
                "forward failure, invalid split spec, concurrent TorchLens execution, "
                "CUDA OOM, or worker/MPS resource contention. "
                f"cause={type(exc).__name__}: {exc}"
            ) from exc


def prepare_split_replay_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    """Prepare an inference-style replay runtime using TorchLens native split."""

    spec = _spec_with_mode(split_spec, mode)
    with torchlens_forward_guard():
        require_torchlens_native_split_api()
        try:
            return tl.prepare_split_replay(model, normalize_example_inputs(example_inputs), spec)
        except Exception as exc:
            raise RuntimeError(
                "TorchLens native split runtime construction failed during "
                "tl.prepare_split_replay. This does not necessarily mean the installed "
                "torchlens wheel is missing the native split API. The cause may be "
                "runtime tracing failure, model forward failure, invalid split spec, "
                "concurrent TorchLens execution, CUDA OOM, or worker/MPS resource "
                f"contention. cause={type(exc).__name__}: {exc}"
            ) from exc


def resolve_split_candidate_metadata(
    model: torch.nn.Module,
    example_inputs: Any,
    split_specs: list[SplitSpec | str] | tuple[SplitSpec | str, ...],
    *,
    mode: str | None = None,
) -> list[SplitCandidateMetadata]:
    """Resolve candidate split ids without constructing SplitRuntime objects.

    This is the only remaining TorchLens plan-metadata path in the adapter. It
    preserves RECAP's existing candidate selection semantics while keeping
    final selected runtime construction on the public ``tl.prepare_split`` API.
    It intentionally does not lower executable prefix/suffix segments.
    """

    specs = [_spec_with_mode(split_spec, mode) for split_spec in split_specs]
    inputs = normalize_example_inputs(example_inputs)
    with torchlens_forward_guard():
        require_torchlens_native_split_api()
        try:
            from torchlens.options import CaptureOptions, VisualizationOptions
            from torchlens.split.planner import plan_split
            from torchlens.split.shape import infer_traced_batch_size
            from torchlens.split.trace_graph import trace_graph_from_model_log
            from torchlens.user_funcs import log_forward_pass
        except Exception as exc:  # pragma: no cover - import failure is wheel-specific.
            raise RuntimeError(
                "TorchLens candidate metadata resolution requires the installed "
                "torchlens wheel to expose split planning metadata internals. "
                "Final runtime construction still uses the public native split API."
            ) from exc
        model_log = log_forward_pass(
            model,
            inputs,
            {},
            capture=CaptureOptions(
                layers_to_save="all",
                keep_unsaved_layers=True,
                detach_saved_tensors=False,
                save_function_args=True,
                intervention_ready=True,
            ),
            visualization=VisualizationOptions(view="none"),
        )
        graph = trace_graph_from_model_log(
            model_log,
            traced_batch_size=infer_traced_batch_size(inputs),
            batch_symbol=specs[0].batch_symbol if specs else "B",
            dynamic_batch=specs[0].dynamic_batch if specs else None,
        )
        resolved: list[SplitCandidateMetadata] = []
        for spec in specs:
            plan = plan_split(graph, spec)
            resolved.append(
                SplitCandidateMetadata(
                    requested_boundary=str(spec.boundary),
                    actual_split_id=str(getattr(plan, "split_id", spec.boundary)),
                    split_label=str(getattr(plan, "split_label", "")),
                    graph_signature=str(getattr(graph, "graph_shape_hash", "") or ""),
                    boundary_nodes=tuple(
                        str(item) for item in getattr(plan, "boundary_nodes", ()) or ()
                    ),
                    prefix_nodes=tuple(
                        str(item) for item in getattr(plan, "prefix_nodes", ()) or ()
                    ),
                    suffix_nodes=tuple(
                        str(item) for item in getattr(plan, "suffix_nodes", ()) or ()
                    ),
                )
            )
        return resolved


def _require_batch_gt1(example_batch: torch.Tensor) -> None:
    if int(example_batch.shape[0]) <= 1:
        raise ValueError("TorchLens batch_gt1 tracing requires example_batch batch size > 1.")


def build_split_runtime(
    model: Any,
    example_batch: torch.Tensor,
    config: SplitRuntimeConfig,
) -> SplitRuntime:
    _require_batch_gt1(example_batch)
    return prepare_split_runtime(model, example_batch, make_split_spec(config), mode=config.mode)


def build_replay_runtime(
    model: Any,
    example_batch: torch.Tensor,
    config: SplitRuntimeConfig,
) -> SplitRuntime:
    _require_batch_gt1(example_batch)
    replay_config = replace(config, trainable=False)
    return prepare_split_replay_runtime(
        model,
        example_batch,
        make_split_spec(replay_config),
        mode=config.mode,
    )


def maybe_warmup_runtime(runtime: Any, batch: torch.Tensor) -> None:
    run_prefix = getattr(runtime, "run_prefix", None)
    run_suffix = getattr(runtime, "run_suffix", None)
    if not callable(run_prefix) or not callable(run_suffix):
        return
    with torch.inference_mode():
        run_suffix(run_prefix(batch))


def trace_signature(runtime: Any) -> str:
    trace_graph = getattr(runtime, "trace_graph", None)
    return str(getattr(trace_graph, "graph_shape_hash", "") or "")


def get_split_runtime_metadata(runtime: Any) -> dict[str, Any]:
    plan = getattr(runtime, "plan", None)
    split_spec = getattr(runtime, "split_spec", None)
    return {
        "actual_split_id": getattr(runtime, "split_id", None),
        "graph_signature": trace_signature(runtime),
        "runtime_backend": "torchlens_native",
        "torchlens_mode": getattr(split_spec, "mode", None),
        "boundary_after": getattr(plan, "split_label", None),
        "boundary_nodes": list(getattr(plan, "boundary_nodes", ()) or ()),
        "prefix_nodes": list(getattr(plan, "prefix_nodes", ()) or ()),
        "suffix_nodes": list(getattr(plan, "suffix_nodes", ()) or ()),
        "boundary_bytes": 0,
        "trainable_suffix": bool(getattr(split_spec, "trainable", True)),
    }


__all__ = [
    "BoundaryPayload",
    "DEFAULT_SPLIT_MODE",
    "ReplayBoundary",
    "SplitRuntime",
    "SplitRuntimeConfig",
    "SplitSpec",
    "SplitCandidateMetadata",
    "build_replay_runtime",
    "build_split_runtime",
    "get_split_runtime_metadata",
    "make_split_spec",
    "maybe_warmup_runtime",
    "normalize_example_inputs",
    "prepare_split_replay_runtime",
    "prepare_split_runtime",
    "require_torchlens_native_split_api",
    "resolve_split_candidate_metadata",
    "trace_signature",
]
