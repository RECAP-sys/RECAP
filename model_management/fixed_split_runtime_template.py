from __future__ import annotations

from typing import Any, Mapping

from model_management.split_candidate import SplitCandidate
from model_management.split_runtime.template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateCache,
    FixedSplitRuntimeTemplateKey,
    FixedSplitRuntimeTemplateLookup,
    bind_request_runtime_from_template,
    fixed_split_runtime_template_key,
    get_fixed_split_runtime_template_cache,
)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_candidate_descriptor,
    reconstruct_candidate_from_descriptor,
)


def describe_split_candidate(candidate: SplitCandidate) -> dict[str, Any]:
    return build_candidate_descriptor(candidate)


def restore_split_candidate(graph: Any, descriptor: Mapping[str, Any] | None) -> SplitCandidate:
    candidate = reconstruct_candidate_from_descriptor(
        graph, descriptor, source="torchlens_template"
    )
    if candidate is None:
        raise RuntimeError("Could not restore split candidate from TorchLens runtime template.")
    return candidate


def compute_graph_trace_signature(graph: Any) -> str:
    if isinstance(graph, str):
        return graph
    return str(getattr(graph, "graph_signature", "") or getattr(graph, "signature", "") or "")


def freeze_trace_timings(trace_timings: Mapping[str, float] | None) -> Mapping[str, float]:
    return {str(key): float(value) for key, value in dict(trace_timings or {}).items()}


def bind_request_splitter_from_template(
    runtime_model,
    template: FixedSplitRuntimeTemplate,
    *,
    example_inputs=None,
    device="cpu",
):
    runtime = bind_request_runtime_from_template(
        template,
        model=runtime_model,
        example_inputs=example_inputs,
        device=device,
    )
    splitter = UniversalModelSplitter(device=device).bind_runtime(
        runtime,
        model=runtime_model,
        split_spec=template.split_spec,
    )
    return splitter, splitter.current_candidate


__all__ = [
    "FixedSplitRuntimeTemplate",
    "FixedSplitRuntimeTemplateCache",
    "FixedSplitRuntimeTemplateKey",
    "FixedSplitRuntimeTemplateLookup",
    "bind_request_runtime_from_template",
    "bind_request_splitter_from_template",
    "compute_graph_trace_signature",
    "describe_split_candidate",
    "fixed_split_runtime_template_key",
    "freeze_trace_timings",
    "get_fixed_split_runtime_template_cache",
    "restore_split_candidate",
]
