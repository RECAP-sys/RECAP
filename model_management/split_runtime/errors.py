from __future__ import annotations

from typing import Any, Mapping

import torch


class SplitRuntimeError(RuntimeError):
    """Base class for RECAP split orchestration failures."""


class BatchPrefixError(SplitRuntimeError):
    """Raised when split prefix execution fails for a batch."""


class BatchSuffixReplayError(SplitRuntimeError):
    """Raised when split suffix replay fails for a batch."""


class SplitTailTrainingError(SplitRuntimeError):
    """Raised when split suffix training fails for a batch."""


class UnsupportedModelAdapterError(SplitRuntimeError):
    """Raised when no detection adapter supports a model family."""


class MissingLossFunctionError(SplitRuntimeError):
    """Raised when split-tail training is requested without a loss function."""


class InvalidOutputStructureError(SplitRuntimeError):
    """Raised when a detector adapter cannot normalize runtime outputs."""


def _shape_of(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, Mapping):
        return {str(key): _shape_of(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_of(item) for item in value]
    return type(value).__name__


def input_shapes(inputs: tuple[Any, ...]) -> list[Any]:
    return [_shape_of(item) for item in inputs]


def boundary_tensor_shapes(boundary: Any) -> dict[str, tuple[int, ...]]:
    tensors = getattr(boundary, "tensors", {}) or {}
    return {
        str(label): tuple(int(dim) for dim in tensor.shape)
        for label, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    }


def error_context(
    *,
    model_name: str | None = None,
    model_family: str | None = None,
    split_id: str | None = None,
    graph_signature: str | None = None,
    batch_size: int | None = None,
    boundary_labels: list[str] | None = None,
    shapes: Any = None,
) -> str:
    parts: list[str] = []
    if model_name:
        parts.append(f"model_name={model_name}")
    if model_family:
        parts.append(f"model_family={model_family}")
    if split_id:
        parts.append(f"split_id={split_id}")
    if graph_signature:
        parts.append(f"graph_signature={graph_signature}")
    if batch_size is not None:
        parts.append(f"batch_size={batch_size}")
    if boundary_labels is not None:
        parts.append(f"boundary_labels={boundary_labels}")
    if shapes is not None:
        parts.append(f"shapes={shapes}")
    return ", ".join(parts)
