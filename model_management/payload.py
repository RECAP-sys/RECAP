from __future__ import annotations

import gzip
import io
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
import torchlens as tl
from torchlens.split import BoundaryTensorSpec

BoundaryPayload = tl.ReplayBoundary
ReplayBoundary = tl.ReplayBoundary


def _metadata_batch_size(tensors: Mapping[str, torch.Tensor], batch_size: int | None) -> int:
    if batch_size is not None:
        return int(batch_size)
    for tensor in tensors.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            return int(tensor.shape[0])
    return 1


def _schema_for_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    batch_symbol: str = "B",
) -> dict[str, BoundaryTensorSpec]:
    schema: dict[str, BoundaryTensorSpec] = {}
    for label, tensor in tensors.items():
        symbolic_shape: tuple[Any, ...]
        if tensor.ndim == 0:
            symbolic_shape = ()
        else:
            symbolic_shape = (batch_symbol, *tuple(int(dim) for dim in tensor.shape[1:]))
        schema[str(label)] = BoundaryTensorSpec(
            canonical_id=str(label),
            torchlens_label=str(label),
            module_path="recap_adapter",
            op_type="tensor",
            shape=symbolic_shape,
            dtype=tensor.dtype,
            requires_grad=bool(tensor.requires_grad),
            role="primary",
            output_index=None,
            device_policy="runtime",
        )
    return schema


def _normalise_spec(
    label: str,
    spec: Any,
    *,
    tensor: torch.Tensor | None = None,
) -> BoundaryTensorSpec:
    if isinstance(spec, BoundaryTensorSpec):
        return spec
    if isinstance(spec, Mapping):
        symbolic_shape = spec.get("symbolic_shape", spec.get("shape"))
        dtype = spec.get("dtype")
        if isinstance(dtype, str) and dtype.startswith("torch."):
            dtype = getattr(torch, dtype.split(".", maxsplit=1)[1], None)
        return BoundaryTensorSpec(
            canonical_id=str(spec.get("canonical_id") or label),
            torchlens_label=str(spec.get("torchlens_label") or label),
            module_path=str(spec.get("module_path") or ""),
            op_type=str(spec.get("op_type") or spec.get("op") or ""),
            shape=tuple(symbolic_shape) if symbolic_shape is not None else None,
            dtype=dtype if isinstance(dtype, torch.dtype) else getattr(tensor, "dtype", None),
            requires_grad=bool(spec.get("requires_grad", getattr(tensor, "requires_grad", False))),
            role=str(spec.get("role") or "primary"),
            output_index=(
                None if spec.get("output_index") is None else int(spec.get("output_index"))
            ),
            device_policy=str(spec.get("device_policy") or "runtime"),
        )
    if tensor is None:
        tensor = torch.empty(())
    return _schema_for_tensors({label: tensor})[label]


def _normalise_schema(
    tensors: Mapping[str, torch.Tensor],
    schema: Mapping[str, Any] | None,
) -> dict[str, BoundaryTensorSpec]:
    if not schema:
        return _schema_for_tensors(tensors)
    result: dict[str, BoundaryTensorSpec] = {}
    for label, tensor in tensors.items():
        result[str(label)] = _normalise_spec(
            str(label),
            dict(schema).get(str(label)),
            tensor=tensor,
        )
    return result


def boundary_payload_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    split_id: str,
    graph_signature: str,
    batch_size: int | None = None,
    schema: Mapping[str, Any] | None = None,
    requires_grad: Mapping[str, bool] | None = None,
    weight_version: int | None = None,
    supports_prefix_backward: bool = False,
    prefix_backward_owner_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryPayload:
    ordered = OrderedDict(
        (str(label), tensor)
        for label, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    )
    if not ordered:
        raise ValueError("BoundaryPayload requires at least one tensor.")
    spec = _normalise_schema(ordered, schema)
    if requires_grad:
        spec = {
            label: replace(
                item,
                requires_grad=bool(dict(requires_grad).get(label, item.requires_grad)),
            )
            for label, item in spec.items()
        }
    batch = _metadata_batch_size(ordered, batch_size)
    meta = dict(metadata or {})
    meta.pop("protocol_version", None)
    meta.update(
        {
            "split_id": str(split_id),
            "split_label": str(meta.get("split_label") or split_id),
            "graph_shape_hash": str(graph_signature),
            "graph_signature": str(graph_signature),
            "batch_size": int(batch),
            "boundary_order": tuple(ordered.keys()),
            "weight_version": weight_version,
            "supports_prefix_backward": bool(supports_prefix_backward),
            "prefix_backward_owner_id": prefix_backward_owner_id,
        }
    )
    return ReplayBoundary(tensors=dict(ordered), spec=spec, metadata=meta)


def serialize_boundary_payload(payload: BoundaryPayload, *, compress: bool = False) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "tensors": dict(payload.tensors),
            "spec": dict(payload.spec),
            "metadata": dict(payload.metadata),
        },
        buffer,
    )
    data = buffer.getvalue()
    return gzip.compress(data) if compress else data


def deserialize_boundary_payload(data: bytes, *, compressed: bool = False) -> BoundaryPayload:
    raw = gzip.decompress(data) if compressed else data
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if isinstance(payload, ReplayBoundary):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported boundary payload transport type: {type(payload)!r}")
    if "tensors" in payload and "spec" in payload:
        tensors = {
            str(label): tensor
            for label, tensor in dict(payload.get("tensors") or {}).items()
            if isinstance(tensor, torch.Tensor)
        }
        metadata = dict(payload.get("metadata") or {})
        split_id = str(payload.get("split_id") or metadata.get("split_id") or "")
        graph_signature = str(
            payload.get("graph_signature")
            or metadata.get("graph_shape_hash")
            or metadata.get("graph_signature")
            or ""
        )
        return boundary_payload_from_tensors(
            tensors,
            split_id=split_id,
            graph_signature=graph_signature,
            batch_size=payload.get("batch_size") or metadata.get("batch_size"),
            schema=payload.get("spec"),
            requires_grad=payload.get("requires_grad"),
            weight_version=payload.get("weight_version") or metadata.get("weight_version"),
            metadata=metadata,
        )
    raise TypeError(
        "Boundary payload transport did not contain ReplayBoundary tensors/spec metadata."
    )


__all__ = [
    "BoundaryPayload",
    "BoundaryTensorSpec",
    "boundary_payload_from_tensors",
    "deserialize_boundary_payload",
    "serialize_boundary_payload",
]
