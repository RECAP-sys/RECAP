from __future__ import annotations

import gzip
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from model_management.payload import BoundaryPayload

BOUNDARY_CACHE_PROTOCOL = "torchlens-native-boundary"


def _runtime_from(value: Any) -> Any:
    if value is None:
        return None
    ensure_runtime = getattr(value, "_ensure_runtime", None)
    if callable(ensure_runtime):
        try:
            return ensure_runtime()
        except RuntimeError:
            pass
    return getattr(value, "runtime", value)


def _runtime_batch_symbol(runtime: Any) -> str:
    trace_graph = getattr(runtime, "trace_graph", None)
    return str(getattr(trace_graph, "batch_symbol", None) or "B")


def _first_module_device(module: Any) -> torch.device | None:
    if not isinstance(module, torch.nn.Module):
        return None
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _runtime_payload_device(runtime: Any) -> torch.device | None:
    if runtime is None:
        return None
    runtime_device = getattr(runtime, "_runtime_device", None)
    if callable(runtime_device):
        device = runtime_device()
        if device is not None:
            return torch.device(device)
    for module_name in ("segments", "suffix_segment", "prefix_segment", "training_prefix_segment"):
        module = getattr(runtime, module_name, None)
        if module_name == "segments":
            module = getattr(module, "suffix", None)
        device = _first_module_device(module)
        if device is not None:
            return device
    model = getattr(runtime, "model", None)
    device = _first_module_device(model)
    return device


def _spec_dtype(spec: Any) -> torch.dtype | None:
    dtype = spec.get("dtype") if isinstance(spec, Mapping) else getattr(spec, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        text = dtype.replace("torch.", "")
        candidate = getattr(torch, text, None)
        if isinstance(candidate, torch.dtype):
            return candidate
    return None


def _payload_values_match_runtime(
    tensors: Mapping[str, Any],
    spec: Mapping[str, Any],
    device: torch.device | None,
) -> bool:
    for label, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        target_dtype = _spec_dtype(spec.get(str(label)))
        if target_dtype is not None and tensor.dtype != target_dtype:
            return False
        if device is not None and tensor.device != device:
            return False
        if not tensor.is_contiguous():
            return False
    return True


def _coerce_payload_tensors_for_runtime(
    tensors: Mapping[str, Any],
    spec: Mapping[str, Any],
    device: torch.device | None,
) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for label, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            coerced[label] = tensor
            continue
        target_dtype = _spec_dtype(spec.get(str(label))) or tensor.dtype
        target_device = device or tensor.device
        moved = tensor.to(device=target_device, dtype=target_dtype)
        coerced[label] = moved if moved.is_contiguous() else moved.contiguous()
    return coerced


def _cpu_cache_payload(payload: BoundaryPayload) -> BoundaryPayload:
    tensors: dict[str, Any] = {}
    for label, tensor in dict(payload.tensors).items():
        if isinstance(tensor, torch.Tensor):
            cpu = tensor.detach().to("cpu")
            tensors[str(label)] = cpu if cpu.is_contiguous() else cpu.contiguous()
        else:
            tensors[str(label)] = tensor
    return replace(payload, tensors=tensors, metadata=dict(payload.metadata))


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    try:
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    except Exception:
        return False


def get_runtime_boundary_codec(runtime: Any) -> "BoundaryPayloadCacheCodec":
    runtime_obj = _runtime_from(runtime)
    if runtime_obj is None:
        return BoundaryPayloadCacheCodec(None)
    codec = getattr(runtime_obj, "_plank_boundary_payload_codec", None)
    if not isinstance(codec, BoundaryPayloadCacheCodec):
        codec = BoundaryPayloadCacheCodec(runtime_obj)
        try:
            setattr(runtime_obj, "_plank_boundary_payload_codec", codec)
        except Exception:
            pass
    return codec


def prepare_boundary_for_runtime(
    runtime: Any,
    boundary: BoundaryPayload,
    *,
    validate: bool,
) -> BoundaryPayload:
    codec = get_runtime_boundary_codec(runtime)
    prepared = codec.to_runtime_device(boundary)
    return codec.validate(prepared) if validate else prepared


class BoundaryPayloadCacheCodec:
    def __init__(self, runtime: Any) -> None:
        self.runtime = _runtime_from(runtime)
        self.batch_symbol = _runtime_batch_symbol(self.runtime)

    def _runtime_spec(self, payload: BoundaryPayload) -> dict[str, Any]:
        runtime_spec = getattr(self.runtime, "boundary_spec", None)
        if isinstance(runtime_spec, Mapping) and set(runtime_spec) == set(payload.tensors):
            return dict(runtime_spec)
        plan_spec = getattr(getattr(self.runtime, "plan", None), "boundary_specs", None)
        if isinstance(plan_spec, Mapping) and set(plan_spec) == set(payload.tensors):
            return dict(plan_spec)
        return dict(getattr(payload, "spec", {}) or {})

    def validate(self, payload: BoundaryPayload) -> BoundaryPayload:
        if not isinstance(payload, BoundaryPayload):
            raise TypeError(f"Expected BoundaryPayload, got {type(payload).__name__}.")
        validate_boundary = getattr(self.runtime, "validate_boundary", None)
        if callable(validate_boundary):
            validate_boundary(payload)
        else:
            expected = self._runtime_spec(payload)
            if expected:
                payload.validate(expected)
        return payload

    def _validate_schema_only(self, payload: BoundaryPayload) -> BoundaryPayload:
        if not isinstance(payload, BoundaryPayload):
            raise TypeError(f"Expected BoundaryPayload, got {type(payload).__name__}.")
        expected = self._runtime_spec(payload)
        if expected:
            payload.validate(expected)
        return payload

    def to_runtime_device(self, payload: BoundaryPayload) -> BoundaryPayload:
        if not isinstance(payload, BoundaryPayload):
            raise TypeError(f"Expected BoundaryPayload, got {type(payload).__name__}.")
        device = _runtime_payload_device(self.runtime)
        spec = self._runtime_spec(payload)
        metadata = dict(payload.metadata)
        tensors = dict(payload.tensors)
        if _payload_values_match_runtime(tensors, spec, device) and spec == dict(payload.spec):
            return payload
        return replace(
            payload,
            tensors=_coerce_payload_tensors_for_runtime(tensors, spec, device),
            spec=spec,
            metadata=metadata,
        )

    def split_batch(
        self,
        payload: BoundaryPayload,
        actual_batch_size: int | None = None,
    ) -> list[BoundaryPayload]:
        payload = self.to_runtime_device(payload)
        self.validate(payload)
        payload_batch_size = int(getattr(payload, "batch_size", 0) or 0)
        if payload_batch_size <= 0:
            raise RuntimeError("BoundaryPayload cache split requires a positive batch_size.")
        actual = payload_batch_size if actual_batch_size is None else int(actual_batch_size)
        if actual < 0 or actual > payload_batch_size:
            raise RuntimeError(
                "Cannot split "
                f"{actual} sample(s) from BoundaryPayload batch_size={payload_batch_size}."
            )

        spec = self._runtime_spec(payload)
        if not spec:
            raise RuntimeError("BoundaryPayload cache split requires TorchLens boundary spec.")

        sample_tensors: list[dict[str, torch.Tensor]] = [dict() for _ in range(actual)]
        for label, tensor in dict(payload.tensors).items():
            if not isinstance(tensor, torch.Tensor):
                continue
            tensor_spec = spec.get(str(label))
            if tensor_spec is None:
                raise RuntimeError(f"Boundary tensor {label!r} is missing from payload spec.")
            plan = self._tensor_batch_plan(str(label), tensor, tensor_spec, payload_batch_size)
            if plan is None:
                for sample in sample_tensors:
                    sample[str(label)] = tensor
                continue
            axis, multiplier = plan
            for sample_index, sample in enumerate(sample_tensors):
                sample[str(label)] = tensor.narrow(axis, sample_index * multiplier, multiplier)

        payloads = []
        for index in range(actual):
            metadata = dict(payload.metadata)
            metadata["batch_size"] = 1
            payloads.append(
                replace(
                    payload,
                    tensors=sample_tensors[index],
                    spec=spec,
                    metadata=metadata,
                )
            )
        for sample_payload in payloads:
            self._validate_schema_only(sample_payload)
        return payloads

    def collate(self, payloads: list[BoundaryPayload]) -> BoundaryPayload:
        if not payloads:
            raise RuntimeError("Cannot collate an empty BoundaryPayload cache batch.")
        payloads = [self.to_runtime_device(payload) for payload in payloads]
        for payload in payloads:
            self._validate_schema_only(payload)

        first = payloads[0]
        spec = self._runtime_spec(first)
        if not spec:
            raise RuntimeError("BoundaryPayload cache collate requires TorchLens boundary spec.")
        for payload in payloads[1:]:
            if str(payload.split_id) != str(first.split_id):
                raise RuntimeError(
                    "Cannot collate BoundaryPayload objects with different split_id."
                )
            if dict(payload.spec) != dict(first.spec):
                raise RuntimeError("Cannot collate BoundaryPayload objects with different spec.")

        batch_size = sum(int(getattr(payload, "batch_size", 0) or 0) for payload in payloads)
        if batch_size <= 0:
            raise RuntimeError("Cannot collate BoundaryPayload objects with empty batch_size.")

        batched_tensors: dict[str, torch.Tensor] = {}
        labels = list(dict(first.tensors).keys())
        for label in labels:
            tensor_spec = spec.get(str(label))
            if tensor_spec is None:
                raise RuntimeError(f"Boundary tensor {label!r} is missing from payload spec.")
            pieces = [dict(payload.tensors)[label] for payload in payloads]
            first_piece = pieces[0]
            if not isinstance(first_piece, torch.Tensor):
                continue
            plan = self._tensor_batch_plan(
                str(label),
                first_piece,
                tensor_spec,
                int(getattr(first, "batch_size", 0) or 0),
            )
            if plan is None:
                for piece in pieces[1:]:
                    if not _same_tensor(first_piece, piece):
                        raise RuntimeError(
                            f"Boundary tensor {label!r} is schema-shared "
                            "but differs across samples."
                        )
                batched_tensors[str(label)] = first_piece
                continue
            axis, multiplier = plan
            for payload, piece in zip(payloads, pieces, strict=True):
                expected = multiplier * int(getattr(payload, "batch_size", 0) or 0)
                if int(piece.shape[axis]) != expected:
                    raise RuntimeError(
                        f"Boundary tensor {label!r} dimension {axis} is {int(piece.shape[axis])}; "
                        f"expected {expected}."
                    )
            target_device = first_piece.device
            batched_tensors[str(label)] = torch.cat(
                [piece.to(target_device) for piece in pieces],
                dim=axis,
            )
        metadata = dict(first.metadata)
        metadata["batch_size"] = batch_size
        result = replace(first, tensors=batched_tensors, spec=spec, metadata=metadata)
        self.validate(result)
        return result

    def save(
        self,
        path: str | os.PathLike[str],
        payload: BoundaryPayload,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = _cpu_cache_payload(self.validate(self.to_runtime_device(payload)))
        record = dict(metadata or {})
        record.update(
            {
                "cache_protocol": BOUNDARY_CACHE_PROTOCOL,
                "intermediate": payload,
            }
        )
        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
        with gzip.open(path, "wb", compresslevel=1) as handle:
            torch.save(record, handle)
        return record

    def load(self, path: str | os.PathLike[str]) -> BoundaryPayload:
        try:
            with gzip.open(path, "rb") as handle:
                record = torch.load(handle, map_location="cpu", weights_only=False)
        except gzip.BadGzipFile:
            record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, Mapping):
            raise TypeError(f"Unsupported boundary cache record: {type(record).__name__}.")
        protocol = str(record.get("cache_protocol") or "")
        if protocol != BOUNDARY_CACHE_PROTOCOL:
            raise RuntimeError(
                f"Unsupported boundary cache protocol {protocol!r}; rebuild feature cache."
            )
        payload = record.get("intermediate")
        if isinstance(payload, BoundaryPayload):
            return self.validate(self.to_runtime_device(payload))
        raise TypeError("Boundary cache record did not contain a BoundaryPayload.")

    def _dimension_multiplier(self, value: Any) -> int | None:
        if isinstance(value, str):
            if value == self.batch_symbol:
                return 1
            prefix = f"{self.batch_symbol}*"
            if value.startswith(prefix):
                try:
                    multiplier = int(value[len(prefix) :])
                except ValueError:
                    return None
                return multiplier if multiplier > 0 else None
            return None
        expression = getattr(value, "expression", None)
        if expression != self.batch_symbol:
            return None
        offset = int(getattr(value, "offset", 0) or 0)
        if offset != 0:
            raise RuntimeError(
                f"Cannot split affine boundary dimension with non-zero offset: {value}."
            )
        multiplier = int(getattr(value, "multiplier", 1) or 1)
        if multiplier <= 0:
            raise RuntimeError(f"Cannot split boundary dimension with multiplier={multiplier}.")
        return multiplier

    def _tensor_batch_plan(
        self,
        label: str,
        tensor: torch.Tensor,
        spec: Any,
        batch_size: int,
    ) -> tuple[int, int] | None:
        symbolic_shape = tuple(
            getattr(spec, "shape", None) or getattr(spec, "symbolic_shape", None) or ()
        )
        if tensor.ndim == 0 or not symbolic_shape:
            return None
        if tensor.ndim != len(symbolic_shape):
            raise RuntimeError(
                f"Boundary tensor {label!r} rank {tensor.ndim} does not match spec rank "
                f"{len(symbolic_shape)}."
            )
        batch_dims: list[tuple[int, int]] = []
        for axis, dim in enumerate(symbolic_shape):
            multiplier = self._dimension_multiplier(dim)
            if multiplier is not None:
                batch_dims.append((axis, multiplier))
        if not batch_dims:
            return None
        if len(batch_dims) > 1:
            raise RuntimeError(f"Boundary tensor {label!r} has multiple batch-derived dimensions.")
        axis, multiplier = batch_dims[0]
        expected = int(batch_size) * multiplier
        actual = int(tensor.shape[axis])
        if actual != expected:
            raise RuntimeError(
                f"Boundary tensor {label!r} dimension {axis} is {actual}; expected {expected}."
            )
        return axis, multiplier


__all__ = [
    "BOUNDARY_CACHE_PROTOCOL",
    "BoundaryPayloadCacheCodec",
    "get_runtime_boundary_codec",
    "prepare_boundary_for_runtime",
]
