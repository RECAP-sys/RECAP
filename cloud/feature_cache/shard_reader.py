from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from cloud.feature_cache.path_utils import fs_path
from cloud.feature_cache.types import (
    NPY_MEMMAP_SHARD,
    SAFETENSORS_SHARD,
    FeatureShardMetadata,
    FeatureShardRef,
)
from model_management.payload import BoundaryPayload, boundary_payload_from_tensors
from model_management.split_runtime import BoundaryPayloadCacheCodec, prepare_boundary_for_runtime

SAMPLE_AXIS_STORAGE_LAYOUT = "sample_axis"


def _read_json(path: str) -> dict[str, Any]:
    with open(fs_path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _contiguous_slice(rows: list[int]) -> slice | None:
    if not rows:
        return None
    start = int(rows[0])
    for offset, row in enumerate(rows):
        if int(row) != start + offset:
            return None
    return slice(start, start + len(rows))


def _metadata_for_ref(ref: FeatureShardRef) -> FeatureShardMetadata:
    if not ref.index_path or not os.path.exists(fs_path(ref.index_path)):
        raise FileNotFoundError(ref.index_path)
    payload = _read_json(ref.index_path)
    return FeatureShardMetadata.from_dict(payload)


def _leaf_spec(metadata: FeatureShardMetadata, leaf_key: str) -> dict[str, Any]:
    spec = metadata.leaf_specs.get(str(leaf_key))
    return dict(spec) if isinstance(spec, Mapping) else {}


def _symbolic_shape(schema: Mapping[str, Any]) -> list[Any]:
    shape = schema.get("symbolic_shape", schema.get("shape"))
    return list(shape or [])


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


def _leaf_batch_plan(leaf_spec: Mapping[str, Any]) -> tuple[int, int]:
    batch_axis = leaf_spec.get("batch_axis")
    batch_multiplier = leaf_spec.get("batch_multiplier")
    if batch_axis is not None:
        try:
            axis = int(batch_axis)
            multiplier = max(1, int(batch_multiplier or 1))
            return axis, multiplier
        except (TypeError, ValueError):
            pass

    schema = leaf_spec.get("schema")
    schema = dict(schema) if isinstance(schema, Mapping) else {}
    for axis, dim in enumerate(_symbolic_shape(schema)):
        multiplier = _batch_dimension_multiplier(dim)
        if multiplier is not None:
            return axis, multiplier
    return 0, 1


def _legacy_physical_rows(
    rows: Sequence[int],
    leaf_spec: Mapping[str, Any],
) -> tuple[list[int], int]:
    axis, multiplier = _leaf_batch_plan(leaf_spec)
    if axis != 0:
        raise RuntimeError(
            "Legacy flat feature shards only support batch-derived dimension at axis 0 "
            f"(got axis={axis})."
        )
    physical_rows: list[int] = []
    for row in rows:
        start = int(row) * int(multiplier)
        physical_rows.extend(range(start, start + int(multiplier)))
    return physical_rows, int(multiplier)


def _schema_from_metadata(metadata: FeatureShardMetadata) -> dict[str, Any]:
    return {
        str(spec.get("original_label") or leaf_key): spec.get("schema") or {}
        for leaf_key, spec in metadata.leaf_specs.items()
        if isinstance(spec, Mapping)
    }


def _is_sample_axis_leaf(leaf_spec: Mapping[str, Any]) -> bool:
    return str(leaf_spec.get("storage_layout") or "") == SAMPLE_AXIS_STORAGE_LAYOUT


class FeatureShardPayloadCache:
    def __init__(
        self, *, enabled: bool = True, max_cpu_bytes: int = 4 * 1024 * 1024 * 1024
    ) -> None:
        self.enabled = bool(enabled)
        self.max_cpu_bytes = max(0, int(max_cpu_bytes))
        self._payloads: dict[tuple[tuple[str, str, int], ...], BoundaryPayload] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, refs: Sequence[FeatureShardRef]) -> BoundaryPayload | None:
        if not self.enabled or not refs:
            return None
        key = self._key(refs)
        payload = self._payloads.get(key)
        if payload is None:
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def put(self, refs: Sequence[FeatureShardRef], payload: BoundaryPayload) -> None:
        if not self.enabled or not refs:
            return
        size = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in dict(payload.tensors or {}).values()
            if isinstance(tensor, torch.Tensor)
        )
        if self.max_cpu_bytes and size > self.max_cpu_bytes:
            return
        while self.max_cpu_bytes and self._bytes + size > self.max_cpu_bytes and self._payloads:
            _old_key, old_payload = self._payloads.popitem()
            self._bytes -= sum(
                int(tensor.numel() * tensor.element_size())
                for tensor in dict(old_payload.tensors or {}).values()
                if isinstance(tensor, torch.Tensor)
            )
        key = self._key(refs)
        self._payloads[key] = payload
        self._bytes += size

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total <= 0 else self.hits / float(total)

    @staticmethod
    def _key(refs: Sequence[FeatureShardRef]) -> tuple[tuple[str, str, int], ...]:
        return tuple((str(ref.shard_id), str(ref.index_path), int(ref.row_id)) for ref in refs)


class NpyMemmapShardReader:
    def __init__(self) -> None:
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self.files_opened = 0

    def read_group(
        self,
        refs: Sequence[FeatureShardRef],
        metadata: FeatureShardMetadata,
    ) -> OrderedDict[str, torch.Tensor]:
        rows = [int(ref.row_id) for ref in refs]
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        if not refs:
            return tensors
        shard_dir = refs[0].shard_dir
        if not shard_dir:
            raise ValueError("npy_memmap_shard ref is missing shard_dir.")
        for leaf_key in refs[0].leaf_keys:
            path = os.path.join(shard_dir, f"{leaf_key}.npy")
            cache_key = (shard_dir, leaf_key)
            array = self._arrays.get(cache_key)
            if array is None:
                array = np.load(fs_path(path), mmap_mode="r")
                self._arrays[cache_key] = array
                self.files_opened += 1
            leaf_spec = _leaf_spec(metadata, leaf_key)
            if _is_sample_axis_leaf(leaf_spec):
                selected = np.asarray(array[rows]).copy()
            else:
                physical_rows, multiplier = _legacy_physical_rows(rows, leaf_spec)
                selected = np.asarray(array[physical_rows]).copy()
                selected = selected.reshape(
                    len(rows),
                    multiplier,
                    *selected.shape[1:],
                )
            label = str(metadata.leaf_specs.get(leaf_key, {}).get("original_label") or leaf_key)
            tensors[label] = torch.from_numpy(selected)
        return tensors


class SafetensorsShardReader:
    def read_group(
        self,
        refs: Sequence[FeatureShardRef],
        metadata: FeatureShardMetadata,
    ) -> OrderedDict[str, torch.Tensor]:
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "safetensors_shard storage requires the 'safetensors' package; "
                "install it or configure npy_memmap_shard."
            ) from exc
        rows = [int(ref.row_id) for ref in refs]
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        if not refs:
            return tensors
        shard_path = refs[0].shard_path
        if not shard_path:
            raise ValueError("safetensors_shard ref is missing shard_path.")
        with safe_open(fs_path(shard_path), framework="pt", device="cpu") as handle:
            for leaf_key in refs[0].leaf_keys:
                leaf_spec = _leaf_spec(metadata, leaf_key)
                if _is_sample_axis_leaf(leaf_spec):
                    contiguous = _contiguous_slice(rows)
                    if contiguous is not None:
                        selected = handle.get_slice(leaf_key)[contiguous]
                    else:
                        full = handle.get_tensor(leaf_key)
                        selected = full.index_select(0, torch.tensor(rows, dtype=torch.long))
                else:
                    physical_rows, multiplier = _legacy_physical_rows(rows, leaf_spec)
                    contiguous = _contiguous_slice(physical_rows)
                    if contiguous is not None:
                        selected = handle.get_slice(leaf_key)[contiguous]
                    else:
                        full = handle.get_tensor(leaf_key)
                        selected = full.index_select(
                            0,
                            torch.tensor(physical_rows, dtype=torch.long),
                        )
                    selected = selected.reshape(
                        len(rows),
                        multiplier,
                        *selected.shape[1:],
                    )
                label = str(metadata.leaf_specs.get(leaf_key, {}).get("original_label") or leaf_key)
                tensors[label] = selected.contiguous()
        return tensors


class ShardFeatureBatchReader:
    def __init__(
        self,
        *,
        payload_cache: FeatureShardPayloadCache | None = None,
        pin_memory: bool = False,
        non_blocking_transfer: bool = True,
    ) -> None:
        self.payload_cache = payload_cache or FeatureShardPayloadCache(enabled=False)
        self.pin_memory = bool(pin_memory)
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self._npy = NpyMemmapShardReader()
        self._safetensors = SafetensorsShardReader()

    def read_batch(
        self,
        refs: Sequence[FeatureShardRef | Mapping[str, object]],
        *,
        device: torch.device | str | None = None,
        runtime: Any | None = None,
    ) -> BoundaryPayload:
        parsed = [
            ref if isinstance(ref, FeatureShardRef) else FeatureShardRef.from_dict(ref)
            for ref in list(refs or [])
        ]
        if not parsed:
            raise RuntimeError("Cannot read an empty feature shard batch.")
        cached = self.payload_cache.get(parsed)
        if cached is not None:
            return self._prepare_for_return(cached, device=device, runtime=runtime)

        groups: dict[tuple[str, str], list[tuple[int, FeatureShardRef]]] = {}
        for position, ref in enumerate(parsed):
            groups.setdefault((ref.storage_format, ref.shard_id), []).append((position, ref))

        per_sample_tensors: list[OrderedDict[str, torch.Tensor] | None] = [None] * len(parsed)
        per_sample_metadata: list[FeatureShardMetadata | None] = [None] * len(parsed)
        for (_format, _shard_id), positioned in groups.items():
            ordered_refs = [ref for _position, ref in positioned]
            metadata = _metadata_for_ref(ordered_refs[0])
            if _format == NPY_MEMMAP_SHARD:
                group_tensors = self._npy.read_group(ordered_refs, metadata)
            elif _format == SAFETENSORS_SHARD:
                group_tensors = self._safetensors.read_group(ordered_refs, metadata)
            else:
                raise ValueError(f"Unsupported feature shard storage_format={_format!r}.")
            for group_index, (position, _ref) in enumerate(positioned):
                per_sample_tensors[position] = OrderedDict(
                    (label, tensor[group_index].contiguous())
                    for label, tensor in group_tensors.items()
                )
                per_sample_metadata[position] = metadata

        sample_payloads: list[BoundaryPayload] = []
        for sample_tensors, metadata in zip(per_sample_tensors, per_sample_metadata, strict=True):
            if sample_tensors is None or metadata is None:
                raise RuntimeError(
                    "Feature shard reader did not reconstruct every requested sample."
                )
            sample_payloads.append(
                boundary_payload_from_tensors(
                    sample_tensors,
                    split_id=str(metadata.boundary_id or "split-tail"),
                    graph_signature=str(
                        metadata.metadata.get("graph_signature") or "feature-shard"
                    ),
                    batch_size=1,
                    schema=_schema_from_metadata(metadata),
                    metadata={
                        "feature_shard_ids": sorted({ref.shard_id for ref in parsed}),
                        "storage_formats": sorted({ref.storage_format for ref in parsed}),
                    },
                )
            )

        payload = BoundaryPayloadCacheCodec(None).collate(sample_payloads)
        if self.pin_memory and torch.cuda.is_available():
            payload = replace(
                payload,
                tensors=OrderedDict(
                    (label, tensor.pin_memory())
                    for label, tensor in dict(payload.tensors or {}).items()
                    if isinstance(tensor, torch.Tensor)
                ),
            )
        self.payload_cache.put(parsed, payload)
        return self._prepare_for_return(payload, device=device, runtime=runtime)

    def _prepare_for_return(
        self,
        payload: BoundaryPayload,
        *,
        device: torch.device | str | None,
        runtime: Any | None,
    ) -> BoundaryPayload:
        if runtime is not None:
            return prepare_boundary_for_runtime(runtime, payload, validate=True)
        return self._move_to_device(payload, device=device)

    def _move_to_device(
        self,
        payload: BoundaryPayload,
        *,
        device: torch.device | str | None,
    ) -> BoundaryPayload:
        if device in (None, "", "cpu"):
            return payload
        target = torch.device(device)
        moved = {
            label: tensor.to(target, non_blocking=self.non_blocking_transfer)
            for label, tensor in dict(payload.tensors or {}).items()
            if isinstance(tensor, torch.Tensor)
        }
        return replace(payload, tensors=moved)
