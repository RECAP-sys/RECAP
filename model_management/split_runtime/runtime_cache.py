from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .torchlens_native_runtime import SplitSpec


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(payload: object) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def symbolize_shape(shape: Any, *, batch_symbol: str = "B") -> list[Any]:
    dims = [int(dim) for dim in tuple(shape)]
    if dims:
        dims[0] = batch_symbol
    return dims


def _walk_tensors(value: Any, path: str, batch_symbol: str) -> list[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        return [
            {
                "path": path or "$",
                "shape": symbolize_shape(value.shape, batch_symbol=batch_symbol),
                "dtype": str(value.dtype),
                "requires_grad": bool(value.requires_grad),
            }
        ]
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key in sorted(value.keys(), key=str):
            rows.extend(
                _walk_tensors(
                    value[key],
                    f"{path}.{key}" if path else str(key),
                    batch_symbol,
                )
            )
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(
                _walk_tensors(
                    item,
                    f"{path}[{index}]" if path else f"[{index}]",
                    batch_symbol,
                )
            )
        return rows
    return []


def symbolic_input_schema(example_inputs: Any, *, batch_symbol: str = "B") -> list[dict[str, Any]]:
    inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(inputs):
        rows.extend(_walk_tensors(item, f"arg{index}", batch_symbol))
    return rows


def split_spec_payload(split_spec: SplitSpec) -> dict[str, Any]:
    return {
        "boundary": split_spec.boundary,
        "batch_symbol": split_spec.batch_symbol,
        "dynamic_batch": (
            list(split_spec.dynamic_batch) if split_spec.dynamic_batch is not None else None
        ),
        "trainable": bool(split_spec.trainable),
        "trace_batch_mode": str(split_spec.trace_batch_mode),
    }


def split_spec_hash(split_spec: SplitSpec) -> str:
    return stable_hash(split_spec_payload(split_spec))


@dataclass(frozen=True)
class RuntimeCacheKey:
    model_name: str
    model_family: str
    graph_signature: str | None
    split_plan_hash: str
    symbolic_input_schema_hash: str
    dynamic_batch: tuple[int, int] | None
    mode: str

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "graph_signature": self.graph_signature,
            "split_plan_hash": self.split_plan_hash,
            "symbolic_input_schema_hash": self.symbolic_input_schema_hash,
            "dynamic_batch": list(self.dynamic_batch) if self.dynamic_batch is not None else None,
            "mode": self.mode,
        }

    def log_string(self) -> str:
        return _stable_json_dumps(self.as_dict())

    def to_log_payload(self) -> dict[str, object]:
        return self.as_dict()


def make_runtime_cache_key(
    *,
    model_name: str,
    model_family: str,
    split_spec: SplitSpec,
    example_inputs: Any,
    graph_signature: str | None = None,
    split_plan_hash_value: str | None = None,
    mode: str = "generated_eager",
) -> RuntimeCacheKey:
    batch_symbol = split_spec.batch_symbol
    schema_hash = stable_hash(symbolic_input_schema(example_inputs, batch_symbol=batch_symbol))
    return RuntimeCacheKey(
        model_name=str(model_name),
        model_family=str(model_family),
        graph_signature=graph_signature,
        split_plan_hash=split_plan_hash_value or split_spec_hash(split_spec),
        symbolic_input_schema_hash=schema_hash,
        dynamic_batch=split_spec.dynamic_batch,
        mode=str(mode),
    )


@dataclass(frozen=True)
class RuntimeCacheLookup:
    runtime: Any
    cache_status: str


class RuntimeCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[Hashable, Any] = {}

    def get_or_create_lookup(self, key: Hashable, builder: Callable[[], Any]) -> RuntimeCacheLookup:
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                return RuntimeCacheLookup(runtime=cached, cache_status="hit")
        runtime = builder()
        with self._lock:
            cached = self._items.setdefault(key, runtime)
        return RuntimeCacheLookup(
            runtime=cached,
            cache_status="miss" if cached is runtime else "hit",
        )

    def get_or_create(self, key: Hashable, builder: Callable[[], Any]) -> Any:
        return self.get_or_create_lookup(key, builder).runtime

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


__all__ = [
    "RuntimeCache",
    "RuntimeCacheKey",
    "RuntimeCacheLookup",
    "make_runtime_cache_key",
    "split_spec_hash",
    "stable_hash",
    "symbolic_input_schema",
    "symbolize_shape",
]
