from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        delete=False,
    )
    try:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
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


def contract_path(
    root_dir: str,
    *,
    edge_id: int | str,
    model_id: str,
    split_config_id: str,
) -> str:
    return os.path.join(
        root_dir,
        f"edge_{_sanitize_segment(edge_id)}",
        _sanitize_segment(model_id),
        f"{_sanitize_segment(split_config_id)}.json",
    )


def normalise_feature_tensors(value: object) -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {"payload": value.detach().cpu()}
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported feature payload: {type(value).__name__}")
    source = value.get("feature") if isinstance(value.get("feature"), Mapping) else value
    if isinstance(source, Mapping) and isinstance(source.get("tensors"), Mapping):
        source = source["tensors"]
    tensors = {
        str(label): tensor.detach().cpu()
        for label, tensor in dict(source or {}).items()
        if isinstance(tensor, torch.Tensor)
    }
    if not tensors:
        raise ValueError("Feature payload did not contain any tensors.")
    return tensors


def feature_layout_from_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for label, tensor in sorted(dict(tensors).items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = [int(dim) for dim in tensor.shape]
        layout[str(label)] = {
            "dtype": str(tensor.dtype),
            "shape_without_batch": shape[1:] if shape else [],
        }
    if not layout:
        raise ValueError("Cannot compute feature layout without tensor features.")
    return layout


def feature_layout_id(layout: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha1(_stable_json(layout).encode("utf-8")).hexdigest()


def _normalise_dtype(value: object) -> str:
    return str(value or "").replace("torch.", "")


def _normalise_shape_dim(value: object) -> int | str:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _symbolic_shape_without_batch(shape: object) -> list[int | str]:
    dims = list(shape or []) if isinstance(shape, (list, tuple)) else []
    if not dims:
        return []
    return [_normalise_shape_dim(dim) for dim in dims[1:]]


def _symbolize_batch_shape(shape: object, *, batch_symbol: str = "B") -> list[int | str]:
    dims = list(shape or []) if isinstance(shape, (list, tuple)) else []
    if not dims:
        return []
    return [batch_symbol, *[_normalise_shape_dim(dim) for dim in dims[1:]]]


@dataclass(frozen=True)
class FeatureAbiSpec:
    model_family: str
    canonical_split_key: str
    graph_signature: str
    boundary_tensor_labels: list[str]
    boundary_tensors: list[dict[str, Any]]
    boundary_schema: dict[str, dict[str, Any]]
    preprocessing_abi: dict[str, Any]
    passthrough_specs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ordered_feature_layout(
    feature_layout: Mapping[str, Mapping[str, Any]] | None,
    labels: list[str],
) -> list[dict[str, Any]]:
    layout = {
        str(label): dict(spec)
        for label, spec in dict(feature_layout or {}).items()
        if isinstance(spec, Mapping)
    }
    ordered_labels = [label for label in labels if label in layout]
    seen = set(ordered_labels)
    ordered_labels.extend(label for label in sorted(layout) if label not in seen)
    tensors: list[dict[str, Any]] = []
    for label in ordered_labels:
        spec = dict(layout.get(label) or {})
        shape_without_batch = list(spec.get("shape_without_batch") or [])
        if not shape_without_batch:
            shape_without_batch = _symbolic_shape_without_batch(
                spec.get("shape") or spec.get("sample_shape")
            )
        normalised_shape = [_normalise_shape_dim(dim) for dim in shape_without_batch]
        rank = spec.get("rank")
        try:
            rank_value = int(rank) if rank is not None else len(normalised_shape) + 1
        except (TypeError, ValueError):
            rank_value = len(normalised_shape) + 1
        tensors.append(
            {
                "label": str(label),
                "dtype": _normalise_dtype(spec.get("dtype")),
                "rank": rank_value,
                "shape_without_batch": normalised_shape,
            }
        )
    return tensors


def _normalise_boundary_schema_for_abi(
    boundary_schema: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    schema = _normalise_boundary_schema(boundary_schema)
    for spec in schema.values():
        spec["symbolic_shape"] = _symbolize_batch_shape(spec.get("symbolic_shape"))
        spec["dtype"] = _normalise_dtype(spec.get("dtype"))
    return schema


def build_feature_abi_spec(
    *,
    model_id: str = "",
    model_family: str = "",
    canonical_split_key: str = "",
    graph_signature: str = "",
    boundary_tensor_labels: list[str] | tuple[str, ...] | None = None,
    boundary_schema: Mapping[str, Any] | None = None,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
    input_tensor_shape: list[int] | tuple[int, ...] | None = None,
    input_resize_mode: str = "",
    passthrough_specs: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = dict(runtime_identity or {})
    runtime_contract = dict(identity.get("runtime_contract") or {})
    labels = [str(label) for label in list(boundary_tensor_labels or [])]
    if not labels:
        labels = [
            str(label) for label in list(runtime_contract.get("boundary_tensor_labels") or [])
        ]
    resolved_boundary_schema = boundary_schema
    if resolved_boundary_schema is None and isinstance(
        runtime_contract.get("boundary_schema"), Mapping
    ):
        resolved_boundary_schema = runtime_contract.get("boundary_schema")  # type: ignore[assignment]
    resolved_model_family = (
        str(model_family or "")
        or str(identity.get("model_family") or "")
        or str(runtime_contract.get("model_family") or "")
        or str(model_id or identity.get("model_id") or runtime_contract.get("model_id") or "")
    )
    resolved_graph_signature = (
        str(graph_signature or "")
        or str(identity.get("graph_signature") or "")
        or str(runtime_contract.get("trace_signature") or "")
    )
    preprocessing_shape = (
        input_tensor_shape
        or identity.get("input_tensor_shape")
        or runtime_contract.get("input_tensor_shape")
        or []
    )
    spec = FeatureAbiSpec(
        model_family=resolved_model_family,
        canonical_split_key=str(
            canonical_split_key
            or identity.get("canonical_split_key")
            or identity.get("cloud_batch_split_id")
            or runtime_contract.get("logical_split_id")
            or ""
        ),
        graph_signature=resolved_graph_signature,
        boundary_tensor_labels=labels,
        boundary_tensors=_ordered_feature_layout(feature_layout, labels),
        boundary_schema=_normalise_boundary_schema_for_abi(resolved_boundary_schema),
        preprocessing_abi={
            "input_tensor_shape": _symbolize_batch_shape(preprocessing_shape),
            "input_resize_mode": str(
                input_resize_mode
                or identity.get("input_resize_mode")
                or runtime_contract.get("input_resize_mode")
                or "direct_resize"
            )
            .strip()
            .lower(),
        },
        passthrough_specs=dict(
            passthrough_specs
            or identity.get("passthrough_specs")
            or identity.get("passthrough_schema")
            or {}
        ),
    )
    return spec.to_dict()


def feature_abi_id(spec: FeatureAbiSpec | Mapping[str, Any]) -> str:
    payload = spec.to_dict() if isinstance(spec, FeatureAbiSpec) else dict(spec)
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def _normalise_boundary_schema(
    boundary_schema: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    normalised: dict[str, dict[str, Any]] = {}
    for label, spec in dict(boundary_schema or {}).items():
        if isinstance(spec, Mapping):
            symbolic_shape = spec.get("symbolic_shape") or spec.get("shape") or ()
            dtype = spec.get("dtype")
            requires_grad = spec.get("requires_grad", False)
            canonical_id = spec.get("canonical_id") or label
            torchlens_label = spec.get("torchlens_label") or label
            module_path = spec.get("module_path") or ""
            op_type = spec.get("op_type") or spec.get("op") or ""
            role = spec.get("role") or "primary"
            output_index = spec.get("output_index")
            device_policy = spec.get("device_policy") or "runtime"
        else:
            symbolic_shape = (
                getattr(spec, "shape", None) or getattr(spec, "symbolic_shape", None) or ()
            )
            dtype = getattr(spec, "dtype", "")
            requires_grad = getattr(spec, "requires_grad", False)
            canonical_id = getattr(spec, "canonical_id", label)
            torchlens_label = getattr(spec, "torchlens_label", label)
            module_path = getattr(spec, "module_path", "")
            op_type = getattr(spec, "op_type", "")
            role = getattr(spec, "role", "primary")
            output_index = getattr(spec, "output_index", None)
            device_policy = getattr(spec, "device_policy", "runtime")
        normalised[str(label)] = {
            "canonical_id": str(canonical_id or label),
            "torchlens_label": str(torchlens_label or label),
            "module_path": str(module_path or ""),
            "op_type": str(op_type or ""),
            "symbolic_shape": [str(dim) for dim in list(symbolic_shape or [])],
            "dtype": str(dtype or ""),
            "requires_grad": bool(requires_grad),
            "role": str(role or "primary"),
            "output_index": None if output_index is None else int(output_index),
            "device_policy": str(device_policy or "runtime"),
        }
    return normalised


def compute_feature_layout_id(
    *,
    model_id: str = "",
    model_version: str = "",
    logical_split_id: str = "",
    trace_signature: str = "",
    input_tensor_shape: list[int] | tuple[int, ...] | None = None,
    input_resize_mode: str = "",
    boundary_tensor_labels: list[str] | tuple[str, ...] | None = None,
    boundary_schema: Mapping[str, Any] | None = None,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    payload = {
        "format": "feature-layout",
        "model_id": str(model_id or ""),
        "model_version": str(model_version or ""),
        "logical_split_id": str(logical_split_id or ""),
        "trace_signature": str(trace_signature or ""),
        "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape or [])],
        "input_resize_mode": str(input_resize_mode or ""),
        "boundary_tensor_labels": [str(label) for label in list(boundary_tensor_labels or [])],
        "boundary_schema": _normalise_boundary_schema(boundary_schema),
        "feature_layout": {
            str(label): dict(spec)
            for label, spec in dict(feature_layout or {}).items()
            if isinstance(spec, Mapping)
        },
    }
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def build_runtime_contract(
    *,
    logical_split_id: str,
    trace_signature: str,
    trace_device_type: str,
    runtime_backend: str,
    boundary_tensor_labels: list[str] | tuple[str, ...],
    boundary_schema: Mapping[str, Any] | None,
    model_id: str,
    model_version: str,
    input_tensor_shape: list[int] | tuple[int, ...],
    input_resize_mode: str,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    labels = [str(label) for label in list(boundary_tensor_labels or [])]
    schema = _normalise_boundary_schema(boundary_schema)
    layout = {
        str(label): dict(spec)
        for label, spec in dict(feature_layout or {}).items()
        if isinstance(spec, Mapping)
    }
    layout_id = compute_feature_layout_id(
        model_id=str(model_id),
        model_version=str(model_version),
        logical_split_id=str(logical_split_id),
        trace_signature=str(trace_signature),
        input_tensor_shape=[int(dim) for dim in list(input_tensor_shape or [])],
        input_resize_mode=str(input_resize_mode or ""),
        boundary_tensor_labels=labels,
        boundary_schema=schema,
        feature_layout=layout,
    )
    abi_spec = build_feature_abi_spec(
        model_id=str(model_id),
        model_family=str(model_id),
        canonical_split_key=str(logical_split_id),
        graph_signature=str(trace_signature),
        boundary_tensor_labels=labels,
        boundary_schema=schema,
        feature_layout=layout,
        input_tensor_shape=[int(dim) for dim in list(input_tensor_shape or [])],
        input_resize_mode=str(input_resize_mode or ""),
    )
    abi_id = feature_abi_id(abi_spec)
    return {
        "logical_split_id": str(logical_split_id),
        "trace_signature": str(trace_signature or ""),
        "trace_device_type": str(trace_device_type or ""),
        "runtime_backend": str(runtime_backend or ""),
        "boundary_tensor_labels": labels,
        "boundary_schema": schema,
        "feature_layout": layout,
        "feature_layout_id": layout_id,
        "feature_abi_id": abi_id,
        "feature_abi_spec": abi_spec,
        "model_id": str(model_id or ""),
        "model_version": str(model_version or ""),
        "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape or [])],
        "input_resize_mode": str(input_resize_mode or ""),
    }


def _contract_payload(contract: Mapping[str, Any] | object | None) -> dict[str, Any]:
    if contract is None:
        return {}
    if isinstance(contract, Mapping):
        return dict(contract)
    to_dict = getattr(contract, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _payload_feature_abi_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("feature_abi_id") or "")


def _payload_runtime_identity_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("runtime_identity_id") or "")


def classify_contract_compatibility(
    edge_contract: Mapping[str, Any] | object | None,
    cloud_contract: Mapping[str, Any] | object | None,
) -> dict[str, Any]:
    edge = _contract_payload(edge_contract)
    cloud = _contract_payload(cloud_contract)
    edge_layout_id = str(edge.get("feature_layout_id") or "")
    cloud_layout_id = str(cloud.get("feature_layout_id") or "")
    edge_abi_id = _payload_feature_abi_id(edge)
    cloud_abi_id = _payload_feature_abi_id(cloud)
    edge_runtime_id = _payload_runtime_identity_id(edge)
    cloud_runtime_id = _payload_runtime_identity_id(cloud)

    compatible = False
    reason = "feature_abi_id"
    if not edge:
        reason = "missing_edge_runtime_contract"
    elif not cloud:
        reason = "missing_cloud_runtime_contract"
    elif not edge_abi_id:
        reason = "missing_edge_feature_abi_id"
    elif not cloud_abi_id:
        reason = "missing_cloud_feature_abi_id"
    else:
        compatible = edge_abi_id == cloud_abi_id
        reason = "compatible" if compatible else "feature_abi_id"

    if compatible and edge_runtime_id and cloud_runtime_id and edge_runtime_id != cloud_runtime_id:
        reason = "runtime_identity_changed_but_feature_abi_compatible"

    return {
        "compatible": compatible,
        "reason": reason,
        "edge_feature_abi_id": edge_abi_id,
        "cloud_feature_abi_id": cloud_abi_id,
        "edge_runtime_identity_id": edge_runtime_id,
        "cloud_runtime_identity_id": cloud_runtime_id,
        "edge_feature_layout_id": edge_layout_id,
        "cloud_feature_layout_id": cloud_layout_id,
        "edge_trace_device_type": str(edge.get("trace_device_type") or ""),
        "cloud_trace_device_type": str(cloud.get("trace_device_type") or ""),
        "edge_boundary_tensor_labels": [
            str(label) for label in list(edge.get("boundary_tensor_labels") or [])
        ],
        "cloud_boundary_tensor_labels": [
            str(label) for label in list(cloud.get("boundary_tensor_labels") or [])
        ],
    }


def classify_feature_layout_compatibility(
    edge_contract: Mapping[str, Any] | None,
    cloud_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return classify_contract_compatibility(edge_contract, cloud_contract)


def _first_tensor_device_type(value: object) -> str:
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


def resolve_cloud_runtime_contract(
    runtime: object,
    candidate: object | None,
    *,
    logical_split_id: str,
    model_id: str,
    model_version: str,
    input_tensor_shape: list[int] | tuple[int, ...],
    input_resize_mode: str,
    sample_input: object | None = None,
    runtime_backend: str | None = None,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_metadata = dict(getattr(candidate, "metadata", {}) or {})
    boundary_labels = (
        list(getattr(candidate, "boundary_tensor_labels", []) or [])
        or list(getattr(candidate, "boundary_nodes", []) or [])
        or list(candidate_metadata.get("boundary_tensor_labels", []) or [])
    )
    boundary_schema = (
        candidate_metadata.get("boundary_schema")
        or getattr(candidate, "boundary_schema", None)
        or {}
    )
    runtime_obj = getattr(runtime, "runtime", runtime)
    trace_device_type = _first_tensor_device_type(sample_input) or str(
        getattr(runtime_obj, "device", "") or ""
    )
    resolved_backend = str(runtime_backend or "")
    if not resolved_backend:
        resolved_backend = (
            "torchlens_native"
            if getattr(runtime_obj, "trace_graph", None) is not None
            else str(getattr(runtime_obj, "mode", "") or "")
        )
    return build_runtime_contract(
        logical_split_id=str(logical_split_id),
        trace_signature=str(
            getattr(getattr(runtime_obj, "trace_graph", None), "graph_shape_hash", "") or ""
        ),
        trace_device_type=trace_device_type,
        runtime_backend=resolved_backend,
        boundary_tensor_labels=[str(label) for label in boundary_labels],
        boundary_schema=boundary_schema,
        model_id=str(model_id),
        model_version=str(model_version),
        input_tensor_shape=[int(dim) for dim in list(input_tensor_shape or [])],
        input_resize_mode=str(input_resize_mode or ""),
        feature_layout=feature_layout,
    )


def runtime_identity_id(identity: Mapping[str, Any]) -> str:
    return hashlib.sha1(_stable_json(dict(identity)).encode("utf-8")).hexdigest()


def _runtime_identity_payload(
    *,
    model_id: str,
    front_version: str,
    split_config_id: str,
    canonical_split_key: str,
    cloud_batch_split_id: str,
    input_tensor_shape: list[int],
    input_resize_mode: str,
    feature_layout_id_value: str,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_id": str(model_id),
        "front_version": str(front_version or "0"),
        "split_config_id": str(split_config_id),
        "canonical_split_key": str(canonical_split_key),
        "cloud_batch_split_id": str(cloud_batch_split_id),
        "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape or [])],
        "input_resize_mode": str(input_resize_mode or "direct_resize"),
        "split_plan_hash": "",
        "symbolic_input_schema_hash": "",
        "dynamic_batch": None,
        "trace_batch_size": None,
        "mode": "",
        "feature_layout_id": str(feature_layout_id_value),
    }
    if runtime_identity:
        payload.update(dict(runtime_identity))
        payload["input_tensor_shape"] = [
            int(dim) for dim in list(payload.get("input_tensor_shape") or [])
        ]
        payload["input_resize_mode"] = str(payload.get("input_resize_mode") or "direct_resize")
        payload["feature_layout_id"] = str(feature_layout_id_value)
    return payload


def feature_layout_matches(
    tensors: Mapping[str, torch.Tensor],
    layout: Mapping[str, Mapping[str, Any]],
) -> bool:
    actual = feature_layout_from_tensors(tensors)
    return _stable_json(actual) == _stable_json(layout)


@dataclass
class SplitRuntimeContract:
    contract_id: str
    edge_id: str
    model_id: str
    split_config_id: str
    canonical_split_key: str
    edge_split_id: str
    cloud_batch_split_id: str
    input_tensor_shape: list[int]
    input_resize_mode: str
    boundary_tensor_labels: list[str]
    feature_layout_id: str
    front_version: str
    tail_version: str | None = None
    feature_layout: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_identity: dict[str, Any] = field(default_factory=dict)
    runtime_identity_id: str = ""
    feature_abi_id: str = ""
    feature_abi_spec: dict[str, Any] = field(default_factory=dict)
    contract_aliases: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        required = {
            "contract_id": self.contract_id,
            "runtime_identity": self.runtime_identity,
            "runtime_identity_id": self.runtime_identity_id,
            "feature_abi_spec": self.feature_abi_spec,
            "feature_abi_id": self.feature_abi_id,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"SplitRuntimeContract is missing current field(s): {', '.join(missing)}."
            )

    @classmethod
    def create(
        cls,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
        canonical_split_key: str,
        edge_split_id: str,
        cloud_batch_split_id: str,
        input_tensor_shape: list[int] | tuple[int, ...],
        input_resize_mode: str,
        boundary_tensor_labels: list[str] | tuple[str, ...],
        front_version: str,
        feature_tensors: Mapping[str, torch.Tensor],
        tail_version: str | None = None,
        runtime_identity: Mapping[str, Any] | None = None,
    ) -> "SplitRuntimeContract":
        layout = feature_layout_from_tensors(feature_tensors)
        runtime_contract = dict(dict(runtime_identity or {}).get("runtime_contract") or {})
        boundary_schema = (
            runtime_contract.get("boundary_schema")
            if isinstance(runtime_contract.get("boundary_schema"), Mapping)
            else {}
        )
        layout_id = str(runtime_contract.get("feature_layout_id") or "")
        if not layout_id:
            layout_id = compute_feature_layout_id(
                model_id=str(model_id),
                model_version=str(
                    dict(runtime_identity or {}).get("model_version") or tail_version or ""
                ),
                logical_split_id=str(cloud_batch_split_id or canonical_split_key),
                trace_signature=str(dict(runtime_identity or {}).get("graph_signature") or ""),
                input_tensor_shape=[int(dim) for dim in input_tensor_shape],
                input_resize_mode=str(input_resize_mode or "direct_resize"),
                boundary_tensor_labels=[str(label) for label in list(boundary_tensor_labels or [])],
                boundary_schema=boundary_schema,
                feature_layout=layout,
            )
        identity = _runtime_identity_payload(
            model_id=str(model_id),
            front_version=str(front_version or "0"),
            split_config_id=str(split_config_id),
            canonical_split_key=str(canonical_split_key),
            cloud_batch_split_id=str(cloud_batch_split_id),
            input_tensor_shape=[int(dim) for dim in input_tensor_shape],
            input_resize_mode=str(input_resize_mode or "direct_resize"),
            feature_layout_id_value=layout_id,
            runtime_identity=runtime_identity,
        )
        identity_id = runtime_identity_id(identity)
        abi_spec = build_feature_abi_spec(
            model_id=str(model_id),
            canonical_split_key=str(canonical_split_key),
            boundary_tensor_labels=[str(label) for label in list(boundary_tensor_labels or [])],
            boundary_schema=boundary_schema,
            feature_layout=layout,
            input_tensor_shape=[int(dim) for dim in input_tensor_shape],
            input_resize_mode=str(input_resize_mode or "direct_resize"),
            runtime_identity=identity,
        )
        abi_id = feature_abi_id(abi_spec)
        return cls(
            contract_id=identity_id,
            edge_id=str(edge_id),
            model_id=str(model_id),
            split_config_id=str(split_config_id),
            canonical_split_key=str(canonical_split_key),
            edge_split_id=str(edge_split_id),
            cloud_batch_split_id=str(cloud_batch_split_id),
            input_tensor_shape=[int(dim) for dim in input_tensor_shape],
            input_resize_mode=str(input_resize_mode or "direct_resize"),
            boundary_tensor_labels=[str(label) for label in boundary_tensor_labels],
            feature_layout_id=layout_id,
            front_version=str(front_version or "0"),
            tail_version=None if tail_version is None else str(tail_version),
            feature_layout=layout,
            runtime_identity=identity,
            runtime_identity_id=identity_id,
            feature_abi_id=abi_id,
            feature_abi_spec=abi_spec,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitRuntimeContract":
        feature_layout_payload = {
            str(label): dict(spec)
            for label, spec in dict(payload.get("feature_layout") or {}).items()
            if isinstance(spec, Mapping)
        }
        layout_id = str(
            payload.get("feature_layout_id") or ""
        )
        if not layout_id:
            raise ValueError("SplitRuntimeContract is missing current field: feature_layout_id.")
        identity = dict(payload.get("runtime_identity") or {})
        if not identity:
            raise ValueError("SplitRuntimeContract is missing current field: runtime_identity.")
        identity_id = str(payload.get("runtime_identity_id") or "")
        if not identity_id:
            raise ValueError("SplitRuntimeContract is missing current field: runtime_identity_id.")
        abi_spec = {
            str(key): value for key, value in dict(payload.get("feature_abi_spec") or {}).items()
        }
        if not abi_spec:
            raise ValueError("SplitRuntimeContract is missing current field: feature_abi_spec.")
        abi_id = str(payload.get("feature_abi_id") or "")
        if not abi_id:
            raise ValueError("SplitRuntimeContract is missing current field: feature_abi_id.")
        contract_id = str(payload.get("contract_id") or "")
        if not contract_id:
            raise ValueError("SplitRuntimeContract is missing current field: contract_id.")
        aliases = [
            dict(item)
            for item in list(payload.get("contract_aliases") or [])
            if isinstance(item, Mapping)
        ]
        return cls(
            contract_id=contract_id,
            edge_id=str(payload["edge_id"]),
            model_id=str(payload["model_id"]),
            split_config_id=str(payload["split_config_id"]),
            canonical_split_key=str(payload["canonical_split_key"]),
            edge_split_id=str(payload["edge_split_id"]),
            cloud_batch_split_id=str(payload["cloud_batch_split_id"]),
            input_tensor_shape=[int(dim) for dim in payload.get("input_tensor_shape", [])],
            input_resize_mode=str(payload.get("input_resize_mode") or "direct_resize"),
            boundary_tensor_labels=[
                str(label) for label in list(payload.get("boundary_tensor_labels", []) or [])
            ],
            feature_layout_id=layout_id,
            front_version=str(payload.get("front_version") or "0"),
            tail_version=(
                None if payload.get("tail_version") is None else str(payload.get("tail_version"))
            ),
            feature_layout=feature_layout_payload,
            runtime_identity=identity,
            runtime_identity_id=identity_id,
            feature_abi_id=abi_id,
            feature_abi_spec=abi_spec,
            contract_aliases=aliases,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, root_dir: str) -> str:
        path = contract_path(
            root_dir,
            edge_id=self.edge_id,
            model_id=self.model_id,
            split_config_id=self.split_config_id,
        )
        _atomic_write_json(path, self.to_dict())
        return path

    @classmethod
    def load(
        cls,
        root_dir: str,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
    ) -> "SplitRuntimeContract" | None:
        path = contract_path(
            root_dir,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unsupported SplitRuntimeContract payload at {path}")
        return cls.from_dict(payload)

    def validate_reference(self, *, split_config_id: str, front_version: str) -> str | None:
        if str(split_config_id) != self.split_config_id:
            return "contract_mismatch"
        if str(front_version or "0") != self.front_version:
            return "front_version_mismatch"
        return None

    def validate_feature_layout(self, tensors: Mapping[str, torch.Tensor]) -> bool:
        return feature_layout_matches(tensors, self.feature_layout)


__all__ = [
    "FeatureAbiSpec",
    "SplitRuntimeContract",
    "build_feature_abi_spec",
    "build_runtime_contract",
    "classify_contract_compatibility",
    "classify_feature_layout_compatibility",
    "compute_feature_layout_id",
    "contract_path",
    "feature_abi_id",
    "feature_layout_from_tensors",
    "feature_layout_id",
    "feature_layout_matches",
    "normalise_feature_tensors",
    "resolve_cloud_runtime_contract",
    "runtime_identity_id",
]
