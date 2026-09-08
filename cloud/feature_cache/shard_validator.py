from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cloud.feature_cache.path_utils import fs_path
from cloud.feature_cache.shard_reader import ShardFeatureBatchReader
from cloud.feature_cache.types import (
    NPY_MEMMAP_SHARD,
    SAFETENSORS_SHARD,
    FeatureShardMetadata,
    FeatureShardRef,
)

_VALIDATION_FIELDS = (
    "valid",
    "missing_shard_file",
    "missing_index",
    "missing_meta",
    "missing_row_id",
    "sample_row_mismatch",
    "abi_compatible",
    "abi_incompatible",
    "label_missing",
    "label_metadata_invalid",
    "unreadable_shard",
)

ABI_REASON_FEATURE_ABI_ID = "feature_abi_id"
ABI_REASON_FEATURE_ABI_SPEC = "feature_abi_spec"
ABI_REASON_FEATURE_LAYOUT = "feature_layout"
ABI_REASON_FEATURE_LAYOUT_ID = "feature_layout_id"
ABI_REASON_LAYOUT_EQUIVALENT_REBIND = "feature_abi_id_mismatch_but_boundary_layout_equivalent"


@dataclass(frozen=True)
class AbiCompatibilityResult:
    compatible: bool
    reason: str
    expected_feature_abi_id: str = ""
    actual_feature_abi_id: str = ""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool = False
    missing_shard_file: bool = False
    missing_index: bool = False
    missing_meta: bool = False
    missing_row_id: bool = False
    sample_row_mismatch: bool = False
    abi_compatible: bool = False
    abi_incompatible: bool = False
    label_missing: bool = False
    label_metadata_invalid: bool = False
    unreadable_shard: bool = False
    reason: str = ""
    feature_ref: FeatureShardRef | None = None
    metadata: FeatureShardMetadata | None = None
    feature_layout: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata_path: str | None = None

    @property
    def status(self) -> str:
        for name in _VALIDATION_FIELDS:
            if name != "valid" and bool(getattr(self, name)):
                return name
        return "valid" if self.valid else (self.reason or "invalid")

    @property
    def unreadable(self) -> bool:
        return bool(
            self.missing_shard_file
            or self.missing_index
            or self.missing_meta
            or self.missing_row_id
            or self.sample_row_mismatch
            or self.unreadable_shard
        )

    def counts(self) -> dict[str, int]:
        return {name: int(bool(getattr(self, name))) for name in _VALIDATION_FIELDS}


def validation_count_fields() -> tuple[str, ...]:
    return _VALIDATION_FIELDS


def shard_feature_layout_from_metadata(
    metadata: FeatureShardMetadata | Mapping[str, object] | None,
) -> dict[str, dict[str, Any]]:
    if metadata is None:
        return {}
    payload = metadata.to_dict() if isinstance(metadata, FeatureShardMetadata) else dict(metadata)
    leaf_specs = dict(payload.get("leaf_specs") or {})
    layout: dict[str, dict[str, Any]] = {}
    for leaf_key, raw_spec in leaf_specs.items():
        if not isinstance(raw_spec, Mapping):
            continue
        spec = dict(raw_spec)
        label = str(spec.get("original_label") or leaf_key)
        feature_shape = spec.get("feature_shape_without_batch")
        if not feature_shape:
            sample_shape = spec.get("sample_shape")
            if not sample_shape:
                shape = list(spec.get("shape") or [])
                sample_shape = shape[1:] if shape else []
            feature_shape = sample_shape
        if not feature_shape:
            shape = list(spec.get("shape") or [])
            feature_shape = shape[1:] if shape else []
        layout[label] = {
            "dtype": str(spec.get("dtype") or payload.get("dtype") or ""),
            "shape_without_batch": [int(dim) for dim in list(feature_shape or [])],
        }
    return layout


def _read_json(path: str) -> dict[str, Any]:
    import json

    with open(fs_path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalise_dtype(value: object) -> str:
    return str(value or "").replace("torch.", "")


def _normalise_layout(
    layout: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, Any]]:
    normalised: dict[str, dict[str, Any]] = {}
    for label, spec in dict(layout or {}).items():
        if not isinstance(spec, Mapping):
            continue
        normalised[str(label)] = {
            "dtype": _normalise_dtype(spec.get("dtype")),
            "shape_without_batch": [
                int(dim) for dim in list(spec.get("shape_without_batch") or [])
            ],
        }
    return normalised


def _layout_matches_by_label(
    actual: Mapping[str, Mapping[str, object]],
    expected: Mapping[str, Mapping[str, object]],
) -> bool:
    actual_norm = _normalise_layout(actual)
    expected_norm = _normalise_layout(expected)
    return bool(actual_norm) and actual_norm == expected_norm


def _layout_matches_ignoring_labels(
    actual: Mapping[str, Mapping[str, object]],
    expected: Mapping[str, Mapping[str, object]],
) -> bool:
    def values(layout: Mapping[str, Mapping[str, object]]) -> list[dict[str, Any]]:
        return sorted(
            _normalise_layout(layout).values(),
            key=lambda item: (item["dtype"], tuple(item["shape_without_batch"])),
        )

    actual_values = values(actual)
    expected_values = values(expected)
    return bool(actual_values) and actual_values == expected_values


def feature_layouts_abi_compatible(
    actual: Mapping[str, Mapping[str, object]] | None,
    expected: Mapping[str, Mapping[str, object]] | None,
    *,
    allow_rename_compatible: bool = False,
) -> bool:
    if _layout_matches_by_label(actual or {}, expected or {}):
        return True
    return bool(allow_rename_compatible) and _layout_matches_ignoring_labels(
        actual or {},
        expected or {},
    )


def _expected_mapping(expected_abi: object) -> dict[str, Any]:
    if expected_abi is None:
        return {}
    if isinstance(expected_abi, Mapping):
        source = dict(expected_abi)
        contract = source.get("split_contract") or source.get("contract")
    else:
        source = {}
        contract = expected_abi
    if contract is not None:
        for key in (
            "contract_id",
            "split_config_id",
            "front_version",
            "feature_layout_id",
            "feature_abi_id",
            "feature_abi_spec",
            "feature_layout",
            "boundary_tensor_labels",
            "canonical_split_key",
            "cloud_batch_split_id",
            "input_tensor_shape",
            "input_resize_mode",
            "runtime_identity",
            "runtime_identity_id",
        ):
            if key not in source and hasattr(contract, key):
                source[key] = getattr(contract, key)
        if "boundary_id" not in source:
            source["boundary_id"] = getattr(contract, "cloud_batch_split_id", None) or getattr(
                contract, "canonical_split_key", None
            )
    return source


def _metadata_path_from_index(ref: FeatureShardRef, index_payload: Mapping[str, object]) -> str:
    value = index_payload.get("metadata_path") or index_payload.get("meta_path")
    if value:
        return str(value)
    if ref.index_path.endswith(".index.json"):
        return ref.index_path[: -len(".index.json")] + ".meta.json"
    return f"{ref.index_path}.meta.json"


def _shard_file_exists(ref: FeatureShardRef, metadata: FeatureShardMetadata) -> bool:
    if ref.storage_format == SAFETENSORS_SHARD:
        shard_path = ref.shard_path or metadata.shard_path
        return bool(shard_path and os.path.exists(fs_path(str(shard_path))))
    if ref.storage_format == NPY_MEMMAP_SHARD:
        shard_dir = ref.shard_dir or metadata.shard_dir
        if not shard_dir or not os.path.isdir(fs_path(str(shard_dir))):
            return False
        leaf_keys = list(ref.leaf_keys or metadata.leaf_specs.keys())
        return all(
            os.path.exists(fs_path(os.path.join(str(shard_dir), f"{leaf}.npy")))
            for leaf in leaf_keys
        )
    return False


def _label_status(expected: Mapping[str, object], ref: FeatureShardRef) -> str | None:
    label_ref = expected.get("label_ref")
    labels = expected.get("labels")
    if isinstance(label_ref, Mapping):
        if str(label_ref.get("sample_id") or ref.sample_id) != ref.sample_id:
            return "label_metadata_invalid"
        labels = label_ref.get("labels") if isinstance(label_ref.get("labels"), Mapping) else labels
    if labels is None:
        return None
    if not isinstance(labels, Mapping):
        return "label_metadata_invalid"
    if "boxes" not in labels or "labels" not in labels:
        return "label_missing"
    return None


def _abi_status(
    *,
    ref: FeatureShardRef,
    metadata: FeatureShardMetadata,
    expected: Mapping[str, object],
    feature_layout: Mapping[str, Mapping[str, object]],
    allow_abi_compatible_migration: bool,
) -> AbiCompatibilityResult:
    expected_split = expected.get("split_config_id")
    if expected_split not in (None, "") and metadata.split_config_id:
        if metadata.split_config_id != str(expected_split):
            return AbiCompatibilityResult(False, "split_config_id")

    expected_boundary = str(
        expected.get("boundary_id")
        or expected.get("cloud_batch_split_id")
        or expected.get("canonical_split_key")
        or ""
    )
    if expected_boundary and metadata.boundary_id and metadata.boundary_id != expected_boundary:
        return AbiCompatibilityResult(False, "boundary_id")

    expected_payload_kind = str(expected.get("payload_kind") or "boundary_payload")
    if (
        expected_payload_kind
        and metadata.payload_kind
        and metadata.payload_kind != expected_payload_kind
    ):
        return AbiCompatibilityResult(False, "payload_kind")

    expected_passthrough = expected.get("passthrough_schema_hash") or expected.get(
        "passthrough_schema_fingerprint"
    )
    if expected_passthrough not in (None, "") and metadata.passthrough_schema_hash not in (
        None,
        "",
    ):
        if str(expected_passthrough) != str(metadata.passthrough_schema_hash):
            return AbiCompatibilityResult(False, "passthrough_schema")

    expected_preprocessing = expected.get("preprocessing_fingerprint")
    if expected_preprocessing not in (None, "") and metadata.preprocessing_fingerprint not in (
        None,
        "",
    ):
        if str(expected_preprocessing) != str(metadata.preprocessing_fingerprint):
            return AbiCompatibilityResult(False, "preprocessing")

    expected_layout = expected.get("feature_layout")
    expected_abi_id = str(expected.get("feature_abi_id") or "")
    actual_abi_id = str(ref.feature_abi_id or metadata.feature_abi_id or "")
    if isinstance(expected_layout, Mapping) and expected_layout:
        if feature_layouts_abi_compatible(
            feature_layout,
            expected_layout,
            allow_rename_compatible=allow_abi_compatible_migration,
        ):
            reason = ABI_REASON_FEATURE_LAYOUT
            if expected_abi_id and actual_abi_id:
                reason = (
                    ABI_REASON_FEATURE_ABI_ID
                    if actual_abi_id == expected_abi_id
                    else ABI_REASON_LAYOUT_EQUIVALENT_REBIND
                )
            return AbiCompatibilityResult(
                True,
                reason,
                expected_feature_abi_id=expected_abi_id,
                actual_feature_abi_id=actual_abi_id,
            )
        return AbiCompatibilityResult(
            False,
            ABI_REASON_FEATURE_LAYOUT,
            expected_feature_abi_id=expected_abi_id,
            actual_feature_abi_id=actual_abi_id,
        )

    if expected_abi_id and actual_abi_id:
        return AbiCompatibilityResult(
            actual_abi_id == expected_abi_id,
            ABI_REASON_FEATURE_ABI_ID,
            expected_feature_abi_id=expected_abi_id,
            actual_feature_abi_id=actual_abi_id,
        )

    expected_abi_spec = expected.get("feature_abi_spec")
    metadata_abi_spec = (
        metadata.metadata.get("feature_abi_spec")
        if isinstance(metadata.metadata, Mapping)
        else None
    )
    if (
        isinstance(expected_abi_spec, Mapping)
        and expected_abi_spec
        and isinstance(metadata_abi_spec, Mapping)
        and metadata_abi_spec
    ):
        import json

        compatible = json.dumps(
            dict(expected_abi_spec), sort_keys=True, separators=(",", ":")
        ) == json.dumps(
            dict(metadata_abi_spec),
            sort_keys=True,
            separators=(",", ":"),
        )
        return AbiCompatibilityResult(
            compatible,
            ABI_REASON_FEATURE_ABI_SPEC,
            expected_feature_abi_id=expected_abi_id,
            actual_feature_abi_id=actual_abi_id,
        )

    expected_layout_id = str(expected.get("feature_layout_id") or "")
    if expected_layout_id:
        compatible = (
            ref.feature_layout_id == expected_layout_id
            or metadata.feature_layout_id == expected_layout_id
        )
        return AbiCompatibilityResult(
            compatible,
            ABI_REASON_FEATURE_LAYOUT_ID,
            expected_feature_abi_id=expected_abi_id,
            actual_feature_abi_id=actual_abi_id,
        )
    return AbiCompatibilityResult(
        True,
        "unspecified",
        expected_feature_abi_id=expected_abi_id,
        actual_feature_abi_id=actual_abi_id,
    )


class ShardFeatureRefValidator:
    def __init__(self) -> None:
        self._reader = ShardFeatureBatchReader()

    def validate_feature_ref(
        self,
        feature_ref: FeatureShardRef | Mapping[str, object],
        expected_abi: object,
        *,
        allow_abi_compatible_migration: bool = False,
        deep_validate_payload: bool = False,
        runtime: object | None = None,
    ) -> ValidationResult:
        try:
            ref = (
                feature_ref
                if isinstance(feature_ref, FeatureShardRef)
                else FeatureShardRef.from_dict(feature_ref)
            )
        except Exception as exc:
            return ValidationResult(unreadable_shard=True, reason=str(exc) or type(exc).__name__)

        if not ref.index_path or not os.path.exists(fs_path(ref.index_path)):
            return ValidationResult(
                missing_index=True,
                reason="missing_index",
                feature_ref=ref,
            )
        try:
            index_payload = _read_json(ref.index_path)
        except Exception as exc:
            return ValidationResult(
                unreadable_shard=True,
                reason=str(exc) or type(exc).__name__,
                feature_ref=ref,
            )

        metadata_path = _metadata_path_from_index(ref, index_payload)
        if not metadata_path or not os.path.exists(fs_path(metadata_path)):
            return ValidationResult(
                missing_meta=True,
                reason="missing_meta",
                feature_ref=ref,
                metadata_path=metadata_path or None,
            )
        try:
            meta_payload = _read_json(metadata_path)
            metadata = FeatureShardMetadata.from_dict({**index_payload, **meta_payload})
        except Exception as exc:
            return ValidationResult(
                unreadable_shard=True,
                reason=str(exc) or type(exc).__name__,
                feature_ref=ref,
                metadata_path=metadata_path,
            )

        if not _shard_file_exists(ref, metadata):
            return ValidationResult(
                missing_shard_file=True,
                reason="missing_shard_file",
                feature_ref=ref,
                metadata=metadata,
                metadata_path=metadata_path,
            )

        row_by_sample = dict(metadata.sample_to_row or {})
        if ref.sample_id not in row_by_sample:
            return ValidationResult(
                missing_row_id=True,
                reason="missing_row_id",
                feature_ref=ref,
                metadata=metadata,
                metadata_path=metadata_path,
            )
        expected_row = int(row_by_sample[ref.sample_id])
        if int(ref.row_id) != expected_row:
            return ValidationResult(
                sample_row_mismatch=True,
                reason="sample_row_mismatch",
                feature_ref=ref,
                metadata=metadata,
                metadata_path=metadata_path,
            )
        if int(ref.row_id) < 0 or (
            int(metadata.num_samples or 0) > 0 and int(ref.row_id) >= int(metadata.num_samples)
        ):
            return ValidationResult(
                missing_row_id=True,
                reason="missing_row_id",
                feature_ref=ref,
                metadata=metadata,
                metadata_path=metadata_path,
            )

        expected = _expected_mapping(expected_abi)
        label_status = _label_status(expected, ref)
        if label_status == "label_missing":
            return ValidationResult(
                label_missing=True,
                reason=label_status,
                feature_ref=ref,
                metadata=metadata,
                feature_layout=shard_feature_layout_from_metadata(metadata),
                metadata_path=metadata_path,
            )
        if label_status == "label_metadata_invalid":
            return ValidationResult(
                label_metadata_invalid=True,
                reason=label_status,
                feature_ref=ref,
                metadata=metadata,
                feature_layout=shard_feature_layout_from_metadata(metadata),
                metadata_path=metadata_path,
            )

        feature_layout = shard_feature_layout_from_metadata(metadata)
        abi_status = _abi_status(
            ref=ref,
            metadata=metadata,
            expected=expected,
            feature_layout=feature_layout,
            allow_abi_compatible_migration=allow_abi_compatible_migration,
        )
        if not abi_status.compatible:
            return ValidationResult(
                abi_incompatible=True,
                reason=abi_status.reason or "abi_incompatible",
                feature_ref=ref,
                metadata=metadata,
                feature_layout=feature_layout,
                metadata_path=metadata_path,
            )

        if deep_validate_payload:
            try:
                self._reader.read_batch([ref], runtime=runtime)
            except Exception as exc:
                return ValidationResult(
                    unreadable_shard=True,
                    reason=str(exc) or type(exc).__name__,
                    feature_ref=ref,
                    metadata=metadata,
                    feature_layout=feature_layout,
                    metadata_path=metadata_path,
                )

        return ValidationResult(
            valid=True,
            abi_compatible=True,
            reason=abi_status.reason or "valid",
            feature_ref=ref,
            metadata=metadata,
            feature_layout=feature_layout,
            metadata_path=metadata_path,
        )


__all__ = [
    "ABI_REASON_LAYOUT_EQUIVALENT_REBIND",
    "AbiCompatibilityResult",
    "ShardFeatureRefValidator",
    "ValidationResult",
    "feature_layouts_abi_compatible",
    "shard_feature_layout_from_metadata",
    "validation_count_fields",
]
