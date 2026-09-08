from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping

SAFETENSORS_SHARD = "safetensors_shard"
NPY_MEMMAP_SHARD = "npy_memmap_shard"
SUPPORTED_STORAGE_FORMATS = {SAFETENSORS_SHARD, NPY_MEMMAP_SHARD}


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_digest(payload: object) -> str:
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureCacheKey:
    """Stable shard grouping key, not a per-sample file path key."""

    sample_id: str
    image_sha1: str | None
    source: str
    model_id: str
    model_family: str
    split_config_id: str
    contract_id: str | None
    feature_layout_id: str
    boundary_id: str
    boundary_payload_schema_hash: str
    prefix_weights_fingerprint: str | None
    preprocessing_fingerprint: str | None
    dtype: str | None
    tensor_shapes_fingerprint: str | None
    passthrough_schema_fingerprint: str | None
    feature_abi_id: str = ""

    def payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return stable_digest(self.payload())


@dataclass(frozen=True)
class FeatureShardMetadata:
    storage_format: str
    model_id: str
    model_family: str
    split_config_id: str
    feature_layout_id: str
    contract_id: str | None
    boundary_id: str
    boundary_schema_hash: str
    passthrough_schema_hash: str | None
    preprocessing_fingerprint: str | None
    dtype: str
    shape_bucket: str
    num_samples: int
    leaf_specs: dict[str, dict[str, object]]
    sample_to_row: dict[str, int]
    payload_kind: str = "boundary_payload"
    shard_id: str = ""
    shard_path: str | None = None
    shard_dir: str | None = None
    index_path: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    feature_abi_id: str = ""
    runtime_identity_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureShardMetadata":
        return cls(
            storage_format=str(payload.get("storage_format") or ""),
            model_id=str(payload.get("model_id") or ""),
            model_family=str(payload.get("model_family") or ""),
            split_config_id=str(payload.get("split_config_id") or ""),
            feature_layout_id=str(payload.get("feature_layout_id") or ""),
            contract_id=(
                None
                if payload.get("contract_id") in (None, "")
                else str(payload.get("contract_id"))
            ),
            boundary_id=str(payload.get("boundary_id") or ""),
            boundary_schema_hash=str(payload.get("boundary_schema_hash") or ""),
            passthrough_schema_hash=(
                None
                if payload.get("passthrough_schema_hash") in (None, "")
                else str(payload.get("passthrough_schema_hash"))
            ),
            preprocessing_fingerprint=(
                None
                if payload.get("preprocessing_fingerprint") in (None, "")
                else str(payload.get("preprocessing_fingerprint"))
            ),
            dtype=str(payload.get("dtype") or ""),
            shape_bucket=str(payload.get("shape_bucket") or ""),
            num_samples=int(payload.get("num_samples") or 0),
            leaf_specs={
                str(key): dict(value)
                for key, value in dict(payload.get("leaf_specs") or {}).items()
                if isinstance(value, Mapping)
            },
            sample_to_row={
                str(key): int(value)
                for key, value in dict(payload.get("sample_to_row") or {}).items()
            },
            payload_kind=str(payload.get("payload_kind") or "boundary_payload"),
            shard_id=str(payload.get("shard_id") or ""),
            shard_path=(
                None if payload.get("shard_path") in (None, "") else str(payload.get("shard_path"))
            ),
            shard_dir=(
                None if payload.get("shard_dir") in (None, "") else str(payload.get("shard_dir"))
            ),
            index_path=str(payload.get("index_path") or ""),
            metadata=dict(payload.get("metadata") or {}),
            feature_abi_id=str(payload.get("feature_abi_id") or ""),
            runtime_identity_id=str(payload.get("runtime_identity_id") or ""),
        )


@dataclass(frozen=True)
class FeatureShardRef:
    storage_format: str
    shard_id: str
    shard_path: str | None
    shard_dir: str | None
    index_path: str
    row_id: int
    sample_id: str
    feature_layout_id: str
    contract_id: str | None
    boundary_id: str
    payload_kind: str
    dtype: str
    shape_bucket: str
    leaf_keys: list[str]
    passthrough_keys: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    feature_abi_id: str = ""
    runtime_identity_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "storage_format": self.storage_format,
            "shard_id": self.shard_id,
            "shard_path": self.shard_path,
            "shard_dir": self.shard_dir,
            "index_path": self.index_path,
            "row_id": int(self.row_id),
            "sample_id": self.sample_id,
            "feature_layout_id": self.feature_layout_id,
            "contract_id": self.contract_id,
            "boundary_id": self.boundary_id,
            "payload_kind": self.payload_kind,
            "dtype": self.dtype,
            "shape_bucket": self.shape_bucket,
            "leaf_keys": list(self.leaf_keys),
            "passthrough_keys": list(self.passthrough_keys),
            "metadata": dict(self.metadata),
            "feature_abi_id": self.feature_abi_id,
            "runtime_identity_id": self.runtime_identity_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureShardRef":
        storage_format = str(payload.get("storage_format") or "")
        if storage_format not in SUPPORTED_STORAGE_FORMATS:
            raise ValueError(f"Unsupported feature shard storage_format={storage_format!r}.")
        sample_id = str(payload.get("sample_id") or "")
        row_id = int(payload.get("row_id") if payload.get("row_id") is not None else -1)
        if not sample_id or row_id < 0:
            raise ValueError("FeatureShardRef requires sample_id and non-negative row_id.")
        return cls(
            storage_format=storage_format,
            shard_id=str(payload.get("shard_id") or ""),
            shard_path=(
                None if payload.get("shard_path") in (None, "") else str(payload.get("shard_path"))
            ),
            shard_dir=(
                None if payload.get("shard_dir") in (None, "") else str(payload.get("shard_dir"))
            ),
            index_path=str(payload.get("index_path") or ""),
            row_id=row_id,
            sample_id=sample_id,
            feature_layout_id=str(payload.get("feature_layout_id") or ""),
            contract_id=(
                None
                if payload.get("contract_id") in (None, "")
                else str(payload.get("contract_id"))
            ),
            boundary_id=str(payload.get("boundary_id") or ""),
            payload_kind=str(payload.get("payload_kind") or "boundary_payload"),
            dtype=str(payload.get("dtype") or ""),
            shape_bucket=str(payload.get("shape_bucket") or ""),
            leaf_keys=[str(key) for key in list(payload.get("leaf_keys") or [])],
            passthrough_keys=[str(key) for key in list(payload.get("passthrough_keys") or [])],
            metadata=dict(payload.get("metadata") or {}),
            feature_abi_id=str(payload.get("feature_abi_id") or ""),
            runtime_identity_id=str(payload.get("runtime_identity_id") or ""),
        )


@dataclass(frozen=True)
class LabelRef:
    sample_id: str
    path: str | None
    codec: str
    label_source: str
    teacher_labeled: bool = False
    pseudo_labeled: bool = False
    size_bytes: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
    labels: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "codec": self.codec,
            "label_source": self.label_source,
            "teacher_labeled": bool(self.teacher_labeled),
            "pseudo_labeled": bool(self.pseudo_labeled),
            "size_bytes": int(self.size_bytes),
            "metadata": dict(self.metadata),
            "labels": dict(self.labels or {}) if self.labels is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LabelRef":
        labels = payload.get("labels")
        return cls(
            sample_id=str(payload.get("sample_id") or ""),
            path=None if payload.get("path") in (None, "") else str(payload.get("path")),
            codec=str(payload.get("codec") or ""),
            label_source=str(payload.get("label_source") or ""),
            teacher_labeled=bool(payload.get("teacher_labeled", False)),
            pseudo_labeled=bool(payload.get("pseudo_labeled", False)),
            size_bytes=int(payload.get("size_bytes") or 0),
            metadata=dict(payload.get("metadata") or {}),
            labels=dict(labels) if isinstance(labels, Mapping) else None,
        )


@dataclass(frozen=True)
class SampleTrainingRef:
    sample_id: str
    sample_type: str
    feature_ref: FeatureShardRef
    label_ref: LabelRef
    metadata_ref: str | None
    teacher_labeled: bool
    pseudo_labeled: bool
    generation: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "sample_type": self.sample_type,
            "feature_ref": self.feature_ref.to_dict(),
            "label_ref": self.label_ref.to_dict(),
            "metadata_ref": self.metadata_ref,
            "teacher_labeled": bool(self.teacher_labeled),
            "pseudo_labeled": bool(self.pseudo_labeled),
            "generation": self.generation,
            "metadata": dict(self.metadata),
        }


@dataclass
class TrainingCacheView:
    view_id: str
    generation: str
    feature_layout_id: str
    contract_id: str
    source: str
    samples: list[SampleTrainingRef]
    manifest_path: str
    metadata_index_path: str
    created_at: float
    feature_abi_id: str = ""
    runtime_identity_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "generation": self.generation,
            "feature_layout_id": self.feature_layout_id,
            "feature_abi_id": self.feature_abi_id,
            "runtime_identity_id": self.runtime_identity_id,
            "contract_id": self.contract_id,
            "source": self.source,
            "sample_count": len(self.samples),
            "created_at": float(self.created_at),
            "manifest_path": self.manifest_path,
            "metadata_index_path": self.metadata_index_path,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass
class FeatureCacheStats:
    requested_samples: int = 0
    existing_reused: int = 0
    existing_rebound: int = 0
    existing_rebuild_required: int = 0
    existing_dropped_incompatible: int = 0
    high_quality_registered: int = 0
    low_quality_reused: int = 0
    low_quality_rebuilt: int = 0
    low_quality_deferred: int = 0
    invalid_dropped: int = 0
    bytes_copied: int = 0
    files_copied: int = 0
    direct_refs_created: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    existing_feature_ref_reused: int = 0
    feature_store_lookup_count: int = 0
    feature_store_register_count: int = 0
    legacy_migration_count: int = 0
    rebuild_batch_size: int = 0
    rebuild_batches: int = 0
    shards_written: int = 0
    total_tensor_bytes: int = 0
    feature_ref_resolve_time: float = 0.0
    feature_store_lookup_time: float = 0.0
    feature_store_register_time: float = 0.0
    label_ref_resolve_time: float = 0.0
    fast_ref_validation_time: float = 0.0
    deep_payload_validation_time: float = 0.0
    rebuild_time: float = 0.0
    cache_write_time: float = 0.0
    atomic_commit_time: float = 0.0
    manifest_write_time: float = 0.0
    metadata_index_time: float = 0.0
    total_prepare_time: float = 0.0
    rebuild_failures: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class FeatureCachePreparePlan:
    view_id: str
    generation: str
    feature_layout_id: str
    contract_id: str
    materialization_mode: str
    feature_abi_id: str = ""
    runtime_identity_id: str = ""
    runtime_context: dict[str, object] = field(default_factory=dict)
    reuse_existing_refs: list[dict[str, object]] = field(default_factory=list)
    register_uploaded_feature_refs: list[dict[str, object]] = field(default_factory=list)
    rebuild_low_quality_from_raw: list[dict[str, object]] = field(default_factory=list)
    defer_unresolved_low_quality: list[dict[str, object]] = field(default_factory=list)
    drop_invalid_samples: list[dict[str, object]] = field(default_factory=list)
    create_training_view: list[dict[str, object]] = field(default_factory=list)
    stats: FeatureCacheStats = field(default_factory=FeatureCacheStats)


@dataclass
class FeatureCachePrepareResult:
    plan: FeatureCachePreparePlan
    view: TrainingCacheView | None = None
    feature_refs: dict[str, FeatureShardRef] = field(default_factory=dict)
    records: dict[str, dict[str, object]] = field(default_factory=dict)
    metadata_by_id: dict[str, dict[str, object]] = field(default_factory=dict)
    bundle_info: dict[str, object] = field(default_factory=dict)
    frame_dir: str | None = None
    stats: FeatureCacheStats = field(default_factory=FeatureCacheStats)
    failed_samples: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureCacheGCResult:
    dry_run: bool
    scanned_files: int = 0
    deleted_files: int = 0
    deleted_bytes: int = 0
    retained_files: int = 0
    orphan_files: list[str] = field(default_factory=list)
    retained_files_preview: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
