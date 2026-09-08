from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence

from loguru import logger

from cloud.feature_cache.shard_store import FeatureShardStore
from cloud.feature_cache.shard_validator import ShardFeatureRefValidator
from cloud.feature_cache.types import (
    FeatureCacheKey,
    FeatureCachePreparePlan,
    FeatureCacheStats,
    FeatureShardRef,
    LabelRef,
    stable_digest,
)
from common.logging_sanitizer import log_diagnostic_debug


def _runtime_value(context: Mapping[str, object], key: str, default: object = "") -> object:
    value = context.get(key)
    return default if value in (None, "") else value


def _sample_id(sample: Mapping[str, object]) -> str:
    return str(sample.get("sample_id") or "").strip()


def _feature_ref_from_sample(sample: Mapping[str, object]) -> FeatureShardRef | None:
    value = sample.get("feature_ref")
    if isinstance(value, FeatureShardRef):
        return value
    if isinstance(value, Mapping):
        try:
            return FeatureShardRef.from_dict(value)
        except Exception:
            return None
    return None


def _label_ref_from_sample(sample: Mapping[str, object]) -> LabelRef | None:
    value = sample.get("label_ref")
    if isinstance(value, LabelRef):
        return value
    if isinstance(value, Mapping):
        try:
            return LabelRef.from_dict(value)
        except Exception:
            return None
    return None


def _candidate_raw_path(sample: Mapping[str, object]) -> str | None:
    for key in ("raw_path", "frame_path", "image_path", "__source_raw_path"):
        value = sample.get(key)
        if value:
            return os.path.abspath(str(value))
    return None


def _label_valid(sample: Mapping[str, object], *, low_quality: bool) -> bool:
    labels = sample.get("labels") or sample.get("label") or sample.get("target")
    if not isinstance(labels, Mapping):
        return False
    if low_quality and str(sample.get("label_source") or "teacher") not in {
        "teacher",
        "teacher_cache",
        "worker_result",
    }:
        return False
    return "boxes" in labels and "labels" in labels


def _label_ref_valid(sample: Mapping[str, object], label_ref: LabelRef | None) -> bool:
    labels: object = label_ref.labels if label_ref is not None else None
    if not isinstance(labels, Mapping):
        labels = sample.get("labels") or sample.get("label") or sample.get("target")
    return isinstance(labels, Mapping) and "boxes" in labels and "labels" in labels


def _key_for_sample(
    sample: Mapping[str, object],
    *,
    runtime_context: Mapping[str, object],
    source: str,
    feature_layout_id: str | None = None,
) -> FeatureCacheKey:
    sample_key = _sample_id(sample)
    front_version = str(_runtime_value(runtime_context, "front_version", "0"))
    preprocessing = {
        "input_tensor_shape": list(_runtime_value(runtime_context, "input_tensor_shape", []) or []),
        "input_resize_mode": str(
            _runtime_value(runtime_context, "input_resize_mode", "direct_resize")
        ),
    }
    if sample.get("input_image_size") is not None:
        preprocessing["input_image_size"] = list(sample.get("input_image_size") or [])
    return FeatureCacheKey(
        sample_id=sample_key,
        image_sha1=(
            None if sample.get("image_sha1") in (None, "") else str(sample.get("image_sha1"))
        ),
        source=source,
        model_id=str(_runtime_value(runtime_context, "model_id", "")),
        model_family=str(_runtime_value(runtime_context, "model_family", "")),
        split_config_id=str(_runtime_value(runtime_context, "split_config_id", "")),
        contract_id=(
            None
            if _runtime_value(runtime_context, "contract_id", None) in (None, "")
            else str(_runtime_value(runtime_context, "contract_id", None))
        ),
        feature_layout_id=str(
            feature_layout_id or _runtime_value(runtime_context, "feature_layout_id", "")
        ),
        feature_abi_id=str(_runtime_value(runtime_context, "feature_abi_id", "")),
        boundary_id=str(_runtime_value(runtime_context, "boundary_id", "")),
        boundary_payload_schema_hash=str(
            _runtime_value(runtime_context, "boundary_payload_schema_hash", stable_digest({}))
        ),
        prefix_weights_fingerprint=str(
            _runtime_value(runtime_context, "prefix_weights_fingerprint", f"front:{front_version}")
        ),
        preprocessing_fingerprint=str(
            _runtime_value(
                runtime_context, "preprocessing_fingerprint", stable_digest(preprocessing)
            )
        ),
        dtype=None if sample.get("dtype") in (None, "") else str(sample.get("dtype")),
        tensor_shapes_fingerprint=(
            None
            if sample.get("tensor_shapes_fingerprint") in (None, "")
            else str(sample.get("tensor_shapes_fingerprint"))
        ),
        passthrough_schema_fingerprint=(
            None
            if _runtime_value(runtime_context, "passthrough_schema_fingerprint", None) in (None, "")
            else str(_runtime_value(runtime_context, "passthrough_schema_fingerprint", None))
        ),
    )


class FeatureCachePlanner:
    def __init__(
        self,
        store: FeatureShardStore,
        *,
        materialization_mode: str = "direct_ref",
        validate_refs: bool = True,
        deep_validate_feature_payload: bool = False,
        deep_validate_sample_rate: float = 0.0,
        log_internal_ids: bool = False,
    ) -> None:
        del deep_validate_feature_payload, deep_validate_sample_rate
        self.store = store
        self.materialization_mode = str(materialization_mode or "direct_ref").strip().lower()
        if self.materialization_mode != "direct_ref":
            raise ValueError("Feature shard views only support materialization_mode='direct_ref'.")
        self.validate_refs = bool(validate_refs)
        self.log_internal_ids = bool(log_internal_ids)
        self._shard_validator = ShardFeatureRefValidator()

    def _validate_feature_ref(
        self,
        ref: FeatureShardRef,
        runtime_context: Mapping[str, object],
        stats: FeatureCacheStats,
    ) -> tuple[bool, str | None]:
        started = time.perf_counter()
        try:
            if ref.storage_format not in self.store.accepted_storage_formats:
                return False, f"storage_format:{ref.storage_format}"
            if not self.validate_refs:
                return True, None
            validation = self._shard_validator.validate_feature_ref(
                ref,
                runtime_context,
                allow_abi_compatible_migration=False,
                deep_validate_payload=False,
            )
            if not validation.valid:
                return False, validation.status
            return True, None
        except Exception as exc:
            return False, str(exc) or type(exc).__name__
        finally:
            stats.fast_ref_validation_time += time.perf_counter() - started

    def _ref_entry(
        self,
        sample: Mapping[str, object],
        runtime_context: Mapping[str, object],
        stats: FeatureCacheStats,
        *,
        source: str,
    ) -> dict[str, object] | None:
        resolve_started = time.perf_counter()
        ref = _feature_ref_from_sample(sample)
        stats.feature_ref_resolve_time += time.perf_counter() - resolve_started
        resolve_started = time.perf_counter()
        label_ref = _label_ref_from_sample(sample)
        stats.label_ref_resolve_time += time.perf_counter() - resolve_started
        if ref is None:
            return None
        valid, reason = self._validate_feature_ref(ref, runtime_context, stats)
        if not valid:
            logger.warning(
                "[FeatureShard][Validate] invalid feature reference: reason={}.",
                reason,
            )
            log_diagnostic_debug(
                self,
                "[FeatureShard][Validate] diagnostics",
                lambda: {
                    "sample_id": _sample_id(sample),
                    "shard_id": ref.shard_id,
                    "shard_path": ref.shard_path,
                    "feature_layout_id": ref.feature_layout_id,
                    "contract_id": ref.contract_id,
                },
            )
            return None
        if not _label_ref_valid(sample, label_ref):
            return None
        key = _key_for_sample(
            sample,
            runtime_context=runtime_context,
            source=source,
            feature_layout_id=ref.feature_layout_id,
        )
        entry: dict[str, object] = {"sample": dict(sample), "feature_ref": ref, "cache_key": key}
        if label_ref is not None:
            entry["label_ref"] = label_ref
        return entry

    def build_plan(
        self,
        *,
        existing_active_samples: Sequence[Mapping[str, object]] | None = None,
        resolved_low_quality_samples: Sequence[Mapping[str, object]] | None = None,
        unresolved_low_quality_samples: Sequence[Mapping[str, object]] | None = None,
        runtime_context: Mapping[str, object],
        view_id: str,
        generation: str,
    ) -> FeatureCachePreparePlan:
        existing = list(existing_active_samples or [])
        resolved_lq = list(resolved_low_quality_samples or [])
        unresolved_lq = list(unresolved_low_quality_samples or [])
        stats = FeatureCacheStats(
            requested_samples=len(existing) + len(resolved_lq) + len(unresolved_lq)
        )
        plan = FeatureCachePreparePlan(
            view_id=str(view_id),
            generation=str(generation),
            feature_layout_id=str(runtime_context.get("feature_layout_id") or ""),
            contract_id=str(runtime_context.get("contract_id") or ""),
            materialization_mode=self.materialization_mode,
            feature_abi_id=str(runtime_context.get("feature_abi_id") or ""),
            runtime_identity_id=str(runtime_context.get("runtime_identity_id") or ""),
            runtime_context=dict(runtime_context),
            stats=stats,
        )

        for sample in existing:
            entry = self._ref_entry(
                sample,
                runtime_context,
                stats,
                source=str(sample.get("feature_source") or "recent_training_window"),
            )
            if entry is None:
                if _candidate_raw_path(sample):
                    stats.existing_rebuild_required += 1
                else:
                    stats.invalid_dropped += 1
                    stats.existing_dropped_incompatible += 1
                    plan.drop_invalid_samples.append(
                        {"sample": dict(sample), "reason": "missing_or_invalid_shard_ref"}
                    )
                continue
            source_contract = str(
                sample.get("source_contract_id") or sample.get("contract_id") or ""
            )
            current_contract = str(runtime_context.get("contract_id") or "")
            is_rebound = bool(sample.get("rebinding_reason")) or bool(
                source_contract and current_contract and source_contract != current_contract
            )
            if is_rebound:
                stats.existing_rebound += 1
            else:
                stats.existing_reused += 1
                plan.reuse_existing_refs.append(entry)
            stats.existing_feature_ref_reused += 1
            plan.create_training_view.append(entry)

        for sample in resolved_lq:
            if not _label_valid(sample, low_quality=True):
                stats.invalid_dropped += 1
                plan.drop_invalid_samples.append(
                    {"sample": dict(sample), "reason": "invalid_teacher_label"}
                )
                continue
            entry = self._ref_entry(
                sample,
                runtime_context,
                stats,
                source=str(sample.get("feature_source") or "cloud_rebuilt"),
            )
            if entry is not None:
                stats.low_quality_reused += 1
                plan.register_uploaded_feature_refs.append(entry)
                plan.create_training_view.append(entry)
                continue
            raw_path = _candidate_raw_path(sample)
            if not raw_path or not os.path.exists(raw_path):
                stats.invalid_dropped += 1
                plan.drop_invalid_samples.append(
                    {"sample": dict(sample), "reason": "missing_raw_for_rebuild"}
                )
                continue
            key = _key_for_sample(
                sample,
                runtime_context=runtime_context,
                source="cloud_rebuilt",
                feature_layout_id=str(runtime_context.get("feature_layout_id") or ""),
            )
            stats.cache_misses += 1
            plan.rebuild_low_quality_from_raw.append(
                {"sample": dict(sample), "raw_path": raw_path, "cache_key": key}
            )

        for sample in unresolved_lq:
            stats.low_quality_deferred += 1
            plan.defer_unresolved_low_quality.append(
                {"sample": dict(sample), "reason": "unresolved_teacher_label"}
            )

        logger.info(
            "[FeatureCache][Plan] requested={} existing_reused={} "
            "existing_rebound={} existing_rebuild_required={} "
            "existing_dropped_incompatible={} high_quality_registered={} "
            "low_quality_reused={} low_quality_rebuild_required={} "
            "low_quality_deferred={} invalid_dropped={} mode=shard_ref",
            stats.requested_samples,
            stats.existing_reused,
            stats.existing_rebound,
            stats.existing_rebuild_required,
            stats.existing_dropped_incompatible,
            stats.high_quality_registered,
            stats.low_quality_reused,
            len(plan.rebuild_low_quality_from_raw),
            stats.low_quality_deferred,
            stats.invalid_dropped,
        )
        return plan


__all__ = ["FeatureCachePlanner"]
