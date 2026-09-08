from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cv2
from loguru import logger

import model_management.model_zoo as model_zoo
from cloud.feature_cache import (
    FeatureCacheMaterializer,
    FeatureCachePlanner,
    FeatureShardStore,
)
from cloud.training.proxy_metadata import (
    is_low_quality_trigger_sample as _is_low_quality_trigger_sample,
)
from common.logging_sanitizer import log_diagnostic_debug
from model_management.split_contract import SplitRuntimeContract


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned or "unknown"


@dataclass(frozen=True)
class FeatureReadinessConfig:
    store_root_dir: str
    storage_format: str
    accepted_storage_formats: tuple[str, ...]
    shard_max_samples: int
    shard_dtype: str | None
    payload_cache_enabled: bool
    payload_cache_max_cpu_bytes: int
    pin_memory: bool
    non_blocking_transfer: bool
    view_root_dir: str
    materialization_mode: str
    feature_rebuild_batch_size: int
    validate_refs: bool
    deep_validate_feature_payload: bool
    deep_validate_sample_rate: float
    log_internal_ids: bool = False


class FeatureReadinessService:
    def __init__(self, config: FeatureReadinessConfig) -> None:
        self.config = config

    def store(self) -> FeatureShardStore:
        cfg = self.config
        return FeatureShardStore(
            cfg.store_root_dir,
            storage_format=cfg.storage_format,
            accepted_storage_formats=cfg.accepted_storage_formats,
            shard_max_samples=cfg.shard_max_samples,
            shard_dtype=cfg.shard_dtype,
            payload_cache_enabled=cfg.payload_cache_enabled,
            payload_cache_max_cpu_bytes=cfg.payload_cache_max_cpu_bytes,
            pin_memory=cfg.pin_memory,
            non_blocking_transfer=cfg.non_blocking_transfer,
            log_internal_ids=cfg.log_internal_ids,
        )

    def materializer(
        self,
        store: FeatureShardStore,
        *,
        rebuild_provider=None,
    ) -> FeatureCacheMaterializer:
        cfg = self.config
        return FeatureCacheMaterializer(
            store,
            view_root_dir=cfg.view_root_dir,
            materialization_mode=cfg.materialization_mode,
            feature_rebuild_batch_size=cfg.feature_rebuild_batch_size,
            rebuild_provider=rebuild_provider,
            deep_validate_feature_payload=cfg.deep_validate_feature_payload,
            deep_validate_sample_rate=cfg.deep_validate_sample_rate,
            log_internal_ids=cfg.log_internal_ids,
        )

    def runtime_context(
        self,
        *,
        contract: SplitRuntimeContract,
        model_name: str,
    ) -> dict[str, object]:
        return {
            "model_id": str(contract.model_id or model_name),
            "model_family": model_zoo.get_model_family(str(model_name)),
            "split_config_id": str(contract.split_config_id),
            "contract_id": str(contract.contract_id),
            "feature_layout_id": str(contract.feature_layout_id),
            "feature_abi_id": str(contract.feature_abi_id),
            "feature_abi_spec": dict(contract.feature_abi_spec or {}),
            "runtime_identity_id": str(contract.runtime_identity_id),
            "feature_layout": dict(contract.feature_layout or {}),
            "boundary_tensor_labels": list(contract.boundary_tensor_labels or []),
            "canonical_split_key": str(contract.canonical_split_key),
            "boundary_id": str(contract.cloud_batch_split_id or contract.canonical_split_key),
            "input_tensor_shape": [int(dim) for dim in list(contract.input_tensor_shape)],
            "input_resize_mode": str(contract.input_resize_mode),
            "front_version": str(contract.front_version),
            "feature_rebuild_batch_size": int(self.config.feature_rebuild_batch_size),
        }

    def low_quality_samples(
        self,
        *,
        bundle_cache_path: str,
        manifest: Mapping[str, object],
        gt_annotations: Mapping[str, Mapping[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        split_plan = dict(manifest.get("split_plan", {}) or {})
        model_meta = dict(manifest.get("model", {}) or {})
        resolved: list[dict[str, object]] = []
        unresolved: list[dict[str, object]] = []
        for sample in list(manifest.get("samples", []) or []):
            if not isinstance(sample, Mapping):
                continue
            if not _is_low_quality_trigger_sample(manifest, sample):
                continue
            sample_id = str(sample.get("sample_id", "") or "").strip()
            raw_relpath = sample.get("raw_relpath")
            if not sample_id or raw_relpath is None:
                continue
            raw_path = os.path.join(
                bundle_cache_path,
                str(raw_relpath).replace("/", os.sep),
            )
            labels = gt_annotations.get(sample_id)
            if labels is None:
                unresolved.append({**dict(sample), "sample_id": sample_id, "raw_path": raw_path})
                continue
            if not os.path.exists(raw_path):
                logger.warning(
                    "[FeatureCache][Plan] low-quality sample cannot be rebuilt: "
                    "reason=missing_raw_image."
                )
                log_diagnostic_debug(
                    self.config,
                    "[FeatureCache][Plan] missing raw image diagnostics",
                    lambda: {"sample_id": sample_id, "raw_path": raw_path},
                )
                unresolved.append({**dict(sample), "sample_id": sample_id, "raw_path": raw_path})
                continue
            input_image_size = sample.get("input_image_size")
            if input_image_size is None:
                frame = cv2.imread(raw_path)
                input_image_size = (
                    [int(frame.shape[0]), int(frame.shape[1])]
                    if frame is not None and frame.ndim >= 2
                    else None
                )
            resolved.append(
                {
                    **dict(sample),
                    "sample_id": sample_id,
                    "sample_source": "low_quality",
                    "label_source": "teacher",
                    "labels": dict(labels),
                    "raw_path": raw_path,
                    "model_id": str(model_meta.get("model_id") or manifest.get("model_id") or ""),
                    "model_version": str(model_meta.get("model_version") or ""),
                    "split_config_id": str(
                        manifest.get("split_config_id") or split_plan.get("split_config_id") or ""
                    ),
                    "front_version": str(
                        manifest.get("front_version") or split_plan.get("front_version") or "0"
                    ),
                    "input_image_size": input_image_size,
                    "input_tensor_shape": list(
                        sample.get("input_tensor_shape")
                        or manifest.get("input_tensor_shape")
                        or split_plan.get("input_tensor_shape", [])
                        or []
                    ),
                    "input_resize_mode": str(
                        sample.get("input_resize_mode")
                        or manifest.get("input_resize_mode")
                        or split_plan.get("input_resize_mode")
                        or "direct_resize"
                    ),
                    "has_raw_sample": True,
                }
            )
        return resolved, unresolved

    def prepare_low_quality_feature_entries(
        self,
        manifest: dict[str, object],
        *,
        bundle_cache_path: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        split_contract: SplitRuntimeContract,
        model_name: str,
        rebuild_provider,
    ) -> list[dict[str, object]]:
        store = self.store()
        runtime_context = self.runtime_context(
            contract=split_contract,
            model_name=model_name,
        )
        resolved_lq, unresolved_lq = self.low_quality_samples(
            bundle_cache_path=bundle_cache_path,
            manifest=manifest,
            gt_annotations=gt_annotations,
        )
        cfg = self.config
        planner = FeatureCachePlanner(
            store,
            materialization_mode=cfg.materialization_mode,
            validate_refs=cfg.validate_refs,
            deep_validate_feature_payload=cfg.deep_validate_feature_payload,
            deep_validate_sample_rate=cfg.deep_validate_sample_rate,
            log_internal_ids=cfg.log_internal_ids,
        )
        plan = planner.build_plan(
            resolved_low_quality_samples=resolved_lq,
            unresolved_low_quality_samples=unresolved_lq,
            runtime_context=runtime_context,
            view_id="low_quality_feature_readiness",
            generation="pending_recent_training_window",
        )
        rebuilt_entries = self.materializer(
            store,
            rebuild_provider=rebuild_provider,
        ).rebuild_low_quality_features_only(plan)
        entries = list(plan.create_training_view)
        entries.extend(rebuilt_entries)
        return entries

    def build_training_cache_view_from_recent_samples(
        self,
        *,
        samples: list[Mapping[str, object]],
        contract: SplitRuntimeContract,
        model_name: str,
        edge_id: int | str,
        pool_annotations_from_labels: Callable[[Mapping[str, object]], dict[str, object]],
    ):
        recent_samples = [dict(sample) for sample in list(samples or [])]
        generation_id = "recent_training_window"
        view_id = (
            f"edge_{_sanitize_segment(edge_id)}_"
            f"{_sanitize_segment(model_name)}_"
            f"{_sanitize_segment(generation_id)}_"
            f"{int(time.time() * 1000)}"
        )
        store = self.store()
        cfg = self.config
        planner = FeatureCachePlanner(
            store,
            materialization_mode=cfg.materialization_mode,
            validate_refs=cfg.validate_refs,
            deep_validate_feature_payload=cfg.deep_validate_feature_payload,
            deep_validate_sample_rate=cfg.deep_validate_sample_rate,
            log_internal_ids=cfg.log_internal_ids,
        )
        plan = planner.build_plan(
            existing_active_samples=recent_samples,
            runtime_context=self.runtime_context(
                contract=contract,
                model_name=model_name,
            ),
            view_id=view_id,
            generation=generation_id,
        )
        if plan.drop_invalid_samples:
            dropped_ids = [
                str(dict(item.get("sample") or {}).get("sample_id") or "")
                for item in plan.drop_invalid_samples[:10]
                if isinstance(item, Mapping)
            ]
            log_diagnostic_debug(
                self.config,
                "[FeatureCache][RecentTrainingWindow] dropped sample diagnostics",
                lambda: {"sample_ids_preview": dropped_ids},
            )
            raise RuntimeError(
                "Recent training-window samples could not all be direct-referenced into "
                f"the training view: dropped={len(plan.drop_invalid_samples)}."
            )
        result = self.materializer(store).prepare(plan)
        if result.view is None:
            raise RuntimeError("Feature cache materializer did not create a TrainingCacheView.")
        active_ids = {str(sample.get("sample_id") or "") for sample in recent_samples}
        view_ids = {sample.sample_id for sample in result.view.samples}
        if active_ids != view_ids:
            log_diagnostic_debug(
                self.config,
                "[FeatureCache][RecentTrainingWindow] mismatch diagnostics",
                lambda: {
                    "missing_from_view": sorted(active_ids - view_ids),
                    "unexpected_in_view": sorted(view_ids - active_ids),
                },
            )
            raise RuntimeError(
                "TrainingCacheView(source=recent_training_window) sample mismatch: "
                f"active={len(active_ids)} view={len(view_ids)}."
            )
        if int(result.stats.files_copied) != 0 or int(result.stats.bytes_copied) != 0:
            raise RuntimeError(
                "TrainingCacheView(source=recent_training_window) must use direct refs "
                f"only; files_copied={result.stats.files_copied} "
                f"bytes_copied={result.stats.bytes_copied}."
            )
        result.view.source = "recent_training_window"
        result.bundle_info["feature_cache_view_source"] = "recent_training_window"
        result.bundle_info["from_recent_training_window"] = True
        logger.info(
            "[FeatureCache][RecentTrainingWindow] generation={} samples={} "
            "source=recent_training_window.",
            generation_id,
            len(active_ids),
        )
        log_diagnostic_debug(
            self.config,
            "[FeatureCache][RecentTrainingWindow] diagnostics",
            lambda: {"view_id": view_id},
        )
        gt_annotations = {
            sample.sample_id: pool_annotations_from_labels(sample.label_ref.labels or {})
            for sample in result.view.samples
        }
        sample_metadata_by_id = {
            sample_id: dict(record) for sample_id, record in result.records.items()
        }
        return (
            result.bundle_info,
            result.frame_dir
            or os.path.join(
                cfg.view_root_dir,
                view_id,
                "frames",
            ),
            result.records,
            gt_annotations,
            sample_metadata_by_id,
            result.view,
            result.stats,
        )


__all__ = [
    "FeatureReadinessConfig",
    "FeatureReadinessService",
]
