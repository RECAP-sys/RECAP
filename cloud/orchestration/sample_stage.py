from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence

import torch
from loguru import logger

from cloud.contracts import validate_high_quality_sync_manifest
from cloud.feature_cache import FeatureShardRef
from cloud.feature_readiness import FeatureReadinessConfig, FeatureReadinessService
from cloud.ingest import load_high_quality_shard_candidates
from cloud.orchestration.fixed_split_dependencies import (
    POOL_LABEL_COORDINATE_SPACE,
    POOL_LABEL_METADATA_FIELDS,
    _read_json_file,
)
from cloud.training.proxy_metadata import (
    original_image_size_from_metadata as _original_image_size_from_metadata,
)
from cloud.training.proxy_metadata import (
    pool_label_metadata_from_record as _pool_label_metadata_from_record,
)
from cloud.training.proxy_metadata import (
    runtime_input_tensor_shape_from_metadata as _runtime_input_tensor_shape_from_metadata,
)
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from grpc_server.workspace import prepare_request_workspace
from model_management.payload import BoundaryPayload
from model_management.split_contract import SplitRuntimeContract
from model_management.universal_model_split import UniversalModelSplitter


class SampleStageMixin:
    @staticmethod
    def _pool_annotations_from_labels(
        labels: Mapping[str, object],
    ) -> dict[str, object]:
        annotations = {
            "boxes": list(labels.get("boxes") or []),
            "labels": list(labels.get("labels") or []),
        }
        if "scores" in labels:
            annotations["scores"] = list(labels.get("scores") or [])
        for field_name in POOL_LABEL_METADATA_FIELDS:
            if labels.get(field_name) is not None:
                annotations[field_name] = labels[field_name]
        return annotations

    @staticmethod
    def _model_input_size_from_record(
        record: Mapping[str, object],
    ) -> tuple[int, int] | None:
        tensor_shape = _runtime_input_tensor_shape_from_metadata(record)
        if tensor_shape is not None:
            return int(tensor_shape[-2]), int(tensor_shape[-1])
        intermediate = record.get("intermediate")
        if isinstance(intermediate, BoundaryPayload):
            candidate_sizes: list[tuple[int, int]] = []
            for tensor in dict(intermediate.tensors or {}).values():
                if not isinstance(tensor, torch.Tensor) or tensor.ndim < 3:
                    continue
                height = int(tensor.shape[-2])
                width = int(tensor.shape[-1])
                if height > 0 and width > 0:
                    candidate_sizes.append((height, width))
            if candidate_sizes:
                return max(candidate_sizes, key=lambda item: item[0] * item[1])
        return None

    def _build_low_quality_staging_candidates(
        self,
        *,
        feature_entries: Sequence[Mapping[str, object]],
        model_input_size: tuple[int, int] | None = None,
        resize_mode: str | None = None,
    ) -> list[dict[str, object]]:
        """Build recent-window candidates from rebuilt low-quality shard refs."""
        processed_samples: list[dict[str, object]] = []
        for entry in list(feature_entries or []):
            if not isinstance(entry, Mapping):
                continue
            sample = dict(entry.get("sample") or {})
            sample_id = str(sample.get("sample_id", "")).strip()
            if not sample_id:
                continue
            labels = sample.get("labels")
            if not isinstance(labels, Mapping):
                continue
            record = dict(entry.get("record") or {})
            feature_ref = entry.get("feature_ref")
            if not isinstance(feature_ref, FeatureShardRef):
                if isinstance(feature_ref, Mapping):
                    feature_ref = FeatureShardRef.from_dict(feature_ref)
                else:
                    logger.warning(
                        "[FeatureCache][Rebuild] low-quality sample has no rebuilt "
                        "feature reference after readiness planning; skipping staging."
                    )
                    log_diagnostic_debug(
                        self,
                        "low-quality staging candidate missing feature reference",
                        lambda: {"sample_id": sample_id},
                    )
                    continue
            original_size = _original_image_size_from_metadata(record)
            if original_size is None and sample.get("input_image_size") is not None:
                original_size = tuple(
                    int(dim) for dim in list(sample.get("input_image_size") or [])[:2]
                )
            resolved_model_input_size = model_input_size or self._model_input_size_from_record(
                record
            )
            if resolved_model_input_size is None:
                tensor_shape = (
                    sample.get("input_tensor_shape") or record.get("input_tensor_shape") or []
                )
                if len(tensor_shape) >= 4:
                    resolved_model_input_size = (int(tensor_shape[-2]), int(tensor_shape[-1]))
            input_tensor_shape = (
                record.get("input_tensor_shape") or sample.get("input_tensor_shape") or []
            )
            metadata_resize_mode = str(
                record.get("input_resize_mode") or sample.get("input_resize_mode") or ""
            )
            resolved_resize_mode = str(resize_mode or metadata_resize_mode or "")
            if (
                original_size is None
                or resolved_model_input_size is None
                or not input_tensor_shape
                or not resolved_resize_mode
            ):
                logger.warning(
                    "[FixedSplitCL] Skipping low-quality sample {} with incomplete "
                    "coordinate metadata "
                    "(input_image_size={}, input_tensor_shape={}, input_resize_mode={}).",
                    sample_id,
                    original_size,
                    input_tensor_shape,
                    resolved_resize_mode,
                )
                continue
            trainable_labels = {
                "boxes": list(labels.get("boxes") or []),
                "labels": list(labels.get("labels") or []),
                **(
                    {"scores": list(labels.get("scores") or [])}
                    if labels.get("scores") is not None
                    else {}
                ),
            }
            trainable_labels.update(
                _pool_label_metadata_from_record(
                    record,
                    model_input_size=resolved_model_input_size,
                    resize_mode=resolved_resize_mode,
                )
            )
            source_split_id = str(feature_ref.boundary_id or record.get("split_label") or "")
            source_graph_signature = str(
                dict(feature_ref.metadata or {}).get("graph_signature")
                or record.get("source_feature_graph_signature")
                or ""
            )
            processed_samples.append(
                {
                    "sample_id": sample_id,
                    "labels": self._pool_annotations_from_labels(trainable_labels),
                    "sample_source": "low_quality",
                    "label_source": "teacher",
                    "feature_ref": feature_ref.to_dict(),
                    "model_id": str(sample.get("model_id") or record.get("model_id") or ""),
                    "split_config_id": str(
                        sample.get("split_config_id") or record.get("split_config_id") or ""
                    ),
                    "front_version": str(
                        sample.get("front_version") or record.get("front_version") or "0"
                    ),
                    "input_image_size": [int(dim) for dim in list(original_size)],
                    "input_tensor_shape": [int(dim) for dim in list(input_tensor_shape)],
                    "input_resize_mode": resolved_resize_mode,
                    "created_at": time.time(),
                    "feature_layout_id": feature_ref.feature_layout_id,
                    "feature_abi_id": feature_ref.feature_abi_id,
                    "runtime_identity_id": feature_ref.runtime_identity_id,
                    "source_feature_abi_id": feature_ref.feature_abi_id,
                    "source_feature_layout_id": feature_ref.feature_layout_id,
                    "source_feature_schema_hash": "",
                    "source_feature_value_schema_hash": "",
                    "source_feature_split_id": source_split_id,
                    "source_feature_graph_signature": source_graph_signature,
                }
            )
        return processed_samples

    def _feature_readiness_service(self) -> FeatureReadinessService:
        return FeatureReadinessService(
            FeatureReadinessConfig(
                store_root_dir=self.feature_cache_store_root_dir,
                storage_format=self.feature_cache_storage_format,
                accepted_storage_formats=tuple(self.feature_cache_accepted_storage_formats),
                shard_max_samples=int(self.feature_cache_shard_max_samples),
                shard_dtype=self.feature_cache_shard_dtype,
                payload_cache_enabled=bool(self.feature_cache_payload_cache_enabled),
                payload_cache_max_cpu_bytes=int(self.feature_cache_payload_cache_max_cpu_bytes),
                pin_memory=bool(self.feature_cache_pin_memory),
                non_blocking_transfer=bool(self.feature_cache_non_blocking_transfer),
                view_root_dir=self.feature_cache_view_root_dir,
                materialization_mode=self.feature_cache_materialization_mode,
                feature_rebuild_batch_size=int(self.feature_cache_feature_rebuild_batch_size),
                validate_refs=bool(self.feature_cache_validate_refs),
                deep_validate_feature_payload=bool(
                    self.feature_cache_deep_validate_feature_payload
                ),
                deep_validate_sample_rate=float(self.feature_cache_deep_validate_sample_rate),
                log_internal_ids=bool(getattr(self, "log_internal_ids", False)),
            )
        )

    def _prepare_low_quality_feature_entries(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_cache_path: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        split_contract: SplitRuntimeContract,
        splitter: UniversalModelSplitter,
        candidate: object,
        model_name: str,
        runtime_batch_size: int | None = None,
    ) -> list[dict[str, object]]:
        provider = self._bundle_batch_feature_provider(
            model,
            manifest,
            bundle_root=bundle_cache_path,
            splitter=splitter,
            candidate=candidate,
            runtime_batch_size=runtime_batch_size,
        )
        return self._feature_readiness_service().prepare_low_quality_feature_entries(
            manifest,
            bundle_cache_path=bundle_cache_path,
            gt_annotations=gt_annotations,
            split_contract=split_contract,
            model_name=model_name,
            rebuild_provider=provider,
        )

    def _build_training_cache_view_from_recent_samples(
        self,
        samples: list[Mapping[str, object]],
        *,
        contract: SplitRuntimeContract,
        model_name: str,
        edge_id: int | str,
    ):
        return self._feature_readiness_service().build_training_cache_view_from_recent_samples(
            samples=samples,
            contract=contract,
            model_name=model_name,
            edge_id=edge_id,
            pool_annotations_from_labels=self._pool_annotations_from_labels,
        )

    def sync_samples(
        self,
        edge_id: int,
        protocol_version: str,
        sync_type: str,
        payload_zip: bytes,
        model_id: str = "",
        model_version: str = "",
        split_config_id: str = "",
    ) -> tuple[bool, str, int]:
        """Append high-quality feature-label samples into the recent training window."""
        workspace = None
        try:
            if str(sync_type or "") != "HIGH_QUALITY_FEATURE_LABEL_SHARD":
                raise RuntimeError(f"Unsupported sample sync type: {sync_type!r}")
            workspace = prepare_request_workspace(
                self.workspace_root,
                edge_id=edge_id,
                request_kind="sample_sync",
                payload_zip=bytes(payload_zip or b""),
                client_cache_path="",
                log_internal_ids=bool(getattr(self, "log_internal_ids", False)),
            )
            bundle_cache_path = str(workspace)
            manifest = _read_json_file(os.path.join(bundle_cache_path, "bundle_manifest.json"))
            manifest = validate_high_quality_sync_manifest(manifest)
            if model_id and not manifest.get("model_id"):
                manifest["model_id"] = str(model_id)
            if model_version and not manifest.get("model_version"):
                manifest["model_version"] = str(model_version)
            if split_config_id and not manifest.get("split_config_id"):
                manifest["split_config_id"] = str(split_config_id)
            manifest.setdefault("edge_id", int(edge_id))
            manifest.setdefault("front_version", "0")
            with open(
                os.path.join(bundle_cache_path, "bundle_manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            self._reset_initial_cloud_state_if_needed(
                edge_id=edge_id,
                manifest=manifest,
                model_name=str(manifest.get("model_id") or model_id or self.edge_model_name),
                fallback_model_version=model_version,
                allow_without_session=True,
            )
            recent_window = self._recent_training_window_for_manifest(
                edge_id=edge_id,
                manifest=manifest,
            )
            pending_candidates, unreadable_ids = load_high_quality_shard_candidates(
                manifest=manifest,
                bundle_cache_path=bundle_cache_path,
                feature_store=self._feature_readiness_service().store(),
                label_coordinate_space=str(
                    manifest.get("label_coordinate_space") or POOL_LABEL_COORDINATE_SPACE
                ),
            )
            append_stats = recent_window.append_samples(
                pending_candidates,
                sample_source="high_quality",
            )
            stage_stats = append_stats.as_dict()
            accepted = int(append_stats.accepted)
            message = (
                f"Appended {accepted} high-quality sample(s) to the recent training window."
            )
            if unreadable_ids:
                stage_stats = dict(stage_stats)
                stage_stats["skipped_unreadable"] = int(
                    stage_stats.get("skipped_unreadable", 0)
                ) + len(unreadable_ids)
            logger.info(
                "[ShardCL][RecentTrainingWindow] high_quality appended: edge={} "
                "accepted={} retained={} required={} skipped_unreadable={}.",
                edge_id,
                accepted,
                append_stats.retained,
                self.training_frame_count,
                int(stage_stats.get("skipped_unreadable", 0) or 0),
            )
            log_diagnostic_debug(
                self,
                "[ShardCL][RecentTrainingWindow] diagnostics",
                lambda: {
                    "window_dir": recent_window.root_dir,
                    "stats": stage_stats,
                    "skipped_unreadable_preview": self._preview_ids(unreadable_ids),
                },
            )
            return True, message, accepted
        except Exception as exc:
            logger.error(
                "[FixedSplitCL] Sample sync failed: edge={} reason={}.",
                edge_id,
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "[FixedSplitCL] sample sync failure",
                lambda error=exc: {"error": repr(error)},
            )
            return False, str(exc), 0
        finally:
            if workspace is not None:
                shutil.rmtree(workspace, ignore_errors=True)
            self._run_feature_cache_gc(reason="sample_sync")
