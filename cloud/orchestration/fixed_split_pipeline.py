from __future__ import annotations

import base64
import json
import time

from loguru import logger

from cloud.ingest import materialize_low_quality_trigger_bundle
from cloud.orchestration.checkpoint_stage import CheckpointStageMixin
from cloud.orchestration.fixed_split_dependencies import (
    _increment_model_version,
    _manifest_model_metadata,
    _normalize_model_version,
)
from cloud.orchestration.logging_utils import StageLoggingMixin
from cloud.orchestration.proxy_stage import ProxyStageMixin
from cloud.orchestration.request_context import RequestContextMixin
from cloud.orchestration.runtime_stage import (
    FIXED_SPLIT_DYNAMIC_BATCH_MIN as _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
)
from cloud.orchestration.runtime_stage import (
    FixedSplitRuntimeContractMixin,
)
from cloud.orchestration.runtime_stage import (
    splitter_dynamic_batch_min as _splitter_dynamic_batch_min,
)
from cloud.orchestration.runtime_template_stage import FixedSplitRuntimeTemplateMixin
from cloud.orchestration.sample_stage import SampleStageMixin
from cloud.orchestration.settings import PipelineLifecycleMixin
from cloud.orchestration.teacher_stage import TeacherAnnotationMixin
from cloud.orchestration.training_stage import TrainingStageMixin
from cloud.training import build_proxy_validation_split
from cloud.training.proxy_metadata import (
    runtime_input_tensor_shape_from_metadata as _runtime_input_tensor_shape_from_metadata,
)
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from model_management.split_model_adapters import (
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
)

# ---------------------------------------------------------------------------
# Cloud-side Continual Learning
# ---------------------------------------------------------------------------


class FixedSplitPipeline(
    PipelineLifecycleMixin,
    RequestContextMixin,
    StageLoggingMixin,
    CheckpointStageMixin,
    TeacherAnnotationMixin,
    FixedSplitRuntimeContractMixin,
    FixedSplitRuntimeTemplateMixin,
    SampleStageMixin,
    ProxyStageMixin,
    TrainingStageMixin,
):
    """Run shard-based annotation, fixed-split retraining, and model publication."""

    def get_ground_truth_and_retrain(
        self,
        edge_id: int,
        frame_indices: list[int],
        cache_path: str,
    ) -> tuple[bool, str, str]:
        del edge_id, frame_indices, cache_path
        message = "full-frame retrain is unavailable; use fixed-split continual learning"
        logger.error("[CL] {}", message)
        return False, "", message

    def get_ground_truth_and_fixed_split_retrain(
        self,
        edge_id: int,
        bundle_cache_path: str,
        *,
        num_epoch: int | None = None,
    ) -> tuple[bool, str, str]:
        effective_num_epoch = self.default_num_epoch if num_epoch is None else int(num_epoch)

        with self._training_job_scope(edge_id):
            total_round_started = time.perf_counter()
            accepted_low_quality_ids_message = ""
            try:
                stage_started = time.perf_counter()
                materialized_manifest = materialize_low_quality_trigger_bundle(
                    bundle_cache_path,
                    feature_store=self._feature_readiness_service().store(),
                )
                if materialized_manifest is None:
                    raise RuntimeError(
                        "Shard-based continual-learning trigger payload must contain "
                        "trigger_manifest.json; bundle_manifest.json uploads are not supported."
                    )
                manifest = materialized_manifest
                self._log_stage_duration("loading bundle manifest", stage_started)
                current_model_name = self._resolve_fixed_split_model_name(manifest)
                manifest_model_metadata = _manifest_model_metadata(manifest)
                manifest_runtime_input_shape = _runtime_input_tensor_shape_from_metadata(manifest)
                early_teacher_requests = self._build_low_quality_raw_teacher_annotation_requests(
                    bundle_cache_path=bundle_cache_path,
                    manifest=manifest,
                    edge_id=edge_id,
                    model_id=current_model_name,
                    target_model_metadata=manifest_model_metadata,
                )
                self._submit_low_quality_teacher_annotations(early_teacher_requests)
                bundle_sample_count = self._count_manifest_training_samples(manifest)
                effective_batch_size = self._resolve_fixed_split_runtime_batch_size(
                    current_model_name,
                    num_train_samples=bundle_sample_count,
                )
                bundle_model_version = _normalize_model_version(
                    manifest.get("model", {}).get("model_version", "0"),
                    field_name="bundle model version",
                )
                recent_window = self._recent_training_window_for_manifest(
                    edge_id=edge_id,
                    manifest=manifest,
                )
                self._reset_initial_cloud_state_if_needed(
                    edge_id=edge_id,
                    manifest=manifest,
                    model_name=current_model_name,
                    fallback_model_version=bundle_model_version,
                    allow_without_session=True,
                )
                recent_window = self._recent_training_window_for_manifest(
                    edge_id=edge_id,
                    manifest=manifest,
                )
                next_checkpoint_model_version = _increment_model_version(
                    bundle_model_version,
                    field_name="bundle model version",
                )
                existing_contract = self._load_split_runtime_contract(
                    edge_id=edge_id,
                    manifest=manifest,
                )
                front_version = str(
                    self._training_window_manifest_context(manifest).get("front_version") or "0"
                )
                if (
                    existing_contract is None
                    and front_version == "0"
                    and bundle_model_version != "0"
                ):
                    message = (
                        "Missing SplitRuntimeContract for front_version=0; refusing "
                        f"to create it from tail checkpoint model_version={bundle_model_version}. "
                        "Run a model_version=0 training job first so the contract is "
                        "created from native pretrained front weights."
                    )
                    logger.warning("[FixedSplitCL] {}", message)
                    self._log_stage_duration("total round time", total_round_started)
                    return False, "", message

                if bundle_model_version == "0":
                    logger.info(
                        "[FixedSplitCL] Bundle model_version=0 for edge {}; "
                        "ignoring any cached {} weights and starting from native "
                        "{} weights.",
                        edge_id,
                        current_model_name,
                        self._native_training_source_label(current_model_name),
                    )
                    tmp_model = self._load_edge_training_model(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        cache_policy="native_only",
                        runtime_input_tensor_shape=manifest_runtime_input_shape,
                        model_metadata=manifest_model_metadata,
                    )
                else:
                    metadata = self._require_matching_edge_weights_metadata(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        bundle_model_version=bundle_model_version,
                    )
                    logger.info(
                        "[FixedSplitCL] Resuming edge {} {} training from persisted "
                        "checkpoint version {}.",
                        edge_id,
                        current_model_name,
                        metadata["checkpoint_model_version"],
                    )
                    tmp_model = self._load_edge_training_model(
                        model_name=current_model_name,
                        edge_id=edge_id,
                        cache_policy="edge_only",
                        runtime_input_tensor_shape=manifest_runtime_input_shape,
                        model_metadata=manifest_model_metadata,
                    )
                weights_metadata = self._build_checkpoint_weights_metadata(
                    edge_id=edge_id,
                    model_name=current_model_name,
                    model=tmp_model,
                    checkpoint_model_version=next_checkpoint_model_version,
                    source_base_model_version=bundle_model_version,
                    runtime_input_tensor_shape=manifest_runtime_input_shape,
                )
                stage_started = time.perf_counter()
                prepared_splitter, prepared_candidate = self._build_bundle_splitter(
                    tmp_model,
                    manifest,
                    bundle_root=bundle_cache_path,
                    runtime_batch_size=effective_batch_size,
                )
                prepared_trace_sample_input = None
                self._log_stage_duration("runtime template load / bind", stage_started)
                pool_runtime_input_tensor_shape = self._infer_pool_runtime_input_tensor_shape(
                    tmp_model,
                    bundle_root=bundle_cache_path,
                    manifest=manifest,
                    prepared_trace_sample_input=prepared_trace_sample_input,
                )
                pool_model_input_size = (
                    (
                        int(pool_runtime_input_tensor_shape[-2]),
                        int(pool_runtime_input_tensor_shape[-1]),
                    )
                    if pool_runtime_input_tensor_shape is not None
                    and len(pool_runtime_input_tensor_shape) >= 3
                    else None
                )
                pool_input_resize_mode = (
                    get_split_runtime_input_resize_mode(get_split_runtime_model(tmp_model))
                    or "direct_resize"
                )
                stage_started = time.perf_counter()
                teacher_requests = self._build_low_quality_raw_teacher_annotation_requests(
                    bundle_cache_path=bundle_cache_path,
                    manifest=manifest,
                    edge_id=edge_id,
                    model_id=current_model_name,
                    target_model_metadata=manifest_model_metadata,
                )
                gt_annotations = self._teacher_annotation_stage().ensure_low_quality(
                    teacher_requests,
                )
                self._log_stage_duration("teacher annotation ensure", stage_started)
                contract_layout_tensors = self._contract_layout_tensors_from_runtime(
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    input_tensor_shape=[
                        int(dim)
                        for dim in list(
                            self._training_window_manifest_context(manifest).get(
                                "input_tensor_shape",
                                [],
                            )
                            or []
                        )
                    ],
                )
                split_contract = self._get_or_create_split_runtime_contract(
                    edge_id=edge_id,
                    manifest=manifest,
                    feature_tensors=contract_layout_tensors,
                    contract_layout_tensors=contract_layout_tensors,
                    model=tmp_model,
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    bundle_root=bundle_cache_path,
                    create_if_missing=True,
                )
                stage_started = time.perf_counter()
                low_quality_feature_entries = self._prepare_low_quality_feature_entries(
                    tmp_model,
                    manifest,
                    bundle_cache_path=bundle_cache_path,
                    gt_annotations=gt_annotations,
                    split_contract=split_contract,
                    splitter=prepared_splitter,
                    candidate=prepared_candidate,
                    model_name=current_model_name,
                    runtime_batch_size=effective_batch_size,
                )
                low_quality_staging_candidates = self._build_low_quality_staging_candidates(
                    feature_entries=low_quality_feature_entries,
                    model_input_size=pool_model_input_size,
                    resize_mode=pool_input_resize_mode,
                )
                append_stats = recent_window.append_samples(
                    low_quality_staging_candidates,
                    sample_source="low_quality",
                )
                uploaded_low_quality_count = int(
                    manifest.get("sample_count", len(low_quality_staging_candidates)) or 0
                )
                accepted_low_quality_sample_ids = [
                    str(sample.get("sample_id") or "")
                    for sample in low_quality_staging_candidates
                    if str(sample.get("sample_id") or "")
                ]
                accepted_low_quality_ids_message = (
                    "accepted_low_quality_sample_ids_json="
                    f"{json.dumps(accepted_low_quality_sample_ids, separators=(',', ':'))}"
                )
                recent_samples = recent_window.training_samples(
                    self.training_frame_count,
                    replay_fraction=float(getattr(self, "training_replay_fraction", 0.0)),
                )
                self._log_stage_duration(
                    "feature readiness + recent training-window append",
                    stage_started,
                )
                logger.info(
                    "[FixedSplitCL] recent training window edge={} accepted={} "
                    "replaced={} retained={} required={} dropped_old={} "
                    "low_quality_candidates={}.",
                    edge_id,
                    append_stats.accepted,
                    append_stats.replaced,
                    append_stats.retained,
                    self.training_frame_count,
                    append_stats.dropped_old,
                    len(low_quality_staging_candidates),
                )
                if len(recent_samples) < int(self.training_frame_count):
                    message = (
                        "Waiting for enough recent training samples: "
                        f"available={len(recent_samples)}, "
                        f"required={int(self.training_frame_count)}, "
                        f"accepted_low_quality_samples={append_stats.accepted}, "
                        f"uploaded_low_quality_samples={uploaded_low_quality_count}; "
                        f"{accepted_low_quality_ids_message}"
                    )
                    logger.info("[FixedSplitCL] {}", message)
                    self._log_stage_duration("total round time", total_round_started)
                    return False, "", message

                stage_started = time.perf_counter()
                (
                    bundle_info,
                    frame_dir,
                    preloaded_records,
                    gt_annotations,
                    sample_metadata_by_id,
                    _training_view,
                    _training_view_stats,
                ) = self._build_training_cache_view_from_recent_samples(
                    recent_samples,
                    contract=split_contract,
                    model_name=current_model_name,
                    edge_id=edge_id,
                )
                self._log_stage_duration("training cache view materialization", stage_started)
                active_sample_count = len(bundle_info["all_sample_ids"])
                if active_sample_count != int(self.training_frame_count):
                    raise RuntimeError(
                        "Recent training-window view must contain exactly "
                        f"{int(self.training_frame_count)} samples; "
                        f"got {active_sample_count}."
                    )
                required_dynamic_batch_min = max(
                    _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
                    _splitter_dynamic_batch_min(prepared_splitter),
                )
                proxy_sample_random_seed = str(
                    bundle_info.get("training_view_id")
                    or bundle_info.get("generation_id")
                    or f"{edge_id}:{current_model_name}"
                )
                validation_split = build_proxy_validation_split(
                    all_sample_ids=bundle_info["all_sample_ids"],
                    gt_annotations=gt_annotations,
                    validation_fraction=float(self.proxy_eval_validation_fraction),
                    max_eval_samples=self.proxy_eval_max_samples,
                    random_seed=proxy_sample_random_seed,
                    min_train_samples=required_dynamic_batch_min,
                )
                train_sample_count = active_sample_count
                validation_sample_count = len(validation_split.validation_sample_ids)
                if (
                    active_sample_count < required_dynamic_batch_min
                    or validation_sample_count <= 0
                ):
                    message = (
                        "Not enough compatible samples for official mAP_50_95 validation split: "
                        f"active_samples={active_sample_count}, "
                        f"train_samples={train_sample_count}, "
                        f"validation_samples={validation_sample_count}, "
                        f"required_train_min={required_dynamic_batch_min}; "
                        f"{accepted_low_quality_ids_message}"
                    )
                    logger.warning("[FixedSplitCL] {}", message)
                    self._log_stage_duration("total round time", total_round_started)
                    return False, "", message
                training_bundle_info = dict(bundle_info)
                training_bundle_info["all_sample_ids"] = list(bundle_info["all_sample_ids"])
                gt_annotations = dict(gt_annotations)
                validation_gt_annotations = dict(validation_split.validation_gt_annotations)
                effective_batch_size = self._resolve_fixed_split_runtime_batch_size(
                    current_model_name,
                    num_train_samples=train_sample_count,
                )
                training_cache_path = str(bundle_info.get("training_view_path") or "")
                effective_batch_size = self._negotiate_cached_split_runtime_batch_size(
                    current_model_name=current_model_name,
                    training_cache_path=training_cache_path,
                    all_sample_ids=training_bundle_info["all_sample_ids"],
                    gt_annotations=gt_annotations,
                    prepared_splitter=prepared_splitter,
                    prepared_candidate=prepared_candidate,
                    configured_batch_size=effective_batch_size,
                    preloaded_records=preloaded_records,
                    manifest=manifest,
                )
                logger.info(
                    "[FixedSplitCL] Training from {} recent-window sample(s) with "
                    "{} validation sample(s) via TrainingCacheView(source=recent_training_window) "
                    "({} train label entry/entries; {} validation label entry/entries).",
                    len(training_bundle_info["all_sample_ids"]),
                    len(validation_split.validation_sample_ids),
                    len(gt_annotations),
                    len(validation_gt_annotations),
                )
                if self.connectivity_smoke_only:
                    stage_started = time.perf_counter()
                    encoded = base64.b64encode(
                        self._serialise_model_bytes(
                            tmp_model,
                            model_name=current_model_name,
                            edge_id=edge_id,
                            weights_metadata=weights_metadata,
                        )
                    ).decode("utf-8")
                    self._log_stage_duration("serialization / encoding", stage_started)
                    self._log_stage_duration("total round time", total_round_started)
                    logger.success(
                        "[FixedSplitCL][ConnectivitySmoke] Connected edge inference, "
                        "sample upload, cloud annotation, feature rebuild, contract "
                        "creation, and tail-runtime preparation for edge {} with {} "
                        "active sample(s) ({} GT-annotated); skipped full retraining.",
                        edge_id,
                        len(bundle_info["all_sample_ids"]),
                        len(gt_annotations),
                    )
                    return (
                        True,
                        encoded,
                        "Fixed split connectivity smoke successful; skipped full retraining; "
                        f"{accepted_low_quality_ids_message}",
                    )
                proxy_evaluator = self._fixed_split_proxy_evaluator()
                proxy_eval_frame_cache = proxy_evaluator.new_frame_cache()

                training_result = self._run_fixed_split_retrain(
                    tmp_model,
                    current_model_name=current_model_name,
                    bundle_info=training_bundle_info,
                    manifest=manifest,
                    bundle_cache_path=bundle_cache_path,
                    training_cache_path=training_cache_path,
                    frame_dir=frame_dir,
                    gt_annotations=gt_annotations,
                    validation_gt_annotations=validation_gt_annotations,
                    validation_sample_ids=list(validation_split.validation_sample_ids),
                    num_epoch=effective_num_epoch,
                    prepared_trace_sample_input=prepared_trace_sample_input,
                    prepared_splitter=prepared_splitter,
                    prepared_candidate=prepared_candidate,
                    effective_batch_size=effective_batch_size,
                    sample_metadata_by_id=sample_metadata_by_id,
                    proxy_eval_frame_cache=proxy_eval_frame_cache,
                    preloaded_records=preloaded_records,
                )
                proxy_metrics_after = dict(training_result.proxy_metrics_after)
                proxy_summary = proxy_evaluator.format_summary(
                    None,
                    proxy_metrics_after,
                )
                if proxy_summary is not None:
                    logger.info("[FixedSplitCL] {}", proxy_summary)
                else:
                    logger.info(
                        "[FixedSplitCL] Proxy mAP_50_95 skipped "
                        "(gt_samples={}, evaluated={}, empty_gt={}, missing_frame={}).",
                        int(proxy_metrics_after.get("total_gt_samples", 0)),
                        int(proxy_metrics_after.get("evaluated_samples", 0)),
                        int(proxy_metrics_after.get("skipped_empty_gt", 0)),
                        int(proxy_metrics_after.get("skipped_missing_frame", 0)),
                    )

                if not training_result.result_available:
                    self._log_stage_duration("total round time", total_round_started)
                    message = (
                        "Fixed split retraining completed without a publishable checkpoint; "
                        "validation proxy_mAP_50_95 was unavailable"
                    )
                    if proxy_summary is not None:
                        message = f"{message}; {proxy_summary}"
                    message = f"{message}; {accepted_low_quality_ids_message}"
                    logger.warning("[FixedSplitCL] {}", message)
                    return True, "", message

                stage_started = time.perf_counter()
                encoded = base64.b64encode(
                    self._serialise_model_bytes(
                        tmp_model,
                        model_name=current_model_name,
                        edge_id=edge_id,
                        weights_metadata=weights_metadata,
                    )
                ).decode("utf-8")
                self._log_stage_duration("serialization / encoding", stage_started)
                self._log_stage_duration("total round time", total_round_started)
                success_message = (
                    f"Fixed split retraining successful; {proxy_summary}"
                    if proxy_summary is not None
                    else "Fixed split retraining successful; proxy_mAP_50_95 skipped"
                )
                success_message = f"{success_message}; {accepted_low_quality_ids_message}"
                logger.success(
                    "[FixedSplitCL] {} done for edge {} with {} train samples and "
                    "{} validation samples.",
                    "Retraining",
                    edge_id,
                    len(training_bundle_info["all_sample_ids"]),
                    len(validation_split.validation_sample_ids),
                )
                return True, encoded, success_message
            except Exception as exc:
                self._log_stage_duration("total round time", total_round_started)
                message = "fixed-split training failed"
                logger.error(
                    "[FixedSplitCL] {} for edge={}: {}.",
                    message,
                    edge_id,
                    safe_error_summary(exc),
                )
                log_diagnostic_debug(
                    self,
                    "[FixedSplitCL] training failure diagnostics",
                    lambda error=exc: {"error": repr(error)},
                    runtime=True,
                )
                detail = f"{message}: {exc}"
                if accepted_low_quality_ids_message:
                    detail = f"{detail}; {accepted_low_quality_ids_message}"
                return False, "", detail
            finally:
                self._run_feature_cache_gc(reason="continual_learning")
