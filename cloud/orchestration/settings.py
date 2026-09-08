from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from cloud.annotation import (
    TeacherAnnotationService,
    TeacherAnnotationWorker,
    TeacherLabelCache,
)
from cloud.feature_cache import FeatureCacheGC
from cloud.orchestration.fixed_split_dependencies import _GLOBAL_TEACHER_ANNOTATION_QUEUE
from cloud.training.proxy_metadata import normalise_shard_dtype as _normalise_shard_dtype
from cloud.workers.gpu_lease_manager import LeaseRequest
from common.logging_sanitizer import log_diagnostic_debug
from model_management.fixed_split_runtime_template import (
    get_fixed_split_runtime_template_cache,
)

if TYPE_CHECKING:
    from model_management.object_detection import Object_Detection


@dataclass(frozen=True)
class FeatureCacheSettings:
    store_root_dir: str
    view_root_dir: str
    storage_format: str
    accepted_storage_formats: tuple[str, ...]
    materialization_mode: str
    view_source: str
    gc_enabled: bool
    gc_dry_run: bool
    gc_max_live_generations: int


@dataclass(frozen=True)
class TeacherAnnotationSettings:
    async_enabled: bool
    cache_enabled: bool
    wait_timeout_sec: float
    worker_batch_size: int
    worker_max_queue_size: int
    worker_max_retries: int
    oom_retry_enabled: bool
    min_worker_batch_size: int
    cache_root_dir: str


@dataclass(frozen=True)
class OrchestrationSettings:
    edge_model_name: str
    workspace_root: str
    default_num_epoch: int
    max_concurrent_jobs: int
    batch_size: int
    trace_batch_size: int
    fixed_split_runtime_smoke_validate: bool
    fixed_split_runtime_diagnostics: bool
    log_internal_ids: bool
    training_frame_count: int
    training_replay_fraction: float
    training_replay_retention_multiplier: int
    recent_training_window_root: str
    split_contract_root: str
    feature_cache: FeatureCacheSettings
    teacher_annotation: TeacherAnnotationSettings

    @classmethod
    def from_config(cls, config: Any) -> "OrchestrationSettings":
        cl_cfg = getattr(config, "continual_learning", None)
        feature_cache_cfg = getattr(cl_cfg, "feature_cache", None) if cl_cfg is not None else None
        workspace_root = os.path.abspath(
            str(getattr(config, "workspace_root", "./cache/server_workspace"))
        )
        teacher_cfg = getattr(cl_cfg, "teacher_annotation", None) if cl_cfg is not None else None
        edge_model_name = str(getattr(config, "edge_model_name", "rfdetr_nano"))
        replay_fraction_by_model = dict(
            getattr(cl_cfg, "training_replay_fraction_by_model", {}) or {}
        )
        training_replay_fraction = float(
            replay_fraction_by_model.get(
                edge_model_name,
                getattr(cl_cfg, "training_replay_fraction", 0.0),
            )
        )
        training_frame_count_by_model = dict(
            getattr(cl_cfg, "training_frame_count_by_model", {}) or {}
        )
        training_frame_count = int(
            training_frame_count_by_model.get(
                edge_model_name,
                getattr(cl_cfg, "training_frame_count", 128),
            )
        )
        return cls(
            edge_model_name=edge_model_name,
            workspace_root=workspace_root,
            default_num_epoch=int(getattr(cl_cfg, "num_epoch", 2)) if cl_cfg else 2,
            max_concurrent_jobs=int(getattr(cl_cfg, "max_concurrent_jobs", 2)) if cl_cfg else 2,
            batch_size=int(getattr(cl_cfg, "batch_size", 2)) if cl_cfg else 2,
            trace_batch_size=int(getattr(cl_cfg, "trace_batch_size", 1)) if cl_cfg else 1,
            fixed_split_runtime_smoke_validate=bool(
                getattr(cl_cfg, "fixed_split_runtime_smoke_validate", False)
            )
            if cl_cfg
            else False,
            fixed_split_runtime_diagnostics=bool(
                getattr(cl_cfg, "fixed_split_runtime_diagnostics", False)
            )
            if cl_cfg
            else False,
            log_internal_ids=bool(getattr(cl_cfg, "log_internal_ids", False)) if cl_cfg else False,
            training_frame_count=max(1, training_frame_count),
            training_replay_fraction=max(0.0, min(0.999999, training_replay_fraction)),
            training_replay_retention_multiplier=max(
                1,
                int(getattr(cl_cfg, "training_replay_retention_multiplier", 4)),
            ),
            recent_training_window_root=os.path.abspath(
                str(
                    getattr(
                        cl_cfg,
                        "recent_training_window_root",
                        os.path.join(workspace_root, "recent_training_windows"),
                    )
                )
            ),
            split_contract_root=os.path.abspath(
                str(getattr(cl_cfg, "split_contract_root", "./cache/split_contracts"))
            ),
            feature_cache=FeatureCacheSettings(
                store_root_dir=os.path.abspath(
                    str(
                        getattr(
                            feature_cache_cfg,
                            "shard_root_dir",
                            getattr(
                                feature_cache_cfg, "store_root_dir", "./cache/cloud_feature_shards"
                            ),
                        )
                    )
                ),
                view_root_dir=os.path.abspath(
                    str(getattr(feature_cache_cfg, "view_root_dir", "./cache/cloud_training_views"))
                ),
                storage_format=str(
                    getattr(feature_cache_cfg, "storage_format", "safetensors_shard")
                )
                .strip()
                .lower(),
                accepted_storage_formats=tuple(
                    str(item).strip().lower()
                    for item in list(
                        getattr(
                            feature_cache_cfg,
                            "accepted_storage_formats",
                            ["safetensors_shard", "npy_memmap_shard"],
                        )
                        or []
                    )
                ),
                materialization_mode=str(
                    getattr(feature_cache_cfg, "materialization_mode", "direct_ref")
                )
                .strip()
                .lower(),
                view_source=str(
                    getattr(feature_cache_cfg, "view_source", "recent_training_window")
                )
                .strip()
                .lower(),
                gc_enabled=bool(getattr(feature_cache_cfg, "gc_enabled", False)),
                gc_dry_run=bool(getattr(feature_cache_cfg, "gc_dry_run", True)),
                gc_max_live_generations=max(
                    0, int(getattr(feature_cache_cfg, "gc_max_live_generations", 3))
                ),
            ),
            teacher_annotation=TeacherAnnotationSettings(
                async_enabled=bool(getattr(teacher_cfg, "async_enabled", False)),
                cache_enabled=bool(getattr(teacher_cfg, "cache_enabled", True)),
                wait_timeout_sec=float(getattr(teacher_cfg, "wait_timeout_sec", 0.5)),
                worker_batch_size=int(getattr(teacher_cfg, "worker_batch_size", 8)),
                worker_max_queue_size=int(getattr(teacher_cfg, "worker_max_queue_size", 4096)),
                worker_max_retries=int(getattr(teacher_cfg, "worker_max_retries", 2)),
                oom_retry_enabled=bool(getattr(teacher_cfg, "oom_retry_enabled", True)),
                min_worker_batch_size=int(getattr(teacher_cfg, "min_worker_batch_size", 1)),
                cache_root_dir=os.path.abspath(
                    str(getattr(teacher_cfg, "cache_root_dir", "./cache/teacher_label_cache"))
                ),
            ),
        )


class PipelineLifecycleMixin:
    def __init__(
        self,
        config,
        large_object_detection: "Object_Detection",
        *,
        gpu_lease_client=None,
        worker_id: str = "",
        teacher_annotation_service=None,
        teacher_annotation_metadata: Mapping[str, object] | None = None,
    ):
        self.config = config
        self.large_od = large_object_detection
        self.gpu_lease_client = gpu_lease_client
        self.worker_id = str(worker_id or "")
        self.teacher_annotation_metadata = dict(teacher_annotation_metadata or {})
        self.lazy_cuda_init = bool(gpu_lease_client is not None)
        self.settings = OrchestrationSettings.from_config(config)
        settings = self.settings
        self.log_internal_ids = settings.log_internal_ids

        # Name of the lightweight model to retrain (mirrors edge model)
        self.edge_model_name = settings.edge_model_name
        self.weight_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "model_management",
            "models",
        )
        os.makedirs(self.weight_folder, exist_ok=True)
        self.device = (
            torch.device("cpu")
            if self.lazy_cuda_init
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve and validate configured weights_path if provided
        configured_weights = str(getattr(config, "weights_path", "") or "").strip()
        if configured_weights:
            # Convert relative path to absolute path
            if not os.path.isabs(configured_weights):
                configured_weights = os.path.abspath(configured_weights)

            if os.path.exists(configured_weights):
                configured_model = self._known_model_name_for_weights_path(configured_weights)
                if (
                    configured_model is not None
                    and configured_model
                    != self._normalize_model_name_for_lookup(self.edge_model_name)
                ):
                    logger.warning(
                        "[CloudCL] configured weights artifact belongs to model={} "
                        "and will be ignored for edge model={}.",
                        configured_model,
                        self.edge_model_name,
                    )
                    log_diagnostic_debug(
                        self,
                        "[CloudCL] ignored configured weights",
                        lambda: {
                            "weights_path": configured_weights,
                            "configured_model": configured_model,
                            "edge_model_name": self.edge_model_name,
                        },
                    )
                else:
                    logger.info(
                        "[CloudCL] Using configured weights for model={}.",
                        self.edge_model_name,
                    )
                    log_diagnostic_debug(
                        self,
                        "[CloudCL] configured weights path",
                        lambda: {
                            "model": self.edge_model_name,
                            "weights_path": configured_weights,
                        },
                    )
                # Update config with resolved absolute path
                config.weights_path = configured_weights
            else:
                logger.error(
                    "[CloudCL] Configured weights artifact is unavailable for model={}.",
                    self.edge_model_name,
                )
                log_diagnostic_debug(
                    self,
                    "[CloudCL] missing configured weights",
                    lambda: {"weights_path": configured_weights},
                )
        else:
            logger.warning(
                "[CloudCL] No weights_path configured for edge model {}. "
                "Will use default pretrained weights which may be incompatible with edge model.",
                self.edge_model_name,
            )

        # Default training hyper-parameters (overridable from config)
        cl_cfg = getattr(config, "continual_learning", None)
        self.default_num_epoch = settings.default_num_epoch
        self.max_concurrent_jobs = settings.max_concurrent_jobs
        self.batch_size = settings.batch_size
        self.trace_batch_size = settings.trace_batch_size
        self.fixed_split_runtime_smoke_validate = settings.fixed_split_runtime_smoke_validate
        self.fixed_split_runtime_diagnostics = settings.fixed_split_runtime_diagnostics
        self.training_frame_count = settings.training_frame_count
        self.training_replay_fraction = settings.training_replay_fraction
        self.training_replay_retention_multiplier = (
            settings.training_replay_retention_multiplier
        )
        self.feature_cache_mode = (
            (str(getattr(cl_cfg, "feature_cache_mode", "auto")) if cl_cfg else "auto")
            .strip()
            .lower()
        )
        if self.feature_cache_mode not in {"auto", "memory", "disk"}:
            raise ValueError(
                "server.continual_learning.feature_cache_mode must be one of: auto, memory, disk."
            )
        feature_cache_cfg = getattr(cl_cfg, "feature_cache", None) if cl_cfg is not None else None
        self.feature_cache_view_source = settings.feature_cache.view_source
        if self.feature_cache_view_source != "recent_training_window":
            raise ValueError(
                "server.continual_learning.feature_cache.view_source must be "
                "'recent_training_window'."
            )
        self.feature_cache_materialization_mode = settings.feature_cache.materialization_mode
        if self.feature_cache_materialization_mode != "direct_ref":
            raise ValueError(
                "server.continual_learning.feature_cache.materialization_mode must be 'direct_ref'."
            )
        self.feature_cache_store_root_dir = settings.feature_cache.store_root_dir
        self.feature_cache_storage_format = settings.feature_cache.storage_format
        self.feature_cache_accepted_storage_formats = list(
            settings.feature_cache.accepted_storage_formats
        )
        self.feature_cache_shard_max_samples = max(
            1,
            int(getattr(feature_cache_cfg, "shard_max_samples", 64)),
        )
        self.feature_cache_shard_dtype = _normalise_shard_dtype(
            getattr(feature_cache_cfg, "shard_dtype", None)
        )
        self.feature_cache_payload_cache_enabled = bool(
            getattr(feature_cache_cfg, "payload_cache_enabled", True)
        )
        self.feature_cache_payload_cache_max_cpu_bytes = int(
            getattr(feature_cache_cfg, "payload_cache_max_cpu_bytes", 4294967296)
        )
        self.feature_cache_pin_memory = bool(getattr(feature_cache_cfg, "pin_memory", True))
        self.feature_cache_non_blocking_transfer = bool(
            getattr(feature_cache_cfg, "non_blocking_transfer", True)
        )
        self.feature_cache_view_root_dir = settings.feature_cache.view_root_dir
        self.feature_cache_validate_refs = bool(getattr(feature_cache_cfg, "validate_refs", True))
        self.feature_cache_deep_validate_feature_payload = bool(
            getattr(feature_cache_cfg, "deep_validate_feature_payload", False)
        )
        self.feature_cache_deep_validate_sample_rate = max(
            0.0,
            min(
                1.0,
                float(getattr(feature_cache_cfg, "deep_validate_sample_rate", 0.0)),
            ),
        )
        self.feature_cache_feature_rebuild_batch_size = max(
            1,
            int(getattr(feature_cache_cfg, "feature_rebuild_batch_size", 16)),
        )
        self.feature_cache_gc_enabled = bool(getattr(feature_cache_cfg, "gc_enabled", False))
        self.feature_cache_gc_dry_run = bool(getattr(feature_cache_cfg, "gc_dry_run", True))
        self.feature_cache_gc_max_live_generations = max(
            0, int(getattr(feature_cache_cfg, "gc_max_live_generations", 3))
        )
        removed_cl_fields = {
            "rebuild_batch_size": (
                "server.continual_learning.rebuild_batch_size has been removed; "
                "use server.continual_learning.batch_size for the shared "
                "cloud continual-learning batch size."
            ),
            "min_wrapper_fixed_split_num_epoch": (
                "server.continual_learning.min_wrapper_fixed_split_num_epoch has been removed; "
                "cloud fixed-split retraining no longer forces a minimum epoch count."
            ),
            "min_rfdetr_fixed_split_num_epoch": (
                "server.continual_learning.min_rfdetr_fixed_split_num_epoch has been removed; "
                "cloud fixed-split retraining no longer forces a minimum epoch count."
            ),
        }
        if cl_cfg:
            for field_name, message in removed_cl_fields.items():
                if getattr(cl_cfg, field_name, None) is not None:
                    raise ValueError(message)
        self.default_split_learning_rate = (
            float(getattr(cl_cfg, "split_learning_rate", 1e-3)) if cl_cfg else 1e-3
        )
        self.teacher_annotation_threshold = (
            float(getattr(cl_cfg, "teacher_annotation_threshold", 0.6)) if cl_cfg else 0.6
        )
        self.teacher_batch_size = (
            int(getattr(cl_cfg, "teacher_batch_size", self.batch_size))
            if cl_cfg
            else self.batch_size
        )
        teacher_settings = settings.teacher_annotation
        self.teacher_annotation_async_enabled = teacher_settings.async_enabled
        self.teacher_annotation_cache_enabled = teacher_settings.cache_enabled
        self.teacher_annotation_wait_timeout_sec = teacher_settings.wait_timeout_sec
        self.teacher_annotation_worker_batch_size = teacher_settings.worker_batch_size
        self.teacher_annotation_worker_max_queue_size = teacher_settings.worker_max_queue_size
        self.teacher_annotation_worker_max_retries = teacher_settings.worker_max_retries
        self.teacher_annotation_oom_retry_enabled = teacher_settings.oom_retry_enabled
        self.teacher_annotation_min_worker_batch_size = teacher_settings.min_worker_batch_size
        self.teacher_annotation_cache_root = teacher_settings.cache_root_dir
        raw_proxy_eval_interval_epochs = (
            getattr(cl_cfg, "proxy_eval_interval_epochs", None) if cl_cfg else None
        )
        if raw_proxy_eval_interval_epochs is None and cl_cfg:
            raw_proxy_eval_interval_epochs = getattr(cl_cfg, "proxy_eval_interval_rounds", 10)
        self.proxy_eval_interval_epochs = (
            int(raw_proxy_eval_interval_epochs)
            if raw_proxy_eval_interval_epochs is not None
            else 10
        )
        self.proxy_eval_interval_rounds = self.proxy_eval_interval_epochs
        self.proxy_eval_patience = int(getattr(cl_cfg, "proxy_eval_patience", 2)) if cl_cfg else 2
        self.proxy_eval_min_delta = (
            float(getattr(cl_cfg, "proxy_eval_min_delta", 0.002)) if cl_cfg else 0.002
        )
        self.wrapper_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "wrapper_fixed_split_learning_rate", 3e-5)) if cl_cfg else 3e-5
        )
        self.tinynext_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "tinynext_fixed_split_learning_rate", 1e-3)) if cl_cfg else 1e-3
        )
        self.rfdetr_fixed_split_learning_rate = (
            float(getattr(cl_cfg, "rfdetr_fixed_split_learning_rate", 1e-4)) if cl_cfg else 1e-4
        )
        self.tinynext_fixed_split_target_steps_per_round = (
            int(getattr(cl_cfg, "tinynext_fixed_split_target_steps_per_round", 4)) if cl_cfg else 4
        )
        self.yolo_fixed_split_target_steps_per_round = (
            int(getattr(cl_cfg, "yolo_fixed_split_target_steps_per_round", 4)) if cl_cfg else 4
        )
        self.rfdetr_fixed_split_target_steps_per_round = (
            int(getattr(cl_cfg, "rfdetr_fixed_split_target_steps_per_round", 4)) if cl_cfg else 4
        )
        raw_proxy_eval_max_samples = (
            getattr(cl_cfg, "proxy_eval_max_samples", None) if cl_cfg else None
        )
        self.proxy_eval_max_samples = (
            128 if raw_proxy_eval_max_samples in (None, "") else int(raw_proxy_eval_max_samples)
        )
        self.proxy_eval_validation_fraction = (
            float(getattr(cl_cfg, "proxy_eval_validation_fraction", 0.2)) if cl_cfg else 0.2
        )
        self.proxy_eval_max_dets = (
            int(getattr(cl_cfg, "proxy_eval_max_dets", 500)) if cl_cfg else 500
        )
        self.proxy_eval_frame_cache_enabled = (
            bool(getattr(cl_cfg, "proxy_eval_frame_cache_enabled", True)) if cl_cfg else True
        )
        self.connectivity_smoke_only = (
            bool(getattr(cl_cfg, "connectivity_smoke_only", False)) if cl_cfg else False
        )
        self.workspace_root = settings.workspace_root
        os.makedirs(self.feature_cache_store_root_dir, exist_ok=True)
        os.makedirs(self.feature_cache_view_root_dir, exist_ok=True)
        self.recent_training_window_root = settings.recent_training_window_root
        os.makedirs(self.recent_training_window_root, exist_ok=True)
        self.split_contract_root = settings.split_contract_root
        os.makedirs(self.split_contract_root, exist_ok=True)
        self._fixed_split_runtime_template_cache = get_fixed_split_runtime_template_cache()

        self._edge_locks_guard = threading.Lock()
        self._edge_locks: dict[str, threading.Lock] = {}
        self._job_state_lock = threading.Lock()
        self._queued_jobs = 0
        self._active_jobs = 0
        self._training_slots = threading.BoundedSemaphore(self.max_concurrent_jobs)
        self._teacher_queue_state = _GLOBAL_TEACHER_ANNOTATION_QUEUE
        self._initial_state_reset_lock = threading.Lock()
        self._initial_state_reset_sessions: dict[str, str] = {}
        self._teacher_weights_fingerprint_cache: str | None = None
        self.teacher_label_cache = TeacherLabelCache(
            self.teacher_annotation_cache_root,
            enabled=self.teacher_annotation_cache_enabled,
            log_internal_ids=self.log_internal_ids,
        )
        self.teacher_annotation_worker: TeacherAnnotationWorker | None = None
        if (
            teacher_annotation_service is None
            and self.teacher_annotation_async_enabled
            and self.teacher_annotation_cache_enabled
        ):
            self.teacher_annotation_worker = TeacherAnnotationWorker(
                label_cache=self.teacher_label_cache,
                batch_inference=getattr(self.large_od, "large_inference_batch", None),
                single_inference=getattr(self.large_od, "large_inference", None),
                label_builder=self._teacher_labels_from_request_prediction,
                teacher_scope=self._teacher_annotation_scope,
                max_queue_size=self.teacher_annotation_worker_max_queue_size,
                worker_batch_size=self.teacher_annotation_worker_batch_size,
                max_retries=self.teacher_annotation_worker_max_retries,
                oom_retry_enabled=self.teacher_annotation_oom_retry_enabled,
                min_worker_batch_size=self.teacher_annotation_min_worker_batch_size,
                log_internal_ids=self.log_internal_ids,
            )
        self.teacher_annotation_service = (
            teacher_annotation_service
            if teacher_annotation_service is not None
            else TeacherAnnotationService(
                label_cache=self.teacher_label_cache,
                worker=self.teacher_annotation_worker,
                log_internal_ids=self.log_internal_ids,
            )
        )
        logger.info(
            "[TeacherAnnotation][Worker] async_enabled={} cache_enabled={} shared_service={} "
            "worker_batch_size={} max_queue_size={}",
            self.teacher_annotation_async_enabled,
            self.teacher_annotation_cache_enabled,
            teacher_annotation_service is not None,
            self.teacher_annotation_worker_batch_size,
            self.teacher_annotation_worker_max_queue_size,
        )
        log_diagnostic_debug(
            self,
            "[TeacherAnnotation][Worker] cache diagnostics",
            lambda: {"cache_root": self.teacher_annotation_cache_root},
        )

    def _run_feature_cache_gc(self, *, reason: str) -> None:
        """Remove feature shards no longer referenced by active training state."""
        if not self.feature_cache_gc_enabled:
            return
        try:
            result = FeatureCacheGC(
                store_root_dir=self.feature_cache_store_root_dir,
                view_root_dir=self.feature_cache_view_root_dir,
                recent_training_window_root_dir=self.recent_training_window_root,
                max_live_generations=self.feature_cache_gc_max_live_generations,
                dry_run=self.feature_cache_gc_dry_run,
            ).collect()
            logger.info(
                "[FeatureCache][GC] reason={} dry_run={} scanned={} retained={} "
                "deleted={} deleted_bytes={}",
                reason,
                result.dry_run,
                result.scanned_files,
                result.retained_files,
                result.deleted_files,
                result.deleted_bytes,
            )
        except Exception as exc:
            # GC must never make an otherwise successful training request fail.
            logger.warning("[FeatureCache][GC] reason={} failed: {}", reason, exc)

    def close(self) -> None:
        if self.teacher_annotation_worker is not None:
            self.teacher_annotation_worker.stop()

    def _edge_lock(self, edge_id: int | str) -> threading.Lock:
        edge_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(edge_id).strip()) or "unknown"
        with self._edge_locks_guard:
            lock = self._edge_locks.get(edge_key)
            if lock is None:
                lock = threading.Lock()
                self._edge_locks[edge_key] = lock
            return lock

    @contextmanager
    def _training_job_scope(self, edge_id: int | str):
        edge_lock = self._edge_lock(edge_id)
        with self._job_state_lock:
            self._queued_jobs += 1

        acquired_slot = False
        with edge_lock:
            try:
                self._training_slots.acquire()
                acquired_slot = True
                with self._job_state_lock:
                    self._queued_jobs = max(0, self._queued_jobs - 1)
                    self._active_jobs += 1
                yield
            finally:
                self._set_current_teacher_ticket(None)
                if acquired_slot:
                    with self._job_state_lock:
                        self._active_jobs = max(0, self._active_jobs - 1)
                    self._training_slots.release()
                else:
                    with self._job_state_lock:
                        self._queued_jobs = max(0, self._queued_jobs - 1)

    def training_queue_state(self) -> tuple[int, int]:
        with self._job_state_lock:
            return self._queued_jobs + self._active_jobs, self.max_concurrent_jobs

    @contextmanager
    def gpu_lease_scope(
        self,
        *,
        edge_id: int,
        job_id: str,
        workspace: str,
        exclusive: bool = False,
    ):
        if self.gpu_lease_client is None:
            yield
            return
        manifest = _read_workspace_manifest(workspace)
        model_meta = dict(manifest.get("model", {}) or {})
        split_plan = dict(manifest.get("split_plan", {}) or {})
        training_config = dict(manifest.get("training_config", {}) or {})
        model_name = str(
            model_meta.get("model_id")
            or model_meta.get("model_name")
            or manifest.get("model_id")
            or manifest.get("model_name")
            or getattr(self, "edge_model_name", "")
        )
        split_key = str(
            split_plan.get("canonical_split_key")
            or manifest.get("canonical_split_key")
            or ""
        )
        train_samples = len(
            [
                sample
                for sample in list(manifest.get("samples", []) or [])
                if isinstance(sample, dict) and str(sample.get("sample_id", "")).strip()
            ]
        )
        batch_size = 0
        if batch_size <= 0:
            batch_size = int(getattr(self, "batch_size", 0) or 0)
        estimate = float(
            getattr(
                getattr(
                    getattr(self.config, "edge_affine_workers", None),
                    "gpu_lease",
                    None,
                ),
                "default_estimated_job_memory_gb",
                18.0,
            )
        )
        handle = self.gpu_lease_client.acquire(
            LeaseRequest(
                edge_id=int(edge_id),
                worker_id=self.worker_id,
                job_id=str(job_id),
                model_name=model_name,
                split_key=split_key,
                batch_size=batch_size,
                train_samples=train_samples,
                estimated_peak_memory_gb=estimate,
                exclusive=bool(exclusive),
            )
        )
        with handle:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                else:
                    self.device = torch.device("cpu")
            except Exception:
                pass
            try:
                yield
            except Exception as exc:
                try:
                    self.gpu_lease_client.mark_oom(job_id=str(job_id), message=str(exc))
                except Exception:
                    pass
                raise
            finally:
                try:
                    if torch.cuda.is_available():
                        handle.observed_peak_memory_gb = torch.cuda.max_memory_reserved() / (
                            1024.0**3
                        )
                        torch.cuda.empty_cache()
                except Exception:
                    pass


def _read_workspace_manifest(workspace: str) -> dict[str, object]:
    for filename in ("trigger_manifest.json",):
        path = Path(workspace) / filename
        if not path.exists():
            continue
        try:
            import json

            return dict(json.loads(path.read_text(encoding="utf-8")) or {})
        except Exception:
            return {}
    return {}
