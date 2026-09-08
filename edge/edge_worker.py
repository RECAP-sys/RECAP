import base64
import io
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any, Callable, Mapping

import cv2
import grpc
import torch
from loguru import logger

from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from edge.box_motion import compensate_boxes_between_frames
from edge.diff import DiffProcessor
from edge.info import TASK_STATE
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    TrainingDecision,
    create_resource_aware_trigger,
    estimate_bandwidth,
    query_cloud_resource,
)
from edge.sample_quality import LOW_QUALITY, EntropyQualityClassifier
from edge.sample_store import EdgeSampleStore
from edge.sample_sync import HighQualitySampleSyncer
from edge.task import Task
from edge.teacher_sample_selector import LowQualityTeacherSampler
from edge.transmit import (
    download_trained_model,
    get_training_job_status,
    report_edge_model_version,
    submit_continual_learning_job,
)
from edge.window_drift_detector import DriftWindowState, WindowDriftDetector
from model_management.fixed_split import (
    SplitConstraints,
    SplitPlan,
    load_or_compute_fixed_split_plan,
)
from model_management.model_delta_payload import require_state_dict_delta_payload
from model_management.object_detection import InferenceArtifacts, Object_Detection
from model_management.split_model_adapters import (
    build_split_training_loss,
    get_split_runtime_input_resize_mode,
)
from model_management.universal_model_split import UniversalModelSplitter
from tools.grpc_options import grpc_message_options

_QUEUE_STOP = object()
_QUEUE_POLL_TIMEOUT_SECONDS = 0.05


class _FixedSplitRuntimeError(RuntimeError):
    pass


def _lower_current_thread_priority() -> bool:
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            get_current_thread = kernel32.GetCurrentThread
            get_current_thread.restype = ctypes.c_void_p
            set_thread_priority = kernel32.SetThreadPriority
            set_thread_priority.argtypes = (ctypes.c_void_p, ctypes.c_int)
            set_thread_priority.restype = ctypes.c_int
            return bool(set_thread_priority(get_current_thread(), -1))
        if hasattr(os, "setpriority") and hasattr(os, "PRIO_PROCESS"):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 5)
            return True
    except (AttributeError, OSError):
        return False
    return False


def _timeout_deadline(timeout: float | None) -> float | None:
    return None if timeout is None else time.monotonic() + max(0.0, float(timeout))


def _remaining_timeout(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _parse_json_message_field(message: str, field_name: str) -> object | None:
    marker = f"{field_name}="
    text = str(message or "")
    start = text.find(marker)
    if start < 0:
        return None
    raw_value = text[start + len(marker) :].strip()
    try:
        return json.loads(raw_value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _accepted_uploaded_sample_ids(message: str, uploaded_sample_ids: list[str]) -> list[str]:
    uploaded = [str(sample_id) for sample_id in uploaded_sample_ids if str(sample_id)]
    if not uploaded:
        return []
    payload = _parse_json_message_field(message, "accepted_low_quality_sample_ids_json")
    if not isinstance(payload, list):
        return []
    uploaded_set = set(uploaded)
    accepted: list[str] = []
    seen: set[str] = set()
    for value in payload:
        sample_id = str(value or "")
        if not sample_id or sample_id not in uploaded_set or sample_id in seen:
            continue
        accepted.append(sample_id)
        seen.add(sample_id)
    return accepted


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _first_tensor_batch_size(value: object) -> int | None:
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    return None


def resize_batch(value: object, current_batch_size: int, target_batch_size: int) -> object:
    current = int(current_batch_size)
    target = int(target_batch_size)
    if target <= current:
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == current:
            return value.narrow(0, 0, target)
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or int(value.shape[0]) != current:
            return value
        pad_count = target - current
        pad = value[-1:].expand(pad_count, *value.shape[1:]).clone()
        return torch.cat([value, pad], dim=0)
    if isinstance(value, Mapping):
        return {
            key: resize_batch(item, current_batch_size, target_batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(resize_batch(item, current_batch_size, target_batch_size) for item in value)
    if isinstance(value, list):
        return [resize_batch(item, current_batch_size, target_batch_size) for item in value]
    return value


def _fixed_split_trace_sample_input(sample_input: object, trace_batch_size: int = 1) -> object:
    current_batch_size = _first_tensor_batch_size(sample_input)
    if current_batch_size is None or current_batch_size >= trace_batch_size:
        return sample_input
    return resize_batch(sample_input, current_batch_size, trace_batch_size)


def _fixed_split_validation_batches(
    fixed_split_cfg: object | None,
    sample_input: object,
) -> list[int]:
    configured: int | None = None
    raw_batches = getattr(fixed_split_cfg, "validation_batches", None)
    if isinstance(raw_batches, (list, tuple)):
        batches: list[int] = []
        for value in raw_batches:
            parsed = _coerce_positive_int(value)
            if parsed is not None and parsed not in batches:
                batches.append(parsed)
        if 1 not in batches:
            batches.insert(0, 1)
        if batches:
            return batches
    for field_name in (
        "configured_training_batch",
        "fixed_split_training_batch",
        "training_batch_size",
    ):
        configured = _coerce_positive_int(getattr(fixed_split_cfg, field_name, None))
        if configured is not None:
            break
    if configured is None:
        configured = _first_tensor_batch_size(sample_input) or 1
    batches = [1]
    if int(configured) not in batches:
        batches.append(int(configured))
    return batches


@dataclass(frozen=True)
class SampleStatsDelta:
    total_samples: int = 1
    high_quality_count: int = 0
    low_quality_count: int = 0
    drift_window_sample_count: int = 0

    @classmethod
    def from_values(
        cls,
        *,
        quality_bucket: str,
        in_drift_window: bool = False,
    ) -> "SampleStatsDelta":
        return cls(
            high_quality_count=1 if quality_bucket != LOW_QUALITY else 0,
            low_quality_count=1 if quality_bucket == LOW_QUALITY else 0,
            drift_window_sample_count=1 if in_drift_window else 0,
        )


@dataclass(frozen=True)
class SampleWriteJob:
    store_kwargs: dict[str, Any]
    stats_delta: SampleStatsDelta


@dataclass(frozen=True)
class SampleCollectionJob:
    sample_id: str
    frame_index: int | None
    frame: Any
    inference: InferenceArtifacts
    split_config_id: str
    model_id: str
    model_version: str
    front_version: str
    split_key: str
    feature_abi_id: str
    runtime_contract: dict[str, Any]


@dataclass
class PendingModelUpdate:
    update_payload: dict[str, Any]
    state_dict: dict[str, Any]
    submitted_model_version: str | None
    next_model_version: str
    job_id: str = ""
    message: str = ""
    report: bool = True
    clear_samples: bool = True
    reset_drift: bool = True
    log_prefix: str = "[EdgeCL]"
    prepared_at: float = field(default_factory=time.time)
    applied_event: threading.Event = field(default_factory=threading.Event)
    applied_version: str | None = None
    apply_elapsed_ms: float | None = None
    error: BaseException | None = None


class AsyncSampleCollector:
    def __init__(
        self,
        handler: Callable[[SampleCollectionJob], None],
        *,
        maxsize: int = 0,
    ) -> None:
        self._handler = handler
        self._queue: Queue = Queue(maxsize=max(0, int(maxsize)))
        self._closed = False
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(
            target=self._run,
            name="edge-sample-collector",
            daemon=False,
        )
        self._thread.start()

    @property
    def errors(self) -> list[BaseException]:
        return list(self._errors)

    def submit(self, job: SampleCollectionJob) -> None:
        self.submit_nowait(job)

    def submit_nowait(self, job: SampleCollectionJob) -> None:
        if self._closed:
            raise RuntimeError("sample collector is closed")
        self._queue.put_nowait(job)

    def submit_blocking(self, job: SampleCollectionJob) -> None:
        if self._closed:
            raise RuntimeError("sample collector is closed")
        self._queue.put(job, block=True)

    def qsize(self) -> int:
        return self._queue.qsize()

    def flush(self, *, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if deadline is None:
                    self._queue.all_tasks_done.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._queue.all_tasks_done.wait(timeout=remaining)
        return True

    def close(self, *, timeout: float | None = None) -> bool:
        deadline = _timeout_deadline(timeout)
        flushed = True
        if not self._closed:
            flushed = self.flush(timeout=_remaining_timeout(deadline))
            self._closed = True
        self._thread.join(timeout=_remaining_timeout(deadline))
        return flushed and not self._thread.is_alive()

    def _run(self) -> None:
        _lower_current_thread_priority()
        while True:
            try:
                item = self._queue.get(block=True, timeout=_QUEUE_POLL_TIMEOUT_SECONDS)
            except Empty:
                if self._closed:
                    return
                continue
            try:
                if item is _QUEUE_STOP:
                    return
                try:
                    self._handler(item)
                except BaseException as exc:  # noqa: BLE001 - preserve worker thread.
                    self._errors.append(exc)
                    logger.error(
                        "Async sample collection failed: {}.",
                        safe_error_summary(exc),
                    )
            finally:
                self._queue.task_done()


class AsyncSampleWriter:
    def __init__(
        self,
        sample_store: EdgeSampleStore,
        *,
        maxsize: int = 0,
        worker_count: int = 1,
        performance_log_every_n_frames: int = 30,
        on_done: Callable[[SampleWriteJob, object | None, BaseException | None], None]
        | None = None,
    ) -> None:
        self.sample_store = sample_store
        self._queue: Queue = Queue(maxsize=max(0, int(maxsize)))
        self._on_done = on_done
        self.performance_log_every_n_frames = max(
            1,
            int(performance_log_every_n_frames),
        )
        self._closed = False
        self._errors: list[BaseException] = []
        workers = max(1, int(worker_count))
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"edge-sample-writer-{index + 1}",
                daemon=False,
            )
            for index in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    @property
    def errors(self) -> list[BaseException]:
        return list(self._errors)

    def qsize(self) -> int:
        return self._queue.qsize()

    def submit(self, job: SampleWriteJob) -> None:
        if self._closed:
            raise RuntimeError("sample writer is closed")
        self._queue.put(job, block=True)

    @staticmethod
    def _is_low_quality(job: SampleWriteJob) -> bool:
        return str(job.store_kwargs.get("quality_bucket", "")) == LOW_QUALITY

    def _drop_queued_low_quality(self) -> SampleWriteJob | None:
        queue_obj = self._queue
        with queue_obj.mutex:
            for index, item in enumerate(queue_obj.queue):
                if item is _QUEUE_STOP or not self._is_low_quality(item):
                    continue
                dropped = item
                del queue_obj.queue[index]
                queue_obj.unfinished_tasks = max(0, queue_obj.unfinished_tasks - 1)
                if queue_obj.unfinished_tasks == 0:
                    queue_obj.all_tasks_done.notify_all()
                queue_obj.not_full.notify()
                return dropped
        return None

    def submit_nowait(self, job: SampleWriteJob) -> tuple[bool, SampleWriteJob | None]:
        if self._closed:
            raise RuntimeError("sample writer is closed")
        try:
            self._queue.put_nowait(job)
            return True, None
        except Full:
            if self._is_low_quality(job):
                return False, None
            dropped = self._drop_queued_low_quality()
            if dropped is None:
                return False, None
            try:
                self._queue.put_nowait(job)
            except Full:
                return False, dropped
            return True, dropped

    def flush(self, *, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if deadline is None:
                    self._queue.all_tasks_done.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._queue.all_tasks_done.wait(timeout=remaining)
        return True

    def close(self, *, timeout: float | None = None) -> bool:
        deadline = _timeout_deadline(timeout)
        flushed = True
        if not self._closed:
            flushed = self.flush(timeout=_remaining_timeout(deadline))
            self._closed = True
        for thread in self._threads:
            thread.join(timeout=_remaining_timeout(deadline))
        return flushed and not any(thread.is_alive() for thread in self._threads)

    def _run(self) -> None:
        _lower_current_thread_priority()
        while True:
            try:
                item = self._queue.get(block=True, timeout=_QUEUE_POLL_TIMEOUT_SECONDS)
            except Empty:
                if self._closed:
                    return
                continue
            try:
                if item is _QUEUE_STOP:
                    return
                job = item
                record = None
                error = None
                try:
                    store_started = time.perf_counter()
                    record = self.sample_store.store_sample(**job.store_kwargs)
                    frame_index = int(job.store_kwargs.get("frame_index") or 0)
                    if frame_index <= 1 or frame_index % self.performance_log_every_n_frames == 0:
                        logger.info(
                            "[EdgePerfAsyncStore] sample_id={} frame={} "
                            "async_sample_store_ms={:.3f} async_writer_queue_size={}",
                            job.store_kwargs.get("sample_id"),
                            frame_index,
                            (time.perf_counter() - store_started) * 1000.0,
                            self.qsize(),
                        )
                except BaseException as exc:  # noqa: BLE001 - preserve worker thread.
                    error = exc
                    self._errors.append(exc)
                    logger.error(
                        "Async sample write failed: {}.",
                        safe_error_summary(exc),
                    )
                finally:
                    if self._on_done is not None:
                        self._on_done(job, record, error)
            finally:
                self._queue.task_done()


class EdgeWorker:
    def _record_experiment_metric(self, event: str, **payload: Any) -> None:
        writer = getattr(self.config, "experiment_metrics_writer", None)
        if writer is None:
            return
        try:
            writer.write(
                {
                    "event": str(event),
                    "timestamp_ms": int(time.time() * 1000),
                    "edge_id": int(self.edge_id),
                    **payload,
                }
            )
        except Exception as exc:
            logger.warning(
                "Experiment metric recording failed: {}.",
                safe_error_summary(exc),
            )

    @staticmethod
    def _resolve_training_poll_interval(config) -> float:
        retrain_cfg = getattr(config, "retrain", None)
        if retrain_cfg is None:
            return 5.0
        try:
            return max(0.5, float(getattr(retrain_cfg, "poll_interval_sec", 5.0)))
        except Exception:
            return 5.0

    @staticmethod
    def _resolve_training_not_found_grace(config) -> float:
        retrain_cfg = getattr(config, "retrain", None)
        if retrain_cfg is None:
            return 60.0
        try:
            return max(0.0, float(getattr(retrain_cfg, "status_not_found_grace_sec", 60.0)))
        except Exception:
            return 60.0

    @staticmethod
    def _resolve_resource_probe_interval(config) -> float:
        ra_cfg = getattr(config, "resource_aware_trigger", None)
        try:
            return max(0.5, float(getattr(ra_cfg, "probe_interval_sec", 5.0)))
        except Exception:
            return 5.0

    @staticmethod
    def _resolve_resource_probe_timeout(config) -> float:
        ra_cfg = getattr(config, "resource_aware_trigger", None)
        try:
            return max(0.1, float(getattr(ra_cfg, "probe_timeout_sec", 3.0)))
        except Exception:
            return 3.0

    @staticmethod
    def _resolve_bandwidth_probe_size(config) -> int:
        ra_cfg = getattr(config, "resource_aware_trigger", None)
        try:
            return max(1, int(getattr(ra_cfg, "bandwidth_probe_size_bytes", 64 * 1024)))
        except Exception:
            return 64 * 1024

    @staticmethod
    def _conservative_cloud_state() -> CloudResourceState:
        return CloudResourceState(
            cpu_utilization=1.0,
            gpu_utilization=1.0,
            memory_utilization=1.0,
            train_queue_size=1,
            max_queue_size=1,
        )

    def __init__(self, config):
        self.config = config
        self.edge_id = config.edge_id
        self.opencv_num_threads = max(
            1,
            int(getattr(config, "opencv_num_threads", 1)),
        )
        cv2.setNumThreads(self.opencv_num_threads)
        self.log_internal_ids = bool(
            getattr(getattr(config, "continual_learning", None), "log_internal_ids", False)
        )

        self.edge_processor = DiffProcessor.str_to_class(config.feature)()
        self.small_object_detection = Object_Detection(config, type="small inference")
        quality_cfg = config.sample_quality
        self.quality_classifier = EntropyQualityClassifier.from_config(quality_cfg)
        self.teacher_sample_selector = LowQualityTeacherSampler.from_config(quality_cfg)
        drift_cfg = config.window_drift
        self.window_drift_detector = WindowDriftDetector(
            window_size=int(drift_cfg.window_size),
            min_window_size=int(drift_cfg.min_window_size),
            low_quality_rate_threshold=float(drift_cfg.low_quality_rate_threshold),
            persistence_windows=int(drift_cfg.persistence_windows),
            evaluation_stride=int(drift_cfg.evaluation_stride),
            recovery_rate_threshold=float(drift_cfg.recovery_rate_threshold),
            recovery_windows=int(drift_cfg.recovery_windows),
            severe_anomaly_rate_threshold=float(drift_cfg.severe_anomaly_rate_threshold),
            recovery_severe_anomaly_rate=float(drift_cfg.recovery_severe_anomaly_rate),
            critical_confidence=float(drift_cfg.critical_confidence),
            critical_output_entropy_ratio=float(drift_cfg.critical_output_entropy_ratio),
            critical_feature_deviation=float(drift_cfg.critical_feature_deviation),
        )
        self.resource_trigger: ResourceAwareCLTrigger | None = None
        self._cloud_state: CloudResourceState | None = None
        self._bandwidth_mbps = 0.0
        self._resource_probe_failure_count = 0
        self._resource_probe_lock = threading.Lock()
        self._resource_probe_requested = threading.Event()
        self._resource_probe_inflight = False
        self._resource_probe_next_allowed_at = 0.0
        self._resource_probe_completed_at = 0.0
        self._resource_probe_required_after = 0.0
        self._drift_probe_active = False
        ra_cfg = getattr(config, "resource_aware_trigger", None)
        self.resource_trigger_enabled = bool(getattr(ra_cfg, "enabled", False)) if ra_cfg else False
        if self.resource_trigger_enabled:
            self.resource_trigger = create_resource_aware_trigger(config)
            logger.info(
                "Resource-aware CL trigger enabled (V={}, lambda_cloud={}, lambda_bw={})",
                self.resource_trigger.V,
                self.resource_trigger.lambda_cloud,
                self.resource_trigger.lambda_bw,
            )

        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)
        self.latest_result_lock = threading.Lock()
        self.latest_result = {
            "frame_index": None,
            "boxes": [],
            "labels": [],
            "scores": [],
            "confidence": 0.0,
            "entropy": 0.0,
            "model_version": "0",
            "result_source": "empty",
            "frame": None,
        }

        self.collect_flag = bool(self.config.retrain.flag)
        self.strict_sample_collection = bool(
            getattr(self.config, "strict_sample_collection", False)
        )
        self.performance_log_every_n_frames = max(
            1,
            int(getattr(self.config, "performance_log_every_n_frames", 30)),
        )
        self.retrain_flag = False
        self.pending_training_decision: TrainingDecision | None = None
        self._pending_model_update_lock = threading.Lock()
        self.pending_model_update: PendingModelUpdate | None = None
        self.edge_session_id = uuid.uuid4().hex
        feature_upload_cfg = self.config.feature_upload
        self.high_quality_sync_enabled = bool(feature_upload_cfg.sync_high_quality)
        self.sample_store = EdgeSampleStore(
            os.path.join(self.config.retrain.cache_path, "sample_store"),
            feature_storage_format=str(
                getattr(feature_upload_cfg, "storage_format", "safetensors_shard")
                or "safetensors_shard"
            ),
            feature_shard_dtype=getattr(feature_upload_cfg, "shard_dtype", None),
        )
        self._pending_sample_stats_lock = threading.Lock()
        self._pending_sample_stats = SampleStatsDelta(total_samples=0)
        self.sample_writer = None
        self.sample_collector = None
        self.sample_writer = AsyncSampleWriter(
            self.sample_store,
            maxsize=int(getattr(config, "local_queue_maxsize", 0) or 0),
            performance_log_every_n_frames=self.performance_log_every_n_frames,
            on_done=self._on_sample_write_done,
        )
        self.sample_collector = AsyncSampleCollector(
            self._collect_data_from_job,
            maxsize=int(getattr(config, "local_queue_maxsize", 0) or 0),
        )
        self.model_id = getattr(self.small_object_detection, "model_name", "edge-model")
        self.model_version = "0"
        self.front_version = "0"
        self.sample_syncer = None
        self.sample_syncer = HighQualitySampleSyncer(
            self.sample_store,
            server_ip=self.config.server_ip,
            edge_id=self.edge_id,
            feature_upload_config=self.config.feature_upload,
            enabled=self.high_quality_sync_enabled,
            context_provider=self._sample_sync_context,
            log_internal_ids=self.log_internal_ids,
        )
        self.bundle_cache_path = os.path.join(self.config.retrain.cache_path, "server_bundle")
        self.min_low_quality_samples = int(self.config.retrain.min_low_quality_samples)
        self.training_poll_interval_sec = self._resolve_training_poll_interval(config)
        self.training_not_found_grace_sec = self._resolve_training_not_found_grace(config)
        self.resource_probe_interval_sec = self._resolve_resource_probe_interval(config)
        self.resource_probe_timeout_sec = self._resolve_resource_probe_timeout(config)
        self.bandwidth_probe_size_bytes = self._resolve_bandwidth_probe_size(config)

        sl_cfg = getattr(config, "split_learning", None)
        self.split_learning_enabled = bool(getattr(sl_cfg, "enabled", False)) if sl_cfg else False
        self.split_learning_disable_reason: str | None = None
        self.universal_split_enabled = False
        self.universal_splitter: UniversalModelSplitter | None = None
        self.fixed_split_plan: SplitPlan | None = None
        self._fixed_split_init_attempted = False
        self._fixed_split_init_lock = threading.Lock()
        self.split_trace_image_size: tuple[int, int] | None = None
        if not self.split_learning_enabled:
            self.split_learning_disable_reason = (
                "disabled_in_config"
            )
        if self.collect_flag and not self.split_learning_enabled:
            self.collect_flag = False
            self._log_split_collection_disabled()

        self.diff = 0.0
        self.key_task = None
        self._stop_event = threading.Event()
        self._retrain_requested = threading.Event()
        self._closed = False
        self.resource_probe_processor: threading.Thread | None = None
        if self.resource_trigger_enabled:
            self.resource_probe_processor = threading.Thread(
                target=self.resource_probe_worker,
                daemon=True,
            )
            self.resource_probe_processor.start()
        if self.sample_syncer is not None:
            self.sample_syncer.start()

        self.diff_processor = threading.Thread(target=self.diff_worker, daemon=False)
        self.local_processor = threading.Thread(target=self.local_worker, daemon=False)
        self.retrain_processor = threading.Thread(target=self.retrain_worker, daemon=False)
        self.diff_processor.start()
        self.local_processor.start()
        self.retrain_processor.start()

    def _init_fixed_split_runtime(
        self,
        frame=None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        sl_cfg = getattr(self.config, "split_learning", None)
        fixed_split_cfg = getattr(sl_cfg, "fixed_split", None) if sl_cfg else None
        self._fixed_split_init_attempted = True
        if not self.split_learning_enabled:
            self.split_learning_disable_reason = "disabled_in_config"
            logger.info("Split learning disabled in config; skipping fixed split initialisation.")
            return

        cache_path = ""
        try:
            split_model = self.small_object_detection.get_split_runtime_model()
            self.universal_splitter = UniversalModelSplitter(
                device=next(split_model.parameters()).device,
            )
            self.universal_splitter.trainability_loss_fn = build_split_training_loss(
                self.small_object_detection.model
            )
            if frame is not None:
                trace_image_size = tuple(int(value) for value in frame.shape[:2])
                sample_input = self.small_object_detection.prepare_splitter_input(frame)
            else:
                trace_image_size = image_size or (224, 224)
                sample_input = self.small_object_detection.build_split_sample_input(
                    trace_image_size
                )
            trace_sample_input = _fixed_split_trace_sample_input(sample_input, 1)
            validation_batches = _fixed_split_validation_batches(
                fixed_split_cfg,
                trace_sample_input,
            )
            constraints = SplitConstraints.from_config(fixed_split_cfg)
            cache_path = os.path.join(self.config.retrain.cache_path, "fixed_split_plan.json")
            plan_started = time.perf_counter()
            logger.info(
                "Loading or computing fixed split plan with trace_batch=1 and "
                "validation_batches={}.",
                validation_batches,
            )
            self.fixed_split_plan = load_or_compute_fixed_split_plan(
                split_model,
                constraints,
                sample_input=trace_sample_input,
                device=next(split_model.parameters()).device,
                model_name=self.model_id,
                cache_path=cache_path,
                splitter=self.universal_splitter,
                validate_cached_plan=False,
                input_resize_mode=(
                    get_split_runtime_input_resize_mode(split_model) or "direct_resize"
                ),
                front_version=str(getattr(self, "front_version", "0") or "0"),
                model_version=str(getattr(self, "model_version", "0") or "0"),
                validation_batches=validation_batches,
            )
            logger.info(
                "Fixed split plan load/compute completed (elapsed={:.3f}s).",
                time.perf_counter() - plan_started,
            )
            self.universal_split_enabled = True
            self.split_trace_image_size = tuple(int(value) for value in trace_image_size)
            self.universal_splitter.prepare_inference_replay(sample_input)
            self._configure_inference_replay_threads(sample_input)
            logger.info("Warming up fixed split runtime.")
            self._warmup_fixed_split_runtime(sample_input)
            logger.info(
                "[EdgeCL] runtime ready: model={} split={} image_size={} elapsed={:.3f}s.",
                self.model_id,
                getattr(self.fixed_split_plan, "canonical_split_key", "auto"),
                self.split_trace_image_size,
                time.perf_counter() - plan_started,
            )
            log_diagnostic_debug(
                self,
                "[EdgeCL] fixed split plan diagnostics",
                lambda: {
                    "split_config_id": self.fixed_split_plan.split_config_id,
                    "cache_path": cache_path,
                    "plan": self.fixed_split_plan.describe(),
                },
                runtime=True,
            )
        except RuntimeError as exc:
            logger.error(
                "Fixed split runtime initialisation failed for model={}: {}.",
                self.model_id,
                safe_error_summary(exc),
            )
            raise _FixedSplitRuntimeError("Fixed split runtime initialisation failed.") from exc
        except Exception as exc:
            logger.error(
                "Failed to initialise fixed split plan: {}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "fixed split initialisation failure diagnostics",
                lambda error=exc: {
                    "cache_path": cache_path,
                    "error": repr(error),
                },
                runtime=True,
            )
            raise _FixedSplitRuntimeError("Failed to initialise fixed split runtime.") from exc

    def _warmup_fixed_split_runtime(self, sample_input) -> None:
        sl_cfg = getattr(self.config, "split_learning", None)
        warmup_iterations = int(getattr(sl_cfg, "warmup_iterations", 1) or 0)
        if warmup_iterations <= 0 or self.universal_splitter is None:
            return
        warmup_started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(warmup_iterations):
                self.universal_splitter.replay_inference(
                    sample_input,
                    return_split_output=True,
                )
        logger.info(
            "Fixed split warmup completed (iterations={}, elapsed={:.3f}s).",
            warmup_iterations,
            time.perf_counter() - warmup_started,
        )

    def _configure_inference_replay_threads(self, sample_input) -> None:
        del sample_input
        if self.universal_splitter is None:
            return
        replay_device = torch.device(getattr(self.universal_splitter, "device", "cpu"))
        if replay_device.type != "cpu":
            return
        sl_cfg = getattr(self.config, "split_learning", None)
        fixed_split_cfg = getattr(sl_cfg, "fixed_split", None) if sl_cfg else None
        threads = int(getattr(fixed_split_cfg, "inference_num_threads", 12))
        if threads <= 0:
            raise ValueError("fixed_split.inference_num_threads must be positive.")
        current_threads = int(torch.get_num_threads())
        torch.set_num_threads(threads)
        logger.info(
            "Configured CPU inference replay torch_num_threads={} (previous={}).",
            threads,
            current_threads,
        )

    def ensure_fixed_split_runtime(
        self,
        frame,
        image_size: tuple[int, int],
    ) -> None:
        init_lock = getattr(self, "_fixed_split_init_lock", None)
        if init_lock is None:
            init_lock = threading.Lock()
            self._fixed_split_init_lock = init_lock
        with init_lock:
            if getattr(self, "_fixed_split_init_attempted", False):
                return
            self._init_fixed_split_runtime(
                frame,
                tuple(int(value) for value in image_size),
            )
            if getattr(self, "collect_flag", False) and not self.split_learning_enabled:
                self.collect_flag = False
                self._log_split_collection_disabled()

    def _next_sample_id(self, task: Task) -> str:
        return f"{task.frame_index}-{int(task.start_time * 1000)}"

    def _apply_pending_sample_stats(self, delta: SampleStatsDelta, *, sign: int) -> None:
        factor = 1 if sign >= 0 else -1
        lock = getattr(self, "_pending_sample_stats_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._pending_sample_stats_lock = lock
        with lock:
            current = getattr(self, "_pending_sample_stats", SampleStatsDelta(total_samples=0))
            self._pending_sample_stats = SampleStatsDelta(
                total_samples=max(0, current.total_samples + factor * delta.total_samples),
                high_quality_count=max(
                    0,
                    current.high_quality_count + factor * delta.high_quality_count,
                ),
                low_quality_count=max(
                    0,
                    current.low_quality_count + factor * delta.low_quality_count,
                ),
                drift_window_sample_count=max(
                    0,
                    current.drift_window_sample_count + factor * delta.drift_window_sample_count,
                ),
            )

    def _on_sample_write_done(
        self,
        job: SampleWriteJob,
        record: object | None,
        error: BaseException | None,
    ) -> None:
        self._apply_pending_sample_stats(job.stats_delta, sign=-1)
        if error is not None:
            logger.error(
                "Dropped queued sample after async write failure: {}.",
                safe_error_summary(error),
            )
            log_diagnostic_debug(
                self,
                "async sample write failure diagnostics",
                lambda: {
                    "sample_id": job.store_kwargs.get("sample_id"),
                    "error": repr(error),
                },
            )
            return
        if record is not None:
            self._notify_sample_syncer(record)

    def _notify_sample_syncer(self, record: object) -> None:
        syncer = getattr(self, "sample_syncer", None)
        if syncer is None:
            return
        try:
            syncer.notify_sample(record)
        except Exception as exc:
            logger.warning(
                "Failed to queue high-quality sample for background sync: {}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "background sample sync queue diagnostics",
                lambda error=exc: {
                    "sample_id": getattr(record, "sample_id", None),
                    "error": repr(error),
                },
            )

    def _current_model_metadata(self) -> dict[str, object]:
        detection = getattr(self, "small_object_detection", None)
        model = getattr(detection, "model", None)
        metadata: dict[str, object] = {}
        num_classes = _coerce_positive_int(getattr(model, "num_classes", None))
        if num_classes is not None:
            metadata["num_classes"] = num_classes
            if str(getattr(self, "model_id", "")).lower().startswith("rfdetr_"):
                metadata["rfdetr_head_num_classes"] = num_classes
        label_schema = str(getattr(model, "label_schema", "") or "").strip()
        if label_schema:
            metadata["label_schema"] = label_schema
        class_names = getattr(self.config, "class_names", None)
        if class_names:
            metadata["class_names"] = [str(name) for name in list(class_names)]
        return metadata

    def _validate_cloud_update_state_compatible(
        self,
        update_payload: Mapping[str, object],
        state_dict: Mapping[str, object],
    ) -> None:
        model = getattr(self.small_object_detection, "model", None)
        if model is None or not hasattr(model, "state_dict"):
            return
        current_state = model.state_dict()
        mismatches: list[str] = []
        for name, value in state_dict.items():
            current_value = current_state.get(name)
            if (
                current_value is None
                or not torch.is_tensor(value)
                or not torch.is_tensor(current_value)
                or tuple(value.shape) == tuple(current_value.shape)
            ):
                continue
            mismatches.append(
                f"{name}: cloud={tuple(value.shape)} edge={tuple(current_value.shape)}"
            )

        if not mismatches:
            return

        raw_metadata = update_payload.get("weights_metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        cloud_head = _coerce_positive_int(
            metadata.get("rfdetr_head_num_classes") or metadata.get("num_classes")
        )
        edge_head = _coerce_positive_int(getattr(model, "num_classes", None))
        if (
            str(getattr(self, "model_id", "")).lower().startswith("rfdetr_")
            and cloud_head is not None
            and edge_head is not None
            and cloud_head != edge_head
        ):
            raise RuntimeError(
                "Cloud RF-DETR update is incompatible with the edge model: "
                f"cloud head has {cloud_head} logits but edge expects {edge_head}. "
                "Check that server.weights_path points to the same custom RF-DETR "
                "checkpoint as client.weights_path."
            )

        preview = "; ".join(mismatches[:4])
        if len(mismatches) > 4:
            preview += f"; ... (+{len(mismatches) - 4} more)"
        raise RuntimeError(
            f"Cloud model update contains tensors with incompatible shapes: {preview}"
        )

    def _sample_sync_context(self) -> dict[str, object]:
        split_plan = getattr(self, "fixed_split_plan", None)
        return {
            "model_id": str(getattr(self, "model_id", "") or ""),
            "model_version": str(getattr(self, "model_version", "") or ""),
            "edge_session_id": str(getattr(self, "edge_session_id", "") or ""),
            "front_version": str(getattr(self, "front_version", "0") or "0"),
            "split_config_id": str(getattr(split_plan, "split_config_id", "") or ""),
            "canonical_split_key": getattr(split_plan, "canonical_split_key", None),
            "edge_split_id": getattr(split_plan, "edge_split_id", None),
            "input_tensor_shape": list(getattr(split_plan, "input_tensor_shape", []) or []),
            "input_resize_mode": getattr(split_plan, "input_resize_mode", None),
            "runtime_contract": dict(getattr(split_plan, "runtime_contract", {}) or {}),
        }

    def _stats_for_training_trigger(self) -> dict[str, Any]:
        stats = dict(self.sample_store.stats())
        if not self.high_quality_sync_enabled:
            stats["high_quality_feature_bytes"] = 0
        lock = getattr(self, "_pending_sample_stats_lock", None)
        if lock is None:
            return stats
        with lock:
            pending = getattr(self, "_pending_sample_stats", SampleStatsDelta(total_samples=0))
        if pending.total_samples <= 0:
            return stats

        base_total = int(stats.get("total_samples", 0) or 0)
        total = base_total + int(pending.total_samples)
        low = int(stats.get("low_quality_count", 0) or 0) + int(pending.low_quality_count)
        high = int(stats.get("high_quality_count", 0) or 0) + int(pending.high_quality_count)

        stats.update(
            {
                "total_samples": total,
                "high_quality_count": high,
                "low_quality_count": low,
                "low_quality_rate": (low / float(total)) if total else 0.0,
                "drift_window_sample_count": int(stats.get("drift_window_sample_count", 0) or 0)
                + int(pending.drift_window_sample_count),
            }
        )
        return stats

    def _submit_sample_write(self, job: SampleWriteJob) -> bool:
        writer = getattr(self, "sample_writer", None)
        if writer is None:
            store_started = time.perf_counter()
            record = self.sample_store.store_sample(**job.store_kwargs)
            self._notify_sample_syncer(record)
            logger.info(
                "[EdgePerfAsyncStore] sample_id={} frame={} "
                "async_sample_store_ms={:.3f} async_writer_queue_size=0",
                job.store_kwargs.get("sample_id"),
                job.store_kwargs.get("frame_index"),
                (time.perf_counter() - store_started) * 1000.0,
            )
            return True
        self._apply_pending_sample_stats(job.stats_delta, sign=1)
        try:
            if bool(getattr(self, "strict_sample_collection", False)):
                writer.submit(job)
                return True
            accepted, dropped = writer.submit_nowait(job)
            if dropped is not None:
                self._apply_pending_sample_stats(dropped.stats_delta, sign=-1)
                logger.warning(
                    "Async sample writer queue full; dropped queued low-quality sample {} "
                    "to preserve higher-quality sample {}.",
                    dropped.store_kwargs.get("sample_id"),
                    job.store_kwargs.get("sample_id"),
                )
            if accepted:
                return True
            self._apply_pending_sample_stats(job.stats_delta, sign=-1)
            logger.warning(
                "Async sample writer queue full; dropped sample {} quality={} "
                "without blocking inference.",
                job.store_kwargs.get("sample_id"),
                job.store_kwargs.get("quality_bucket"),
            )
            return False
        except Exception as exc:
            self._apply_pending_sample_stats(job.stats_delta, sign=-1)
            logger.warning(
                "Async sample writer unavailable; dropped sample: {}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "async sample writer failure diagnostics",
                lambda error=exc: {
                    "sample_id": job.store_kwargs.get("sample_id"),
                    "error": repr(error),
                },
            )
            return False

    def _submit_sample_collection(self, job: SampleCollectionJob) -> bool:
        collector = getattr(self, "sample_collector", None)
        if collector is None:
            logger.warning(
                "Async sample collector unavailable; dropped sample {}.",
                job.sample_id,
            )
            return False
        try:
            if bool(getattr(self, "strict_sample_collection", False)):
                collector.submit_blocking(job)
            else:
                collector.submit_nowait(job)
            return True
        except Full:
            logger.warning(
                "Async sample collector queue full; dropped sample {} without blocking inference.",
                job.sample_id,
            )
            log_diagnostic_debug(
                self,
                "full sample collector diagnostics",
                lambda: {"sample_id": job.sample_id},
            )
            return False
        except Exception as exc:
            logger.warning(
                "Async sample collector unavailable; dropped sample: {}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "sample collector failure diagnostics",
                lambda error=exc: {
                    "sample_id": job.sample_id,
                    "error": repr(error),
                },
            )
            return False

    def _flush_sample_collector(self, *, timeout: float = 10.0) -> bool:
        collector = getattr(self, "sample_collector", None)
        if collector is None:
            return True
        try:
            return bool(collector.flush(timeout=timeout))
        except Exception as exc:
            logger.error("Failed to flush async sample collector: {}.", safe_error_summary(exc))
            return False

    def _flush_sample_writer(self, *, timeout: float = 10.0) -> bool:
        writer = getattr(self, "sample_writer", None)
        if writer is None:
            return True
        try:
            return bool(writer.flush(timeout=timeout))
        except Exception as exc:
            logger.error("Failed to flush async sample writer: {}.", safe_error_summary(exc))
            return False

    def _flush_sample_syncer(self, *, timeout: float = 10.0) -> bool:
        syncer = getattr(self, "sample_syncer", None)
        if syncer is None:
            return True
        try:
            return bool(syncer.flush(timeout=timeout, include_partial=True))
        except Exception as exc:
            logger.error("Failed to flush high-quality sample syncer: {}.", safe_error_summary(exc))
            return False

    def _feature_upload_shard_size(self) -> int | None:
        feature_upload_cfg = getattr(self.config, "feature_upload", None)
        if feature_upload_cfg is None:
            return None
        try:
            return max(1, int(getattr(feature_upload_cfg, "shard_max_samples")))
        except Exception:
            return None

    def submit_task(self, task: Task) -> Task:
        self.frame_cache.put(task, block=True)
        return task

    def _snapshot_result(self, task: Task) -> tuple[list, list, list]:
        detection_boxes, detection_class, detection_score = task.get_result()
        return (
            [list(box) for box in detection_boxes],
            list(detection_class),
            list(detection_score),
        )

    def _task_artifact_snapshot(
        self,
        task: Task,
        inference: InferenceArtifacts | None = None,
        *,
        result_source: str | None = None,
    ) -> dict[str, object]:
        boxes, labels, scores = self._snapshot_result(task)
        confidence = getattr(inference, "confidence", None) if inference is not None else None
        logit_entropy = getattr(inference, "logit_entropy", None) if inference is not None else None
        feature_spectral_entropy = (
            getattr(inference, "feature_spectral_entropy", None) if inference is not None else None
        )
        if logit_entropy is not None:
            entropy = logit_entropy
            entropy_source = "logit_entropy"
        elif feature_spectral_entropy is not None:
            entropy = feature_spectral_entropy
            entropy_source = "feature_spectral_entropy"
        else:
            entropy = None
            entropy_source = "missing"
        try:
            confidence_value = float(
                confidence if confidence is not None else max(scores, default=0.0)
            )
        except (TypeError, ValueError):
            confidence_value = 0.0
        try:
            entropy_value = float(entropy if entropy is not None else 0.0)
        except (TypeError, ValueError):
            entropy_value = 0.0
        return {
            "boxes": boxes,
            "labels": labels,
            "scores": [float(score) for score in scores],
            "confidence": confidence_value,
            "entropy": entropy_value,
            "entropy_source": entropy_source,
            "logit_entropy": logit_entropy,
            "feature_spectral_entropy": feature_spectral_entropy,
            "model_version": str(getattr(self, "model_version", "0") or "0"),
            "result_source": str(result_source or task.result_source or "pending"),
        }

    def _set_task_inference_artifacts(
        self,
        task: Task,
        inference: InferenceArtifacts | None = None,
        *,
        result_source: str | None = None,
    ) -> None:
        task.set_inference_artifacts(
            self._task_artifact_snapshot(task, inference, result_source=result_source)
        )

    def _remember_latest_result(self, task: Task) -> None:
        detection_boxes, detection_class, detection_score = self._snapshot_result(task)
        frame = getattr(task, "frame_edge", None)
        artifacts = dict(getattr(task, "inference_artifacts", {}) or {})
        with self.latest_result_lock:
            self.latest_result = {
                "frame_index": task.frame_index,
                "boxes": detection_boxes,
                "labels": detection_class,
                "scores": detection_score,
                "confidence": float(artifacts.get("confidence", 0.0) or 0.0),
                "entropy": float(artifacts.get("entropy", 0.0) or 0.0),
                "model_version": str(artifacts.get("model_version", self.model_version) or "0"),
                "result_source": str(artifacts.get("result_source", task.result_source) or ""),
                # VideoProcessor returns a new frame buffer for every read and
                # downstream rendering copies before drawing. Keeping this
                # buffer alive avoids a second 1080p copy on task completion.
                "frame": frame,
            }

    def _should_log_performance(self, frame_index: int | None) -> bool:
        index = int(frame_index or 0)
        interval = max(
            1,
            int(getattr(self, "performance_log_every_n_frames", 30)),
        )
        return index <= 1 or index % interval == 0

    def _reuse_latest_result(self, task: Task) -> None:
        with self.latest_result_lock:
            cached = {
                "frame_index": self.latest_result["frame_index"],
                "boxes": [list(box) for box in self.latest_result["boxes"]],
                "labels": list(self.latest_result["labels"]),
                "scores": list(self.latest_result["scores"]),
                "confidence": float(self.latest_result.get("confidence", 0.0) or 0.0),
                "entropy": float(self.latest_result.get("entropy", 0.0) or 0.0),
                "model_version": str(self.latest_result.get("model_version", "0") or "0"),
                "frame": self.latest_result.get("frame"),
            }
        boxes = cached["boxes"]
        labels = cached["labels"]
        scores = cached["scores"]
        if boxes and cached["frame"] is not None and getattr(task, "frame_edge", None) is not None:
            compensated_boxes, keep_indices = compensate_boxes_between_frames(
                boxes,
                cached["frame"],
                task.frame_edge,
            )
            kept = [
                (box, labels[index], scores[index])
                for box, index in zip(compensated_boxes, keep_indices)
                if index < len(labels) and index < len(scores)
            ]
            boxes = [item[0] for item in kept]
            labels = [item[1] for item in kept]
            scores = [item[2] for item in kept]
        elif boxes:
            boxes = []
            labels = []
            scores = []
        task.replace_result(
            boxes,
            labels,
            scores,
        )
        if cached["frame_index"] is not None:
            task.ref = cached["frame_index"]
            task.result_source = "cached"
        else:
            task.result_source = "empty"
        task.set_inference_artifacts(
            {
                "boxes": [list(box) for box in boxes],
                "labels": list(labels),
                "scores": [float(score) for score in scores],
                "confidence": cached["confidence"],
                "entropy": cached["entropy"],
                "model_version": cached["model_version"],
                "result_source": task.result_source,
            }
        )

    def _set_task_terminal_state(
        self,
        task: Task,
        state: TASK_STATE,
        *,
        result_source: str,
    ) -> None:
        task.state = state
        task.result_source = result_source
        self._finalize_task(task)

    def _finalize_task(self, task: Task) -> None:
        artifacts = dict(getattr(task, "inference_artifacts", {}) or {})
        if artifacts and str(artifacts.get("result_source", "pending")) != "pending":
            artifacts["result_source"] = str(task.result_source or artifacts["result_source"])
            artifacts["model_version"] = str(getattr(self, "model_version", "0") or "0")
            task.set_inference_artifacts(artifacts)
        else:
            self._set_task_inference_artifacts(task, result_source=task.result_source)
        if task.state == TASK_STATE.FINISHED and task.result_source == "inference":
            self._remember_latest_result(task)
        if hasattr(task, "set_timing") and hasattr(task, "created_perf"):
            task.set_timing(
                "task_complete_ms",
                (time.perf_counter() - task.created_perf) * 1000.0,
            )
        task.end_time = time.time()
        task.mark_done()

    def _log_split_collection_disabled(self) -> None:
        if self.split_learning_disable_reason == "disabled_in_config":
            logger.info(
                "Continual learning sample collection disabled because "
                "split_learning.enabled is false."
            )
            return
        logger.warning(
            "Continual learning sample collection disabled for edge model {}: {}.",
            self.model_id,
            safe_error_summary(self.split_learning_disable_reason or "split learning unavailable"),
        )

    def _reset_pending_training_cycle(self) -> None:
        self.pending_training_decision = None
        self.retrain_flag = False
        self.collect_flag = True
        self._drift_probe_active = False
        self._retrain_requested.clear()

    def _delete_uploaded_low_quality_samples(
        self,
        sample_ids: list[str],
    ) -> int:
        sample_store = getattr(self, "sample_store", None)
        if sample_store is None or not sample_ids:
            return 0
        deleted = int(sample_store.delete_samples(sample_ids, quality_bucket=LOW_QUALITY))
        if deleted:
            logger.info(
                "[EdgeCL] deleted uploaded low-quality samples from edge cache: "
                "deleted={} requested={}.",
                deleted,
                len(sample_ids),
            )
        return deleted

    def _delete_cloud_accepted_low_quality_samples(
        self,
        message: str,
        uploaded_sample_ids: list[str],
        *,
        job_id: str,
    ) -> int:
        accepted_sample_ids = _accepted_uploaded_sample_ids(message, uploaded_sample_ids)
        if uploaded_sample_ids and not accepted_sample_ids:
            logger.warning(
                "[EdgeCL] cloud did not return accepted low-quality sample ids; "
                "keeping uploaded samples for retry."
            )
        deleted = self._delete_uploaded_low_quality_samples(accepted_sample_ids)
        if deleted:
            self._record_experiment_metric(
                "uploaded_low_quality_samples_deleted",
                timestamp_ms=int(time.time() * 1000),
                job_id=job_id,
                deleted_sample_count=deleted,
            )
        return deleted

    def apply_model_update(
        self,
        model_b64: str,
        *,
        submitted_model_version: str | None = None,
        result_model_version: str | None = None,
        job_id: str = "",
        message: str = "",
        report: bool = True,
        clear_samples: bool = True,
        reset_drift: bool = True,
        log_prefix: str = "[EdgeCL]",
    ) -> str:
        update = self._prepare_model_update(
            model_b64,
            submitted_model_version=submitted_model_version,
            result_model_version=result_model_version,
            job_id=job_id,
            message=message,
            report=report,
            clear_samples=clear_samples,
            reset_drift=reset_drift,
            log_prefix=log_prefix,
        )
        with self.small_object_detection.model_lock:
            self._apply_prepared_model_update_locked(update)
        self._finish_applied_model_update(update)
        return str(update.applied_version or self.model_version)

    def _prepare_model_update(
        self,
        model_b64: str,
        *,
        submitted_model_version: str | None = None,
        result_model_version: str | None = None,
        job_id: str = "",
        message: str = "",
        report: bool = True,
        clear_samples: bool = True,
        reset_drift: bool = True,
        log_prefix: str = "[EdgeCL]",
    ) -> PendingModelUpdate:
        if not model_b64:
            raise RuntimeError("model update payload is empty")
        expected_version = None if submitted_model_version is None else str(submitted_model_version)
        current_version = str(self.model_version)
        if expected_version is not None and current_version != expected_version:
            raise RuntimeError(
                "stale model update: "
                f"submitted_version={expected_version} current_version={current_version}"
            )

        buf = io.BytesIO(base64.b64decode(model_b64))
        next_version = str(result_model_version or "")
        if not next_version:
            try:
                next_version = str(int(current_version) + 1)
            except (TypeError, ValueError):
                next_version = "1"
        logger.info(
            "{} model update received: version={} size={:.1f}MB.",
            log_prefix,
            next_version,
            len(buf.getbuffer()) / (1024.0 * 1024.0),
        )
        update_payload = dict(
            require_state_dict_delta_payload(
                torch.load(buf, map_location="cpu", weights_only=False)
            )
        )
        state_dict = dict(update_payload["state_dict"])
        weight_keys = [
            name
            for name in state_dict
            if name not in {"plank_threshold_low", "plank_threshold_high"}
        ]
        if not weight_keys:
            logger.warning(
                "{} cloud model update contains only threshold metadata; "
                "model weights will not change.",
                log_prefix,
            )
        return PendingModelUpdate(
            update_payload=update_payload,
            state_dict=state_dict,
            submitted_model_version=expected_version,
            next_model_version=next_version,
            job_id=job_id,
            message=message,
            report=report,
            clear_samples=clear_samples,
            reset_drift=reset_drift,
            log_prefix=log_prefix,
        )

    def _apply_prepared_model_update_locked(self, update: PendingModelUpdate) -> str:
        current_version = str(self.model_version)
        expected_version = update.submitted_model_version
        if expected_version is not None and current_version != expected_version:
            raise RuntimeError(
                "stale model update: "
                f"submitted_version={expected_version} current_version={current_version}"
            )
        apply_started = time.perf_counter()
        self._validate_cloud_update_state_compatible(
            update.update_payload,
            update.state_dict,
        )
        load_result = self.small_object_detection.model.load_state_dict(
            update.state_dict,
            strict=False,
        )
        self.small_object_detection.model.eval()
        self.small_object_detection.get_split_runtime_model().eval()
        self.small_object_detection.refresh_thresholds_from_model()
        if self.fixed_split_plan is not None:
            logger.info(
                "{} reusing fixed split plan after model update: split={}.",
                update.log_prefix,
                getattr(self.fixed_split_plan, "canonical_split_key", "auto"),
            )
            log_diagnostic_debug(
                self,
                f"{update.log_prefix} reused split plan diagnostics",
                lambda: {"split_config_id": self.fixed_split_plan.split_config_id},
            )
        self.model_version = update.next_model_version
        update.applied_version = self.model_version
        update.apply_elapsed_ms = (time.perf_counter() - apply_started) * 1000.0
        weight_keys = [
            name
            for name in update.state_dict
            if name not in {"plank_threshold_low", "plank_threshold_high"}
        ]
        logger.info(
            "{} model update applied between frames: version={} state_keys={} "
            "weight_keys={} missing_keys={} unexpected_keys={} elapsed={:.3f}s.",
            update.log_prefix,
            self.model_version,
            len(update.state_dict),
            len(weight_keys),
            len(list(getattr(load_result, "missing_keys", ()) or ())),
            len(list(getattr(load_result, "unexpected_keys", ()) or ())),
            time.perf_counter() - apply_started,
        )
        logger.success(
            "{} model update successful: version={} -> {}.",
            update.log_prefix,
            current_version,
            self.model_version,
        )
        return self.model_version

    def _finish_applied_model_update(self, update: PendingModelUpdate) -> None:
        if update.error is not None:
            raise RuntimeError("pending model update failed") from update.error
        if update.applied_version is None:
            raise RuntimeError("pending model update has not been applied")
        if update.clear_samples:
            self.sample_store.clear()
        if update.reset_drift:
            self.window_drift_detector.reset()
            self.quality_classifier.reset()
            self.teacher_sample_selector.on_model_update()
        if update.report:
            reported, report_message = report_edge_model_version(
                self.config.server_ip,
                edge_id=self.edge_id,
                model_id=self.model_id,
                model_version=self.model_version,
            )
            if not reported:
                logger.warning(
                    "{} model version report was not acknowledged: {}.",
                    update.log_prefix,
                    safe_error_summary(report_message),
                )
        log_diagnostic_debug(
            self,
            f"{update.log_prefix} model update diagnostics",
            lambda: {"job_id": update.job_id, "message": update.message},
        )
        self._record_experiment_metric(
            "model_update_applied",
            job_id=update.job_id,
            model_version=str(update.applied_version),
            model_apply_ms=update.apply_elapsed_ms,
            message=update.message,
        )

    def _queue_pending_model_update(self, update: PendingModelUpdate) -> None:
        with self._pending_model_update_lock:
            if self.pending_model_update is not None:
                raise RuntimeError("another model update is already pending")
            self.pending_model_update = update
        logger.info(
            "{} model update prepared and queued for a non-blocking between-frame apply: "
            "version={}.",
            update.log_prefix,
            update.next_model_version,
        )

    def _try_apply_pending_model_update(self) -> bool:
        with self._pending_model_update_lock:
            update = self.pending_model_update
        if update is None:
            return False
        model_lock = self.small_object_detection.model_lock
        if not model_lock.acquire(blocking=False):
            logger.debug(
                "{} model lock busy; pending version {} deferred to a later frame.",
                update.log_prefix,
                update.next_model_version,
            )
            return False
        with self._pending_model_update_lock:
            if self.pending_model_update is not update:
                model_lock.release()
                return False
        try:
            self._apply_prepared_model_update_locked(update)
        except BaseException as exc:  # noqa: BLE001 - report to retrain worker.
            update.error = exc
            logger.error(
                "{} failed to apply pending model update: {}.",
                update.log_prefix,
                safe_error_summary(exc),
            )
        finally:
            model_lock.release()
            with self._pending_model_update_lock:
                if self.pending_model_update is update:
                    self.pending_model_update = None
            update.applied_event.set()
        return update.error is None

    def _wait_for_pending_model_update(self, update: PendingModelUpdate) -> bool:
        while not self._stop_event.is_set():
            if update.applied_event.wait(timeout=_QUEUE_POLL_TIMEOUT_SECONDS):
                break
        if not update.applied_event.is_set():
            return False
        if update.error is not None:
            raise RuntimeError("pending model update failed") from update.error
        self._finish_applied_model_update(update)
        return True

    def _resolve_active_splitter(self, current_frame, frame_image_size: tuple[int, int]):
        if self.split_learning_enabled and not getattr(self, "_fixed_split_init_attempted", False):
            self.ensure_fixed_split_runtime(current_frame, frame_image_size)

        active_splitter = self.universal_splitter if self.universal_split_enabled else None
        if self.split_learning_enabled and active_splitter is None:
            raise _FixedSplitRuntimeError(
                "Split learning is enabled but its runtime is unavailable."
            )
        effective_image_size = tuple(int(value) for value in frame_image_size)
        if (
            active_splitter is None
            or self.split_trace_image_size is None
            or effective_image_size == self.split_trace_image_size
        ):
            return active_splitter

        raise _FixedSplitRuntimeError(
            "Split runtime input size changed "
            f"from {self.split_trace_image_size} to {effective_image_size}."
        )

    def _update_resource_probe_cache(
        self,
        *,
        cloud_state: CloudResourceState | None = None,
        bandwidth_mbps: float | None = None,
    ) -> None:
        completed_at = time.time()
        lock = getattr(self, "_resource_probe_lock", None)
        if lock is None:
            if cloud_state is not None:
                self._cloud_state = cloud_state
            if bandwidth_mbps is not None:
                self._bandwidth_mbps = float(bandwidth_mbps)
            self._resource_probe_completed_at = completed_at
            return
        with lock:
            if cloud_state is not None:
                self._cloud_state = cloud_state
            if bandwidth_mbps is not None:
                self._bandwidth_mbps = float(bandwidth_mbps)
            self._resource_probe_completed_at = completed_at

    def _resource_probe_snapshot(self) -> tuple[CloudResourceState, float]:
        lock = getattr(self, "_resource_probe_lock", None)
        if lock is None:
            cloud_state = getattr(self, "_cloud_state", None)
            bandwidth_mbps = getattr(self, "_bandwidth_mbps", 0.0)
        else:
            with lock:
                cloud_state = getattr(self, "_cloud_state", None)
                bandwidth_mbps = getattr(self, "_bandwidth_mbps", 0.0)

        max_age_sec = max(
            30.0,
            float(getattr(self, "resource_probe_interval_sec", 5.0)) * 2.0,
        )
        if cloud_state is None or cloud_state.is_stale(max_age_sec):
            cloud_state = self._conservative_cloud_state()
        return cloud_state, max(0.0, float(bandwidth_mbps or 0.0))

    def _resource_probe_cached_age_ms(self) -> float:
        lock = getattr(self, "_resource_probe_lock", None)
        if lock is None:
            completed_at = float(getattr(self, "_resource_probe_completed_at", 0.0))
        else:
            with lock:
                completed_at = float(getattr(self, "_resource_probe_completed_at", 0.0))
        if completed_at <= 0.0:
            return -1.0
        return max(0.0, (time.time() - completed_at) * 1000.0)

    def _resource_probe_ready_for_decision(self) -> bool:
        lock = getattr(self, "_resource_probe_lock", None)
        if lock is None:
            cloud_state = getattr(self, "_cloud_state", None)
            completed_at = float(getattr(self, "_resource_probe_completed_at", 0.0))
            required_after = float(getattr(self, "_resource_probe_required_after", 0.0))
        else:
            with lock:
                cloud_state = getattr(self, "_cloud_state", None)
                completed_at = float(getattr(self, "_resource_probe_completed_at", 0.0))
                required_after = float(getattr(self, "_resource_probe_required_after", 0.0))

        if cloud_state is None or completed_at < required_after:
            return False
        max_age_sec = max(
            30.0,
            float(getattr(self, "resource_probe_interval_sec", 5.0)) * 2.0,
        )
        return not cloud_state.is_stale(max_age_sec)

    def _request_resource_probe(self) -> bool:
        requested = getattr(self, "_resource_probe_requested", None)
        if requested is None:
            return False
        now = time.time()
        lock = getattr(self, "_resource_probe_lock", None)
        if lock is None:
            if getattr(self, "_resource_probe_inflight", False):
                return True
            if now < float(getattr(self, "_resource_probe_next_allowed_at", 0.0)):
                return False
            self._resource_probe_inflight = True
            requested.set()
            return True
        with lock:
            if getattr(self, "_resource_probe_inflight", False):
                return True
            if now < float(getattr(self, "_resource_probe_next_allowed_at", 0.0)):
                return False
            self._resource_probe_inflight = True
            requested.set()
            return True

    def _finish_resource_probe(self, success: bool) -> None:
        base_interval = float(getattr(self, "resource_probe_interval_sec", 5.0))
        next_allowed_at = 0.0
        lock = getattr(self, "_resource_probe_lock", None)
        if lock is None:
            if success:
                self._resource_probe_failure_count = 0
            else:
                self._resource_probe_failure_count += 1
                backoff_sec = min(
                    max(30.0, base_interval),
                    base_interval * (2**self._resource_probe_failure_count),
                )
                next_allowed_at = time.time() + backoff_sec
            self._resource_probe_next_allowed_at = next_allowed_at
            self._resource_probe_inflight = False
            return
        with lock:
            if success:
                self._resource_probe_failure_count = 0
            else:
                self._resource_probe_failure_count += 1
                backoff_sec = min(
                    max(30.0, base_interval),
                    base_interval * (2**self._resource_probe_failure_count),
                )
                next_allowed_at = time.time() + backoff_sec
            self._resource_probe_next_allowed_at = next_allowed_at
            self._resource_probe_inflight = False

    def _refresh_resource_probe_cache(self) -> bool:
        timeout_sec = float(getattr(self, "resource_probe_timeout_sec", 3.0))
        try:
            cloud_state = query_cloud_resource(
                self.config.server_ip,
                edge_id=self.edge_id,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            logger.warning(
                "Resource probe cloud-state refresh failed; using conservative cache: {}.",
                safe_error_summary(exc),
            )
            self._update_resource_probe_cache(
                cloud_state=self._conservative_cloud_state(),
                bandwidth_mbps=0.0,
            )
            return False

        bandwidth_mbps = estimate_bandwidth(
            self.config.server_ip,
            probe_size_bytes=int(getattr(self, "bandwidth_probe_size_bytes", 64 * 1024)),
            timeout_sec=timeout_sec,
        )
        self._update_resource_probe_cache(
            cloud_state=cloud_state,
            bandwidth_mbps=bandwidth_mbps,
        )
        return True

    def resource_probe_worker(self) -> None:
        while not self._stop_event.is_set():
            requested = getattr(self, "_resource_probe_requested", None)
            if requested is None:
                return
            requested.wait()
            if self._stop_event.is_set():
                return
            requested.clear()
            success = False
            try:
                success = self._refresh_resource_probe_cache()
            finally:
                self._finish_resource_probe(success)

    def _make_training_decision(
        self,
        *,
        stats: PendingTrainingStats,
        drift_detected: bool,
    ) -> TrainingDecision:
        if self.resource_trigger_enabled and self.resource_trigger is not None:
            cloud_state, bandwidth_mbps = self._resource_probe_snapshot()
            return self.resource_trigger.decide(
                drift_detected=bool(drift_detected),
                cloud_state=cloud_state,
                bandwidth_mbps=bandwidth_mbps,
                sample_stats=stats,
            )

        return TrainingDecision(
            train_now=False,
            send_low_conf_features=False,
            urgency=0.0,
            compute_pressure=0.0,
            bandwidth_pressure=0.0,
            bundle_cap_bytes=0,
            reason="Resource-aware continual-learning trigger is disabled.",
        )

    def _bootstrap_training_ready(self, stats: PendingTrainingStats) -> bool:
        resource_cfg = self.config.resource_aware_trigger
        if not self.resource_trigger_enabled or not resource_cfg.bootstrap_without_drift:
            return False
        if str(self.model_version) != "0":
            return False
        if self.resource_trigger is None:
            raise RuntimeError("resource-aware trigger is enabled but not initialized")
        min_samples = int(self.resource_trigger.min_training_samples)
        return int(stats.low_quality_count) >= max(1, min_samples)

    def _should_store_quality_sample(
        self,
        *,
        quality_bucket: str,
        raw_sample_selected: bool,
    ) -> bool:
        if str(quality_bucket) == LOW_QUALITY:
            return bool(raw_sample_selected)
        return self.high_quality_sync_enabled

    @staticmethod
    def _apply_window_quality_rate(
        stats: PendingTrainingStats,
        drift_state: DriftWindowState,
    ) -> float:
        retained_sample_rate = float(stats.low_quality_rate)
        stats.low_quality_rate = float(drift_state.low_quality_rate)
        return retained_sample_rate

    def collect_data(self, task: Task, frame, inference: InferenceArtifacts) -> bool:
        split_plan = self.fixed_split_plan
        if split_plan is None:
            return False
        runtime_contract = dict(getattr(split_plan, "runtime_contract", {}) or {})
        split_key = str(
            getattr(split_plan, "canonical_split_key", "")
            or runtime_contract.get("logical_split_id")
            or getattr(split_plan, "split_config_id", "")
            or ""
        )
        feature_abi_id = str(runtime_contract.get("feature_abi_id") or "")
        if not feature_abi_id:
            raise RuntimeError("Fixed split plan is missing the current feature ABI identifier.")
        job = SampleCollectionJob(
            sample_id=self._next_sample_id(task),
            frame_index=task.frame_index,
            frame=frame,
            inference=inference,
            split_config_id=str(split_plan.split_config_id),
            model_id=str(self.model_id),
            model_version=str(self.model_version),
            front_version=str(getattr(self, "front_version", "0") or "0"),
            split_key=split_key,
            feature_abi_id=feature_abi_id,
            runtime_contract=runtime_contract,
        )
        return self._submit_sample_collection(job)

    def _collect_data_from_job(self, job: SampleCollectionJob) -> None:
        inference = job.inference
        frame = job.frame
        confidence = float(inference.confidence)
        observables_ms = float(
            dict(getattr(inference, "timing_ms", {}) or {}).get("observables_ms", 0.0)
        )
        async_quality_ms = 0.0
        async_drift_ms = 0.0
        writer_queued = False
        try:
            quality_classifier = getattr(self, "quality_classifier", None)
            if quality_classifier is None:
                quality_classifier = EntropyQualityClassifier.from_config(
                    getattr(getattr(self, "config", None), "sample_quality", None)
                )
                self.quality_classifier = quality_classifier
            window_detector = getattr(self, "window_drift_detector", WindowDriftDetector())
            quality_started = time.perf_counter()
            quality = quality_classifier.classify(
                inference,
                inference.intermediate,
                model_name=job.model_id,
                split_key=job.split_key,
                feature_abi_id=job.feature_abi_id,
            )
            async_quality_ms = (time.perf_counter() - quality_started) * 1000.0
            drift_started = time.perf_counter()
            drift_state = window_detector.update(
                quality,
                feature_stats={
                    "feature_spectral_entropy": getattr(
                        inference, "feature_spectral_entropy", None
                    ),
                    "logit_entropy": getattr(inference, "logit_entropy", None),
                    "logit_margin": getattr(inference, "logit_margin", None),
                    "logit_energy": getattr(inference, "logit_energy", None),
                },
            )
            async_drift_ms = (time.perf_counter() - drift_started) * 1000.0
            teacher_selector = self.teacher_sample_selector
            teacher_selection = teacher_selector.select(quality)
            save_raw = quality.quality_bucket == LOW_QUALITY and teacher_selection.selected
            should_store = self._should_store_quality_sample(
                quality_bucket=quality.quality_bucket,
                raw_sample_selected=save_raw,
            )
            retrain_cfg = self.config.retrain
            persist_debug_stats = bool(quality_classifier.persist_debug_stats)
            if should_store:
                store_kwargs = {
                    "sample_id": job.sample_id,
                    "frame_index": job.frame_index,
                    "confidence": confidence,
                    "split_config_id": job.split_config_id,
                    "model_id": job.model_id,
                    "model_version": job.model_version,
                    "front_version": job.front_version,
                    "quality_bucket": quality.quality_bucket,
                    "quality_metadata": quality.quality_metadata(
                        persist_debug_stats=persist_debug_stats
                    ),
                    "window_id": quality.window_id,
                    "in_drift_window": quality.in_drift_window,
                    "inference_result": inference.to_inference_result(),
                    "intermediate": inference.intermediate,
                    "raw_frame": frame if save_raw else None,
                    "raw_jpeg_quality": int(retrain_cfg.raw_jpeg_quality),
                    "input_image_size": list(frame.shape[:2]),
                    "input_tensor_shape": inference.input_tensor_shape,
                    "input_resize_mode": inference.input_resize_mode,
                    "runtime_contract": job.runtime_contract,
                }
                writer_queued = self._submit_sample_write(
                    SampleWriteJob(
                        store_kwargs=store_kwargs,
                        stats_delta=SampleStatsDelta.from_values(
                            quality_bucket=quality.quality_bucket,
                            in_drift_window=quality.in_drift_window,
                        ),
                    )
                )

            self._record_experiment_metric(
                "sample_quality_summary",
                frame_id=int(job.frame_index),
                quality_bucket=str(quality.quality_bucket),
                quality_reason=str(quality.reason),
                window_id=str(quality.window_id or ""),
                in_drift_window=bool(quality.in_drift_window),
                confidence=confidence,
                output_entropy=quality.output_entropy,
                output_entropy_threshold=quality.output_entropy_threshold,
                feature_entropy_deviation=quality.feature_entropy_deviation,
                raw_sample_selected=bool(save_raw),
                teacher_sample_reason=str(teacher_selection.reason),
                teacher_sample_selection_rate=float(teacher_selector.effective_selection_rate),
            )
            trigger_stats = PendingTrainingStats.from_mapping(self._stats_for_training_trigger())
            self._record_experiment_metric(
                "drift_window_summary",
                frame_id=int(job.frame_index),
                window_id=str(quality.window_id or ""),
                drift_detected=bool(drift_state.drift_detected),
                drift_started=bool(drift_state.drift_started),
                drift_ended=bool(drift_state.drift_ended),
                evaluation_performed=bool(drift_state.evaluation_performed),
                drift_score=float(drift_state.drift_score),
                window_low_quality_rate=float(drift_state.low_quality_rate),
                window_severe_anomaly_rate=float(drift_state.severe_anomaly_rate),
                low_quality_count=int(trigger_stats.low_quality_count),
                total_samples=int(trigger_stats.total_samples),
            )

            if self.retrain_flag:
                return

            bootstrap_ready = self._bootstrap_training_ready(trigger_stats)
            effective_drift = bool(drift_state.drift_detected or bootstrap_ready)
            if not effective_drift:
                self._drift_probe_active = False
                return

            if self.resource_trigger_enabled and self.resource_trigger is not None:
                if not getattr(self, "_drift_probe_active", False):
                    self._drift_probe_active = True
                    self._resource_probe_required_after = time.time()
                if not self._resource_probe_ready_for_decision():
                    self._request_resource_probe()
                    return

            stats = PendingTrainingStats.from_mapping(self._stats_for_training_trigger())
            retained_sample_low_quality_rate = self._apply_window_quality_rate(
                stats,
                drift_state,
            )
            bootstrap_ready = self._bootstrap_training_ready(stats)
            effective_drift = bool(drift_state.drift_detected or bootstrap_ready)
            stats.drift_detected = effective_drift
            decision = self._make_training_decision(
                stats=stats,
                drift_detected=effective_drift,
            )
            self._record_experiment_metric(
                "resource_trigger_decision",
                frame_id=int(job.frame_index),
                window_id=str(quality.window_id or ""),
                train_now=bool(decision.train_now),
                trigger_decision=bool(decision.train_now),
                trigger_reason=str(decision.reason),
                send_low_conf_features=bool(decision.send_low_conf_features),
                bandwidth_mbps=float(decision.bandwidth_mbps),
                cloud_compute_pressure=float(decision.compute_pressure),
                bandwidth_pressure=float(decision.bandwidth_pressure),
                bundle_cap_bytes=int(decision.bundle_cap_bytes or 0),
                window_low_quality_rate=float(stats.low_quality_rate),
                retained_sample_low_quality_rate=float(retained_sample_low_quality_rate),
                cooldown_remaining_decisions=int(decision.cooldown_remaining_decisions),
                cooldown_remaining_sec=float(decision.cooldown_remaining_sec),
                bootstrap_trigger_eligible=bool(bootstrap_ready),
            )
            if decision.train_now and stats.low_quality_count > 0:
                self.pending_training_decision = decision
                self.retrain_flag = True
                self.collect_flag = False
                self._retrain_requested.set()
                logger.info(
                    "Continual learning triggered (samples={}, low_quality={}, "
                    "send_low_conf_features={}, reason={})",
                    stats.total_samples,
                    stats.low_quality_count,
                    decision.send_low_conf_features,
                    decision.reason,
                )
        finally:
            writer = getattr(self, "sample_writer", None)
            writer_queue_size = writer.qsize() if writer is not None else 0
            if self._should_log_performance(job.frame_index):
                logger.info(
                    "[EdgePerfAsyncCollect] sample_id={} frame={} observables_ms={:.3f} "
                    "async_quality_ms={:.3f} async_drift_ms={:.3f} writer_queued={} "
                    "async_writer_queue_size={} resource_probe_cached_age_ms={:.3f}",
                    job.sample_id,
                    job.frame_index,
                    observables_ms,
                    async_quality_ms,
                    async_drift_ms,
                    writer_queued,
                    writer_queue_size,
                    self._resource_probe_cached_age_ms(),
                )

    def retrain_worker(self):
        while not self._stop_event.is_set():
            self._retrain_requested.wait()
            if self._stop_event.is_set():
                return
            self._retrain_requested.clear()

            if not self.retrain_flag:
                continue

            decision = self.pending_training_decision
            if self.fixed_split_plan is None or decision is None:
                if decision is not None and self.resource_trigger is not None:
                    self.resource_trigger.cancel_trigger(decision)
                self._reset_pending_training_cycle()
                continue

            submitted_model_version = str(self.model_version)
            success = False
            model_b64 = ""
            terminal_message = ""
            last_status = ""
            uploaded_low_quality_sample_ids: list[str] = []
            training_channel = grpc.insecure_channel(
                self.config.server_ip,
                options=grpc_message_options(),
            )
            try:
                if not self._flush_sample_collector(timeout=30.0):
                    logger.warning(
                        "Timed out while flushing pending sample collection before "
                        "continual learning upload."
                    )
                if not self._flush_sample_writer(timeout=30.0):
                    logger.warning(
                        "Timed out while flushing pending sample writes before "
                        "continual learning upload."
                    )
                if not self._flush_sample_syncer(timeout=30.0):
                    logger.warning(
                        "Timed out while syncing high-quality samples before "
                        "continual learning upload."
                    )
                upload_metrics: dict[str, Any] = {}
                (
                    accepted,
                    job_id,
                    msg,
                    uploaded_low_quality_sample_ids,
                ) = submit_continual_learning_job(
                    self.config.server_ip,
                    edge_id=self.edge_id,
                    sample_store=self.sample_store,
                    split_plan=self.fixed_split_plan,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    model_metadata=self._current_model_metadata(),
                    edge_session_id=str(getattr(self, "edge_session_id", "") or ""),
                    send_low_conf_features=decision.send_low_conf_features,
                    bundle_cap_bytes=decision.bundle_cap_bytes,
                    max_samples=self.resource_trigger.max_training_samples,
                    trigger_shard_size=self._feature_upload_shard_size(),
                    bandwidth_mbps=decision.bandwidth_mbps,
                    channel=training_channel,
                    log_internal_ids=self.log_internal_ids,
                    metrics_callback=upload_metrics.update,
                )
                if not accepted or not job_id:
                    logger.error(
                        "[EdgeCL] cloud continual learning submission failed: {}.",
                        safe_error_summary(msg),
                    )
                    if self.resource_trigger is not None:
                        self.resource_trigger.cancel_trigger(decision)
                    self._reset_pending_training_cycle()
                    continue

                if self.resource_trigger is None:
                    raise RuntimeError("resource-aware trigger is unavailable")
                self.resource_trigger.confirm_trigger(decision)

                self._record_experiment_metric(
                    "bundle_built",
                    job_id=job_id,
                    model_version=str(self.model_version),
                    timestamp_ms=upload_metrics.get("upload_started_at_ms")
                    or int(time.time() * 1000),
                )
                self._record_experiment_metric(
                    "bundle_upload_started",
                    job_id=job_id,
                    model_version=str(self.model_version),
                    timestamp_ms=upload_metrics.get("upload_started_at_ms")
                    or int(time.time() * 1000),
                    send_low_conf_features=bool(decision.send_low_conf_features),
                    bundle_cap_bytes=int(decision.bundle_cap_bytes or 0),
                )
                self._record_experiment_metric(
                    "bundle_upload_done",
                    job_id=job_id,
                    model_version=str(self.model_version),
                    timestamp_ms=upload_metrics.get("upload_done_at_ms") or int(time.time() * 1000),
                    **{
                        key: value
                        for key, value in upload_metrics.items()
                        if key not in {"upload_started_at_ms", "upload_done_at_ms"}
                    },
                )
                self._record_experiment_metric(
                    "training_job_submitted",
                    job_id=job_id,
                    model_version=str(self.model_version),
                )
                logger.info(
                    "[EdgeCL] training accepted: edge={} model_version={}.",
                    self.edge_id,
                    self.model_version,
                )
                log_diagnostic_debug(
                    self,
                    "[EdgeCL] accepted training job diagnostics",
                    lambda: {"job_id": job_id},
                )

                terminal_message = msg
                not_found_since = None
                not_found_count = 0
                while not self._stop_event.is_set():
                    reply = get_training_job_status(
                        self.config.server_ip,
                        edge_id=self.edge_id,
                        job_id=job_id,
                        channel=training_channel,
                    )
                    if reply is None:
                        if self._stop_event.wait(self.training_poll_interval_sec):
                            break
                        continue

                    if not bool(reply.found):
                        now = time.monotonic()
                        if not_found_since is None:
                            not_found_since = now
                        not_found_count += 1
                        elapsed = now - not_found_since
                        if (
                            self.training_not_found_grace_sec > 0.0
                            and elapsed <= self.training_not_found_grace_sec
                        ):
                            logger.warning(
                                "[EdgeCL] training temporarily not visible on cloud "
                                "poll={} elapsed={:.1f}/{:.1f}s; retrying.",
                                not_found_count,
                                elapsed,
                                self.training_not_found_grace_sec,
                            )
                            if self._stop_event.wait(self.training_poll_interval_sec):
                                break
                            continue

                        terminal_message = f"Training job not found on cloud after {elapsed:.1f}s."
                        logger.error(
                            "[EdgeCL] cloud training unavailable: elapsed={:.1f}s.",
                            elapsed,
                        )
                        log_diagnostic_debug(
                            self,
                            "[EdgeCL] missing training job diagnostics",
                            lambda: {"job_id": job_id},
                        )
                        break

                    not_found_since = None
                    not_found_count = 0
                    status = str(reply.status or "")
                    if status != last_status:
                        queue_position = int(getattr(reply, "queue_position", -1))
                        logger.info(
                            "[EdgeCL] training status={} queue_position={}.",
                            status,
                            queue_position,
                        )
                        log_diagnostic_debug(
                            self,
                            "[EdgeCL] training status diagnostics",
                            lambda: {"job_id": job_id, "status": status},
                        )
                        last_status = status
                        if status == "RUNNING":
                            self._record_experiment_metric(
                                "training_job_started",
                                timestamp_ms=int(
                                    getattr(reply, "started_at_ms", 0) or time.time() * 1000
                                ),
                                job_id=job_id,
                                status=status,
                            )

                    if status in {"QUEUED", "RUNNING"}:
                        if self._stop_event.wait(self.training_poll_interval_sec):
                            break
                        continue

                    if status == "SUCCEEDED":
                        self._record_experiment_metric(
                            "training_job_succeeded",
                            timestamp_ms=int(
                                getattr(reply, "finished_at_ms", 0) or time.time() * 1000
                            ),
                            job_id=job_id,
                            status=status,
                        )
                        self._delete_cloud_accepted_low_quality_samples(
                            str(reply.message or ""),
                            uploaded_low_quality_sample_ids,
                            job_id=job_id,
                        )
                        download_started = time.perf_counter()
                        success, model_b64, terminal_message = download_trained_model(
                            self.config.server_ip,
                            edge_id=self.edge_id,
                            job_id=job_id,
                            channel=training_channel,
                        )
                        model_update_download_ms = (time.perf_counter() - download_started) * 1000.0
                        if not success:
                            logger.error(
                                "[EdgeCL] model update download failed: reason={}.",
                                safe_error_summary(terminal_message),
                            )
                        else:
                            try:
                                model_update_download_bytes = len(
                                    base64.b64decode(model_b64, validate=True)
                                )
                            except (TypeError, ValueError):
                                model_update_download_bytes = len(str(model_b64).encode("utf-8"))
                            self._record_experiment_metric(
                                "model_update_downloaded",
                                job_id=job_id,
                                model_version=str(self.model_version),
                                model_update_download_bytes=model_update_download_bytes,
                                model_update_download_ms=model_update_download_ms,
                            )
                        break

                    if status == "WAITING_FOR_SAMPLES":
                        terminal_message = str(
                            reply.message or "Training job skipped while waiting for samples."
                        )
                        logger.info(
                            "[EdgeCL] cloud training waiting for samples: {}.",
                            safe_error_summary(terminal_message),
                        )
                        self._record_experiment_metric(
                            "training_job_waiting_for_samples",
                            timestamp_ms=int(
                                getattr(reply, "finished_at_ms", 0) or time.time() * 1000
                            ),
                            job_id=job_id,
                            status=status,
                        )
                        self._delete_cloud_accepted_low_quality_samples(
                            terminal_message,
                            uploaded_low_quality_sample_ids,
                            job_id=job_id,
                        )
                        break

                    terminal_message = str(
                        reply.message or f"Training job ended with status {status}"
                    )
                    self._delete_cloud_accepted_low_quality_samples(
                        terminal_message,
                        uploaded_low_quality_sample_ids,
                        job_id=job_id,
                    )
                    logger.error(
                        "[EdgeCL] cloud training failed: status={} reason={}.",
                        status,
                        safe_error_summary(terminal_message),
                    )
                    log_diagnostic_debug(
                        self,
                        "[EdgeCL] failed training job diagnostics",
                        lambda: {"job_id": job_id, "message": terminal_message},
                    )
                    break
            finally:
                training_channel.close()

            if success and model_b64:
                # Stale detection: if our model version has advanced since
                # we submitted this job, the result is based on an older model
                # and should not be applied.
                current_version = str(self.model_version)
                if current_version != submitted_model_version:
                    logger.warning(
                        "[EdgeCL] discarding stale training result: submitted_version={} "
                        "current_version={}.",
                        submitted_model_version,
                        current_version,
                    )
                    log_diagnostic_debug(
                        self,
                        "[EdgeCL] stale result diagnostics",
                        lambda: {"job_id": job_id},
                    )
                    self._reset_pending_training_cycle()
                    continue

                try:
                    update = self._prepare_model_update(
                        model_b64,
                        submitted_model_version=submitted_model_version,
                        result_model_version="",
                        job_id=job_id,
                        message=terminal_message,
                        log_prefix="[EdgeCL]",
                    )
                    self._queue_pending_model_update(update)
                    if not self._wait_for_pending_model_update(update):
                        logger.warning(
                            "[EdgeCL] pending model update was not applied before shutdown."
                        )
                except Exception as exc:
                    logger.error(
                        "[EdgeCL] failed to prepare or apply cloud model update: {}.",
                        safe_error_summary(exc),
                    )
                    log_diagnostic_debug(
                        self,
                        "[EdgeCL] model update failure diagnostics",
                        lambda error=exc: {
                            "job_id": job_id,
                            "error": repr(error),
                        },
                    )
            elif not self._stop_event.is_set() and last_status != "WAITING_FOR_SAMPLES":
                logger.error(
                    "[EdgeCL] cloud continual learning failed: {}.",
                    safe_error_summary(terminal_message),
                )

            self._reset_pending_training_cycle()

    def decision_worker(self, task):
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            self._set_task_terminal_state(
                task,
                TASK_STATE.TIMEOUT,
                result_source="shutdown",
            )
            return
        task.edge_process = True
        task.local_queue_enqueued_perf = time.perf_counter()
        self.local_queue.put(task, block=True)

    def _stop_after_fatal_inference_error(self) -> None:
        self._stop_event.set()
        retrain_requested = getattr(self, "_retrain_requested", None)
        if retrain_requested is not None:
            retrain_requested.set()
        for queue_name in ("frame_cache", "local_queue"):
            queue_obj = getattr(self, queue_name, None)
            if queue_obj is None:
                continue
            while True:
                try:
                    pending = queue_obj.get_nowait()
                except Empty:
                    break
                if pending is not _QUEUE_STOP:
                    self._set_task_terminal_state(
                        pending,
                        TASK_STATE.TIMEOUT,
                        result_source="inference_error",
                    )
            try:
                queue_obj.put_nowait(_QUEUE_STOP)
            except Full:
                pass

    def close(self, *, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._retrain_requested.set()
        probe_requested = getattr(self, "_resource_probe_requested", None)
        if probe_requested is not None:
            probe_requested.set()
        for queue_obj in (self.frame_cache, self.local_queue):
            inserted = False
            while not inserted:
                try:
                    queue_obj.put_nowait(_QUEUE_STOP)
                    inserted = True
                except Full:
                    try:
                        queue_obj.get_nowait()
                    except Empty:
                        inserted = True
                except Exception:
                    inserted = True
        for thread in (
            getattr(self, "diff_processor", None),
            getattr(self, "local_processor", None),
            getattr(self, "retrain_processor", None),
            getattr(self, "resource_probe_processor", None),
        ):
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)
        collector = getattr(self, "sample_collector", None)
        if collector is not None:
            if not collector.close(timeout=timeout):
                logger.warning("Timed out while closing async sample collector.")
        writer = getattr(self, "sample_writer", None)
        if writer is not None:
            if not writer.close(timeout=timeout):
                logger.warning("Timed out while closing async sample writer.")
        syncer = getattr(self, "sample_syncer", None)
        if syncer is not None:
            if not syncer.close(timeout=timeout):
                logger.warning("Timed out while closing high-quality sample syncer.")

    def diff_worker(self):
        if not self.config.diff_flag:
            while not self._stop_event.is_set():
                task = self.frame_cache.get(block=True)
                if task is _QUEUE_STOP:
                    return
                task.set_timing(
                    "frame_queue_wait",
                    (time.perf_counter() - task.created_perf) * 1000.0,
                )
                task.set_timing("diff", 0.0)
                self.decision_worker(task)
            return

        task = self.frame_cache.get(block=True)
        if task is _QUEUE_STOP:
            return
        frame = task.frame_edge
        task.set_timing(
            "frame_queue_wait",
            (time.perf_counter() - task.created_perf) * 1000.0,
        )
        diff_started = time.perf_counter()
        self.pre_frame_feature = self.edge_processor.get_frame_feature(frame)
        task.record_timing("diff", (time.perf_counter() - diff_started) * 1000.0)
        self.key_task = task
        self.decision_worker(task)

        while not self._stop_event.is_set():
            task = self.frame_cache.get(block=True)
            if task is _QUEUE_STOP:
                return
            frame = task.frame_edge
            task.set_timing(
                "frame_queue_wait",
                (time.perf_counter() - task.created_perf) * 1000.0,
            )
            diff_started = time.perf_counter()
            self.frame_feature = self.edge_processor.get_frame_feature(frame)
            self.diff += self.edge_processor.cal_frame_diff(
                self.frame_feature,
                self.pre_frame_feature,
            )
            self.pre_frame_feature = self.frame_feature
            task.record_timing("diff", (time.perf_counter() - diff_started) * 1000.0)
            if self.diff >= self.config.diff_thresh:
                self.diff = 0.0
                self.key_task = task
                self.decision_worker(task)
            else:
                self._reuse_latest_result(task)
                self._set_task_terminal_state(
                    task,
                    TASK_STATE.FINISHED,
                    result_source=task.result_source,
                )
                self._try_apply_pending_model_update()

    def local_worker(self):
        while not self._stop_event.is_set():
            task = self.local_queue.get(block=True)
            if task is _QUEUE_STOP:
                return
            task.set_timing(
                "queue_wait_ms",
                (time.perf_counter() - task.created_perf) * 1000.0,
            )
            if time.time() - task.start_time >= self.config.wait_thresh:
                self._set_task_terminal_state(
                    task,
                    TASK_STATE.TIMEOUT,
                    result_source="timeout",
                )
                continue

            local_enqueued = float(getattr(task, "local_queue_enqueued_perf", time.perf_counter()))
            task.set_timing("local_queue_wait", (time.perf_counter() - local_enqueued) * 1000.0)
            current_frame = task.frame_edge
            frame_image_size = tuple(int(value) for value in current_frame.shape[:2])
            try:
                split_resolve_started = time.perf_counter()
                active_splitter = self._resolve_active_splitter(
                    current_frame,
                    frame_image_size,
                )
                task.set_timing(
                    "split_resolve",
                    (time.perf_counter() - split_resolve_started) * 1000.0,
                )
                inference = self.small_object_detection.infer_sample(
                    current_frame,
                    splitter=active_splitter,
                )
            except _FixedSplitRuntimeError as exc:
                logger.error(
                    "Fatal fixed split inference failure for frame={}: {}.",
                    task.frame_index,
                    safe_error_summary(exc),
                )
                self._set_task_terminal_state(
                    task,
                    TASK_STATE.TIMEOUT,
                    result_source="inference_error",
                )
                self._stop_after_fatal_inference_error()
                return
            except Exception as exc:
                logger.error(
                    "Edge inference failed for frame={}: {}.",
                    task.frame_index,
                    safe_error_summary(exc),
                )
                self._set_task_terminal_state(
                    task,
                    TASK_STATE.TIMEOUT,
                    result_source="inference_error",
                )
                continue
            for name, value in dict(getattr(inference, "timing_ms", {}) or {}).items():
                task.record_timing(name, value)

            task.add_result(
                inference.final_detection_boxes or None,
                inference.final_detection_labels or None,
                inference.final_detection_scores or None,
            )
            self._set_task_inference_artifacts(
                task,
                inference,
                result_source="inference",
            )

            self._set_task_terminal_state(
                task,
                TASK_STATE.FINISHED,
                result_source="inference",
            )
            timing = dict(getattr(task, "timing_ms", {}) or {})
            if self._should_log_performance(task.frame_index):
                logger.info(
                    "[EdgePerfSync] frame={} queue_wait_ms={:.3f} "
                    "split_preprocess_ms={:.3f} split_prefix_ms={:.3f} "
                    "split_suffix_ms={:.3f} observables_ms={:.3f} postprocess_ms={:.3f} "
                    "parse_filter_ms={:.3f} task_complete_ms={:.3f}",
                    task.frame_index,
                    float(timing.get("queue_wait_ms", 0.0)),
                    float(timing.get("split_preprocess_ms", 0.0)),
                    float(timing.get("split_prefix_ms", 0.0)),
                    float(timing.get("split_suffix_ms", 0.0)),
                    float(timing.get("observables_ms", 0.0)),
                    float(timing.get("postprocess_ms", 0.0)),
                    float(timing.get("parse_filter_ms", 0.0)),
                    float(timing.get("task_complete_ms", 0.0)),
                )

            if (
                self.collect_flag
                and self.split_learning_enabled
                and inference.intermediate is not None
                and self.fixed_split_plan is not None
            ):
                collection_started = time.perf_counter()
                sample_id = self._next_sample_id(task)
                queued = self.collect_data(task, current_frame, inference)
                if self._should_log_performance(task.frame_index):
                    logger.info(
                        "[EdgePerfEnqueue] sample_id={} frame={} "
                        "sample_collect_enqueue_ms={:.3f} queued={}",
                        sample_id,
                        task.frame_index,
                        (time.perf_counter() - collection_started) * 1000.0,
                        queued,
                    )
            else:
                if self._should_log_performance(task.frame_index):
                    logger.info(
                        "[EdgePerfEnqueue] sample_id=<disabled> frame={} "
                        "sample_collect_enqueue_ms=0.000 queued=False",
                        task.frame_index,
                    )

            self._try_apply_pending_model_update()
