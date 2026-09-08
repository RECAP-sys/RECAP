from __future__ import annotations

import threading
import shutil
import time
import uuid
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from grpc_server import message_transmission_pb2
from grpc_server.workspace import prepare_request_workspace

if TYPE_CHECKING:
    from cloud.edge_registry import EdgeRegistry
    from cloud_server import CloudContinualLearner


JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_SUCCEEDED = "SUCCEEDED"
JOB_STATUS_FAILED = "FAILED"
JOB_STATUS_WAITING_FOR_SAMPLES = "WAITING_FOR_SAMPLES"
JOB_STATUS_STALE = "STALE"
JOB_STATUS_CANCELLED = "CANCELLED"
TERMINAL_JOB_STATUSES = {
    JOB_STATUS_SUCCEEDED,
    JOB_STATUS_FAILED,
    JOB_STATUS_WAITING_FOR_SAMPLES,
    JOB_STATUS_STALE,
    JOB_STATUS_CANCELLED,
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _format_bytes(size: int) -> str:
    value = float(max(0, int(size or 0)))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}GB"


@dataclass(slots=True)
class TrainingJob:
    job_id: str
    edge_id: int
    request_id: str
    job_type: int
    workspace: str
    protocol_version: str = ""
    workspace_root: str = "./cache/server_workspace"
    request_kind: str = ""
    payload_zip: bytes = b""
    send_low_conf_features: bool = False
    frame_indices: tuple[int, ...] = ()
    exclusive_gpu_lease: bool = False
    status: str = JOB_STATUS_QUEUED
    message: str = ""
    model_data: str = ""
    submitted_at_ms: int = field(default_factory=_now_ms)
    started_at_ms: int = 0
    finished_at_ms: int = 0
    base_model_version: str = "0"
    result_model_version: str = ""
    worker_id: str = ""


class TrainingJobManager:
    """Async in-memory scheduler for cloud-side edge training jobs."""

    def __init__(
        self,
        *,
        continual_learner: "CloudContinualLearner",
        max_concurrent_jobs: int,
        edge_registry: "EdgeRegistry | None" = None,
        log_internal_ids: bool = False,
    ) -> None:
        self.continual_learner = continual_learner
        self.max_concurrent_jobs = max(1, int(max_concurrent_jobs))
        self.edge_registry = edge_registry
        self.log_internal_ids = bool(log_internal_ids)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._jobs: dict[str, TrainingJob] = {}
        self._request_index: dict[tuple[int, str], str] = {}
        self._pending_by_edge: dict[int, deque[str]] = {}
        self._edge_round_robin: deque[int] = deque()
        self._running_jobs: set[str] = set()
        self._worker_threads: list[threading.Thread] = []
        self._active_edges: set[int] = set()
        self._edge_model_versions: dict[int, str] = {}
        self._closed = False

        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="training-job-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    def close(self, *, timeout: float = 5.0) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._dispatcher.is_alive():
            self._dispatcher.join(timeout=timeout)
        deadline = time.monotonic() + max(0.0, float(timeout))
        for worker in list(self._worker_threads):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if worker.is_alive():
                worker.join(timeout=remaining)

    def submit(
        self,
        *,
        edge_id: int,
        request_id: str,
        job_type: int,
        workspace: str,
        protocol_version: str = "",
        workspace_root: str = "./cache/server_workspace",
        request_kind: str = "",
        payload_zip: bytes = b"",
        send_low_conf_features: bool = False,
        frame_indices: list[int] | tuple[int, ...] | None = None,
        exclusive_gpu_lease: bool = False,
        base_model_version: str = "0",
    ) -> tuple[TrainingJob, bool]:
        normalized_request_id = str(request_id or "").strip()
        with self._cv:
            if normalized_request_id:
                request_key = (int(edge_id), normalized_request_id)
                existing_job_id = self._request_index.get(request_key)
                if existing_job_id is not None:
                    return self._jobs[existing_job_id], False
            if int(job_type) == message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING:
                for existing in self._jobs.values():
                    if existing.edge_id != int(edge_id):
                        continue
                    if existing.job_type != int(job_type):
                        continue
                    if existing.base_model_version != str(base_model_version or "0"):
                        continue
                    if existing.status in TERMINAL_JOB_STATUSES:
                        continue
                    logger.info(
                        "[TrainingJob] Reusing existing continual-learning job: "
                        "edge={} existing_job={}",
                        int(edge_id),
                        existing.job_id,
                    )
                    return existing, False

            job_id = uuid.uuid4().hex
            job = TrainingJob(
                job_id=job_id,
                edge_id=int(edge_id),
                request_id=normalized_request_id,
                job_type=int(job_type),
                workspace=str(workspace),
                protocol_version=str(protocol_version or ""),
                workspace_root=str(workspace_root or "./cache/server_workspace"),
                request_kind=str(request_kind or ""),
                payload_zip=bytes(payload_zip or b""),
                send_low_conf_features=bool(send_low_conf_features),
                frame_indices=tuple(int(value) for value in (frame_indices or [])),
                exclusive_gpu_lease=bool(exclusive_gpu_lease),
                base_model_version=str(base_model_version or "0"),
                worker_id=str(getattr(self.continual_learner, "worker_id", "") or ""),
            )
            self._jobs[job_id] = job
            if normalized_request_id:
                self._request_index[(job.edge_id, normalized_request_id)] = job_id

            queue = self._pending_by_edge.setdefault(job.edge_id, deque())
            queue.append(job_id)
            if job.edge_id not in self._edge_round_robin:
                self._edge_round_robin.append(job.edge_id)
            self._cv.notify_all()
            return job, True

    def get_job(self, *, edge_id: int, job_id: str) -> TrainingJob | None:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None or job.edge_id != int(edge_id):
                return None
            return job

    def download_result(self, *, edge_id: int, job_id: str) -> tuple[bool, TrainingJob | None, str]:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None or job.edge_id != int(edge_id):
                return False, None, "Training job not found."
            if job.status == JOB_STATUS_STALE:
                return False, job, f"Training job is STALE: {job.message}"
            if job.status != JOB_STATUS_SUCCEEDED:
                return False, job, f"Training job is not ready: {job.status}"
            if not job.model_data:
                return False, job, "Training job completed without model data."
            return True, job, job.message or "Training job completed."

    def cancel_job(self, *, edge_id: int, job_id: str) -> tuple[bool, str]:
        """Cancel a queued or running job.  Returns (cancelled, message)."""
        with self._cv:
            job = self._jobs.get(str(job_id))
            if job is None or job.edge_id != int(edge_id):
                return False, "Training job not found."
            if job.status in TERMINAL_JOB_STATUSES:
                return False, f"Job already in terminal state: {job.status}"
            if job.status == JOB_STATUS_QUEUED:
                # Remove from pending queue
                queue = self._pending_by_edge.get(job.edge_id)
                if queue:
                    try:
                        queue.remove(job.job_id)
                    except ValueError:
                        pass
                    if not queue:
                        self._pending_by_edge.pop(job.edge_id, None)
                        self._remove_edge_from_round_robin_locked(job.edge_id)
            job.status = JOB_STATUS_CANCELLED
            job.message = "Cancelled by edge request."
            job.finished_at_ms = _now_ms()
            self._running_jobs.discard(job_id)
            self._active_edges.discard(job.edge_id)
            self._cv.notify_all()
            return True, "Training job cancelled."

    def update_edge_model_version(self, edge_id: int, model_version: str) -> None:
        """Record that an edge has applied a model update to a new version.

        This is called when the edge successfully downloads and applies
        training results, so the server can detect stale jobs.
        """
        with self._lock:
            self._edge_model_versions[int(edge_id)] = str(model_version)

    def queue_position(self, job_id: str) -> int:
        with self._lock:
            return self._queue_position_locked(str(job_id))

    def training_queue_state(self) -> tuple[int, int]:
        with self._lock:
            queued = sum(len(queue) for queue in self._pending_by_edge.values())
            total = queued + len(self._running_jobs)
            return total, self.max_concurrent_jobs

    def _dispatch_loop(self) -> None:
        while True:
            with self._cv:
                while not self._closed:
                    job = self._next_dispatchable_job_locked()
                    if job is not None:
                        break
                    self._cv.wait(timeout=0.5)
                else:
                    return

            worker = threading.Thread(
                target=self._run_job,
                args=(job.job_id,),
                name=f"training-job-{job.job_id}",
                daemon=True,
            )
            with self._lock:
                self._worker_threads.append(worker)
            worker.start()

    def _next_dispatchable_job_locked(self) -> TrainingJob | None:
        if len(self._running_jobs) >= self.max_concurrent_jobs:
            return None
        if not self._edge_round_robin:
            return None

        max_attempts = len(self._edge_round_robin)
        for _ in range(max_attempts):
            edge_id = self._edge_round_robin[0]
            self._edge_round_robin.rotate(-1)

            if edge_id in self._active_edges:
                continue

            queue = self._pending_by_edge.get(edge_id)
            if not queue:
                self._pending_by_edge.pop(edge_id, None)
                self._remove_edge_from_round_robin_locked(edge_id)
                continue

            job_id = queue.popleft()
            if not queue:
                self._pending_by_edge.pop(edge_id, None)
                self._remove_edge_from_round_robin_locked(edge_id)

            job = self._jobs[job_id]
            job.status = JOB_STATUS_RUNNING
            job.started_at_ms = _now_ms()
            self._running_jobs.add(job_id)
            self._active_edges.add(edge_id)
            return job

        return None

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            edge_id = job.edge_id
            job_type = job.job_type
            workspace = job.workspace
            workspace_root = job.workspace_root
            request_kind = job.request_kind
            payload_zip = job.payload_zip
            frame_indices = list(job.frame_indices)
            send_low_conf_features = job.send_low_conf_features
            exclusive_gpu_lease = job.exclusive_gpu_lease
            base_model_version = job.base_model_version
        logger.info(
            "Training job started: edge={} type={}.",
            edge_id,
            request_kind or self._request_kind_for_job_type(job_type),
        )
        log_diagnostic_debug(
            self,
            "training job started details",
            lambda: {
                "job_id": job_id,
                "workspace": workspace,
                "payload_zip_bytes": len(payload_zip or b""),
            },
        )

        temporary_workspace = None
        try:
            if payload_zip:
                temporary_workspace = prepare_request_workspace(
                    workspace_root,
                    edge_id=edge_id,
                    request_kind=(request_kind or self._request_kind_for_job_type(job_type)),
                    payload_zip=payload_zip,
                    client_cache_path=workspace,
                    log_internal_ids=self.log_internal_ids,
                )
                workspace = str(temporary_workspace)
            lease_scope = getattr(self.continual_learner, "gpu_lease_scope", None)
            context = (
                lease_scope(
                    edge_id=edge_id,
                    job_id=job_id,
                    workspace=workspace,
                    exclusive=exclusive_gpu_lease,
                )
                if callable(lease_scope)
                else nullcontext()
            )
            with context:
                success, model_data, message = self._execute_job(
                    edge_id=edge_id,
                    job_type=job_type,
                    workspace=workspace,
                    frame_indices=frame_indices,
                    send_low_conf_features=send_low_conf_features,
                    base_model_version=base_model_version,
                )
        except Exception as exc:
            logger.error(
                "Training job failed: edge={} type={} reason={}.",
                edge_id,
                request_kind or self._request_kind_for_job_type(job_type),
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "training job failure details",
                lambda error=exc: {"job_id": job_id, "error": repr(error)},
            )
            success = False
            model_data = ""
            message = str(exc)
        finally:
            if temporary_workspace is not None:
                shutil.rmtree(temporary_workspace, ignore_errors=True)

        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = _status_for_job_result(
                success=success,
                model_data=model_data,
                message=message,
            )
            job.model_data = model_data or ""
            job.message = str(message or "")
            job.finished_at_ms = _now_ms()

            # Model version tracking: assign result version on success
            if success and job.model_data:
                try:
                    result_model_version = str(int(job.base_model_version) + 1)
                except (ValueError, TypeError):
                    result_model_version = "1"
                job.result_model_version = result_model_version

                # Stale detection: if the edge has already advanced past the
                # base version this job was trained on, mark it STALE.
                current_edge_version = self._edge_model_versions.get(job.edge_id, "0")
                try:
                    if int(current_edge_version) > int(job.base_model_version):
                        job.status = JOB_STATUS_STALE
                        job.message = (
                            f"STALE: edge model advanced to version {current_edge_version} "
                            f"while job was based on version {job.base_model_version}"
                        )
                        logger.warning(
                            "Training job marked STALE: edge={} base_version={} "
                            "current_version={}.",
                            job.edge_id,
                            job.base_model_version,
                            current_edge_version,
                        )
                        log_diagnostic_debug(
                            self,
                            "stale training job details",
                            lambda: {"job_id": job_id},
                        )
                except (ValueError, TypeError):
                    pass

            self._running_jobs.discard(job_id)
            self._active_edges.discard(job.edge_id)

            # Notify edge registry of completion
            if self.edge_registry is not None:
                self.edge_registry.record_job_completed(
                    job.edge_id,
                    success=job.status == JOB_STATUS_SUCCEEDED,
                )

            logger.info(
                "Training job completed: edge={} status={} model_size={}.",
                job.edge_id,
                job.status,
                _format_bytes(len((job.model_data or "").encode("utf-8"))),
            )
            log_diagnostic_debug(
                self,
                "training job completed details",
                lambda: {
                    "job_id": job_id,
                    "request_id": job.request_id,
                    "message": job.message,
                },
            )

            self._cv.notify_all()

    def _execute_job(
        self,
        *,
        edge_id: int,
        job_type: int,
        workspace: str,
        frame_indices: list[int],
        send_low_conf_features: bool,
        base_model_version: str = "0",
    ) -> tuple[bool, str, str]:
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING:
            return self.continual_learner.get_ground_truth_and_fixed_split_retrain(
                edge_id,
                workspace,
            )
        raise ValueError(f"Unsupported training job type: {job_type!r}")

    @staticmethod
    def _request_kind_for_job_type(job_type: int) -> str:
        if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING:
            return "continual_learning"
        raise ValueError(f"Unsupported training job type: {job_type!r}")

    def _queue_position_locked(self, job_id: str) -> int:
        job = self._jobs.get(job_id)
        if job is None:
            return -1
        if job.status == JOB_STATUS_RUNNING:
            return 0
        if job.status in TERMINAL_JOB_STATUSES:
            return -1

        queued_job_ids: list[str] = []
        for queue in self._pending_by_edge.values():
            queued_job_ids.extend(queue)
        try:
            return queued_job_ids.index(job_id) + 1
        except ValueError:
            return -1

    def _remove_edge_from_round_robin_locked(self, edge_id: int) -> None:
        if edge_id not in self._edge_round_robin:
            return
        self._edge_round_robin = deque(
            value for value in self._edge_round_robin if value != edge_id
        )


def _status_for_job_result(*, success: bool, model_data: str, message: str) -> str:
    if success:
        return JOB_STATUS_SUCCEEDED
    if _is_waiting_for_samples_result(model_data=model_data, message=message):
        return JOB_STATUS_WAITING_FOR_SAMPLES
    return JOB_STATUS_FAILED


def _is_waiting_for_samples_result(*, model_data: str, message: str) -> bool:
    if str(model_data or ""):
        return False
    return str(message or "").startswith("Waiting for enough recent training samples:")
