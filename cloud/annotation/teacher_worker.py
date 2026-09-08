from __future__ import annotations

import contextlib
import gc
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from cloud.annotation.label_cache import TeacherLabelCache
from cloud.annotation.types import (
    TeacherAnnotationRequest,
    TeacherAnnotationResult,
    TeacherAnnotationStatus,
    TeacherAnnotationSubmitResult,
)
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary

BatchInference = Callable[[Sequence[np.ndarray], float], Sequence[object]]
SingleInference = Callable[[np.ndarray, float], object]
LabelBuilder = Callable[[TeacherAnnotationRequest, np.ndarray, object], Mapping[str, Any] | None]
TeacherScopeFactory = Callable[..., contextlib.AbstractContextManager[None]]


class _BatchUnsupported(RuntimeError):
    pass


@dataclass
class _QueueEntry:
    request: TeacherAnnotationRequest
    attempts: int = 0


@contextlib.contextmanager
def _null_teacher_scope(*_args, **_kwargs):
    yield


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text)


def _is_retryable_annotation_error(exc: BaseException) -> bool:
    return bool(getattr(exc, "retryable", False))


def _release_cuda_cache_after_oom() -> None:
    """Release failed-attempt tensors before retrying with a smaller batch."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()


def _default_label_builder(
    request: TeacherAnnotationRequest,
    frame: np.ndarray,
    prediction: object,
) -> Mapping[str, Any] | None:
    del request, frame
    if not isinstance(prediction, (list, tuple)):
        return None
    pred_boxes = prediction[0] if len(prediction) >= 1 else None
    pred_labels = prediction[1] if len(prediction) >= 2 else None
    pred_scores = prediction[2] if len(prediction) >= 3 else None
    if pred_boxes is None or pred_labels is None:
        return None
    labels: dict[str, Any] = {
        "boxes": list(pred_boxes or []),
        "labels": list(pred_labels or []),
    }
    if pred_scores is not None:
        labels["scores"] = list(pred_scores or [])
    return labels


def _request_signature(request: TeacherAnnotationRequest) -> tuple[object, ...]:
    return (
        str(request.teacher_model_name),
        str(request.teacher_weights_fingerprint),
        str(request.teacher_label_schema),
        int(request.teacher_num_classes),
        float(request.teacher_annotation_threshold),
        str(request.label_coordinate_space),
    )


class TeacherAnnotationWorker:
    def __init__(
        self,
        *,
        label_cache: TeacherLabelCache,
        batch_inference: BatchInference | None = None,
        single_inference: SingleInference | None = None,
        label_builder: LabelBuilder | None = None,
        teacher_scope: TeacherScopeFactory | None = None,
        max_queue_size: int = 4096,
        worker_batch_size: int = 8,
        max_retries: int = 2,
        oom_retry_enabled: bool = True,
        min_worker_batch_size: int = 1,
        auto_start: bool = True,
        log_internal_ids: bool = False,
    ) -> None:
        self.label_cache = label_cache
        self.batch_inference = batch_inference
        self.single_inference = single_inference
        self.label_builder = label_builder or _default_label_builder
        self.teacher_scope = teacher_scope or _null_teacher_scope
        self.max_queue_size = max(1, int(max_queue_size))
        self.worker_batch_size = max(1, int(worker_batch_size))
        self.max_retries = max(0, int(max_retries))
        self.oom_retry_enabled = bool(oom_retry_enabled)
        self.min_worker_batch_size = max(1, int(min_worker_batch_size))
        self.min_worker_batch_size = min(self.min_worker_batch_size, self.worker_batch_size)
        self.log_internal_ids = bool(log_internal_ids)

        self._condition = threading.Condition()
        self._queue: deque[_QueueEntry] = deque()
        self._active_keys: set[str] = set()
        self._failed: dict[str, str] = {}
        self._retryable_failures: dict[str, str] = {}
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._stats: dict[str, int] = {
            "teacher_batch_size": self.worker_batch_size,
            "teacher_batches": 0,
            "batch_fallback_count": 0,
            "oom_retry_count": 0,
            "failed_count": 0,
        }
        if auto_start:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopped = False
            self._thread = threading.Thread(
                target=self._run_loop,
                name="teacher-annotation-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float | None = 2.0) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def snapshot_stats(self) -> dict[str, int]:
        with self._condition:
            return dict(self._stats)

    @staticmethod
    def stats_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
        return {
            "teacher_batch_size": int(after.get("teacher_batch_size", 0) or 0),
            "teacher_batches": int(after.get("teacher_batches", 0) or 0)
            - int(before.get("teacher_batches", 0) or 0),
            "batch_fallback_count": int(after.get("batch_fallback_count", 0) or 0)
            - int(before.get("batch_fallback_count", 0) or 0),
            "oom_retry_count": int(after.get("oom_retry_count", 0) or 0)
            - int(before.get("oom_retry_count", 0) or 0),
            "failed_count": int(after.get("failed_count", 0) or 0)
            - int(before.get("failed_count", 0) or 0),
        }

    def submit_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> TeacherAnnotationSubmitResult:
        requested = list(requests or [])
        results: list[TeacherAnnotationResult] = []
        submitted = 0
        duplicates = 0
        failures = 0
        with self._condition:
            for request in requested:
                cache_key = request.cache_key().digest
                if self.label_cache.exists(request):
                    results.append(
                        TeacherAnnotationResult(
                            request=request,
                            status=TeacherAnnotationStatus.CACHE_HIT,
                            labels=self.label_cache.read(request),
                        )
                    )
                    continue
                if cache_key in self._active_keys:
                    duplicates += 1
                    results.append(
                        TeacherAnnotationResult(
                            request=request,
                            status=TeacherAnnotationStatus.DUPLICATE,
                        )
                    )
                    continue
                if len(self._queue) >= self.max_queue_size:
                    failures += 1
                    self._failed[cache_key] = "teacher annotation queue is full"
                    results.append(
                        TeacherAnnotationResult(
                            request=request,
                            status=TeacherAnnotationStatus.FAILED,
                            error="teacher annotation queue is full",
                        )
                    )
                    continue
                self._queue.append(_QueueEntry(request=request))
                self._active_keys.add(cache_key)
                self._failed.pop(cache_key, None)
                self._retryable_failures.pop(cache_key, None)
                submitted += 1
                results.append(
                    TeacherAnnotationResult(
                        request=request,
                        status=TeacherAnnotationStatus.SUBMITTED,
                    )
                )
            self._condition.notify_all()
        return TeacherAnnotationSubmitResult(
            requested_samples=len(requested),
            submitted=submitted,
            duplicate=duplicates,
            failed_count=failures,
            results=results,
        )

    def wait_for(
        self,
        requests: Sequence[TeacherAnnotationRequest],
        *,
        timeout_sec: float | None,
    ) -> float:
        requests_by_key = {request.cache_key().digest: request for request in list(requests or [])}
        pending_keys = set(requests_by_key)
        started = time.perf_counter()
        deadline = None if timeout_sec is None else started + max(0.0, float(timeout_sec))
        with self._condition:
            while pending_keys:
                pending_keys = {
                    key
                    for key in pending_keys
                    if key in self._active_keys
                    and key not in self._failed
                    and not self.label_cache.exists(requests_by_key[key])
                }
                if not pending_keys:
                    break
                remaining = None if deadline is None else deadline - time.perf_counter()
                if remaining is not None and remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
        return time.perf_counter() - started

    def retryable_failure_reasons(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> dict[str, str]:
        reasons: dict[str, str] = {}
        with self._condition:
            for request in list(requests or []):
                cache_key = request.cache_key().digest
                reason = self._retryable_failures.get(cache_key)
                if not reason and cache_key in self._active_keys:
                    reason = "teacher annotation still pending"
                if reason:
                    reasons[str(request.sample_id)] = str(reason)
        return reasons

    def process_pending_once(self) -> int:
        entries = self._drain_queue()
        if not entries:
            return 0
        self._process_entries(entries)
        return len(entries)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
            self.process_pending_once()

    def _drain_queue(self) -> list[_QueueEntry]:
        with self._condition:
            entries = list(self._queue)
            self._queue.clear()
            return entries

    def _process_entries(self, entries: Sequence[_QueueEntry]) -> None:
        grouped: dict[tuple[object, ...], list[_QueueEntry]] = {}
        for entry in entries:
            if self.label_cache.exists(entry.request):
                self._mark_done(entry.request)
                continue
            grouped.setdefault(_request_signature(entry.request), []).append(entry)
        for group_entries in grouped.values():
            self._process_group(group_entries)

    def _process_group(self, entries: Sequence[_QueueEntry]) -> None:
        index = 0
        attempt_batch_size = max(1, int(self.worker_batch_size))
        while index < len(entries):
            actual_size = min(attempt_batch_size, len(entries) - index)
            chunk = list(entries[index : index + actual_size])
            try:
                self._annotate_chunk(chunk)
            except Exception as exc:
                if _is_retryable_annotation_error(exc):
                    self._mark_retryable(chunk, str(exc) or type(exc).__name__)
                    index += actual_size
                    continue
                if _is_cuda_oom(exc) and self.oom_retry_enabled:
                    with self._condition:
                        self._stats["oom_retry_count"] += 1
                    _release_cuda_cache_after_oom()
                    if attempt_batch_size > self.min_worker_batch_size:
                        next_size = max(
                            self.min_worker_batch_size,
                            int(attempt_batch_size // 2),
                        )
                        logger.warning(
                            "[TeacherAnnotation][OOMRetry] teacher_batch_size={} "
                            "next_batch_size={} samples={} error={}",
                            actual_size,
                            next_size,
                            len(chunk),
                            safe_error_summary(exc),
                        )
                        attempt_batch_size = next_size
                        continue
                    logger.warning(
                        "[TeacherAnnotation][OOMRetry] batch_size=1 still failed; "
                        "marking sample(s) failed. error={}",
                        safe_error_summary(exc),
                    )
                self._handle_chunk_failure(chunk, exc)
            index += actual_size

    def _annotate_chunk(self, chunk: Sequence[_QueueEntry]) -> None:
        requests = [entry.request for entry in chunk]
        frames: list[np.ndarray] = []
        readable_requests: list[TeacherAnnotationRequest] = []
        for request in requests:
            frame = cv2.imread(str(request.image_path))
            if frame is None:
                self._mark_failed(request, f"unreadable image: {request.image_path}")
                continue
            readable_requests.append(request)
            frames.append(frame)
        if not readable_requests:
            return

        threshold = float(readable_requests[0].teacher_annotation_threshold)
        with self.teacher_scope(
            "teacher annotation worker batch",
            sample_count=len(readable_requests),
        ):
            predictions = self._run_batch_or_fallback(frames, threshold)
        if len(predictions) != len(readable_requests):
            raise RuntimeError(
                "Teacher batch inference returned an invalid result count "
                f"(expected={len(readable_requests)}, got={len(predictions)})."
            )
        write_entries: list[tuple[TeacherAnnotationRequest, Mapping[str, Any]]] = []
        for request, frame, prediction in zip(readable_requests, frames, predictions):
            labels = self.label_builder(request, frame, prediction)
            if labels is None and bool(dict(request.metadata or {}).get("include_empty", True)):
                labels = {"boxes": [], "labels": []}
            if labels is None:
                self._mark_failed(request, "teacher prediction produced no labels")
                continue
            write_entries.append((request, labels))
        write_started = time.perf_counter()
        for request, labels in write_entries:
            self.label_cache.write(request, labels, source="worker")
            self._mark_done(request)
        cache_write_time = time.perf_counter() - write_started
        with self._condition:
            self._stats["teacher_batches"] += 1
        logger.info(
            "[TeacherAnnotation][Batch] teacher_batch_size={} teacher_batches=1 "
            "batch_fallback_count=0 oom_retry_count=0 failed_count=0 "
            "cache_write_time={:.3f}s",
            len(readable_requests),
            cache_write_time,
        )

    def _run_batch_or_fallback(
        self,
        frames: Sequence[np.ndarray],
        threshold: float,
    ) -> list[object]:
        if self.batch_inference is not None:
            try:
                predictions = self._call_batch_inference(frames, threshold)
            except _BatchUnsupported as exc:
                logger.warning(
                    "[TeacherAnnotation][Batch] batch inference unavailable; "
                    "falling back to per-sample inference. error={}",
                    safe_error_summary(exc),
                )
            else:
                return list(predictions)
        if self.single_inference is None:
            raise RuntimeError("No teacher batch or single-sample inference callable is available.")
        with self._condition:
            self._stats["batch_fallback_count"] += 1
        predictions = [self._call_single_inference(frame, threshold) for frame in frames]
        logger.info(
            "[TeacherAnnotation][Batch] teacher_batch_size={} teacher_batches=1 "
            "batch_fallback_count=1",
            len(frames),
        )
        return predictions

    def _call_batch_inference(
        self,
        frames: Sequence[np.ndarray],
        threshold: float,
    ) -> Sequence[object]:
        if self.batch_inference is None:
            raise _BatchUnsupported("missing batch inference callable")
        try:
            return self.batch_inference(frames, threshold)
        except (AttributeError, NotImplementedError) as exc:
            raise _BatchUnsupported(str(exc)) from exc
        except TypeError:
            try:
                return self.batch_inference(frames, threshold=threshold)
            except (AttributeError, NotImplementedError) as exc:
                raise _BatchUnsupported(str(exc)) from exc
            except TypeError as exc:
                raise _BatchUnsupported(str(exc)) from exc

    def _call_single_inference(self, frame: np.ndarray, threshold: float) -> object:
        if self.single_inference is None:
            raise RuntimeError("missing single-sample inference callable")
        try:
            return self.single_inference(frame, threshold)
        except TypeError:
            try:
                return self.single_inference(frame, threshold=threshold)
            except TypeError:
                return self.single_inference(frame)

    def _handle_chunk_failure(self, chunk: Sequence[_QueueEntry], exc: BaseException) -> None:
        for entry in chunk:
            if entry.attempts < self.max_retries and not _is_cuda_oom(exc):
                entry.attempts += 1
                with self._condition:
                    self._queue.append(entry)
                    self._condition.notify_all()
                continue
            self._mark_failed(entry.request, str(exc) or type(exc).__name__)

    def _mark_done(self, request: TeacherAnnotationRequest) -> None:
        cache_key = request.cache_key().digest
        with self._condition:
            self._active_keys.discard(cache_key)
            self._failed.pop(cache_key, None)
            self._retryable_failures.pop(cache_key, None)
            self._condition.notify_all()

    def _mark_failed(self, request: TeacherAnnotationRequest, error: str) -> None:
        cache_key = request.cache_key().digest
        with self._condition:
            self._active_keys.discard(cache_key)
            self._failed[cache_key] = str(error)
            self._retryable_failures.pop(cache_key, None)
            self._stats["failed_count"] += 1
            self._condition.notify_all()
        logger.warning(
            "[TeacherAnnotation][Worker] annotation failed: reason={}.",
            safe_error_summary(error),
        )
        log_diagnostic_debug(
            self,
            "[TeacherAnnotation][Worker] failure diagnostics",
            lambda: {
                "sample_id": request.sample_id,
                "cache_key": cache_key,
                "error": error,
            },
        )

    def _mark_retryable(self, chunk: Sequence[_QueueEntry], error: str) -> None:
        with self._condition:
            for entry in chunk:
                cache_key = entry.request.cache_key().digest
                self._active_keys.discard(cache_key)
                self._failed.pop(cache_key, None)
                self._retryable_failures[cache_key] = str(error)
            self._condition.notify_all()
