from __future__ import annotations

import time
from collections.abc import Sequence

from cloud.annotation.label_cache import TeacherLabelCache
from cloud.annotation.teacher_worker import TeacherAnnotationWorker
from cloud.annotation.types import (
    TeacherAnnotationEnsureResult,
    TeacherAnnotationRequest,
    TeacherAnnotationResult,
    TeacherAnnotationStatus,
    TeacherAnnotationSubmitResult,
)


class TeacherAnnotationService:
    def __init__(
        self,
        *,
        label_cache: TeacherLabelCache,
        worker: TeacherAnnotationWorker | None = None,
        log_internal_ids: bool = False,
    ) -> None:
        self.label_cache = label_cache
        self.worker = worker
        self.log_internal_ids = bool(log_internal_ids)

    def lookup_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> TeacherAnnotationSubmitResult:
        requested = list(requests or [])
        results, cache_read_time = self.label_cache.lookup_many(requested)
        cache_hits = sum(
            1 for result in results if result.status == TeacherAnnotationStatus.CACHE_HIT
        )
        cache_misses = len(results) - cache_hits
        return TeacherAnnotationSubmitResult(
            requested_samples=len(requested),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            results=results,
            cache_read_time=cache_read_time,
        )

    def submit_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> TeacherAnnotationSubmitResult:
        requested = list(requests or [])
        lookup = self.lookup_many(requested)
        misses = [
            result.request
            for result in lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_MISS
        ]
        submitted = 0
        duplicate = 0
        failed_count = 0
        worker_results: list[TeacherAnnotationResult] = []
        if misses and self.worker is not None:
            worker_submit = self.worker.submit_many(misses)
            submitted = int(worker_submit.submitted)
            duplicate = int(worker_submit.duplicate)
            failed_count = int(worker_submit.failed_count)
            worker_results = list(worker_submit.results)
        elif misses:
            worker_results = [
                TeacherAnnotationResult(
                    request=request,
                    status=TeacherAnnotationStatus.UNRESOLVED,
                    error="teacher annotation worker is not configured",
                )
                for request in misses
            ]
            failed_count = len(worker_results)

        results = [
            result
            for result in lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_HIT
        ] + worker_results
        return TeacherAnnotationSubmitResult(
            requested_samples=len(requested),
            cache_hits=lookup.cache_hits,
            cache_misses=lookup.cache_misses,
            submitted=submitted,
            duplicate=duplicate,
            failed_count=failed_count,
            results=results,
            cache_read_time=lookup.cache_read_time,
        )

    def ensure_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
        *,
        wait: bool,
        timeout_sec: float | None,
    ) -> TeacherAnnotationEnsureResult:
        requested = list(requests or [])
        ensure_started = time.perf_counter()
        worker_stats_before = self.worker.snapshot_stats() if self.worker is not None else {}

        initial_lookup = self.lookup_many(requested)
        cache_read_time = initial_lookup.cache_read_time
        results: list[TeacherAnnotationResult] = [
            result
            for result in initial_lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_HIT
        ]
        resolved_ids = {result.sample_id for result in results if result.labels is not None}
        misses = [
            result.request
            for result in initial_lookup.results
            if result.status == TeacherAnnotationStatus.CACHE_MISS
        ]

        submitted = 0
        if misses:
            submit_result = self.submit_many(misses)
            submitted = int(submit_result.submitted)

        waited_sec = 0.0
        if wait and misses and self.worker is not None:
            waited_sec = self.worker.wait_for(misses, timeout_sec=timeout_sec)

        remaining = [request for request in misses if str(request.sample_id) not in resolved_ids]
        if remaining:
            post_lookup = self.lookup_many(remaining)
            cache_read_time += post_lookup.cache_read_time
            for result in post_lookup.results:
                if result.status == TeacherAnnotationStatus.CACHE_HIT and result.labels is not None:
                    results.append(
                        TeacherAnnotationResult(
                            request=result.request,
                            status=TeacherAnnotationStatus.WORKER_RESULT,
                            labels=result.labels,
                        )
                    )
                    resolved_ids.add(result.sample_id)
            remaining = [
                result.request
                for result in post_lookup.results
                if result.status == TeacherAnnotationStatus.CACHE_MISS
                and str(result.request.sample_id) not in resolved_ids
            ]

        cache_write_time = 0.0

        unresolved_requests = [
            request for request in requested if str(request.sample_id) not in resolved_ids
        ]
        retryable_errors_by_sample_id = (
            self.worker.retryable_failure_reasons(unresolved_requests)
            if self.worker is not None
            else {}
        )
        unresolved_results = [
            TeacherAnnotationResult(
                request=request,
                status=TeacherAnnotationStatus.UNRESOLVED,
            )
            for request in unresolved_requests
        ]
        results.extend(unresolved_results)

        worker_stats_after = self.worker.snapshot_stats() if self.worker is not None else {}
        stats_delta = (
            self.worker.stats_delta(worker_stats_before, worker_stats_after)
            if self.worker is not None
            else {
                "teacher_batch_size": 0,
                "teacher_batches": 0,
                "batch_fallback_count": 0,
                "oom_retry_count": 0,
                "failed_count": 0,
            }
        )
        annotation_time = time.perf_counter() - ensure_started
        ensure_result = TeacherAnnotationEnsureResult(
            requested_samples=len(requested),
            cache_hits=initial_lookup.cache_hits,
            cache_misses=initial_lookup.cache_misses,
            submitted=submitted,
            waited_sec=waited_sec,
            unresolved_count=len(unresolved_requests),
            annotation_time=annotation_time,
            cache_read_time=cache_read_time,
            cache_write_time=cache_write_time,
            teacher_batch_size=int(stats_delta.get("teacher_batch_size", 0) or 0),
            teacher_batches=int(stats_delta.get("teacher_batches", 0) or 0),
            batch_fallback_count=int(stats_delta.get("batch_fallback_count", 0) or 0),
            oom_retry_count=int(stats_delta.get("oom_retry_count", 0) or 0),
            failed_count=int(stats_delta.get("failed_count", 0) or 0),
            retryable_errors_by_sample_id=retryable_errors_by_sample_id,
            results=results,
            unresolved_requests=unresolved_requests,
        )
        return ensure_result
