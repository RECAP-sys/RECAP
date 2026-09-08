from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cloud.annotation.types import (
    TeacherAnnotationEnsureResult,
    TeacherAnnotationRequest,
    TeacherAnnotationResult,
    TeacherAnnotationStatus,
    TeacherAnnotationSubmitResult,
)
from cloud.workers.worker_protocol import JsonRpcServer, post_json


def _request_to_payload(request: TeacherAnnotationRequest) -> dict[str, Any]:
    return {
        "sample_id": str(request.sample_id),
        "edge_id": request.edge_id,
        "model_id": str(request.model_id),
        "image_path": str(request.image_path),
        "image_sha1": str(request.image_sha1),
        "teacher_model_name": str(request.teacher_model_name),
        "teacher_weights_fingerprint": str(request.teacher_weights_fingerprint),
        "teacher_label_schema": str(request.teacher_label_schema),
        "teacher_num_classes": int(request.teacher_num_classes),
        "teacher_annotation_threshold": float(request.teacher_annotation_threshold),
        "label_coordinate_space": str(request.label_coordinate_space),
        "metadata": dict(request.metadata or {}),
    }


def _request_from_payload(payload: Mapping[str, Any]) -> TeacherAnnotationRequest:
    return TeacherAnnotationRequest(
        sample_id=str(payload.get("sample_id", "")),
        edge_id=payload.get("edge_id", ""),
        model_id=str(payload.get("model_id", "")),
        image_path=str(payload.get("image_path", "")),
        image_sha1=str(payload.get("image_sha1", "")),
        teacher_model_name=str(payload.get("teacher_model_name", "")),
        teacher_weights_fingerprint=str(payload.get("teacher_weights_fingerprint", "")),
        teacher_label_schema=str(payload.get("teacher_label_schema", "")),
        teacher_num_classes=int(payload.get("teacher_num_classes", 0) or 0),
        teacher_annotation_threshold=float(
            payload.get("teacher_annotation_threshold", 0.0) or 0.0
        ),
        label_coordinate_space=str(payload.get("label_coordinate_space", "")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _result_to_payload(result: TeacherAnnotationResult) -> dict[str, Any]:
    return {
        "request": _request_to_payload(result.request),
        "status": result.status.value,
        "labels": None if result.labels is None else dict(result.labels),
        "error": result.error,
    }


def _result_from_payload(payload: Mapping[str, Any]) -> TeacherAnnotationResult:
    labels = payload.get("labels")
    return TeacherAnnotationResult(
        request=_request_from_payload(dict(payload.get("request", {}) or {})),
        status=TeacherAnnotationStatus(str(payload.get("status", "unresolved"))),
        labels=dict(labels) if isinstance(labels, Mapping) else None,
        error=None if payload.get("error") is None else str(payload.get("error")),
    )


def _submit_to_payload(result: TeacherAnnotationSubmitResult) -> dict[str, Any]:
    return {
        "requested_samples": int(result.requested_samples),
        "cache_hits": int(result.cache_hits),
        "cache_misses": int(result.cache_misses),
        "submitted": int(result.submitted),
        "duplicate": int(result.duplicate),
        "failed_count": int(result.failed_count),
        "cache_read_time": float(result.cache_read_time),
        "results": [_result_to_payload(item) for item in result.results],
    }


def _submit_from_payload(payload: Mapping[str, Any]) -> TeacherAnnotationSubmitResult:
    return TeacherAnnotationSubmitResult(
        requested_samples=int(payload.get("requested_samples", 0) or 0),
        cache_hits=int(payload.get("cache_hits", 0) or 0),
        cache_misses=int(payload.get("cache_misses", 0) or 0),
        submitted=int(payload.get("submitted", 0) or 0),
        duplicate=int(payload.get("duplicate", 0) or 0),
        failed_count=int(payload.get("failed_count", 0) or 0),
        cache_read_time=float(payload.get("cache_read_time", 0.0) or 0.0),
        results=[
            _result_from_payload(dict(item))
            for item in list(payload.get("results", []) or [])
            if isinstance(item, Mapping)
        ],
    )


def _ensure_to_payload(result: TeacherAnnotationEnsureResult) -> dict[str, Any]:
    return {
        "requested_samples": int(result.requested_samples),
        "cache_hits": int(result.cache_hits),
        "cache_misses": int(result.cache_misses),
        "submitted": int(result.submitted),
        "waited_sec": float(result.waited_sec),
        "unresolved_count": int(result.unresolved_count),
        "annotation_time": float(result.annotation_time),
        "cache_read_time": float(result.cache_read_time),
        "cache_write_time": float(result.cache_write_time),
        "teacher_batch_size": int(result.teacher_batch_size),
        "teacher_batches": int(result.teacher_batches),
        "batch_fallback_count": int(result.batch_fallback_count),
        "oom_retry_count": int(result.oom_retry_count),
        "failed_count": int(result.failed_count),
        "retryable_errors_by_sample_id": dict(result.retryable_errors_by_sample_id),
        "results": [_result_to_payload(item) for item in result.results],
        "unresolved_requests": [
            _request_to_payload(request) for request in result.unresolved_requests
        ],
    }


def _ensure_from_payload(payload: Mapping[str, Any]) -> TeacherAnnotationEnsureResult:
    return TeacherAnnotationEnsureResult(
        requested_samples=int(payload.get("requested_samples", 0) or 0),
        cache_hits=int(payload.get("cache_hits", 0) or 0),
        cache_misses=int(payload.get("cache_misses", 0) or 0),
        submitted=int(payload.get("submitted", 0) or 0),
        waited_sec=float(payload.get("waited_sec", 0.0) or 0.0),
        unresolved_count=int(payload.get("unresolved_count", 0) or 0),
        annotation_time=float(payload.get("annotation_time", 0.0) or 0.0),
        cache_read_time=float(payload.get("cache_read_time", 0.0) or 0.0),
        cache_write_time=float(payload.get("cache_write_time", 0.0) or 0.0),
        teacher_batch_size=int(payload.get("teacher_batch_size", 0) or 0),
        teacher_batches=int(payload.get("teacher_batches", 0) or 0),
        batch_fallback_count=int(payload.get("batch_fallback_count", 0) or 0),
        oom_retry_count=int(payload.get("oom_retry_count", 0) or 0),
        failed_count=int(payload.get("failed_count", 0) or 0),
        retryable_errors_by_sample_id={
            str(key): str(value)
            for key, value in dict(
                payload.get("retryable_errors_by_sample_id", {}) or {}
            ).items()
        },
        results=[
            _result_from_payload(dict(item))
            for item in list(payload.get("results", []) or [])
            if isinstance(item, Mapping)
        ],
        unresolved_requests=[
            _request_from_payload(dict(item))
            for item in list(payload.get("unresolved_requests", []) or [])
            if isinstance(item, Mapping)
        ],
    )


class SharedTeacherAnnotationRpcServer:
    """Expose one process-local teacher queue to all edge worker processes."""

    def __init__(
        self,
        *,
        service,
        metadata_provider: Callable[[], Mapping[str, Any]],
        listen_address: str = "127.0.0.1:0",
    ) -> None:
        self.service = service
        self.metadata_provider = metadata_provider
        self._server = JsonRpcServer(
            listen_address=str(listen_address),
            routes={
                "/teacher_annotation/metadata": self._metadata,
                "/teacher_annotation/submit": self._submit,
                "/teacher_annotation/ensure": self._ensure,
            },
            health_payload={"ok": True, "service": "shared_teacher_annotation"},
        )

    @property
    def listen_address(self) -> str:
        return self._server.listen_address

    def start(self) -> None:
        self._server.start()

    def shutdown(self) -> None:
        self._server.shutdown()

    def _metadata(self, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"metadata": dict(self.metadata_provider())}

    def _submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        requests = self._requests(payload)
        return {"result": _submit_to_payload(self.service.submit_many(requests))}

    def _ensure(self, payload: dict[str, Any]) -> dict[str, Any]:
        requests = self._requests(payload)
        result = self.service.ensure_many(
            requests,
            wait=bool(payload.get("wait", True)),
            timeout_sec=(
                None
                if payload.get("timeout_sec") is None
                else float(payload.get("timeout_sec", 0.0) or 0.0)
            ),
        )
        return {"result": _ensure_to_payload(result)}

    @staticmethod
    def _requests(payload: Mapping[str, Any]) -> list[TeacherAnnotationRequest]:
        return [
            _request_from_payload(dict(item))
            for item in list(payload.get("requests", []) or [])
            if isinstance(item, Mapping)
        ]


class RemoteTeacherAnnotationService:
    """TeacherAnnotationService-compatible client for edge worker processes."""

    def __init__(self, endpoint: str, *, timeout_sec: float = 600.0) -> None:
        self.endpoint = str(endpoint)
        self.timeout_sec = float(timeout_sec)
        self.label_cache = None
        self.log_internal_ids = False
        self._metadata_cache: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        if self._metadata_cache is None:
            response = post_json(
                self.endpoint,
                "/teacher_annotation/metadata",
                {},
                timeout=self.timeout_sec,
            )
            self._metadata_cache = dict(response.get("metadata", {}) or {})
        return dict(self._metadata_cache)

    def submit_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> TeacherAnnotationSubmitResult:
        response = post_json(
            self.endpoint,
            "/teacher_annotation/submit",
            {"requests": [_request_to_payload(item) for item in list(requests or [])]},
            timeout=self.timeout_sec,
        )
        return _submit_from_payload(dict(response.get("result", {}) or {}))

    def ensure_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
        *,
        wait: bool,
        timeout_sec: float | None,
    ) -> TeacherAnnotationEnsureResult:
        response = post_json(
            self.endpoint,
            "/teacher_annotation/ensure",
            {
                "requests": [_request_to_payload(item) for item in list(requests or [])],
                "wait": bool(wait),
                "timeout_sec": timeout_sec,
            },
            timeout=self.timeout_sec,
        )
        return _ensure_from_payload(dict(response.get("result", {}) or {}))
