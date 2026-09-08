from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TeacherAnnotationRetryableError(RuntimeError):
    """Raised when annotation should be retried by the caller later."""


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_digest(payload: object) -> str:
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


_TARGET_LABEL_METADATA_KEYS = (
    "model_id",
    "model_version",
    "label_schema",
    "class_names",
    "num_classes",
    "rfdetr_head_num_classes",
    "yolo_head_num_classes",
    "tinynext_head_num_classes",
    "head_num_classes",
    "class_logits",
)


def _normalise_cache_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_cache_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_cache_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _target_label_mapping_payload(
    model_id: object,
    metadata: Mapping[str, Any] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"request_model_id": str(model_id)}
    request_metadata = metadata if isinstance(metadata, Mapping) else {}

    target_model_metadata = request_metadata.get("target_model_metadata")
    if isinstance(target_model_metadata, Mapping):
        for key in _TARGET_LABEL_METADATA_KEYS:
            if key in target_model_metadata and target_model_metadata[key] is not None:
                payload[key] = _normalise_cache_value(target_model_metadata[key])

    if "include_empty" in request_metadata:
        payload["include_empty"] = bool(request_metadata.get("include_empty"))

    return payload


@dataclass(frozen=True)
class TeacherLabelCacheKey:
    edge_id: str
    image_sha1: str
    teacher_model_name: str
    teacher_weights_fingerprint: str
    teacher_label_schema: str
    teacher_num_classes: int
    teacher_annotation_threshold: float
    label_coordinate_space: str
    target_label_mapping: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        return {
            "edge_id": str(self.edge_id),
            "image_sha1": str(self.image_sha1),
            "teacher_model_name": str(self.teacher_model_name),
            "teacher_weights_fingerprint": str(self.teacher_weights_fingerprint),
            "teacher_label_schema": str(self.teacher_label_schema),
            "teacher_num_classes": int(self.teacher_num_classes),
            "teacher_annotation_threshold": float(self.teacher_annotation_threshold),
            "label_coordinate_space": str(self.label_coordinate_space),
            "target_label_mapping": _normalise_cache_value(self.target_label_mapping),
        }

    @property
    def digest(self) -> str:
        return _stable_digest(self.payload())

    @property
    def teacher_model_fingerprint(self) -> str:
        return str(self.teacher_weights_fingerprint or self.teacher_model_name or "unknown")


@dataclass(frozen=True)
class TeacherAnnotationRequest:
    sample_id: str
    edge_id: str | int
    model_id: str
    image_path: str
    image_sha1: str
    teacher_model_name: str
    teacher_weights_fingerprint: str
    teacher_label_schema: str
    teacher_num_classes: int
    teacher_annotation_threshold: float
    label_coordinate_space: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def cache_key(self) -> TeacherLabelCacheKey:
        return TeacherLabelCacheKey(
            edge_id=str(self.edge_id),
            image_sha1=str(self.image_sha1),
            teacher_model_name=str(self.teacher_model_name),
            teacher_weights_fingerprint=str(self.teacher_weights_fingerprint),
            teacher_label_schema=str(self.teacher_label_schema),
            teacher_num_classes=int(self.teacher_num_classes),
            teacher_annotation_threshold=float(self.teacher_annotation_threshold),
            label_coordinate_space=str(self.label_coordinate_space),
            target_label_mapping=_target_label_mapping_payload(
                self.model_id,
                self.metadata,
            ),
        )


class TeacherAnnotationStatus(str, Enum):
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    SUBMITTED = "submitted"
    DUPLICATE = "duplicate"
    WORKER_RESULT = "worker_result"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


@dataclass(frozen=True)
class TeacherAnnotationResult:
    request: TeacherAnnotationRequest
    status: TeacherAnnotationStatus
    labels: Mapping[str, Any] | None = None
    error: str | None = None

    @property
    def sample_id(self) -> str:
        return str(self.request.sample_id)

    @property
    def cache_key(self) -> str:
        return self.request.cache_key().digest


@dataclass
class TeacherAnnotationSubmitResult:
    requested_samples: int
    cache_hits: int = 0
    cache_misses: int = 0
    submitted: int = 0
    duplicate: int = 0
    failed_count: int = 0
    results: list[TeacherAnnotationResult] = field(default_factory=list)
    cache_read_time: float = 0.0

    @property
    def labels_by_sample_id(self) -> dict[str, dict[str, Any]]:
        return {
            result.sample_id: dict(result.labels or {})
            for result in self.results
            if result.labels is not None
        }


@dataclass
class TeacherAnnotationEnsureResult:
    requested_samples: int
    cache_hits: int = 0
    cache_misses: int = 0
    submitted: int = 0
    waited_sec: float = 0.0
    unresolved_count: int = 0
    annotation_time: float = 0.0
    cache_read_time: float = 0.0
    cache_write_time: float = 0.0
    teacher_batch_size: int = 0
    teacher_batches: int = 0
    batch_fallback_count: int = 0
    oom_retry_count: int = 0
    failed_count: int = 0
    retryable_errors_by_sample_id: dict[str, str] = field(default_factory=dict)
    results: list[TeacherAnnotationResult] = field(default_factory=list)
    unresolved_requests: list[TeacherAnnotationRequest] = field(default_factory=list)

    @property
    def labels_by_sample_id(self) -> dict[str, dict[str, Any]]:
        return {
            result.sample_id: dict(result.labels or {})
            for result in self.results
            if result.labels is not None
        }

    @property
    def unresolved_sample_ids(self) -> list[str]:
        return [str(request.sample_id) for request in self.unresolved_requests]

    @property
    def retryable_count(self) -> int:
        return len(self.retryable_errors_by_sample_id)
