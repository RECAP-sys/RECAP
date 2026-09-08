from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from cloud.annotation.service import TeacherAnnotationService
from cloud.annotation.types import TeacherAnnotationRequest, TeacherAnnotationRetryableError


@dataclass(frozen=True)
class RawFrameAnnotationSample:
    sample_id: str
    edge_id: int | str
    model_id: str
    raw_frame: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CloudBatchTeacherAnnotator:
    """Batch teacher annotation facade backed by the shared annotation service."""

    def __init__(
        self,
        *,
        service: TeacherAnnotationService,
        teacher_model_name: str,
        teacher_weights_fingerprint: str,
        teacher_label_schema: str = "coco_91",
        teacher_num_classes: int = 91,
        teacher_annotation_threshold: float = 0.5,
        label_coordinate_space: str = "original_xyxy",
        wait_timeout_sec: float | None = None,
        staging_root_dir: str | Path | None = None,
        owned_worker: object | None = None,
        manages_gpu_lease: bool = False,
    ) -> None:
        self.service = service
        self.teacher_model_name = str(teacher_model_name or "rtdetr_x")
        self.teacher_weights_fingerprint = str(
            teacher_weights_fingerprint or _stable_fingerprint(
                {
                    "teacher_model_name": self.teacher_model_name,
                    "teacher_label_schema": teacher_label_schema,
                    "teacher_num_classes": int(teacher_num_classes or 91),
                }
            )
        )
        self.teacher_label_schema = str(teacher_label_schema or "coco_91")
        self.teacher_num_classes = int(teacher_num_classes or 91)
        self.teacher_annotation_threshold = float(teacher_annotation_threshold)
        self.label_coordinate_space = str(label_coordinate_space or "original_xyxy")
        self.wait_timeout_sec = wait_timeout_sec
        cache_root = getattr(getattr(service, "label_cache", None), "root_dir", "")
        self.staging_root_dir = Path(
            staging_root_dir or cache_root or "./cache/teacher_label_cache"
        ) / "raw_frame_inputs"
        self._owned_worker = owned_worker
        self.manages_gpu_lease = bool(manages_gpu_lease)
        self.last_ensure_result = None

    def close(self) -> None:
        stop = getattr(self._owned_worker, "stop", None)
        if callable(stop):
            stop()

    def annotate_raw_frames(
        self,
        samples: Sequence[RawFrameAnnotationSample | Mapping[str, Any] | object],
        *,
        threshold: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        sample_list = [_coerce_sample(sample) for sample in list(samples or [])]
        sample_list = [sample for sample in sample_list if sample.raw_frame]
        if not sample_list:
            self.last_ensure_result = None
            return {}

        annotation_threshold = (
            self.teacher_annotation_threshold if threshold is None else float(threshold)
        )
        requests = self._requests_for_samples(
            sample_list,
            threshold=annotation_threshold,
        )
        if not requests:
            self.last_ensure_result = None
            return {}
        result = self.service.ensure_many(
            requests,
            wait=True,
            timeout_sec=self.wait_timeout_sec,
        )
        self.last_ensure_result = result
        labels = {
            str(sample_id): dict(value)
            for sample_id, value in result.labels_by_sample_id.items()
        }
        if result.unresolved_count:
            if result.retryable_count:
                reasons = "; ".join(
                    f"{sample_id}: {reason}"
                    for sample_id, reason in sorted(
                        result.retryable_errors_by_sample_id.items()
                    )[:5]
                )
                raise TeacherAnnotationRetryableError(
                    "teacher annotation deferred for "
                    f"{result.retryable_count} sample(s): {reasons}"
                )
            missing = ", ".join(result.unresolved_sample_ids[:5])
            logger.warning(
                "cloud_batch_teacher_annotation_unresolved requested={} unresolved={}",
                result.requested_samples,
                result.unresolved_count,
            )
            raise RuntimeError(
                "teacher annotation unresolved for "
                f"{result.unresolved_count} sample(s): {missing}"
            )
        return labels

    def _requests_for_samples(
        self,
        samples: Sequence[RawFrameAnnotationSample],
        *,
        threshold: float,
    ) -> list[TeacherAnnotationRequest]:
        requests: list[TeacherAnnotationRequest] = []
        for sample in samples:
            raw_frame = bytes(sample.raw_frame or b"")
            image_sha1 = hashlib.sha1(raw_frame).hexdigest()
            image_path = self._staged_frame_path(image_sha1)
            _write_staged_frame(image_path, raw_frame)
            metadata = dict(sample.metadata or {})
            metadata.setdefault("include_empty", True)
            requests.append(
                TeacherAnnotationRequest(
                    sample_id=str(sample.sample_id),
                    edge_id=sample.edge_id,
                    model_id=str(sample.model_id or ""),
                    image_path=str(image_path),
                    image_sha1=image_sha1,
                    teacher_model_name=self.teacher_model_name,
                    teacher_weights_fingerprint=self.teacher_weights_fingerprint,
                    teacher_label_schema=self.teacher_label_schema,
                    teacher_num_classes=self.teacher_num_classes,
                    teacher_annotation_threshold=float(threshold),
                    label_coordinate_space=self.label_coordinate_space,
                    metadata=metadata,
                )
            )
        return requests

    def _staged_frame_path(self, image_sha1: str) -> Path:
        digest = str(image_sha1)
        return self.staging_root_dir / digest[:2] / f"{digest}.jpg"


def _write_staged_frame(path: Path, raw_frame: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_frame)
        os.replace(tmp_name, path)
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass


def _coerce_sample(sample: RawFrameAnnotationSample | Mapping[str, Any] | object):
    if isinstance(sample, RawFrameAnnotationSample):
        return sample
    if isinstance(sample, Mapping):
        return RawFrameAnnotationSample(
            sample_id=str(sample.get("sample_id", sample.get("frame_id", ""))),
            edge_id=sample.get("edge_id", ""),
            model_id=str(sample.get("model_id", sample.get("model_name", "")) or ""),
            raw_frame=bytes(sample.get("raw_frame", b"") or b""),
            metadata=dict(sample.get("metadata", {}) or {}),
        )
    return RawFrameAnnotationSample(
        sample_id=str(getattr(sample, "sample_id", getattr(sample, "frame_id", ""))),
        edge_id=getattr(sample, "edge_id", ""),
        model_id=str(getattr(sample, "model_id", getattr(sample, "model_name", "")) or ""),
        raw_frame=bytes(getattr(sample, "raw_frame", b"") or b""),
        metadata=dict(getattr(sample, "metadata", {}) or {}),
    )


def _stable_fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
