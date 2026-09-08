from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import torch
from loguru import logger

from cloud.annotation import TeacherAnnotationRequest, TeacherAnnotationService
from cloud.orchestration.fixed_split_dependencies import (
    POOL_LABEL_COORDINATE_SPACE,
    _file_sha1,
    _json_fingerprint,
)
from cloud.training.proxy_metadata import (
    class_names_from_metadata as _class_names_from_metadata,
)
from cloud.training.proxy_metadata import (
    coerce_positive_int as _coerce_positive_int,
)
from cloud.training.proxy_metadata import (
    is_low_quality_trigger_sample as _is_low_quality_trigger_sample,
)
from cloud.training.proxy_metadata import (
    label_name_from_schema as _label_name_from_schema,
)
from cloud.training.proxy_metadata import (
    normalise_class_name as _normalise_class_name,
)
from cloud.training.proxy_metadata import (
    normalise_label_schema as _normalise_label_schema,
)
from cloud.training.proxy_metadata import (
    runtime_image_size_from_metadata as _runtime_image_size_from_metadata,
)
from cloud.training.proxy_metadata import (
    runtime_input_tensor_shape_from_metadata as _runtime_input_tensor_shape_from_metadata,
)
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from model_management.model_info import COCO_INSTANCE_CATEGORY_NAMES, model_lib
from model_management.split_model_adapters import prepare_split_runtime_input
from model_management.split_runtime.torchlens_forward_guard import torchlens_forward_guard


class TeacherAnnotationStage:
    def __init__(
        self,
        service: TeacherAnnotationService,
        *,
        wait_timeout_sec: float,
    ) -> None:
        self.service = service
        self.wait_timeout_sec = float(wait_timeout_sec)

    def ensure_low_quality(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> dict[str, dict[str, object]]:
        ensure_result = self.service.ensure_many(
            list(requests),
            wait=True,
            timeout_sec=self.wait_timeout_sec,
        )
        if ensure_result.unresolved_count:
            logger.info(
                "[TeacherAnnotation][Ensure] deferring unresolved low-quality "
                "samples before canonical staging: unresolved_count={}.",
                ensure_result.unresolved_count,
            )
            log_diagnostic_debug(
                self.service,
                "[TeacherAnnotation][Ensure] deferred sample diagnostics",
                lambda: {
                    "sample_ids_preview": [
                        str(sample_id) for sample_id in ensure_result.unresolved_sample_ids[:5]
                    ]
                },
            )
        return {
            str(sample_id): dict(labels)
            for sample_id, labels in ensure_result.labels_by_sample_id.items()
        }


class TeacherAnnotationMixin:
    def _reserve_teacher_ticket(self) -> int:
        queue_state = self._teacher_queue_state
        with queue_state.condition:
            ticket = int(queue_state.next_ticket)
            queue_state.next_ticket += 1
            queue_state.ticket_states[ticket] = "reserved"
            return ticket

    def _advance_teacher_queue_locked(self) -> None:
        queue_state = self._teacher_queue_state
        while queue_state.ticket_states.get(int(queue_state.serving_ticket)) in {"done", "skipped"}:
            queue_state.ticket_states.pop(int(queue_state.serving_ticket), None)
            queue_state.serving_ticket = int(queue_state.serving_ticket) + 1

    def _set_current_teacher_ticket(self, ticket: int | None) -> None:
        queue_state = self._teacher_queue_state
        if ticket is None:
            if hasattr(queue_state.ticket_local, "ticket"):
                delattr(queue_state.ticket_local, "ticket")
            return
        queue_state.ticket_local.ticket = int(ticket)

    def _current_teacher_ticket(self) -> int:
        queue_state = self._teacher_queue_state
        ticket = getattr(queue_state.ticket_local, "ticket", None)
        if ticket is not None:
            with queue_state.condition:
                state = queue_state.ticket_states.get(int(ticket))
            if state in {"reserved", "active"}:
                return int(ticket)
            delattr(queue_state.ticket_local, "ticket")
        ticket = self._reserve_teacher_ticket()
        queue_state.ticket_local.ticket = int(ticket)
        logger.warning(
            "[FixedSplitCL] Reserved ad-hoc teacher slot outside training-job scope."
        )
        log_diagnostic_debug(
            self,
            "[FixedSplitCL] ad-hoc teacher ticket diagnostics",
            lambda: {"ticket": ticket},
        )
        return int(ticket)

    @contextmanager
    def _teacher_annotation_scope(
        self,
        stage_label: str,
        *,
        sample_count: int | None = None,
    ):
        """Serialize only teacher inference globally while preserving batched requests."""
        queue_state = self._teacher_queue_state
        ticket = self._current_teacher_ticket()
        wait_started = time.perf_counter()
        logger.info(
            "[FixedSplitCL] waiting for teacher slot: stage={} samples={}.",
            stage_label,
            sample_count,
        )
        with queue_state.condition:
            while True:
                self._advance_teacher_queue_locked()
                state = queue_state.ticket_states.get(int(ticket))
                if state is None:
                    raise RuntimeError(
                        f"Teacher ticket {ticket} is no longer pending for stage {stage_label!r}."
                    )
                if int(ticket) == int(queue_state.serving_ticket) and state == "reserved":
                    queue_state.ticket_states[int(ticket)] = "active"
                    break
                queue_state.condition.wait()
        wait_elapsed = time.perf_counter() - wait_started
        logger.info(
            "[FixedSplitCL] acquired teacher slot: stage={} wait_time={:.3f}s.",
            stage_label,
            wait_elapsed,
        )
        execution_started = time.perf_counter()
        try:
            with torchlens_forward_guard():
                yield
        finally:
            execution_elapsed = time.perf_counter() - execution_started
            with queue_state.condition:
                if queue_state.ticket_states.get(int(ticket)) == "active":
                    queue_state.ticket_states[int(ticket)] = "done"
                self._advance_teacher_queue_locked()
                queue_state.condition.notify_all()
            logger.info(
                "[FixedSplitCL] released teacher slot: stage={} "
                "wait_time={:.3f}s execution_time={:.3f}s.",
                stage_label,
                wait_elapsed,
                execution_elapsed,
            )
            log_diagnostic_debug(
                self,
                "[FixedSplitCL] teacher slot diagnostics",
                lambda: {"ticket": ticket, "stage": stage_label},
            )

    def _teacher_label_schema(self) -> str:
        metadata = getattr(self, "teacher_annotation_metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("teacher_label_schema"):
            return _normalise_label_schema(metadata.get("teacher_label_schema"))
        teacher_model = getattr(getattr(self, "large_od", None), "model", None)
        return _normalise_label_schema(getattr(teacher_model, "label_schema", "coco_91"))

    def _teacher_model_name(self) -> str:
        metadata = getattr(self, "teacher_annotation_metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("teacher_model_name"):
            return str(metadata.get("teacher_model_name"))
        return str(
            getattr(getattr(self, "large_od", None), "model_name", "")
            or getattr(self.config, "golden", "")
            or "unknown"
        )

    def _teacher_weights_fingerprint(self) -> str:
        if self._teacher_weights_fingerprint_cache:
            return self._teacher_weights_fingerprint_cache
        metadata = getattr(self, "teacher_annotation_metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("teacher_weights_fingerprint"):
            self._teacher_weights_fingerprint_cache = str(
                metadata.get("teacher_weights_fingerprint")
            )
            return self._teacher_weights_fingerprint_cache
        model_name = self._teacher_model_name()
        model_info = model_lib.get(model_name, {})
        artifact_name = str(model_info.get("model_path", "") or "").strip()
        candidate_paths = []
        if artifact_name:
            candidate_paths.append(os.path.join(self.weight_folder, artifact_name))
        model = getattr(getattr(self, "large_od", None), "model", None)
        for attr_name in ("weights_path", "ckpt_path", "checkpoint_path"):
            value = getattr(model, attr_name, None)
            if value:
                candidate_paths.append(str(value))
        for path in candidate_paths:
            if path and os.path.exists(path) and os.path.isfile(path):
                try:
                    self._teacher_weights_fingerprint_cache = _file_sha1(path)
                    return self._teacher_weights_fingerprint_cache
                except Exception:
                    continue
        class_names = self._teacher_class_names()
        self._teacher_weights_fingerprint_cache = _json_fingerprint(
            {
                "teacher_model_name": model_name,
                "teacher_label_schema": self._teacher_label_schema(),
                "teacher_class_count": len(class_names),
                "artifact_name": artifact_name,
            }
        )
        return self._teacher_weights_fingerprint_cache

    def _teacher_class_names(self) -> list[str]:
        metadata = getattr(self, "teacher_annotation_metadata", {})
        if isinstance(metadata, Mapping):
            class_names = metadata.get("teacher_class_names")
            if isinstance(class_names, Mapping):
                return _class_names_from_metadata({"class_names": class_names})
            if isinstance(class_names, (list, tuple)):
                return [str(item) for item in class_names]
        teacher_model = getattr(getattr(self, "large_od", None), "model", None)
        class_names = getattr(teacher_model, "class_names", None)
        if isinstance(class_names, Mapping):
            return _class_names_from_metadata({"class_names": class_names})
        if isinstance(class_names, (list, tuple)):
            return [str(item) for item in class_names]
        return []

    def _teacher_num_classes(self) -> int:
        metadata = getattr(self, "teacher_annotation_metadata", {})
        if isinstance(metadata, Mapping):
            value = _coerce_positive_int(metadata.get("teacher_num_classes"))
            if value is not None:
                return value
        teacher_model = getattr(getattr(self, "large_od", None), "model", None)
        for attr_name in ("num_classes", "nc"):
            value = _coerce_positive_int(getattr(teacher_model, attr_name, None))
            if value is not None:
                return value
        class_names = self._teacher_class_names()
        if class_names:
            return len(class_names)
        return len(COCO_INSTANCE_CATEGORY_NAMES)

    def _map_teacher_label_for_target(
        self,
        label: object,
        *,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> int | None:
        try:
            label_index = int(label)
        except (TypeError, ValueError):
            return None

        if not isinstance(target_model_metadata, Mapping):
            return label_index

        target_schema = _normalise_label_schema(
            target_model_metadata.get("label_schema"),
        )
        teacher_schema = self._teacher_label_schema()
        if target_schema != "zero_based":
            return label_index

        if teacher_schema == "zero_based":
            return label_index

        target_class_names = _class_names_from_metadata(target_model_metadata)
        if not target_class_names:
            return None

        teacher_name = _label_name_from_schema(
            label_index,
            label_schema=teacher_schema,
            class_names=self._teacher_class_names(),
        )
        if teacher_name is None:
            return None

        target_lookup = {
            _normalise_class_name(name): index for index, name in enumerate(target_class_names)
        }
        return target_lookup.get(_normalise_class_name(teacher_name))

    def _build_teacher_targets_from_prediction(
        self,
        pred_boxes,
        pred_class,
        pred_score=None,
        *,
        image_size: tuple[int, int] | list[int] | None = None,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        if pred_boxes is None or pred_class is None:
            return None

        boxes = list(pred_boxes)
        labels = list(pred_class)
        if not boxes or not labels:
            return None

        count = min(len(boxes), len(labels))
        if count <= 0:
            return None

        image_height: float | None = None
        image_width: float | None = None
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            image_height = float(image_size[0])
            image_width = float(image_size[1])
            if image_height <= 0.0 or image_width <= 0.0:
                image_height = None
                image_width = None

        target_boxes: list[list[float]] = []
        target_labels: list[int] = []
        target_scores: list[float] = []
        scores = list(pred_score) if pred_score is not None else None
        for index in range(count):
            try:
                values = [float(value) for value in list(boxes[index])[:4]]
            except (TypeError, ValueError):
                continue
            if len(values) != 4:
                continue
            if image_height is not None and image_width is not None:
                values[0] = max(0.0, min(float(image_width), values[0]))
                values[2] = max(0.0, min(float(image_width), values[2]))
                values[1] = max(0.0, min(float(image_height), values[1]))
                values[3] = max(0.0, min(float(image_height), values[3]))
            if values[2] <= values[0] or values[3] <= values[1]:
                continue
            target_label = self._map_teacher_label_for_target(
                labels[index],
                target_model_metadata=target_model_metadata,
            )
            if target_label is None:
                continue
            target_boxes.append(values)
            target_labels.append(int(target_label))
            if scores is not None and index < len(scores):
                target_scores.append(float(scores[index]))

        if not target_boxes:
            return None

        targets: dict[str, object] = {
            "boxes": target_boxes,
            "labels": target_labels,
        }
        if scores is not None:
            targets["scores"] = target_scores
        return targets

    @staticmethod
    def _runtime_image_size_from_metadata(
        metadata: Mapping[str, object] | None,
    ) -> tuple[int, int] | None:
        return _runtime_image_size_from_metadata(metadata)

    def _prepare_split_runtime_input(
        self,
        model: torch.nn.Module,
        frame,
        *,
        sample_metadata: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ):
        return prepare_split_runtime_input(
            model,
            frame,
            device=self.device if device is None else device,
            input_tensor_shape=_runtime_input_tensor_shape_from_metadata(sample_metadata),
        )

    def _teacher_inference(self, frame, threshold: float | None = None):
        threshold = self.teacher_annotation_threshold if threshold is None else float(threshold)
        try:
            return self.large_od.large_inference(
                frame,
                threshold=threshold,
            )
        except TypeError:
            return self.large_od.large_inference(frame)

    def _teacher_labels_from_request_prediction(
        self,
        request: TeacherAnnotationRequest,
        frame,
        prediction: object,
    ) -> dict[str, object] | None:
        pred_boxes = pred_class = pred_score = None
        if isinstance(prediction, (list, tuple)):
            if len(prediction) >= 1:
                pred_boxes = prediction[0]
            if len(prediction) >= 2:
                pred_class = prediction[1]
            if len(prediction) >= 3:
                pred_score = prediction[2]
        metadata = dict(request.metadata or {})
        target_model_metadata = metadata.get("target_model_metadata")
        return self._build_teacher_targets_from_prediction(
            pred_boxes,
            pred_class,
            pred_score,
            image_size=tuple(int(value) for value in frame.shape[:2]),
            target_model_metadata=(
                target_model_metadata if isinstance(target_model_metadata, Mapping) else None
            ),
        )

    def _build_teacher_targets(
        self,
        frame,
        *,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        pred_boxes, pred_class, pred_score = self._teacher_inference(frame)
        return self._build_teacher_targets_from_prediction(
            pred_boxes,
            pred_class,
            pred_score,
            image_size=tuple(int(value) for value in frame.shape[:2]),
            target_model_metadata=target_model_metadata,
        )

    def _build_teacher_annotation_request(
        self,
        *,
        sample_id: object,
        image_path: str,
        edge_id: int | str | None,
        model_id: str | None,
        target_model_metadata: Mapping[str, object] | None = None,
        include_empty: bool = True,
    ) -> TeacherAnnotationRequest | None:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            image_sha1 = _file_sha1(image_path)
        except Exception as exc:
            logger.warning(
                "[TeacherAnnotation][Submit] skipped unreadable image: reason={}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "[TeacherAnnotation][Submit] unreadable image diagnostics",
                lambda error=exc: {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "error": repr(error),
                },
            )
            return None
        return TeacherAnnotationRequest(
            sample_id=str(sample_id),
            edge_id="" if edge_id is None else edge_id,
            model_id=str(model_id or self.edge_model_name),
            image_path=str(image_path),
            image_sha1=image_sha1,
            teacher_model_name=self._teacher_model_name(),
            teacher_weights_fingerprint=self._teacher_weights_fingerprint(),
            teacher_label_schema=self._teacher_label_schema(),
            teacher_num_classes=self._teacher_num_classes(),
            teacher_annotation_threshold=float(self.teacher_annotation_threshold),
            label_coordinate_space=POOL_LABEL_COORDINATE_SPACE,
            metadata={
                "target_model_metadata": dict(target_model_metadata or {}),
                "include_empty": bool(include_empty),
            },
        )

    def _build_teacher_annotation_requests_from_frame_dir(
        self,
        frame_dir: str,
        sample_ids,
        *,
        edge_id: int | str | None = None,
        model_id: str | None = None,
        missing_raw_message: str | None = None,
        include_empty: bool = True,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> list[TeacherAnnotationRequest]:
        requests: list[TeacherAnnotationRequest] = []
        for sample_id in sample_ids:
            img_path = os.path.join(frame_dir, f"{sample_id}.jpg")
            if not os.path.exists(img_path):
                if missing_raw_message is not None:
                    logger.warning(
                        "[TeacherAnnotation][Submit] skipped sample with missing raw image."
                    )
                    log_diagnostic_debug(
                        self,
                        "[TeacherAnnotation][Submit] missing raw image diagnostics",
                        lambda: {"sample_id": sample_id, "image_path": img_path},
                    )
                continue
            request = self._build_teacher_annotation_request(
                sample_id=sample_id,
                image_path=img_path,
                edge_id=edge_id,
                model_id=model_id,
                target_model_metadata=target_model_metadata,
                include_empty=include_empty,
            )
            if request is not None:
                requests.append(request)
        return requests

    def _build_low_quality_raw_teacher_annotation_requests(
        self,
        *,
        bundle_cache_path: str,
        manifest: Mapping[str, object],
        edge_id: int | str | None,
        model_id: str | None,
        target_model_metadata: Mapping[str, object] | None = None,
    ) -> list[TeacherAnnotationRequest]:
        requests: list[TeacherAnnotationRequest] = []
        for sample in list(manifest.get("samples", []) or []):
            if not isinstance(sample, Mapping):
                continue
            if not _is_low_quality_trigger_sample(manifest, sample):
                continue
            sample_id = str(sample.get("sample_id", "") or "").strip()
            raw_relpath = sample.get("raw_relpath")
            if not sample_id or raw_relpath is None:
                continue
            image_path = os.path.join(
                bundle_cache_path,
                str(raw_relpath).replace("/", os.sep),
            )
            request = self._build_teacher_annotation_request(
                sample_id=sample_id,
                image_path=image_path,
                edge_id=edge_id,
                model_id=model_id,
                target_model_metadata=target_model_metadata,
                include_empty=True,
            )
            if request is not None:
                requests.append(request)
        return requests

    def _submit_low_quality_teacher_annotations(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> None:
        if (
            not self.teacher_annotation_async_enabled
            or not self.teacher_annotation_cache_enabled
            or not requests
        ):
            return
        result = self.teacher_annotation_service.submit_many(list(requests))
        logger.info(
            "[TeacherAnnotation][Submit] low_quality_raw requested_samples={} cache_hits={} "
            "cache_misses={} submitted={} duplicate={} failed_count={}",
            result.requested_samples,
            result.cache_hits,
            result.cache_misses,
            result.submitted,
            result.duplicate,
            result.failed_count,
        )

    def _teacher_annotation_stage(self) -> TeacherAnnotationStage:
        return TeacherAnnotationStage(
            self.teacher_annotation_service,
            wait_timeout_sec=self.teacher_annotation_wait_timeout_sec,
        )

    def _collect_teacher_annotations(
        self,
        frame_dir: str,
        sample_ids,
        *,
        missing_raw_message: str | None = None,
        key_transform=None,
        include_empty: bool = False,
        target_model_metadata: Mapping[str, object] | None = None,
        edge_id: int | str | None = None,
        model_id: str | None = None,
    ) -> dict:
        transform = key_transform or (lambda sample_id: sample_id)
        requests = self._build_teacher_annotation_requests_from_frame_dir(
            frame_dir,
            sample_ids,
            edge_id=edge_id,
            model_id=model_id,
            missing_raw_message=missing_raw_message,
            include_empty=include_empty,
            target_model_metadata=target_model_metadata,
        )
        if not requests:
            return {}
        labels_by_sample_id = self._teacher_annotation_stage().ensure_low_quality(
            requests,
        )
        return {
            transform(sample_id): labels_by_sample_id[str(sample_id)]
            for sample_id in sample_ids
            if str(sample_id) in labels_by_sample_id
        }
