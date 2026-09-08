#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.drift_detection_validity.experiment_io import (  # noqa: E402
    output_dir,
    require_bool,
    require_float,
    require_mapping,
    require_text,
    resolve_project_path,
)


def _optional_checkpoint(path_value: Any) -> Path | None:
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None
    path = resolve_project_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Configured checkpoint is unavailable: {path}")
    return path


class RealWeatherBackend:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        run_cfg = require_mapping(config, "run")
        data_cfg = require_mapping(config, "data")
        self.device = require_text(run_cfg, "device", context="run")
        video_path = resolve_project_path(require_text(data_cfg, "video_path", context="data"))
        if not video_path.exists():
            raise FileNotFoundError(f"Configured data.video_path is unavailable: {video_path}")
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for real weather evaluation.") from exc
        self.cv2 = cv2
        self.video_path = video_path
        self._capture = cv2.VideoCapture(str(video_path))
        if not self._capture.isOpened():
            raise FileNotFoundError(f"Could not open configured video: {video_path}")
        self._last_source_id: int | None = None
        self._last_frame: np.ndarray | None = None
        self.student = None
        self.teacher = None
        self.splitter = None
        self._load_models()

    def close(self) -> None:
        self._capture.release()

    def _load_models(self) -> None:
        model_cfg = require_mapping(self.config, "models")
        student_checkpoint = _optional_checkpoint(model_cfg.get("student_checkpoint"))
        teacher_checkpoint = _optional_checkpoint(model_cfg.get("teacher_checkpoint"))
        import model_management.object_detection as object_detection_runtime
        from model_management.object_detection import Object_Detection

        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
        object_detection_runtime.device = device
        threshold = require_float(model_cfg, "confidence_threshold", context="models")
        student_cfg = SimpleNamespace(
            lightweight=require_text(model_cfg, "student_model", context="models"),
            weights_path=str(student_checkpoint) if student_checkpoint else None,
            tinynext_input_size=640,
            final_detection_threshold=threshold,
        )
        teacher_cfg = SimpleNamespace(
            golden=require_text(model_cfg, "teacher_model", context="models"),
            weights_path=str(teacher_checkpoint) if teacher_checkpoint else None,
            tinynext_input_size=640,
            final_detection_threshold=threshold,
        )
        self.student = Object_Detection(student_cfg, "small inference")
        self.teacher = Object_Detection(teacher_cfg, "large inference")

    def frame(self, source_frame_id: int) -> np.ndarray:
        if self._last_source_id == int(source_frame_id) and self._last_frame is not None:
            return self._last_frame.copy()
        self._capture.set(self.cv2.CAP_PROP_POS_FRAMES, int(source_frame_id))
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read source frame {source_frame_id}.")
        self._last_source_id = int(source_frame_id)
        self._last_frame = frame.copy()
        return frame

    def _ensure_splitter(self, frame: np.ndarray) -> None:
        split_cfg = require_mapping(self.config, "split_boundary")
        if not require_bool(split_cfg, "enabled", context="split_boundary") or self.splitter is not None:
            return
        if self.student is None:
            raise RuntimeError("Student model must be loaded before split runtime setup.")
        from model_management.fixed_split import SplitConstraints, load_or_compute_fixed_split_plan
        from model_management.model_zoo import get_model_family
        from model_management.split_model_adapters import get_split_runtime_input_resize_mode
        from model_management.universal_model_split import UniversalModelSplitter

        sample_input = self.student.prepare_splitter_input(frame)
        split_model = self.student.get_split_runtime_model()
        splitter = UniversalModelSplitter(device=self.device)
        split_point = split_cfg.get("split_point")
        if split_point:
            splitter.trace(
                split_model,
                sample_input,
                boundary=str(split_point),
                model_name=self.student.model_name,
                model_family=get_model_family(self.student.model_name),
            )
        else:
            records_dir = output_dir(self.config) / "records"
            plan_path = records_dir / "fixed_split_plan.json"
            constraints = SplitConstraints()
            resize_mode = get_split_runtime_input_resize_mode(split_model)
            if not resize_mode:
                raise RuntimeError("Split runtime input resize mode is unavailable.")
            load_or_compute_fixed_split_plan(
                split_model,
                constraints,
                sample_input=sample_input,
                device=self.device,
                model_name=self.student.model_name,
                cache_path=str(plan_path),
                splitter=splitter,
                input_resize_mode=resize_mode,
                validate_cached_plan=False,
            )
        splitter.prepare_inference_replay(sample_input)
        self.splitter = splitter

    def infer(
        self,
        frame: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        if self.student is None or self.teacher is None:
            raise RuntimeError("Real weather backend models are not loaded.")
        self._ensure_splitter(frame)
        artifacts = self.student.infer_sample(frame, splitter=self.splitter)
        student_prediction = {
            "boxes": artifacts.final_detection_boxes or [],
            "labels": artifacts.final_detection_labels or [],
            "scores": artifacts.final_detection_scores or [],
            "output_entropy": artifacts.logit_entropy,
            "logit_entropy": artifacts.logit_entropy,
        }
        model_cfg = require_mapping(self.config, "models")
        teacher_threshold = require_float(model_cfg, "confidence_threshold", context="models")
        boxes, labels, scores = self.teacher.large_inference(frame, threshold=teacher_threshold)
        teacher_prediction = {
            "boxes": boxes or [],
            "labels": labels or [],
            "scores": scores or [],
        }
        return student_prediction, teacher_prediction, artifacts.intermediate
