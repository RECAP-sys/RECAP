from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_management.detectors import legacy_model_zoo as _zoo
from model_management.detectors import legacy_split_model_adapters as _split
from model_management.detectors.base import DetectionBackend

RTDETRDetectionModel = _zoo.RTDETRDetectionModel
_RTDETR_MODELS = _zoo._RTDETR_MODELS


class RTDETRBackend(DetectionBackend):
    family = "rtdetr"

    def matches_name(self, name: str) -> bool:
        return _zoo._normalise_model_name(name) in _RTDETR_MODELS

    def matches_model(self, model: torch.nn.Module) -> bool:
        return isinstance(model, RTDETRDetectionModel)

    def list_model_names(self) -> list[str]:
        return list(_RTDETR_MODELS)

    def build(
        self,
        name: str,
        *,
        num_classes: int = 91,
        pretrained: bool = True,
        device: str | torch.device = "cpu",
        weights_path: str | None = None,
        confidence: float = 0.01,
        **kwargs: Any,
    ) -> torch.nn.Module:
        return _zoo.build_detection_model(
            name,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device,
            weights_path=weights_path,
            confidence=confidence,
            **kwargs,
        )

    def ensure_local_model_artifact(self, name: str) -> Path:
        return _zoo.ensure_local_model_artifact(name)

    def get_split_runtime_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return _split.get_split_runtime_model(model)

    def build_split_runtime_sample_input(
        self,
        model: torch.nn.Module,
        *,
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device = "cpu",
    ) -> Any:
        return _split.build_split_runtime_sample_input(model, image_size=image_size, device=device)

    def get_split_runtime_input_resize_mode(self, model: torch.nn.Module) -> str | None:
        return _split.get_split_runtime_input_resize_mode(model)

    def prepare_split_runtime_input(
        self,
        model: torch.nn.Module,
        frame: np.ndarray,
        *,
        device: str | torch.device,
        input_tensor_shape: tuple[int, ...] | list[int] | None = None,
    ) -> Any:
        return _split.prepare_split_runtime_input(
            model,
            frame,
            device=device,
            input_tensor_shape=input_tensor_shape,
        )

    def postprocess_split_runtime_output(
        self,
        model: torch.nn.Module,
        outputs: Any,
        *,
        threshold: float,
        model_input: Any | None = None,
        orig_image: np.ndarray | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        return _split.postprocess_split_runtime_output(
            model,
            outputs,
            threshold=threshold,
            model_input=model_input,
            orig_image=orig_image,
        )

    def build_split_training_loss(self, model: torch.nn.Module):
        return _split.build_split_training_loss(model)

    def summarize_split_runtime_observables(
        self,
        model: torch.nn.Module,
        outputs: Any,
        split_payload: Any | None = None,
        *,
        include_feature_spectral_entropy: bool = True,
    ) -> dict[str, float | None]:
        return _split.summarize_split_runtime_observables(
            model,
            outputs,
            split_payload,
            include_feature_spectral_entropy=include_feature_spectral_entropy,
        )

    def get_detection_thresholds(self, model_name: str) -> tuple[float, float]:
        return _zoo.get_detection_thresholds(model_name)


BACKEND = RTDETRBackend()
