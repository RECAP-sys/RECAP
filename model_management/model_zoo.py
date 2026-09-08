"""Compatibility facade for detection model construction.

Model-family behavior lives under :mod:`model_management.detectors`.  This
module keeps the historic public import surface stable for edge/cloud code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from model_management.detectors import legacy_model_zoo as _legacy
from model_management.detectors.detr import DETRDetectionModel
from model_management.detectors.registry import (
    get_backend_by_name,
    get_backend_for_model,
    list_backends,
)
from model_management.detectors.rfdetr import (
    RFDETRDetectionModel,
    ensure_rfdetr_serialization_state,
    has_compatible_rfdetr_cache_state,
    infer_rfdetr_state_dict_num_classes,
)
from model_management.detectors.rtdetr import RTDETRDetectionModel
from model_management.detectors.tinynext import (
    infer_tinynext_state_dict_num_classes,
)
from model_management.detectors.yolo import (
    YOLODetectionModel,
    infer_ultralytics_state_dict_num_classes,
)

COCO_80_TO_91 = _legacy.COCO_80_TO_91

_DETECTION_THRESHOLD_LOW_BUFFER = _legacy._DETECTION_THRESHOLD_LOW_BUFFER
_DETECTION_THRESHOLD_HIGH_BUFFER = _legacy._DETECTION_THRESHOLD_HIGH_BUFFER

get_models_dir = _legacy.get_models_dir
get_model_artifact_path = _legacy.get_model_artifact_path
get_detection_thresholds = _legacy.get_detection_thresholds
get_model_detection_thresholds = _legacy.get_model_detection_thresholds
set_model_detection_thresholds = _legacy.set_model_detection_thresholds
ensure_detection_threshold_state = _legacy.ensure_detection_threshold_state
invalidate_wrapper_predictor = _legacy.invalidate_wrapper_predictor
model_has_roi_heads = _legacy.model_has_roi_heads

# Compatibility helpers used by server-side cache inspection code.
_normalise_model_name = _legacy._normalise_model_name
_load_tinynext_checkpoint = _legacy._load_tinynext_checkpoint
_extract_tinynext_checkpoint_state_dict = _legacy._extract_tinynext_checkpoint_state_dict
_load_rfdetr_checkpoint = _legacy._load_rfdetr_checkpoint
_extract_rfdetr_checkpoint_state_dict = _legacy._extract_rfdetr_checkpoint_state_dict
_load_ultralytics_checkpoint = _legacy._load_ultralytics_checkpoint
_extract_ultralytics_checkpoint_state_dict = _legacy._extract_ultralytics_checkpoint_state_dict
_infer_ultralytics_checkpoint_num_classes = _legacy._infer_ultralytics_checkpoint_num_classes


def build_detection_model(
    name: str,
    num_classes: int = 91,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    weights_path: Optional[str] = None,
    confidence: float = 0.01,
    **kwargs: Any,
) -> nn.Module:
    try:
        backend = get_backend_by_name(name)
    except KeyError as exc:
        raise ValueError(
            f"Unknown detection model: '{name}'.  Available: {list_available_models()}"
        ) from exc
    return backend.build(
        name,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
        weights_path=weights_path,
        confidence=confidence,
        **kwargs,
    )


def ensure_local_model_artifact(name: str) -> Path:
    try:
        return get_backend_by_name(name).ensure_local_model_artifact(name)
    except KeyError:
        return _legacy.ensure_local_model_artifact(name)


def list_available_models() -> List[str]:
    names: list[str] = []
    for backend in list_backends():
        names.extend(backend.list_model_names())
    return sorted(set(names))


def build_model_sample_input(
    model_or_name,
    *,
    image_size: Tuple[int, int] = (224, 224),
    device: str | torch.device = "cpu",
):
    return _legacy.build_model_sample_input(
        model_or_name,
        image_size=image_size,
        device=device,
    )


def is_wrapper_model(model_or_name) -> bool:
    return _legacy.is_wrapper_model(model_or_name)


def get_model_family(name: str) -> str:
    return _legacy.get_model_family(name)


def get_backend_family_for_model(model: torch.nn.Module) -> str:
    return get_backend_for_model(model).family


__all__ = [
    "COCO_80_TO_91",
    "YOLODetectionModel",
    "DETRDetectionModel",
    "RFDETRDetectionModel",
    "RTDETRDetectionModel",
    "build_detection_model",
    "build_model_sample_input",
    "ensure_detection_threshold_state",
    "ensure_local_model_artifact",
    "ensure_rfdetr_serialization_state",
    "get_backend_family_for_model",
    "get_detection_thresholds",
    "get_model_artifact_path",
    "get_model_detection_thresholds",
    "get_model_family",
    "get_models_dir",
    "has_compatible_rfdetr_cache_state",
    "infer_rfdetr_state_dict_num_classes",
    "infer_tinynext_state_dict_num_classes",
    "infer_ultralytics_state_dict_num_classes",
    "invalidate_wrapper_predictor",
    "is_wrapper_model",
    "list_available_models",
    "model_has_roi_heads",
    "set_model_detection_thresholds",
]
