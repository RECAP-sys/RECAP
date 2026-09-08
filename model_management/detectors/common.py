from __future__ import annotations

from typing import Any

import torch

from model_management.detectors import legacy_model_zoo as _zoo
from model_management.detectors import legacy_split_model_adapters as _split

COCO_80_TO_91 = _zoo.COCO_80_TO_91
COCO_91_TO_80 = _split.COCO_91_TO_80

normalise_model_name = _zoo._normalise_model_name

get_models_dir = _zoo.get_models_dir
get_model_artifact_path = _zoo.get_model_artifact_path
ensure_local_model_artifact = _zoo.ensure_local_model_artifact

get_detection_thresholds = _zoo.get_detection_thresholds
ensure_detection_threshold_state = _zoo.ensure_detection_threshold_state
get_model_detection_thresholds = _zoo.get_model_detection_thresholds
set_model_detection_thresholds = _zoo.set_model_detection_thresholds

is_anchor_detector = _split._is_anchor_detector
is_ultralytics_detection_core = _split._is_ultralytics_detection_core

iter_tensors = _split._iter_tensors
contiguous_tensor_tree = _split._contiguous_tensor_tree
first_tensor_device = _split._first_tensor_device


def empty_detection_result(device: torch.device) -> list[dict[str, torch.Tensor]]:
    return _split._empty_detection_result(device)


def summarize_split_runtime_observables(
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
