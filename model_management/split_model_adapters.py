from __future__ import annotations

from typing import Any

import numpy as np
import torch

from model_management.detectors import legacy_split_model_adapters as _legacy
from model_management.detectors.registry import get_backend_for_model

TorchvisionAnchorDetectorReplay = _legacy.TorchvisionAnchorDetectorReplay
RFDETRReplay = _legacy.RFDETRReplay

COCO_91_TO_80 = _legacy.COCO_91_TO_80
_RFDETR_PACKED_AUX_OUTPUTS_MARKER = _legacy._RFDETR_PACKED_AUX_OUTPUTS_MARKER

build_anchor_detector_training_loss = _legacy.build_anchor_detector_training_loss
build_ssd_split_training_loss = _legacy.build_ssd_split_training_loss

_pack_rfdetr_aux_outputs = _legacy._pack_rfdetr_aux_outputs
_unpack_rfdetr_aux_outputs = _legacy._unpack_rfdetr_aux_outputs
_patch_rfdetr_decoder_batch_polymorphism = _legacy._patch_rfdetr_decoder_batch_polymorphism


def _backend_or_none(model: torch.nn.Module):
    try:
        return get_backend_for_model(model)
    except KeyError:
        return None


def get_split_runtime_model(model: torch.nn.Module) -> torch.nn.Module:
    backend = _backend_or_none(model)
    if backend is not None:
        return backend.get_split_runtime_model(model)
    return _legacy.get_split_runtime_model(model)


def build_split_runtime_sample_input(
    model: torch.nn.Module,
    *,
    image_size: tuple[int, int] = (224, 224),
    device: str | torch.device = "cpu",
):
    backend = _backend_or_none(model)
    if backend is not None:
        return backend.build_split_runtime_sample_input(
            model,
            image_size=image_size,
            device=device,
        )
    return _legacy.build_split_runtime_sample_input(
        model,
        image_size=image_size,
        device=device,
    )


def get_split_runtime_input_resize_mode(model: torch.nn.Module) -> str | None:
    backend = _backend_or_none(model)
    if backend is not None:
        return backend.get_split_runtime_input_resize_mode(model)
    return _legacy.get_split_runtime_input_resize_mode(model)


def prepare_split_runtime_input(
    model: torch.nn.Module,
    frame: np.ndarray,
    *,
    device: str | torch.device,
    input_tensor_shape: tuple[int, ...] | list[int] | None = None,
):
    backend = _backend_or_none(model)
    if backend is not None:
        return backend.prepare_split_runtime_input(
            model,
            frame,
            device=device,
            input_tensor_shape=input_tensor_shape,
        )
    return _legacy.prepare_split_runtime_input(
        model,
        frame,
        device=device,
        input_tensor_shape=input_tensor_shape,
    )


def postprocess_split_runtime_output(
    model: torch.nn.Module,
    outputs: Any,
    *,
    threshold: float,
    model_input: Any | None = None,
    orig_image: np.ndarray | None = None,
) -> list[dict[str, torch.Tensor]]:
    backend = _backend_or_none(model)
    if backend is not None:
        return backend.postprocess_split_runtime_output(
            model,
            outputs,
            threshold=threshold,
            model_input=model_input,
            orig_image=orig_image,
        )
    return _legacy.postprocess_split_runtime_output(
        model,
        outputs,
        threshold=threshold,
        model_input=model_input,
        orig_image=orig_image,
    )


def summarize_split_runtime_observables(
    model: torch.nn.Module,
    outputs: Any,
    split_payload: Any | None = None,
    *,
    include_feature_spectral_entropy: bool = True,
) -> dict[str, float | None]:
    backend = _backend_or_none(model)
    if backend is not None:
        return backend.summarize_split_runtime_observables(
            model,
            outputs,
            split_payload,
            include_feature_spectral_entropy=include_feature_spectral_entropy,
        )
    return _legacy.summarize_split_runtime_observables(
        model,
        outputs,
        split_payload,
        include_feature_spectral_entropy=include_feature_spectral_entropy,
    )


def build_split_training_loss(model: torch.nn.Module):
    backend = _backend_or_none(model)
    if backend is not None:
        return backend.build_split_training_loss(model)
    return _legacy.build_split_training_loss(model)


# Compatibility aliases for tests and older internal imports.
_empty_detection_result = _legacy._empty_detection_result
_map_wrapper_labels = _legacy._map_wrapper_labels
_clamp_xyxy_boxes = _legacy._clamp_xyxy_boxes
_postprocess_yolo_output = _legacy._postprocess_yolo_output
_postprocess_rtdetr_output = _legacy._postprocess_rtdetr_output
_postprocess_detr_output = _legacy._postprocess_detr_output
_postprocess_rfdetr_output = _legacy._postprocess_rfdetr_output
_postprocess_anchor_detector_output = _legacy._postprocess_anchor_detector_output
_iter_payload_tensors = _legacy._iter_payload_tensors
_feature_matrix_from_tensor = _legacy._feature_matrix_from_tensor
_spectral_entropy_from_matrix = _legacy._spectral_entropy_from_matrix
_summarize_payload_spectral_entropy = _legacy._summarize_payload_spectral_entropy
_summarize_runtime_output_spectral_entropy = _legacy._summarize_runtime_output_spectral_entropy
_summarize_logits_statistics = _legacy._summarize_logits_statistics
_extract_runtime_logits = _legacy._extract_runtime_logits
_extract_yolo_runtime_aux = _legacy._extract_yolo_runtime_aux
_extract_yolo_runtime_scores = _legacy._extract_yolo_runtime_scores
_extract_yolo_runtime_feats = _legacy._extract_yolo_runtime_feats
_extract_detr_outputs = _legacy._extract_detr_outputs
_extract_rfdetr_outputs = _legacy._extract_rfdetr_outputs
_extract_anchor_detector_outputs = _legacy._extract_anchor_detector_outputs
_extract_rtdetr_loss_outputs = _legacy._extract_rtdetr_loss_outputs
_iter_tensors = _legacy._iter_tensors
_contiguous_tensor_tree = _legacy._contiguous_tensor_tree
_first_tensor_device = _legacy._first_tensor_device
_build_ultralytics_training_batch = _legacy._build_ultralytics_training_batch
_build_detr_training_labels = _legacy._build_detr_training_labels
_build_rfdetr_training_labels = _legacy._build_rfdetr_training_labels


__all__ = [
    "RFDETRReplay",
    "TorchvisionAnchorDetectorReplay",
    "build_anchor_detector_training_loss",
    "build_split_runtime_sample_input",
    "build_split_training_loss",
    "build_ssd_split_training_loss",
    "get_split_runtime_input_resize_mode",
    "get_split_runtime_model",
    "postprocess_split_runtime_output",
    "prepare_split_runtime_input",
    "summarize_split_runtime_observables",
]
