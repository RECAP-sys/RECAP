from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from .errors import InvalidOutputStructureError, UnsupportedModelAdapterError


class DetectionSplitAdapter(Protocol):
    model_family: str

    def get_runtime_model(self, model: torch.nn.Module) -> torch.nn.Module: ...

    def prepare_trace_batch(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        device: str | torch.device,
    ) -> torch.Tensor: ...

    def prepare_runtime_batch(
        self,
        frames: list[np.ndarray],
        device: str | torch.device,
    ) -> torch.Tensor: ...

    def build_targets(self, annotations: Any, metadata: Any, device: str | torch.device) -> Any: ...

    def loss_fn(self, outputs: Any, targets: Any) -> torch.Tensor: ...

    def postprocess(self, outputs: Any, model_input: Any, metadata: Any) -> Any: ...


def _infer_model_family(model: Any, model_name: str | None) -> str:
    if model_name:
        try:
            from model_management.model_zoo import get_model_family

            return get_model_family(model_name)
        except Exception:
            pass
    cls_name = model.__class__.__name__.lower()
    if "yolo" in cls_name:
        return "yolo"
    if "rfdetr" in cls_name or "rf_detr" in cls_name:
        return "rfdetr"
    if "tinynext" in cls_name or hasattr(model, "compute_loss"):
        return "tinynext"
    if hasattr(model, "rfdetr"):
        return "rfdetr"
    if hasattr(model, "yolo"):
        return "yolo"
    return "generic"


def _ensure_batched_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            return value.unsqueeze(0)
        if value.ndim == 4:
            return value
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, torch.Tensor) for item in value)
    ):
        tensors = [item if item.ndim == 3 else item.squeeze(0) for item in value]
        return torch.stack(tensors, dim=0)
    raise TypeError(f"Adapter expected tensor batch input, got {type(value)!r}.")


def _repeat_to_batch(sample: torch.Tensor, batch_size: int) -> torch.Tensor:
    sample = _ensure_batched_tensor(sample)
    if int(sample.shape[0]) == int(batch_size):
        return sample
    if int(sample.shape[0]) == 1:
        repeats = [int(batch_size), *([1] * (sample.ndim - 1))]
        return sample.repeat(*repeats)
    if int(sample.shape[0]) > int(batch_size):
        return sample[:batch_size]
    copies = [sample]
    while sum(int(item.shape[0]) for item in copies) < int(batch_size):
        copies.append(sample[:1].clone())
    return torch.cat(copies, dim=0)[:batch_size]


def _default_loss(outputs: Any, targets: Any) -> torch.Tensor:
    del targets
    tensors: list[torch.Tensor] = []

    def walk(value: Any) -> None:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(outputs)
    if not tensors:
        raise InvalidOutputStructureError(
            "Detector output does not contain differentiable tensors."
        )
    loss = tensors[0].sum() * 0.0
    for tensor in tensors:
        if tensor.numel():
            loss = loss + tensor.float().square().mean()
    return loss / max(1, len(tensors))


@dataclass
class PlankDetectionSplitAdapter:
    model: torch.nn.Module
    model_family: str = "generic"

    def get_runtime_model(self, model: torch.nn.Module | None = None) -> torch.nn.Module:
        from model_management.split_model_adapters import get_split_runtime_model

        return get_split_runtime_model(model or self.model)

    def prepare_trace_batch(
        self,
        batch_size: int = 2,
        image_size: tuple[int, int] = (640, 640),
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        from model_management.split_model_adapters import build_split_runtime_sample_input

        sample = build_split_runtime_sample_input(
            self.model,
            image_size=image_size,
            device=device,
        )
        return _repeat_to_batch(sample, int(batch_size)).to(device)

    def prepare_runtime_batch(
        self,
        frames: list[np.ndarray],
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        from model_management.split_model_adapters import prepare_split_runtime_input

        prepared = [
            _ensure_batched_tensor(prepare_split_runtime_input(self.model, frame, device=device))
            for frame in frames
        ]
        if not prepared:
            raise ValueError("prepare_runtime_batch requires at least one frame.")
        return torch.cat(prepared, dim=0).to(device)

    def build_targets(
        self,
        annotations: Any,
        metadata: Any = None,
        device: str | torch.device = "cpu",
    ) -> Any:
        del metadata
        target_device = torch.device(device)

        def tensorize(record: Any) -> Any:
            if not isinstance(record, dict):
                return record
            converted = dict(record)
            if "boxes" in converted:
                converted["boxes"] = torch.as_tensor(
                    converted["boxes"],
                    dtype=torch.float32,
                    device=target_device,
                ).reshape(-1, 4)
            if "labels" in converted:
                converted["labels"] = torch.as_tensor(
                    converted["labels"],
                    dtype=torch.int64,
                    device=target_device,
                ).reshape(-1)
            return converted

        if isinstance(annotations, dict):
            return tensorize(annotations)
        if isinstance(annotations, (list, tuple)):
            return [tensorize(item) for item in annotations]
        return annotations

    def loss_fn(self, outputs: Any, targets: Any) -> torch.Tensor:
        try:
            from model_management.split_model_adapters import build_split_training_loss

            loss_fn = build_split_training_loss(self.model)
            if loss_fn is not None:
                return loss_fn(outputs, targets)
        except Exception:
            pass
        return _default_loss(outputs, targets)

    def postprocess(self, outputs: Any, model_input: Any = None, metadata: Any = None) -> Any:
        threshold = 0.0
        orig_image = None
        if isinstance(metadata, dict):
            threshold = float(metadata.get("threshold", threshold) or threshold)
            orig_image = metadata.get("orig_image")
        try:
            from model_management.split_model_adapters import postprocess_split_runtime_output

            return postprocess_split_runtime_output(
                self.model,
                outputs,
                threshold=threshold,
                model_input=model_input,
                orig_image=orig_image,
            )
        except Exception:
            return outputs


def select_detection_adapter(
    model: torch.nn.Module,
    *,
    model_name: str | None = None,
) -> PlankDetectionSplitAdapter:
    family = _infer_model_family(model, model_name)
    if family == "unknown":
        raise UnsupportedModelAdapterError(f"Unsupported detection model adapter: {model_name!r}.")
    return PlankDetectionSplitAdapter(model=model, model_family=family)


__all__ = [
    "DetectionSplitAdapter",
    "PlankDetectionSplitAdapter",
    "select_detection_adapter",
]
