from __future__ import annotations

from typing import Any

import cv2
import torch
import torch.nn.functional as F

from model_management.object_detection import Object_Detection


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class RuntimeInputAdapter:
    def __init__(
        self,
        detector: Object_Detection,
        target_shape: tuple[int, ...],
        *,
        device: torch.device,
    ) -> None:
        if len(target_shape) != 4 or int(target_shape[1]) != 3:
            raise ValueError(f"Expected BCHW model input shape, got {target_shape}.")
        self.detector = detector
        self.height = int(target_shape[-2])
        self.width = int(target_shape[-1])
        self.device = device
        self.is_rfdetr = hasattr(getattr(detector.model, "rfdetr", None), "means")
        self.mean = None
        self.std = None
        if self.is_rfdetr:
            means = getattr(detector.model.rfdetr, "means")
            stds = getattr(detector.model.rfdetr, "stds")
            self.mean = torch.as_tensor(means, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            self.std = torch.as_tensor(stds, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        else:
            transform = getattr(detector.model, "transform", None)
            means = getattr(transform, "image_mean", None)
            stds = getattr(transform, "image_std", None)
            if means is not None and stds is not None:
                self.mean = torch.as_tensor(means, dtype=torch.float32, device=device).view(
                    1, 3, 1, 1
                )
                self.std = torch.as_tensor(stds, dtype=torch.float32, device=device).view(
                    1, 3, 1, 1
                )

    def to_runtime_input(self, rgb_image: torch.Tensor) -> torch.Tensor:
        x = rgb_image
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW image tensor, got {tuple(x.shape)}.")
        if tuple(x.shape[-2:]) != (self.height, self.width):
            x = F.interpolate(
                x,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std.clamp_min(1.0e-12)
        return x

    def from_runtime_input(self, model_input: torch.Tensor, *, clamp: bool = False) -> torch.Tensor:
        x = model_input
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW model input tensor, got {tuple(x.shape)}.")
        if self.mean is not None and self.std is not None:
            x = x * self.std + self.mean
        if clamp:
            x = x.clamp(0.0, 1.0)
        return x


def prediction_on_reconstruction(
    teacher: Object_Detection,
    image_tensor: torch.Tensor,
    *,
    threshold: float | None,
) -> dict[str, Any]:
    rgb = image_tensor.detach().cpu().clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor((rgb * 255.0).round().astype("uint8"), cv2.COLOR_RGB2BGR)
    boxes, labels, scores = teacher.large_inference(bgr, threshold=threshold)
    from experiments.privacy_reconstruction_attack.attack_dataset import prediction_to_json

    return prediction_to_json(boxes or [], labels or [], scores or [], image_size=bgr.shape[:2])
