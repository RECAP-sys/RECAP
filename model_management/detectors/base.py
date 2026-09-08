from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class DetectionBackend(ABC):
    """Backend contract for a detection model family.

    Backends own family-specific model construction and split-runtime adapter
    behavior. Public compatibility modules delegate through this interface.
    """

    family: str

    @abstractmethod
    def matches_name(self, name: str) -> bool:
        """Return True when this backend owns a user-facing model name."""

    @abstractmethod
    def matches_model(self, model: torch.nn.Module) -> bool:
        """Return True when this backend owns a model instance."""

    @abstractmethod
    def list_model_names(self) -> list[str]:
        """Return user-facing model names supported by this backend."""

    @abstractmethod
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
        """Build a model whose public forward returns torchvision detections."""

    @abstractmethod
    def ensure_local_model_artifact(self, name: str) -> Path:
        """Ensure the local artifact for a model name exists."""

    @abstractmethod
    def get_split_runtime_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the trace/replay model used by split runtime."""

    @abstractmethod
    def build_split_runtime_sample_input(
        self,
        model: torch.nn.Module,
        *,
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device = "cpu",
    ) -> Any:
        """Return a representative input for tracing split runtime."""

    @abstractmethod
    def get_split_runtime_input_resize_mode(self, model: torch.nn.Module) -> str | None:
        """Return coordinate resize mode used for split runtime inputs."""

    @abstractmethod
    def prepare_split_runtime_input(
        self,
        model: torch.nn.Module,
        frame: np.ndarray,
        *,
        device: str | torch.device,
        input_tensor_shape: tuple[int, ...] | list[int] | None = None,
    ) -> Any:
        """Prepare one BGR frame for split-runtime replay."""

    @abstractmethod
    def postprocess_split_runtime_output(
        self,
        model: torch.nn.Module,
        outputs: Any,
        *,
        threshold: float,
        model_input: Any | None = None,
        orig_image: np.ndarray | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode split-runtime output to torchvision detection format."""

    @abstractmethod
    def build_split_training_loss(self, model: torch.nn.Module):
        """Build the split-tail training loss for this family."""

    @abstractmethod
    def summarize_split_runtime_observables(
        self,
        model: torch.nn.Module,
        outputs: Any,
        split_payload: Any | None = None,
        *,
        include_feature_spectral_entropy: bool = True,
    ) -> dict[str, float | None]:
        """Summarize split-runtime observables used by inference artifacts."""

    def get_detection_thresholds(self, model_name: str) -> tuple[float, float]:
        del model_name
        return 0.2, 0.6
