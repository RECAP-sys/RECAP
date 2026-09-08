from __future__ import annotations

from model_management.detectors.base import DetectionBackend
from model_management.detectors.registry import (
    get_backend_by_name,
    get_backend_for_model,
    list_backends,
    register_backend,
)

__all__ = [
    "DetectionBackend",
    "get_backend_by_name",
    "get_backend_for_model",
    "list_backends",
    "register_backend",
]
