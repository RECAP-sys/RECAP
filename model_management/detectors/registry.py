from __future__ import annotations

from importlib import import_module

import torch

from model_management.detectors.base import DetectionBackend

_BACKENDS: list[DetectionBackend] = []
_DEFAULTS_LOADED = False

_DEFAULT_BACKEND_MODULES = (
    "model_management.detectors.yolo",
    "model_management.detectors.rtdetr",
    "model_management.detectors.rfdetr",
    "model_management.detectors.tinynext",
    "model_management.detectors.detr",
    "model_management.detectors.torchvision_anchor",
)


def _normalise_model_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


def register_backend(backend: DetectionBackend) -> None:
    """Register a backend singleton, replacing any previous backend by family."""
    global _BACKENDS
    _BACKENDS = [item for item in _BACKENDS if item.family != backend.family]
    _BACKENDS.append(backend)


def _ensure_default_backends() -> None:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    for module_name in _DEFAULT_BACKEND_MODULES:
        module = import_module(module_name)
        register_backend(module.BACKEND)
    _DEFAULTS_LOADED = True


def list_backends() -> list[DetectionBackend]:
    _ensure_default_backends()
    return list(_BACKENDS)


def get_backend_by_name(model_name: str) -> DetectionBackend:
    _ensure_default_backends()
    name_lower = _normalise_model_name(model_name)
    for backend in _BACKENDS:
        if backend.matches_name(name_lower):
            return backend
    raise KeyError(f"No detection backend registered for model name: {model_name!r}")


def get_backend_for_model(model: torch.nn.Module) -> DetectionBackend:
    _ensure_default_backends()
    for backend in _BACKENDS:
        if backend.matches_model(model):
            return backend
    raise KeyError(f"No detection backend registered for model instance: {type(model)!r}")
