from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

MODEL_DELTA_PAYLOAD_FORMAT = "state_dict_delta"
_THRESHOLD_STATE_NAMES = frozenset(
    {
        "plank_threshold_low",
        "plank_threshold_high",
    }
)
_WRAPPER_INNER_MODULE_PATHS = (
    ("yolo", "model"),
    ("rtdetr", "model"),
    ("detr",),
    ("rfdetr", "model", "model"),
)


def _resolve_attr_path(root: object, path: tuple[str, ...]) -> object | None:
    current = root
    for attr in path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _iter_named_parameters(module: object):
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return
    try:
        yield from named_parameters()
    except TypeError:
        yield from named_parameters(recurse=True)


def _state_key_for_parameter_name(
    parameter_name: str,
    state_keys: set[str],
) -> str | None:
    if parameter_name in state_keys:
        return parameter_name

    if "." in parameter_name:
        stripped = parameter_name.split(".", 1)[1]
        if stripped in state_keys:
            return stripped

    suffix_matches = [key for key in state_keys if key.endswith(f".{parameter_name}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def _trainable_state_dict_names(
    model: torch.nn.Module,
    state: Mapping[str, object],
) -> set[str]:
    """Return trainable parameter names as they appear in ``model.state_dict()``.

    Several detector wrappers expose ``state_dict()`` from an inner model while
    ``named_parameters()`` either carries a wrapper prefix or is empty because
    the wrapped detector is not an ``nn.Module`` child.  The edge update payload
    must use state-dict keys, otherwise the cloud sends a delta that contains no
    real weights.
    """
    state_keys = {str(key) for key in state.keys()}
    selected: set[str] = set()

    modules: list[object] = [model]
    for path in _WRAPPER_INNER_MODULE_PATHS:
        inner = _resolve_attr_path(model, path)
        if isinstance(inner, torch.nn.Module):
            modules.append(inner)

    for module in modules:
        for name, parameter in _iter_named_parameters(module) or ():
            if not bool(getattr(parameter, "requires_grad", False)):
                continue
            state_key = _state_key_for_parameter_name(str(name), state_keys)
            if state_key is not None:
                selected.add(state_key)

    if not selected:
        # Some detector wrappers expose their state_dict but no usable
        # named_parameters() surface. A non-empty floating-state fallback is
        # preferable to producing an update payload the edge cannot apply.
        selected.update(
            str(name)
            for name, value in state.items()
            if torch.is_tensor(value) and value.is_floating_point()
        )
    return selected


def build_state_dict_delta_payload(
    model: torch.nn.Module,
    *,
    model_name: str,
    base_model_version: str,
    result_model_version: str,
) -> dict[str, Any]:
    state = model.state_dict()
    trainable_names = _trainable_state_dict_names(model, state)
    selected_names = trainable_names | {name for name in state if name in _THRESHOLD_STATE_NAMES}
    return {
        "format": MODEL_DELTA_PAYLOAD_FORMAT,
        "model_name": str(model_name),
        "base_model_version": str(base_model_version),
        "result_model_version": str(result_model_version),
        "state_dict": {
            name: value.detach().cpu() if torch.is_tensor(value) else value
            for name, value in state.items()
            if name in selected_names
        },
    }


def require_state_dict_delta_payload(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("Cloud model update must be a state_dict_delta payload.")
    if payload.get("format") != MODEL_DELTA_PAYLOAD_FORMAT:
        raise RuntimeError(
            "Unsupported cloud model update format: "
            f"{payload.get('format', '<missing>')!r}; expected {MODEL_DELTA_PAYLOAD_FORMAT!r}."
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("Cloud model update payload is missing a non-empty state_dict delta.")
    return payload
