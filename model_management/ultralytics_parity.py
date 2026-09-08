from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils.checks import check_imgsz


def _parameter_grad_state(module: Any) -> dict[str, bool]:
    if not isinstance(module, torch.nn.Module):
        return {}
    return {name: parameter.requires_grad for name, parameter in module.named_parameters()}


def _restore_parameter_grad_state(module: Any, grad_state: dict[str, bool]) -> None:
    if not isinstance(module, torch.nn.Module):
        return
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(grad_state.get(name, parameter.requires_grad))


def _predictor_overrides(
    engine: Any,
    *,
    conf: float,
    imgsz: int | tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    custom = {
        "conf": float(conf),
        "batch": 1,
        "save": False,
        "mode": "predict",
        "rect": True,
    }
    if imgsz is not None:
        custom["imgsz"] = list(imgsz) if isinstance(imgsz, tuple) else imgsz
    return {**getattr(engine, "overrides", {}), **custom}


def ensure_predictor(
    engine: Any,
    *,
    conf: float,
    imgsz: int | tuple[int, int] | list[int] | None = None,
):
    args = _predictor_overrides(engine, conf=conf, imgsz=imgsz)
    predictor = getattr(engine, "predictor", None)
    current_device = getattr(getattr(predictor, "args", None), "device", None)
    requested_device = args.get("device", current_device)

    if predictor is None or current_device != requested_device:
        predictor = engine._smart_load("predictor")(overrides=args, _callbacks=engine.callbacks)
        model_for_predictor = (
            deepcopy(engine.model) if isinstance(engine.model, torch.nn.Module) else engine.model
        )
        predictor.setup_model(model=model_for_predictor, verbose=False)
        engine.predictor = predictor
    else:
        predictor.args = get_cfg(predictor.args, args)

    predictor.imgsz = check_imgsz(
        predictor.args.imgsz,
        stride=predictor.model.stride,
        min_dim=2,
    )
    return predictor


def invalidate_predictor(engine: Any) -> None:
    if hasattr(engine, "predictor"):
        engine.predictor = None


def rgb_tensor_to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    rgb = image.detach().cpu()
    if rgb.ndim != 3:
        raise RuntimeError(f"Expected CHW image tensor, got shape {tuple(rgb.shape)!r}.")
    rgb = rgb.permute(1, 2, 0).contiguous().numpy()
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype("uint8")
    else:
        rgb = np.clip(rgb, 0, 255).astype("uint8")
    return np.ascontiguousarray(rgb[..., ::-1])


def preprocess_bgr_images(
    engine: Any,
    images_bgr: list[np.ndarray],
    *,
    conf: float,
    imgsz: int | tuple[int, int] | list[int] | None = None,
) -> tuple[Any, torch.Tensor]:
    grad_state = _parameter_grad_state(getattr(engine, "model", None))
    try:
        predictor = ensure_predictor(engine, conf=conf, imgsz=imgsz)
        predictor.batch = ([f"image{i}.jpg" for i in range(len(images_bgr))], None, None)
        prepared = [np.ascontiguousarray(image) for image in images_bgr]
        return predictor, predictor.preprocess(prepared)
    finally:
        _restore_parameter_grad_state(getattr(engine, "model", None), grad_state)


def postprocess_predictions(
    engine: Any,
    predictions: Any,
    model_input: torch.Tensor,
    images_bgr: list[np.ndarray],
    *,
    conf: float,
):
    grad_state = _parameter_grad_state(getattr(engine, "model", None))
    try:
        predictor = ensure_predictor(engine, conf=conf)
        predictor.batch = ([f"image{i}.jpg" for i in range(len(images_bgr))], None, None)
        prepared = [np.ascontiguousarray(image) for image in images_bgr]
        return predictor.postprocess(predictions, model_input, prepared)
    finally:
        _restore_parameter_grad_state(getattr(engine, "model", None), grad_state)
