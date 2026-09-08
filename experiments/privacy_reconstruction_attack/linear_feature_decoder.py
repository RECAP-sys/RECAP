from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from experiments.privacy_reconstruction_attack.runtime_input_adapter import RuntimeInputAdapter
from model_management.payload import BoundaryPayload


@dataclass(frozen=True)
class LinearFeatureDecodeResult:
    image: torch.Tensor
    label: str
    module_name: str
    regularization: float
    feature_mse: float


def _candidate_roots(detector: Any) -> list[tuple[str, Any]]:
    roots: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def add(name: str, value: Any) -> None:
        if value is None or id(value) in seen or not hasattr(value, "named_modules"):
            return
        seen.add(id(value))
        roots.append((name, value))

    add("model", getattr(detector, "model", None))
    rfdetr = getattr(getattr(detector, "model", None), "rfdetr", None)
    add("model.rfdetr", rfdetr)
    context = getattr(rfdetr, "model", None)
    add("model.rfdetr.model", context)
    add("model.rfdetr.model.model", getattr(context, "model", None))
    return roots


def _conv_output_size(input_size: tuple[int, int], conv: torch.nn.Conv2d) -> tuple[int, int]:
    height, width = input_size
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    ph, pw = conv.padding
    dh, dw = conv.dilation
    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return int(out_h), int(out_w)


def _find_matching_conv(
    detector: Any,
    *,
    feature: torch.Tensor,
    target_shape: tuple[int, ...],
) -> tuple[str, torch.nn.Conv2d] | None:
    if feature.ndim != 4 or len(target_shape) != 4 or int(target_shape[1]) != 3:
        return None
    _, channels, out_h, out_w = (int(dim) for dim in feature.shape)
    input_h, input_w = int(target_shape[-2]), int(target_shape[-1])
    for root_name, root in _candidate_roots(detector):
        for module_name, module in root.named_modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue
            if int(module.in_channels) != 3 or int(module.out_channels) != channels:
                continue
            if int(module.groups) != 1:
                continue
            if _conv_output_size((input_h, input_w), module) != (out_h, out_w):
                continue
            name = f"{root_name}.{module_name}" if module_name else root_name
            return name, module
    return None


def _find_rfdetr_patch_embedding(detector: Any) -> tuple[str, Any, torch.nn.Conv2d] | None:
    for root_name, root in _candidate_roots(detector):
        for module_name, module in root.named_modules():
            patch_embeddings = getattr(module, "patch_embeddings", None)
            projection = getattr(patch_embeddings, "projection", None)
            position_embeddings = getattr(module, "position_embeddings", None)
            if not isinstance(projection, torch.nn.Conv2d):
                continue
            if not isinstance(position_embeddings, torch.Tensor):
                continue
            if int(projection.in_channels) != 3:
                continue
            name = f"{root_name}.{module_name}" if module_name else root_name
            return name, module, projection
    return None


def _patch_grid(target_shape: tuple[int, ...], conv: torch.nn.Conv2d) -> tuple[int, int]:
    return _conv_output_size((int(target_shape[-2]), int(target_shape[-1])), conv)


def _position_embeddings_for_grid(
    embeddings: Any,
    *,
    token_count: int,
    target_shape: tuple[int, ...],
    conv: torch.nn.Conv2d,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    position_embeddings = getattr(embeddings, "position_embeddings")
    position_embeddings = position_embeddings.detach().to(device=device, dtype=dtype)
    if int(position_embeddings.shape[1]) == token_count:
        return position_embeddings

    interpolate = getattr(embeddings, "interpolate_pos_encoding", None)
    if callable(interpolate):
        dummy = torch.empty(
            (1, int(token_count), int(position_embeddings.shape[-1])),
            device=device,
            dtype=dtype,
        )
        return interpolate(
            dummy,
            int(target_shape[-2]),
            int(target_shape[-1]),
        ).detach()

    grid_h, grid_w = _patch_grid(target_shape, conv)
    patch_pos = position_embeddings[:, 1:]
    source_hw = int(round(float(patch_pos.shape[1]) ** 0.5))
    patch_pos = patch_pos.reshape(1, source_hw, source_hw, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(),
        size=(grid_h, grid_w),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=dtype)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
    return torch.cat((position_embeddings[:, :1], patch_pos), dim=1)


def _global_tokens_to_conv_feature(
    tokens: torch.Tensor,
    *,
    embeddings: Any,
    conv: torch.nn.Conv2d,
    target_shape: tuple[int, ...],
) -> torch.Tensor | None:
    if tokens.ndim != 3:
        return None
    batch, token_count, channels = (int(dim) for dim in tokens.shape)
    if batch < 1 or channels != int(conv.out_channels):
        return None
    grid_h, grid_w = _patch_grid(target_shape, conv)
    if token_count != grid_h * grid_w + 1:
        return None
    dtype = tokens.dtype
    pos = _position_embeddings_for_grid(
        embeddings,
        token_count=token_count,
        target_shape=target_shape,
        conv=conv,
        dtype=dtype,
        device=tokens.device,
    )
    patch_tokens = tokens - pos
    patch_tokens = patch_tokens[:, 1:]
    return patch_tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)


def _window_tokens_to_conv_feature(
    tokens: torch.Tensor,
    *,
    embeddings: Any,
    conv: torch.nn.Conv2d,
    target_shape: tuple[int, ...],
) -> torch.Tensor | None:
    if tokens.ndim != 3:
        return None
    windows_batch, token_count, channels = (int(dim) for dim in tokens.shape)
    num_windows = int(getattr(getattr(embeddings, "config", None), "num_windows", 1) or 1)
    if num_windows <= 1 or channels != int(conv.out_channels):
        return None
    if windows_batch % (num_windows * num_windows) != 0:
        return None
    grid_h, grid_w = _patch_grid(target_shape, conv)
    if grid_h % num_windows != 0 or grid_w % num_windows != 0:
        return None
    h_per = grid_h // num_windows
    w_per = grid_w // num_windows
    if token_count != h_per * w_per + 1:
        return None

    batch = windows_batch // (num_windows * num_windows)
    pixel_tokens_with_pos = tokens[:, 1:]
    pixel_tokens_with_pos = pixel_tokens_with_pos.reshape(
        batch * num_windows,
        num_windows,
        h_per,
        w_per,
        channels,
    )
    pixel_tokens_with_pos = pixel_tokens_with_pos.permute(0, 2, 1, 3, 4)
    pixel_tokens_with_pos = pixel_tokens_with_pos.reshape(batch, grid_h, grid_w, channels)
    pixel_tokens_with_pos = pixel_tokens_with_pos.reshape(batch, grid_h * grid_w, channels)

    dtype = tokens.dtype
    pos = _position_embeddings_for_grid(
        embeddings,
        token_count=grid_h * grid_w + 1,
        target_shape=target_shape,
        conv=conv,
        dtype=dtype,
        device=tokens.device,
    )
    patch_tokens = pixel_tokens_with_pos - pos[:, 1:]
    return patch_tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)


def _tokens_to_conv_feature(
    detector: Any,
    *,
    tokens: torch.Tensor,
    target_shape: tuple[int, ...],
) -> tuple[str, torch.Tensor, torch.nn.Conv2d] | None:
    match = _find_rfdetr_patch_embedding(detector)
    if match is None:
        return None
    module_name, embeddings, conv = match

    global_feature = _global_tokens_to_conv_feature(
        tokens,
        embeddings=embeddings,
        conv=conv,
        target_shape=target_shape,
    )
    if global_feature is not None:
        return f"{module_name}.tokens", global_feature, conv

    window_feature = _window_tokens_to_conv_feature(
        tokens,
        embeddings=embeddings,
        conv=conv,
        target_shape=target_shape,
    )
    if window_feature is not None:
        return f"{module_name}.window_tokens", window_feature, conv
    return None


def _invert_conv_feature(
    feature: torch.Tensor,
    conv: torch.nn.Conv2d,
    *,
    target_shape: tuple[int, ...],
    regularization: float,
) -> tuple[torch.Tensor, float]:
    batch, _channels, _out_h, _out_w = (int(dim) for dim in feature.shape)
    device = feature.device
    dtype = torch.float32
    weight = conv.weight.detach().to(device=device, dtype=dtype)
    matrix = weight.reshape(int(conv.out_channels), -1)
    eye = torch.eye(matrix.shape[0], device=device, dtype=dtype)
    decoder = matrix.t() @ torch.linalg.inv(matrix @ matrix.t() + float(regularization) * eye)
    values = feature.detach().to(device=device, dtype=dtype).reshape(
        batch,
        -1,
        feature.shape[-2] * feature.shape[-1],
    )
    if conv.bias is not None:
        bias = conv.bias.detach().to(device=device, dtype=dtype).view(1, -1, 1)
        values = values - bias
    patches = torch.einsum("do,bol->bdl", decoder, values)
    fold_kwargs = {
        "output_size": (int(target_shape[-2]), int(target_shape[-1])),
        "kernel_size": conv.kernel_size,
        "dilation": conv.dilation,
        "padding": conv.padding,
        "stride": conv.stride,
    }
    reconstructed = F.fold(patches, **fold_kwargs)
    divisor = F.fold(torch.ones_like(patches), **fold_kwargs).clamp_min(1.0)
    reconstructed = reconstructed / divisor
    with torch.no_grad():
        replay = conv(reconstructed.to(device=weight.device, dtype=weight.dtype)).float()
        feature_mse = F.mse_loss(
            replay,
            feature.detach().to(replay.device).float(),
        ).item()
    return reconstructed, float(feature_mse)


def reconstruct_from_linear_conv_feature(
    target_payload: BoundaryPayload,
    *,
    detector: Any,
    runtime_adapter: RuntimeInputAdapter,
    target_shape: tuple[int, ...],
    regularization: float = 1.0e-5,
) -> LinearFeatureDecodeResult | None:
    best: LinearFeatureDecodeResult | None = None
    for label, tensor in dict(target_payload.tensors).items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        if tensor.ndim == 4:
            match = _find_matching_conv(detector, feature=tensor, target_shape=target_shape)
            if match is None:
                continue
            module_name, conv = match
            feature = tensor
        elif tensor.ndim == 3:
            token_match = _tokens_to_conv_feature(
                detector,
                tokens=tensor,
                target_shape=target_shape,
            )
            if token_match is None:
                continue
            module_name, feature, conv = token_match
        else:
            continue

        model_input, feature_mse = _invert_conv_feature(
            feature,
            conv,
            target_shape=target_shape,
            regularization=float(regularization),
        )
        image = runtime_adapter.from_runtime_input(model_input, clamp=True).detach()
        result = LinearFeatureDecodeResult(
            image=image,
            label=str(label),
            module_name=module_name,
            regularization=float(regularization),
            feature_mse=feature_mse,
        )
        if best is None or result.feature_mse < best.feature_mse:
            best = result
    return best
