from __future__ import annotations

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from timm.layers import trunc_normal_
from torch import nn
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.ops.misc import Conv2dNormActivation

TINYNEXT_VARIANTS: dict[str, dict[str, object]] = {
    "tinynext_s": {
        "cfg": [
            ["mv2", 32, 3, 2.0],
            ["mv2", 64, 3, 2.0],
            ["former", 96, 8, 2.0],
            ["se", 192, 3, 2.0],
        ],
        "feature_channels": (96, 192),
    },
    "tinynext_m": {
        "cfg": [
            ["mv2", 32, 4, 2.0],
            ["mv2", 64, 4, 2.0],
            ["former", 128, 9, 2.0],
            ["se", 256, 4, 1.5],
        ],
        "feature_channels": (128, 256),
    },
}


class Add(nn.Module):
    def forward(self, identity: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return identity + residual


class Mul(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs * rhs


class MatMul(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Stem(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            ConvBNReLU(in_channels, out_channels // 2, 3, 2),
            ConvBNReLU(out_channels // 2, out_channels, 3, 2),
        )


class MV2Block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int, expand_channel: int):
        super().__init__()
        self.shortcut = stride == 1 and in_channel == out_channel
        self.layers = nn.Sequential(
            ConvBNReLU(in_channel, expand_channel, kernel_size=1),
            ConvBNReLU(expand_channel, expand_channel, stride=stride, groups=expand_channel),
            nn.Conv2d(expand_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.add = Add() if self.shortcut else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut:
            return self.add(x, self.layers(x))
        return self.layers(x)


class Embed(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(MV2Block(in_channels, out_channels, stride=2, expand_channel=out_channels))


class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**-0.5
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.matmul1 = MatMul()
        self.matmul2 = MatMul()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        flattened = x.view(batch, channels, -1).transpose(-1, -2).contiguous()
        attn = self.matmul1(self.linear1(flattened), flattened.transpose(-2, -1))
        attn = self.softmax(attn * self.scale)
        mixed = self.matmul2(attn, self.linear2(flattened))
        return mixed.transpose(-1, -2).view(batch, channels, height, width).contiguous()


class Mlp(nn.Sequential):
    def __init__(self, in_channels: int, ratio: float):
        hidden_channels = int(ratio * in_channels)
        super().__init__(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
        )


class FormerBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.attention = nn.Sequential(nn.BatchNorm2d(dim), Attention(dim))
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.mlp = nn.Sequential(nn.BatchNorm2d(dim), Mlp(dim, mlp_ratio))
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add1(x, self.attention(x))
        x = self.add2(x, self.local(x))
        x = self.add3(x, self.mlp(x))
        return x


class SeModule(nn.Module):
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        hidden_channel = max(channel // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, channel, kernel_size=1, bias=False),
            nn.Hardsigmoid(),
        )
        self.mul = Mul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mul(x, self.se(x))


class SeBlock(nn.Module):
    def __init__(self, in_channels: int, mlp_ratio: float, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(nn.BatchNorm2d(in_channels), SeModule(in_channels, reduction))
        self.local = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.mlp = nn.Sequential(nn.BatchNorm2d(in_channels), Mlp(in_channels, mlp_ratio))
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add1(x, self.se(x))
        x = self.add2(x, self.local(x))
        x = self.add3(x, self.mlp(x))
        return x


def _gen_block(name: str, channels: int, ratio: float) -> nn.Module:
    expand_channel = int(ratio * channels)
    if name == "mv2":
        return MV2Block(
            in_channel=channels, out_channel=channels, stride=1, expand_channel=expand_channel
        )
    if name == "former":
        return FormerBlock(dim=channels, mlp_ratio=ratio)
    if name == "se":
        return SeBlock(in_channels=channels, mlp_ratio=ratio, reduction=4)
    raise ValueError(f"Invalid TinyNeXt block type: {name}")


def _normal_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


def _extra_block(
    in_channels: int, out_channels: int, norm_layer: Callable[..., nn.Module]
) -> nn.Sequential:
    activation = nn.ReLU6
    intermediate_channels = out_channels // 2
    return nn.Sequential(
        Conv2dNormActivation(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
        Conv2dNormActivation(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            groups=intermediate_channels,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
        Conv2dNormActivation(
            intermediate_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
    )


def _resolve_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected TinyNeXt checkpoint type: {type(checkpoint)!r}")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected TinyNeXt state dict type: {type(state_dict)!r}")
    if any(str(key).startswith("backbone.") for key in state_dict.keys()):
        return {
            str(key).split("backbone.", 1)[1]: value
            for key, value in state_dict.items()
            if str(key).startswith("backbone.")
        }
    return {str(key): value for key, value in state_dict.items()}


class TinyNeXtBackbone(nn.Module):
    def __init__(
        self,
        *,
        cfg: list[list[object]],
        pretrained_path: str | Path | None = None,
        out_indices: tuple[int, int] = (3, 4),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_indices = out_indices

        input_channel = int(self.cfg[0][1])
        self.embeds = nn.ModuleList([Stem(3, input_channel)])
        self.stages = nn.ModuleList()
        for index in range(4):
            name, width, depth, ratio = self.cfg[index]
            width = int(width)
            depth = int(depth)
            ratio = float(ratio)
            if index > 0:
                self.embeds.append(Embed(input_channel, width))
            stage = nn.Sequential(*[_gen_block(str(name), width, ratio) for _ in range(depth)])
            self.stages.append(stage)
            input_channel = width

        self._initialize_weights()
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def _initialize_weights(self) -> None:
        for _, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def load_pretrained(self, checkpoint_path: str | Path) -> None:
        state_dict = _resolve_checkpoint_state_dict(checkpoint_path)
        filtered = {key: value for key, value in state_dict.items() if key in self.state_dict()}
        self.load_state_dict(filtered, strict=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        for index in range(4):
            x = self.embeds[index](x)
            x = self.stages[index](x)
            if (index + 1) in self.out_indices:
                outputs.append(x)
        return tuple(outputs)


class TinyNeXtFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: TinyNeXtBackbone,
        *,
        feature_channels: tuple[int, int],
        extra_channels: tuple[int, int, int, int] = (512, 256, 256, 128),
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        self.backbone = backbone
        self.out_channels = list(feature_channels) + list(extra_channels)

        in_channels = int(feature_channels[-1])
        extras = []
        for out_channels in extra_channels:
            extras.append(_extra_block(in_channels, out_channels, norm_layer))
            in_channels = out_channels
        self.extra = nn.ModuleList(extras)
        _normal_init(self.extra)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = list(self.backbone(x))
        current = features[-1]
        for block in self.extra:
            current = block(current)
            features.append(current)
        return OrderedDict((str(index), tensor) for index, tensor in enumerate(features))


def build_tinynext_detector(
    variant: str,
    *,
    num_classes: int = 91,
    device: str | torch.device = "cpu",
    backbone_weights_path: str | Path | None = None,
    image_size: int | tuple[int, int] = 320,
) -> SSD:
    name = variant.lower().replace("-", "_")
    if name not in TINYNEXT_VARIANTS:
        raise ValueError(f"Unsupported TinyNeXt detector variant: {variant}")
    if isinstance(image_size, tuple):
        if len(image_size) != 2:
            raise ValueError("TinyNeXt image_size tuple must be (height, width).")
        height, width = int(image_size[0]), int(image_size[1])
        if height != width:
            raise ValueError("TinyNeXt SSD detector requires a square fixed image_size.")
        detector_size = height
    else:
        detector_size = int(image_size)
    if detector_size <= 0:
        raise ValueError("TinyNeXt image_size must be positive.")

    variant_cfg = TINYNEXT_VARIANTS[name]
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    backbone = TinyNeXtBackbone(
        cfg=variant_cfg["cfg"],  # type: ignore[arg-type]
        pretrained_path=backbone_weights_path,
        out_indices=(3, 4),
    )
    extractor = TinyNeXtFeatureExtractor(
        backbone,
        feature_channels=variant_cfg["feature_channels"],  # type: ignore[arg-type]
        norm_layer=norm_layer,
    )

    anchor_generator = DefaultBoxGenerator(
        [[2, 3] for _ in range(6)],
        scales=[
            48 / detector_size,
            100 / detector_size,
            150 / detector_size,
            202 / detector_size,
            253 / detector_size,
            304 / detector_size,
            1.0,
        ],
        steps=[16, 32, 64, 107, 160, 320],
    )
    head = SSDLiteHead(
        extractor.out_channels,
        anchor_generator.num_anchors_per_location(),
        num_classes,
        norm_layer,
    )
    model = SSD(
        extractor,
        anchor_generator,
        (detector_size, detector_size),
        num_classes,
        head=head,
        score_thresh=0.02,
        nms_thresh=0.45,
        detections_per_img=200,
        topk_candidates=1000,
        # Mirror the upstream MMDetection preprocessor:
        # RGB tensors are in [0, 1], so mean/std need to be scaled from 0-255.
        image_mean=[128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0],
        image_std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
    )
    model.to(device)
    return model
