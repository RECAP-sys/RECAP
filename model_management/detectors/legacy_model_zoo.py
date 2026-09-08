"""
Unified Model Zoo — Detection Model Factory
============================================

Provides a single ``build_detection_model(name)`` entry-point that returns a
PyTorch model **whose forward() always produces the torchvision detection
output format**::

    [{"boxes": Tensor[N,4], "labels": Tensor[N], "scores": Tensor[N]}]

This makes YOLO, DETR, RF-DETR, RT-DETR, TinyNeXt, and any future model
family a drop-in replacement for the existing detection pipeline.

Supported model families
------------------------

**Torchvision built-in (RetinaNet / FCOS)**

    Directly instantiated via ``torchvision.models.detection.*``.
    These already follow the target format natively.

**DETR family (via torchvision or huggingface)**

    - ``detr_resnet50``  — Facebook DETR (torchvision hub / HF)
    - ``detr_resnet101``
    - ``conditional_detr_resnet50``
    - ``rtdetr_r50vd`` / ``rtdetr_r101vd`` — Baidu RT-DETR (ultralytics)

**YOLO family (via ultralytics)**

    - ``yolov5s`` / ``yolov5m`` / ``yolov5l`` / ``yolov5x``
    - ``yolov8n`` / ``yolov8s`` / ``yolov8m`` / ``yolov8l`` / ``yolov8x``
    - ``yolov10n`` / ``yolov10s`` / ``yolov10m`` / ``yolov10l`` / ``yolov10x``
    - ``yolo11n`` / ``yolo11s`` / ``yolo11m`` / ``yolo11l`` / ``yolo11x``
    - ``yolo12n`` / ``yolo12s`` / ``yolo12m`` / ``yolo12l`` / ``yolo12x``

Usage
-----
>>> from model_management.model_zoo import build_detection_model
>>> model = build_detection_model("yolov8s", num_classes=91, pretrained=True)
>>> model.eval()
>>> results = model([img_tensor])
>>> results[0]["boxes"], results[0]["labels"], results[0]["scores"]
"""

from __future__ import annotations

import hashlib
import re
import sys
import types
from collections import OrderedDict
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger
from torch.hub import download_url_to_file
from torchvision import transforms
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.ops import batched_nms

# ---------------------------------------------------------------------------
# Required dependencies
# ---------------------------------------------------------------------------
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

from common.logging_sanitizer import safe_error_summary
from model_management.model_info import model_lib
from model_management.tinynext import build_tinynext_detector
from model_management.ultralytics_parity import (
    invalidate_predictor,
    postprocess_predictions,
    preprocess_bgr_images,
    rgb_tensor_to_bgr_uint8,
)

try:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    _HAS_RFDETR = True
except Exception:
    RFDETRNano = RFDETRSmall = RFDETRMedium = RFDETRLarge = None
    _HAS_RFDETR = False

_HAS_ULTRALYTICS = find_spec("ultralytics") is not None
_HAS_HF_DETR = True
_HAS_TV_DETECTION = True


# ═══════════════════════════════════════════════════════════════════════
#  COCO 80-class → 91-index mapping (used by YOLO → torchvision label)
# ═══════════════════════════════════════════════════════════════════════

# COCO dataset has 80 categories but uses IDs 1–90 (with gaps).  YOLO
# outputs a contiguous 0-79 class index; torchvision detection models
# use the original COCO IDs (1-indexed).  This LUT maps yolo_cls →
# torchvision label.
COCO_80_TO_91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


# ═══════════════════════════════════════════════════════════════════════
#  1. YOLO Wrapper  (ultralytics)
# ═══════════════════════════════════════════════════════════════════════


class YOLODetectionModel(nn.Module):
    """Wraps an Ultralytics YOLO model to produce torchvision-compatible output.

    Parameters
    ----------
    model_name : str
        Any name accepted by ``ultralytics.YOLO()``, e.g.
        ``"yolov8s.pt"``, ``"yolov5m.pt"``, ``"yolo11n.pt"``.
    confidence : float
        Minimum confidence threshold for NMS.
    device : str or torch.device
        Target device.
    num_classes : int
        Number of classes (used for label mapping).
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence: float = 0.01,
        device: str | torch.device = "cpu",
        num_classes: int = 91,
    ):
        super().__init__()
        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is required for YOLO models.  Install: pip install ultralytics"
            )
        from ultralytics import YOLO

        self.yolo = YOLO(model_name)
        internal_num_classes = max(int(num_classes), 1)
        if internal_num_classes == 91:
            internal_num_classes = 80
        model_source = str(model_name)
        if Path(model_source).suffix.lower() in {".yaml", ".yml"}:
            model_yaml = getattr(getattr(self.yolo, "model", None), "yaml", {}) or {}
            try:
                current_num_classes = int(model_yaml.get("nc"))
            except (TypeError, ValueError):
                current_num_classes = None
            if current_num_classes is not None and current_num_classes != internal_num_classes:
                from ultralytics.cfg import DEFAULT_CFG_DICT
                from ultralytics.nn.tasks import DetectionModel

                self.yolo.model = DetectionModel(
                    model_source,
                    nc=internal_num_classes,
                    verbose=False,
                )
                self.yolo.model.args = {
                    **DEFAULT_CFG_DICT,
                    **getattr(self.yolo, "overrides", {}),
                }
                self.yolo.model.task = getattr(self.yolo, "task", "detect")
        self.yolo.to(device)
        if isinstance(getattr(self.yolo, "model", None), torch.nn.Module):
            self.yolo.model.eval()
        self.confidence = confidence
        self._device = device
        self.num_classes = num_classes
        self.label_schema = "coco_91" if int(num_classes) >= 91 else "zero_based"
        # If COCO 80-class, use the mapping; otherwise use native class IDs.
        self._map_labels = num_classes >= 91

    def forward(self, images: List[torch.Tensor], targets=None) -> List[Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        images : list of Tensor[C, H, W]
            Batch of images (float, 0–1 range).
        targets : ignored (for compatibility with train() mode)

        Returns
        -------
        list of dict  with keys ``boxes``, ``labels``, ``scores``
        """
        images_bgr = [rgb_tensor_to_bgr_uint8(image) for image in images]
        _, model_input = preprocess_bgr_images(
            self.yolo,
            images_bgr,
            conf=self.confidence,
        )
        preds = self.yolo.model(model_input)
        results = postprocess_predictions(
            self.yolo,
            preds,
            model_input,
            images_bgr,
            conf=self.confidence,
        )

        detections: List[Dict[str, torch.Tensor]] = []
        for result in results:
            if result.boxes is None or result.boxes.data.numel() == 0:
                detections.append(
                    {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                        "scores": torch.zeros((0,), dtype=torch.float32),
                    }
                )
                continue

            boxes_xyxy = result.boxes.xyxy.detach().cpu().float()
            scores = result.boxes.conf.detach().cpu().float()
            cls_ids = result.boxes.cls.detach().cpu().long()

            if self._map_labels:
                labels = torch.tensor(
                    [COCO_80_TO_91[c] if c < 80 else int(c) + 1 for c in cls_ids.tolist()],
                    dtype=torch.int64,
                )
            else:
                labels = cls_ids

            detections.append(
                {
                    "boxes": boxes_xyxy,
                    "labels": labels,
                    "scores": scores,
                }
            )
        return detections

    def train(self, mode: bool = True):
        """Keep the wrapper and its Ultralytics core in the same runtime mode."""
        self.yolo.model.train(mode)
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        """Return underlying YOLO parameters for device placement etc."""
        return self.yolo.model.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.yolo.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        result = self.yolo.model.load_state_dict(state_dict, strict=strict)
        invalidate_predictor(self.yolo)
        return result

    def to(self, device):
        self._device = device
        self.yolo.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════
#  2. DETR Wrapper  (HuggingFace transformers)
# ═══════════════════════════════════════════════════════════════════════


class DETRDetectionModel(nn.Module):
    """Wraps a HuggingFace DETR model to produce torchvision-compatible output.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id, e.g. ``"facebook/detr-resnet-50"``.
    confidence : float
        Score threshold for filtering predictions.
    device : str or torch.device
    num_classes : int
        Number of detection classes (including background for DETR).
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/detr-resnet-50",
        confidence: float = 0.5,
        device: str | torch.device = "cpu",
        num_classes: int = 91,
        pretrained: bool = True,
    ):
        super().__init__()
        if not _HAS_HF_DETR:
            raise ImportError(
                "transformers is required for DETR models.  Install: pip install transformers"
            )
        if pretrained:
            self.processor = DetrImageProcessor.from_pretrained(model_name_or_path)
            self.detr = DetrForObjectDetection.from_pretrained(model_name_or_path)
        else:
            cfg = DetrConfig(
                num_labels=num_classes,
                use_timm_backbone=False,
                backbone=None,
                backbone_config={
                    "model_type": "resnet",
                    "num_channels": 3,
                    "embedding_size": 64,
                    "hidden_sizes": [64, 128, 256, 512],
                    "depths": [2, 2, 2, 2],
                    "hidden_act": "relu",
                },
            )
            self.processor = DetrImageProcessor()
            self.detr = DetrForObjectDetection(cfg)
        self.detr.to(device)
        self.confidence = confidence
        self._device = device
        self.num_classes = num_classes

    def forward(self, images: List[torch.Tensor], targets=None) -> List[Dict[str, torch.Tensor]]:
        results: List[Dict[str, torch.Tensor]] = []
        for img_tensor in images:
            # Convert tensor → PIL for the processor
            pil_img = transforms.ToPILImage()(img_tensor.cpu())
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            if self.training:
                outputs = self.detr(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.detr(**inputs)

            # Post-process: convert logits → boxes in image coords
            target_sizes = torch.tensor([pil_img.size[::-1]], device=self._device)  # (H, W)
            post = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence,
            )[0]

            results.append(
                {
                    "boxes": post["boxes"].cpu().float(),
                    "labels": post["labels"].cpu().long(),
                    "scores": post["scores"].cpu().float(),
                }
            )
        return results

    def train(self, mode: bool = True):
        self.detr.train(mode)
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        return self.detr.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        return self.detr.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.detr.load_state_dict(state_dict, strict=strict)

    def to(self, device):
        self._device = device
        self.detr.to(device)
        return self


class RFDETRDetectionModel(nn.Module):
    """Wraps the official rfdetr package with torchvision-style outputs."""

    _VARIANTS = {
        "rfdetr_nano": RFDETRNano,
        "rfdetr_small": RFDETRSmall,
        "rfdetr_medium": RFDETRMedium,
        "rfdetr_large": RFDETRLarge,
    }

    def __init__(
        self,
        model_name: str = "rfdetr_nano",
        confidence: float = 0.5,
        device: str | torch.device = "cpu",
        num_classes: int = 91,
        pretrained: bool = True,
        **model_kwargs: Any,
    ):
        super().__init__()
        if not _HAS_RFDETR:
            raise ImportError("rfdetr is required for RF-DETR models. Install: pip install rfdetr")

        model_name = _normalise_model_name(model_name)
        variant_cls = self._VARIANTS.get(model_name)
        if variant_cls is None:
            raise ValueError(f"Unsupported RF-DETR variant: {model_name}")

        self.model_name = model_name
        self.confidence = confidence
        self._device = torch.device(device)
        self.num_classes = num_classes
        self.internal_num_classes = max(int(num_classes) - 1, 1)
        self.label_schema = "coco_91" if int(num_classes) >= 91 else "zero_based"

        kwargs = {
            "device": str(self._device),
            "num_classes": self.internal_num_classes,
        }
        kwargs.update(model_kwargs)
        if not pretrained:
            kwargs["pretrain_weights"] = None
        self.rfdetr = variant_cls(**kwargs)
        self.rfdetr.model.model.to(self._device)
        ensure_detection_threshold_state(self, self.model_name)
        ensure_rfdetr_serialization_state(self)

    @property
    def resolution(self) -> int:
        return int(self.rfdetr.model.resolution)

    @property
    def patch_size(self) -> int:
        return int(self.rfdetr.model_config.patch_size)

    @property
    def num_windows(self) -> int:
        return int(self.rfdetr.model_config.num_windows)

    def _prepare_batch(
        self, images: List[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        processed: list[torch.Tensor] = []
        original_sizes: list[tuple[int, int]] = []
        resize_shape = [self.resolution, self.resolution]

        for image in images:
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Unsupported RF-DETR input type: {type(image)!r}")
            if image.ndim != 3:
                raise TypeError(f"RF-DETR expects CHW tensors, got shape {tuple(image.shape)}")
            image = image.to(self._device, dtype=torch.float32)
            height, width = int(image.shape[-2]), int(image.shape[-1])
            original_sizes.append((height, width))
            resized = F.resize(image, resize_shape)
            normalized = F.normalize(resized, self.rfdetr.means, self.rfdetr.stds)
            processed.append(normalized)
        return torch.stack(processed, dim=0), original_sizes

    def forward(self, images: List[torch.Tensor], targets=None) -> List[Dict[str, torch.Tensor]]:
        batch_tensor, original_sizes = self._prepare_batch(images)
        if self.training:
            predictions = self.rfdetr.model.model(batch_tensor)
        else:
            with torch.no_grad():
                predictions = self.rfdetr.model.model(batch_tensor)
        if isinstance(predictions, tuple):
            predictions = {
                "pred_logits": predictions[1],
                "pred_boxes": predictions[0],
            }

        target_sizes = torch.as_tensor(original_sizes, device=self._device)
        return _postprocess_rfdetr_predictions(
            predictions,
            target_sizes=target_sizes,
            threshold=float(self.confidence),
            num_classes=self.num_classes,
            label_schema=getattr(self, "label_schema", "coco_91"),
            num_select=getattr(
                self.rfdetr.model.postprocess, "num_select", predictions["pred_logits"].shape[1]
            ),
            device=self._device,
        )

    def train(self, mode: bool = True):
        self.rfdetr.model.model.train(mode)
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        return self.rfdetr.model.model.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        combined = OrderedDict(self.rfdetr.model.model.state_dict(*args, **kwargs))
        wrapper_state = nn.Module.state_dict(self, *args, **kwargs)
        for key in (
            _DETECTION_THRESHOLD_LOW_BUFFER,
            _DETECTION_THRESHOLD_HIGH_BUFFER,
            _RFDETR_CACHE_VERSION_BUFFER,
        ):
            value = wrapper_state.get(key)
            if torch.is_tensor(value):
                combined[key] = value
        return combined

    def load_state_dict(self, state_dict, strict=True):
        if not isinstance(state_dict, Mapping):
            return self.rfdetr.model.model.load_state_dict(state_dict, strict=strict)
        wrapper_state, core_state = _split_rfdetr_serialized_state_dict(state_dict)
        if wrapper_state:
            nn.Module.load_state_dict(self, wrapper_state, strict=False)
        return self.rfdetr.model.model.load_state_dict(core_state, strict=strict)

    def to(self, device):
        self._device = torch.device(device)
        self.rfdetr.model.model.to(self._device)
        return self


# ═══════════════════════════════════════════════════════════════════════
#  3. RT-DETR Wrapper  (ultralytics)
# ═══════════════════════════════════════════════════════════════════════


class RTDETRDetectionModel(nn.Module):
    """Wraps an Ultralytics RT-DETR model (real-time DETR) to produce
    torchvision-compatible output.

    Parameters
    ----------
    model_name : str
        e.g. ``"rtdetr-l.pt"``, ``"rtdetr-x.pt"``
    confidence : float
    device : str or torch.device
    num_classes : int
    """

    def __init__(
        self,
        model_name: str = "rtdetr-l.pt",
        confidence: float = 0.01,
        device: str | torch.device = "cpu",
        num_classes: int = 91,
    ):
        super().__init__()
        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is required for RT-DETR models.  Install: pip install ultralytics"
            )
        from ultralytics import RTDETR

        self.rtdetr = RTDETR(model_name)
        self.rtdetr.to(device)
        self.confidence = confidence
        self._device = device
        self.num_classes = num_classes
        self.label_schema = "coco_91" if int(num_classes) >= 91 else "zero_based"
        self._map_labels = num_classes >= 91

    def forward(self, images: List[torch.Tensor], targets=None) -> List[Dict[str, torch.Tensor]]:
        images_bgr = [rgb_tensor_to_bgr_uint8(image) for image in images]
        _, model_input = preprocess_bgr_images(
            self.rtdetr,
            images_bgr,
            conf=self.confidence,
        )
        preds = self.rtdetr.model(model_input)
        results = postprocess_predictions(
            self.rtdetr,
            preds,
            model_input,
            images_bgr,
            conf=self.confidence,
        )

        detections: List[Dict[str, torch.Tensor]] = []
        for result in results:
            if result.boxes is None or result.boxes.data.numel() == 0:
                detections.append(
                    {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                        "scores": torch.zeros((0,), dtype=torch.float32),
                    }
                )
                continue

            boxes_xyxy = result.boxes.xyxy.detach().cpu().float()
            scores = result.boxes.conf.detach().cpu().float()
            cls_ids = result.boxes.cls.detach().cpu().long()

            if self._map_labels:
                labels = torch.tensor(
                    [COCO_80_TO_91[c] if c < 80 else int(c) + 1 for c in cls_ids.tolist()],
                    dtype=torch.int64,
                )
            else:
                labels = cls_ids

            detections.append(
                {
                    "boxes": boxes_xyxy,
                    "labels": labels,
                    "scores": scores,
                }
            )
        return detections

    def train(self, mode: bool = True):
        self.rtdetr.model.train(mode)
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        return self.rtdetr.model.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.rtdetr.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        result = self.rtdetr.model.load_state_dict(state_dict, strict=strict)
        invalidate_predictor(self.rtdetr)
        return result

    def to(self, device):
        self._device = device
        self.rtdetr.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════
#  4. Torchvision built-in (RetinaNet / FCOS)
# ═══════════════════════════════════════════════════════════════════════

# These already output the standard torchvision detection format, so we
# just need a factory function to instantiate them.

_TORCHVISION_BUILTIN = (
    {
        # RetinaNet
        "retinanet_resnet50_fpn": lambda **kw: retinanet_resnet50_fpn(**kw),
        # FCOS
        "fcos_resnet50_fpn": lambda **kw: fcos_resnet50_fpn(**kw),
    }
    if _HAS_TV_DETECTION
    else {}
)

_TORCHVISION_WEIGHT_DEFAULTS = {
    "retinanet_resnet50_fpn": RetinaNet_ResNet50_FPN_Weights.DEFAULT,
    "fcos_resnet50_fpn": FCOS_ResNet50_FPN_Weights.DEFAULT,
}


# ═══════════════════════════════════════════════════════════════════════
#  5. YOLO / DETR name mapping
# ═══════════════════════════════════════════════════════════════════════

# Maps user-facing config names → (wrapper_class, constructor_arg)
_YOLO_MODELS: Dict[str, str] = {
    # YOLOv5
    "yolov5n": "yolov5nu.pt",
    "yolov5s": "yolov5su.pt",
    "yolov5m": "yolov5mu.pt",
    "yolov5l": "yolov5lu.pt",
    "yolov5x": "yolov5xu.pt",
    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    # YOLOv10
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10l": "yolov10l.pt",
    "yolov10x": "yolov10x.pt",
    # YOLO11
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    # YOLO12
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt",
    "yolo12x": "yolo12x.pt",
    # YOLO26
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
}

_DETR_MODELS: Dict[str, str] = {
    "detr_resnet50": "facebook/detr-resnet-50",
    "detr_resnet101": "facebook/detr-resnet-101",
    "conditional_detr_resnet50": "microsoft/conditional-detr-resnet-50",
}

_RTDETR_MODELS: Dict[str, str] = {
    "rtdetr_l": "rtdetr-l.pt",
    "rtdetr_x": "rtdetr-x.pt",
}

_RFDETR_MODELS: Dict[str, str] = {
    "rfdetr_nano": "rf-detr-nano.pth",
    "rfdetr_small": "rf-detr-small.pth",
    "rfdetr_medium": "rf-detr-medium.pth",
    "rfdetr_large": "rf-detr-large-2026.pth",
}

_TINYNEXT_MODELS: Dict[str, str] = {
    "tinynext_s": "tinynext_s.pth",
    "tinynext_m": "tinynext_m.pth",
}
_TINYNEXT_REPO_MODELS: Dict[str, str] = {
    "tinynext_s": "ssdlite_tinynext_s_coco.pth",
    "tinynext_m": "ssdlite_tinynext_m_coco.pth",
}
_TINYNEXT_ANCHORS_PER_LOCATION = 6

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _normalise_model_name(name: str) -> str:
    return name.lower().replace("-", "_")


def get_models_dir() -> Path:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _MODELS_DIR


def get_model_artifact_path(name: str) -> Path:
    name_lower = _normalise_model_name(name)
    model_info = model_lib.get(name_lower, {})
    artifact_name = model_info.get("model_path", name_lower)
    return get_models_dir() / artifact_name


def _expected_hash_prefix(filename: str) -> str | None:
    stem = Path(filename).name
    if "." not in stem or "-" not in stem:
        return None
    suffix = stem.rsplit("-", 1)[-1].split(".", 1)[0].lower()
    if suffix and all(ch in "0123456789abcdef" for ch in suffix):
        return suffix
    return None


def _matches_expected_hash(path: Path, hash_prefix: str | None) -> bool:
    if hash_prefix is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().startswith(hash_prefix)


def _matches_md5(path: Path, expected_md5: str | None) -> bool:
    if expected_md5 is None:
        return True
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_md5.lower()


def _download_file(url: str, destination: Path, hash_prefix: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".download")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        download_url_to_file(url, str(tmp_path), hash_prefix=hash_prefix, progress=True)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return destination


def _download_http_file_with_resume(
    url: str,
    destination: Path,
    *,
    expected_md5: str | None = None,
    max_attempts: int = 5,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".download")
    for attempt in range(1, max_attempts + 1):
        existing_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status()
                if existing_bytes and response.status_code != 206:
                    tmp_path.unlink(missing_ok=True)
                    existing_bytes = 0
                mode = "ab" if existing_bytes and response.status_code == 206 else "wb"
                with tmp_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            if not _matches_md5(tmp_path, expected_md5):
                logger.warning(
                    "Downloaded model artifact failed MD5 validation on attempt {}/{}.",
                    attempt,
                    max_attempts,
                )
                tmp_path.unlink(missing_ok=True)
                continue

            tmp_path.replace(destination)
            return destination
        except (requests.RequestException, OSError) as exc:
            logger.warning(
                "Download attempt {}/{} failed for {}: {}",
                attempt,
                max_attempts,
                destination.name,
                safe_error_summary(exc),
            )
            if attempt == max_attempts:
                break

    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {destination.name} after {max_attempts} attempts")


def _ensure_torchvision_artifact(name: str) -> Path:
    name_lower = _normalise_model_name(name)
    artifact_path = get_model_artifact_path(name_lower)
    hash_prefix = _expected_hash_prefix(artifact_path.name)
    if artifact_path.is_file() and _matches_expected_hash(artifact_path, hash_prefix):
        return artifact_path
    if artifact_path.exists():
        logger.warning("Existing {} weights failed validation; redownloading.", name_lower)
        artifact_path.unlink()
    weights_enum = _TORCHVISION_WEIGHT_DEFAULTS[name_lower]
    logger.info("Downloading {} weights.", name_lower)
    return _download_file(weights_enum.url, artifact_path, hash_prefix=hash_prefix)


def _has_hf_snapshot(local_dir: Path) -> bool:
    if not local_dir.is_dir():
        return False
    has_config = (local_dir / "config.json").exists()
    has_weights = any(
        (local_dir / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin")
    )
    has_processor = (local_dir / "preprocessor_config.json").exists()
    return has_config and has_weights and has_processor


def _ensure_detr_artifact(name: str) -> Path:
    name_lower = _normalise_model_name(name)
    artifact_path = get_model_artifact_path(name_lower)
    if _has_hf_snapshot(artifact_path):
        return artifact_path
    artifact_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} weights.", name_lower)
    snapshot_download(
        repo_id=_DETR_MODELS[name_lower],
        local_dir=artifact_path,
        cache_dir=get_models_dir() / ".hf_cache",
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite"],
    )
    return artifact_path


def _ensure_ultralytics_artifact(name: str) -> Path:
    name_lower = _normalise_model_name(name)
    artifact_path = get_model_artifact_path(name_lower)
    if artifact_path.is_file():
        return artifact_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} weights.", name_lower)
    from ultralytics.utils.downloads import attempt_download_asset

    # Let Ultralytics choose the correct release tag for newer families such as
    # YOLO26. Forcing ``latest`` can point at a release that does not publish
    # the requested asset and leaves us with a non-existent path.
    downloaded = Path(attempt_download_asset(str(artifact_path)))
    if downloaded.resolve() != artifact_path.resolve():
        artifact_path.write_bytes(downloaded.read_bytes())
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Ultralytics download did not produce the expected artifact: {artifact_path}"
        )
    return artifact_path


def _ensure_tinynext_artifact(name: str) -> Path:
    name_lower = _normalise_model_name(name)
    artifact_path = get_model_artifact_path(name_lower)
    repo_filename = _TINYNEXT_REPO_MODELS[name_lower]
    if artifact_path.is_file():
        try:
            if _tinynext_checkpoint_has_detector_weights(artifact_path):
                return artifact_path
            logger.info(
                "Replacing {} backbone-only weights with detector checkpoint.",
                name_lower,
            )
        except Exception as exc:
            logger.warning(
                "Failed to inspect {} weights; refreshing detector checkpoint: {}.",
                name_lower,
                safe_error_summary(exc),
            )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} detector weights.", name_lower)
    downloaded = hf_hub_download(
        repo_id="yuffeenn/TinyNeXt",
        filename=repo_filename,
        local_dir=artifact_path.parent,
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != artifact_path.resolve():
        artifact_path.write_bytes(downloaded_path.read_bytes())
    return artifact_path


@contextmanager
def _tinynext_mmengine_checkpoint_shim():
    """Provide the minimal mmengine surface needed to unpickle TinyNeXt ckpts."""
    previous_modules = {
        name: sys.modules.get(name)
        for name in (
            "mmengine",
            "mmengine.logging",
            "mmengine.logging.history_buffer",
        )
    }

    mmengine_module = types.ModuleType("mmengine")
    logging_module = types.ModuleType("mmengine.logging")
    history_module = types.ModuleType("mmengine.logging.history_buffer")

    class HistoryBuffer:
        max_length = 1_000_000

        def __init__(self, *args, **kwargs):
            self._log_history = []

        @classmethod
        def min(cls, *args, **kwargs):
            return None

        @classmethod
        def max(cls, *args, **kwargs):
            return None

        @classmethod
        def mean(cls, *args, **kwargs):
            return None

        @classmethod
        def current(cls, *args, **kwargs):
            return None

    history_module.HistoryBuffer = HistoryBuffer
    logging_module.history_buffer = history_module
    mmengine_module.logging = logging_module

    sys.modules["mmengine"] = mmengine_module
    sys.modules["mmengine.logging"] = logging_module
    sys.modules["mmengine.logging.history_buffer"] = history_module
    try:
        yield
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_tinynext_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> object:
    checkpoint_path = str(path)
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except (ModuleNotFoundError, AttributeError) as exc:
        message = str(exc)
        missing_mmengine = isinstance(exc, ModuleNotFoundError) and str(
            getattr(exc, "name", "")
        ).startswith("mmengine")
        needs_history_buffer_shim = "HistoryBuffer" in message or "mmengine" in message
        if not missing_mmengine and not needs_history_buffer_shim:
            raise
        with _tinynext_mmengine_checkpoint_shim():
            return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _extract_tinynext_checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected TinyNeXt state dict type: {type(state_dict)!r}")
    return {str(key): value for key, value in state_dict.items()}


def _looks_like_tinynext_official_detector_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return (
        any(key.startswith("bbox_head.cls_convs.") for key in state_dict.keys())
        and any(key.startswith("neck.extra_layers.") for key in state_dict.keys())
        and any(key.startswith("backbone.") for key in state_dict.keys())
    )


def _looks_like_tinynext_internal_detector_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(
        key.startswith("head.")
        or key.startswith("backbone.backbone.")
        or key.startswith("backbone.extra.")
        for key in state_dict.keys()
    )


def infer_tinynext_state_dict_num_classes(state_dict: object) -> int | None:
    if not isinstance(state_dict, Mapping):
        return None
    class_counts: list[int] = []
    classifier_pattern = re.compile(
        r"(?:^bbox_head\.cls_convs\.\d+\.1|^head\.classification_head\.module_list\.\d+\.1)\.(?:weight|bias)$"
    )
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value) or value.ndim < 1:
            continue
        if not classifier_pattern.search(key):
            continue
        out_channels = int(value.shape[0])
        if out_channels > 0 and out_channels % _TINYNEXT_ANCHORS_PER_LOCATION == 0:
            class_counts.append(out_channels // _TINYNEXT_ANCHORS_PER_LOCATION)

    unique_counts = set(class_counts)
    if len(unique_counts) != 1:
        return None
    return unique_counts.pop()


def _tinynext_checkpoint_has_detector_weights(path: str | Path) -> bool:
    checkpoint = _load_tinynext_checkpoint(path, device="cpu")
    state_dict = _extract_tinynext_checkpoint_state_dict(checkpoint)
    return _looks_like_tinynext_official_detector_state_dict(
        state_dict
    ) or _looks_like_tinynext_internal_detector_state_dict(state_dict)


def _convert_tinynext_classifier_tensor(
    tensor: torch.Tensor,
    *,
    target_num_classes: int,
    source_num_classes: int = 81,
    label_schema: str = "coco_91",
) -> torch.Tensor:
    """Map TinyNeXt MMDetection SSD logits to torchvision's SSD label layout."""
    source_background_index = source_num_classes - 1
    if tensor.shape[0] % source_num_classes != 0:
        raise RuntimeError(
            f"Unexpected TinyNeXt classifier tensor shape {tuple(tensor.shape)}; "
            f"expected the leading dimension to be divisible by {source_num_classes}."
        )

    num_anchors = tensor.shape[0] // source_num_classes
    converted_shape = (num_anchors * target_num_classes, *tensor.shape[1:])
    converted = tensor.new_zeros(converted_shape)

    for anchor_index in range(num_anchors):
        source_start = anchor_index * source_num_classes
        dest_start = anchor_index * target_num_classes
        background_value = tensor[source_start + source_background_index].clone()
        for dest_class in range(target_num_classes):
            converted[dest_start + dest_class] = background_value
        converted[dest_start] = background_value
        for source_class in range(source_background_index):
            if str(label_schema or "coco_91").strip().lower() == "zero_based":
                mapped_label = source_class + 1
            else:
                mapped_label = COCO_80_TO_91[source_class]
            if mapped_label >= target_num_classes:
                continue
            converted[dest_start + mapped_label] = tensor[source_start + source_class]
    return converted


def _map_tinynext_official_extra_key(key: str) -> str | None:
    parts = key.split(".")
    if len(parts) < 6:
        return None
    layer_index = parts[2]
    block_index = parts[3]
    if block_index == "0":
        block_target = "0"
        branch_kind = parts[4]
        suffix = ".".join(parts[5:])
        if branch_kind == "conv":
            return f"backbone.extra.{layer_index}.{block_target}.0.{suffix}"
        if branch_kind == "bn":
            return f"backbone.extra.{layer_index}.{block_target}.1.{suffix}"
        return None

    if block_index != "1" or len(parts) < 7:
        return None
    branch_kind = parts[4]
    op_kind = parts[5]
    suffix = ".".join(parts[6:])
    if branch_kind == "depthwise_conv":
        target_block = "1"
    elif branch_kind == "pointwise_conv":
        target_block = "2"
    else:
        return None
    if op_kind == "conv":
        return f"backbone.extra.{layer_index}.{target_block}.0.{suffix}"
    if op_kind == "bn":
        return f"backbone.extra.{layer_index}.{target_block}.1.{suffix}"
    return None


def _map_tinynext_official_head_key(key: str, *, head_name: str) -> str | None:
    parts = key.split(".")
    if len(parts) < 5:
        return None
    layer_index = parts[2]
    branch_index = parts[3]
    base = f"head.{head_name}_head.module_list.{layer_index}"
    if branch_index == "0" and len(parts) >= 6:
        op_kind = parts[4]
        suffix = ".".join(parts[5:])
        if op_kind == "conv":
            return f"{base}.0.0.{suffix}"
        if op_kind == "bn":
            return f"{base}.0.1.{suffix}"
        return None
    if branch_index == "1":
        suffix = ".".join(parts[4:])
        return f"{base}.1.{suffix}"
    return None


def _convert_tinynext_official_detector_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    target_num_classes: int,
    source_num_classes: int = 81,
    label_schema: str = "coco_91",
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key: str | None = None
        converted_value = value
        if key.startswith("backbone."):
            new_key = f"backbone.backbone.{key.split('backbone.', 1)[1]}"
        elif key.startswith("neck.extra_layers."):
            new_key = _map_tinynext_official_extra_key(key)
        elif key.startswith("bbox_head.reg_convs."):
            new_key = _map_tinynext_official_head_key(key, head_name="regression")
        elif key.startswith("bbox_head.cls_convs."):
            new_key = _map_tinynext_official_head_key(key, head_name="classification")
            if key.endswith(".1.weight") or key.endswith(".1.bias"):
                converted_value = _convert_tinynext_classifier_tensor(
                    value,
                    target_num_classes=target_num_classes,
                    source_num_classes=source_num_classes,
                    label_schema=label_schema,
                )
        if new_key is not None:
            converted[new_key] = converted_value
    return converted


def _remap_tinynext_public_detections(model: nn.Module, detections: object) -> object:
    if str(getattr(model, "label_schema", "coco_91")).strip().lower() != "zero_based":
        return detections
    if not isinstance(detections, (list, tuple)):
        return detections

    remapped: list[object] = []
    changed = False
    for detection in detections:
        if not isinstance(detection, Mapping):
            remapped.append(detection)
            continue
        labels = detection.get("labels")
        if not torch.is_tensor(labels):
            remapped.append(detection)
            continue
        copied = dict(detection)
        copied["labels"] = (labels.to(dtype=torch.int64) - 1).clamp_min(0)
        remapped.append(copied)
        changed = True
    if not changed:
        return detections
    return tuple(remapped) if isinstance(detections, tuple) else remapped


def _tinynext_forward_label_hook(module: nn.Module, _inputs: object, output: object) -> object:
    return _remap_tinynext_public_detections(module, output)


def _configure_tinynext_label_schema(model: nn.Module, *, num_classes: int) -> None:
    setattr(model, "num_classes", int(num_classes))
    setattr(model, "label_schema", "coco_91" if int(num_classes) >= 91 else "zero_based")
    if hasattr(model, "register_forward_hook") and not bool(
        getattr(model, "_plank_public_label_hook_registered", False)
    ):
        model.register_forward_hook(_tinynext_forward_label_hook)
        setattr(model, "_plank_public_label_hook_registered", True)


def _load_rfdetr_checkpoint(artifact_path: Path, *, device: str | torch.device = "cpu") -> object:
    return torch.load(artifact_path, map_location=device, weights_only=False)


def _load_ultralytics_checkpoint(
    artifact_path: Path, *, device: str | torch.device = "cpu"
) -> object:
    return torch.load(artifact_path, map_location=device, weights_only=False)


def _extract_ultralytics_checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "ema", "state_dict"):
            nested = checkpoint.get(key)
            if hasattr(nested, "state_dict"):
                return {
                    str(state_key): value
                    for state_key, value in nested.state_dict().items()
                    if isinstance(state_key, str) and torch.is_tensor(value)
                }
            if isinstance(nested, Mapping):
                return {
                    str(state_key): value
                    for state_key, value in nested.items()
                    if isinstance(state_key, str) and torch.is_tensor(value)
                }
        if any(torch.is_tensor(value) for value in checkpoint.values()):
            return {
                str(state_key): value
                for state_key, value in checkpoint.items()
                if isinstance(state_key, str) and torch.is_tensor(value)
            }
    return {}


def infer_ultralytics_state_dict_num_classes(state_dict: object) -> int | None:
    if not isinstance(state_dict, Mapping):
        return None

    class_counts: list[int] = []
    yolo_head_pattern = re.compile(r"(?:^|\.)(?:one2one_)?cv3\.\d+\.2\.(?:weight|bias)$")
    rtdetr_head_pattern = re.compile(
        r"(?:^|\.)(?:denoising_class_embed|enc_score_head|dec_score_head\.\d+)\.(?:weight|bias)$"
    )
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value) or value.ndim < 1:
            continue
        if yolo_head_pattern.search(key) or rtdetr_head_pattern.search(key):
            count = int(value.shape[0])
            if count > 0:
                class_counts.append(count)

    unique_counts = set(class_counts)
    if len(unique_counts) != 1:
        return None
    return unique_counts.pop()


def _infer_ultralytics_checkpoint_num_classes(checkpoint: object) -> int | None:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "ema"):
            model_obj = checkpoint.get(key)
            if model_obj is None:
                continue
            try:
                model_nc = int(getattr(model_obj, "nc"))
            except (AttributeError, TypeError, ValueError):
                model_nc = 0
            if model_nc > 0:
                return model_nc

            names = getattr(model_obj, "names", None)
            if isinstance(names, Mapping) and len(names) > 0:
                return len(names)
            if isinstance(names, (list, tuple)) and names:
                return len(names)

    state_count = infer_ultralytics_state_dict_num_classes(
        _extract_ultralytics_checkpoint_state_dict(checkpoint)
    )
    if state_count is not None:
        return state_count
    return None


def _maybe_infer_ultralytics_num_classes(
    artifact_path: Path | None,
    *,
    device: str | torch.device,
    model_label: str,
    current_num_classes: int,
) -> int:
    if artifact_path is None or not artifact_path.is_file():
        return int(current_num_classes)
    try:
        inferred_class_count = _infer_ultralytics_checkpoint_num_classes(
            _load_ultralytics_checkpoint(artifact_path, device=device)
        )
    except Exception as exc:
        logger.warning(
            "[ModelZoo] Failed to inspect {} weights: {}",
            model_label,
            safe_error_summary(exc),
        )
        return int(current_num_classes)
    if inferred_class_count is None or inferred_class_count == 80:
        return int(current_num_classes)
    logger.info(
        "[ModelZoo] Inferred {} {} class(es); using native zero-based labels.",
        inferred_class_count,
        model_label,
    )
    return int(inferred_class_count)


def _normalise_rfdetr_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping):
        return {}
    tensor_items = {
        str(key): value
        for key, value in state_dict.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if not tensor_items:
        return {}

    model_prefix = "model."
    prefixed_count = sum(1 for key in tensor_items if key.startswith(model_prefix))
    if prefixed_count and prefixed_count == len(tensor_items):
        return {key[len(model_prefix) :]: value for key, value in tensor_items.items()}
    return dict(tensor_items)


def _extract_rfdetr_checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        nested_model = checkpoint.get("model")
        if isinstance(nested_model, dict):
            model_state = _normalise_rfdetr_state_dict_keys(nested_model)
            if model_state:
                return model_state

        nested_state = checkpoint.get("state_dict")
        if isinstance(nested_state, dict):
            state = _normalise_rfdetr_state_dict_keys(nested_state)
            if state:
                return state

        if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
            state = _normalise_rfdetr_state_dict_keys(checkpoint)
            if state:
                return state

    raise ValueError("RF-DETR checkpoint does not contain a readable model state_dict.")


def _rfdetr_checkpoint_class_count(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    for key in ("class_embed.bias", "class_embed.weight"):
        value = state_dict.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    return None


def infer_rfdetr_state_dict_num_classes(state_dict: object) -> int | None:
    if not isinstance(state_dict, Mapping):
        return None
    return _rfdetr_checkpoint_class_count(_normalise_rfdetr_state_dict_keys(state_dict))


def _validate_rfdetr_artifact_checkpoint(
    artifact_path: Path,
    checkpoint: object,
    *,
    expected_class_count: int | None = None,
) -> dict[str, torch.Tensor]:
    state = _extract_rfdetr_checkpoint_state_dict(checkpoint)
    class_count = _rfdetr_checkpoint_class_count(state)
    if (
        expected_class_count is not None
        and class_count is not None
        and class_count != int(expected_class_count)
    ):
        raise ValueError(
            "RF-DETR checkpoint class head has "
            f"{class_count} logits, expected {expected_class_count} for {artifact_path.name}"
        )
    return state


def _validate_rfdetr_state_dict_matches_model(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    artifact_path: Path,
) -> None:
    rfdetr = getattr(model, "rfdetr", None)
    model_context = getattr(rfdetr, "model", None)
    inner_model = getattr(model_context, "model", None)
    if inner_model is None or not hasattr(inner_model, "state_dict"):
        return

    wrapper_keys = {
        _DETECTION_THRESHOLD_LOW_BUFFER,
        _DETECTION_THRESHOLD_HIGH_BUFFER,
        _RFDETR_CACHE_VERSION_BUFFER,
    }
    incoming_keys = set(state_dict) - wrapper_keys
    model_keys = set(inner_model.state_dict().keys())
    if incoming_keys and not (incoming_keys & model_keys):
        raise RuntimeError(
            "RF-DETR weights did not match the model architecture: "
            f"{artifact_path} has {len(incoming_keys)} tensor key(s), "
            "but none match the inner RF-DETR model. "
            "The checkpoint may use an unsupported key prefix or belong to a different model."
        )


def _ensure_rfdetr_artifact(name: str) -> Path:
    name_lower = _normalise_model_name(name)
    artifact_path = get_model_artifact_path(name_lower)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_RFDETR:
        raise ImportError("rfdetr is required for RF-DETR models. Install: pip install rfdetr")

    from rfdetr.assets.model_weights import ModelWeights

    asset = ModelWeights.from_filename(artifact_path.name)
    if asset is None:
        raise RuntimeError(f"No upstream RF-DETR asset metadata found for {artifact_path.name}")

    if artifact_path.is_file():
        if _matches_md5(artifact_path, asset.md5_hash):
            return artifact_path
        try:
            state = _validate_rfdetr_artifact_checkpoint(
                artifact_path,
                _load_rfdetr_checkpoint(artifact_path),
            )
            class_count = _rfdetr_checkpoint_class_count(state)
            logger.info(
                "Reusing readable {} weights despite MD5 mismatch{}.",
                name_lower,
                (f" (class head logits={class_count})" if class_count is not None else ""),
            )
            return artifact_path
        except Exception as exc:
            logger.warning(
                "Existing {} weights are unreadable; re-downloading: {}.",
                name_lower,
                safe_error_summary(exc),
            )
            artifact_path.unlink(missing_ok=True)

    logger.info("Downloading {} weights.", name_lower)
    _download_http_file_with_resume(
        asset.url,
        artifact_path,
        expected_md5=asset.md5_hash,
    )
    return artifact_path


def ensure_local_model_artifact(name: str) -> Path:
    name_lower = _normalise_model_name(name)
    if name_lower in _TORCHVISION_WEIGHT_DEFAULTS:
        return _ensure_torchvision_artifact(name_lower)
    if name_lower in _DETR_MODELS:
        return _ensure_detr_artifact(name_lower)
    if name_lower in _YOLO_MODELS or name_lower in _RTDETR_MODELS:
        return _ensure_ultralytics_artifact(name_lower)
    if name_lower in _RFDETR_MODELS:
        return _ensure_rfdetr_artifact(name_lower)
    if name_lower in _TINYNEXT_MODELS:
        return _ensure_tinynext_artifact(name_lower)
    return get_model_artifact_path(name_lower)


# ═══════════════════════════════════════════════════════════════════════
#  6. Unified factory
# ═══════════════════════════════════════════════════════════════════════


def build_detection_model(
    name: str,
    num_classes: int = 91,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    weights_path: Optional[str] = None,
    confidence: float = 0.01,
    **kwargs,
) -> nn.Module:
    """Build any supported detection model by name.

    Parameters
    ----------
    name : str
        Model identifier (case-insensitive).  See the module docstring
        for the full list.
    num_classes : int
        Number of detection classes.  91 = COCO (default).
    pretrained : bool
        If ``True``, load pre-trained weights (COCO).
    device : str or torch.device
    weights_path : str, optional
        Path to a local ``.pth`` / ``.pt`` weights file.
        If provided, overrides the default pre-trained weights.
    confidence : float
        Confidence threshold for wrapper models (YOLO / DETR).
    **kwargs
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    nn.Module
        A model whose ``forward([img_tensor])`` returns the standard
        torchvision detection output format.
    """
    name_lower = _normalise_model_name(name)
    explicit_weights_path = weights_path is not None
    artifact_path = Path(weights_path) if explicit_weights_path else None
    if artifact_path is not None and not artifact_path.exists():
        raise FileNotFoundError(f"Weights path does not exist: {artifact_path}")

    # ── 1. Torchvision built-in ──
    if name_lower in _TORCHVISION_BUILTIN:
        build_fn = _TORCHVISION_BUILTIN[name_lower]
        try:
            model = build_fn(weights=None, weights_backbone=None, **kwargs)
        except TypeError:
            # Some constructors do not expose weights_backbone.
            try:
                model = build_fn(weights=None, **kwargs)
            except TypeError:
                model = build_fn(pretrained=False, pretrained_backbone=False, **kwargs)

        ensure_detection_threshold_state(model, name_lower)

        if artifact_path is None and pretrained:
            artifact_path = ensure_local_model_artifact(name_lower)

        if artifact_path is not None and artifact_path.is_file():
            state = torch.load(artifact_path, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)
            logger.info("[ModelZoo] Loaded weights for {}.", name)

        model.to(device)
        model.eval()
        logger.info("[ModelZoo] Built torchvision model: {}", name)
        return model

    # ── 2. YOLO ──
    if name_lower in _YOLO_MODELS:
        if artifact_path is None and pretrained:
            artifact_path = ensure_local_model_artifact(name_lower)
        num_classes = _maybe_infer_ultralytics_num_classes(
            artifact_path,
            device=device,
            model_label="YOLO",
            current_num_classes=num_classes,
        )
        model_source = str(artifact_path) if artifact_path is not None else f"{name_lower}.yaml"
        model = YOLODetectionModel(
            model_name=model_source,
            confidence=confidence,
            device=device,
            num_classes=num_classes,
        )
        logger.info("[ModelZoo] Built YOLO model: {}.", name)
        return model

    # ── 3. DETR (HuggingFace) ──
    if name_lower in _DETR_MODELS:
        model_source = str(artifact_path) if artifact_path is not None else _DETR_MODELS[name_lower]
        if pretrained and artifact_path is None:
            artifact_path = ensure_local_model_artifact(name_lower)
            model_source = str(artifact_path)
        model = DETRDetectionModel(
            model_name_or_path=model_source,
            confidence=confidence,
            device=device,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        logger.info("[ModelZoo] Built DETR model: {}.", name)
        return model

    # ── 4. RT-DETR (ultralytics) ──
    if name_lower in _RFDETR_MODELS:
        if artifact_path is None and pretrained:
            artifact_path = ensure_local_model_artifact(name_lower)
        should_load_local_weights = artifact_path is not None and artifact_path.is_file()
        rfdetr_checkpoint = None
        rfdetr_state = None
        if should_load_local_weights:
            rfdetr_checkpoint = _load_rfdetr_checkpoint(artifact_path, device=device)
            rfdetr_state = _extract_rfdetr_checkpoint_state_dict(rfdetr_checkpoint)
            inferred_class_count = _rfdetr_checkpoint_class_count(rfdetr_state)
            if inferred_class_count is not None and inferred_class_count != int(num_classes):
                logger.info(
                    "[ModelZoo] Inferred {} RF-DETR logits; building {} with {} "
                    "foreground class(es).",
                    inferred_class_count,
                    name,
                    max(1, inferred_class_count - 1),
                )
                num_classes = inferred_class_count
        model = RFDETRDetectionModel(
            model_name=name_lower,
            confidence=confidence,
            device=device,
            num_classes=num_classes,
            pretrained=pretrained and not should_load_local_weights,
            **kwargs,
        )
        if should_load_local_weights:
            state = rfdetr_state
            if state is None:
                state = _extract_rfdetr_checkpoint_state_dict(
                    rfdetr_checkpoint
                    if rfdetr_checkpoint is not None
                    else _load_rfdetr_checkpoint(artifact_path, device=device)
                )
            _validate_rfdetr_state_dict_matches_model(
                model,
                state,
                artifact_path=artifact_path,
            )
            model.load_state_dict(state, strict=False)
            logger.info("[ModelZoo] Loaded weights for {}.", name)
        logger.info("[ModelZoo] Built RF-DETR model: {}", name)
        return model

    if name_lower in _TINYNEXT_MODELS:
        checkpoint_state_dict: dict[str, torch.Tensor] | None = None
        backbone_weights_path = None
        tinynext_source_num_classes = 81
        tinynext_label_schema = "coco_91" if int(num_classes) >= 91 else "zero_based"
        tinynext_input_size = int(kwargs.pop("tinynext_input_size", kwargs.pop("image_size", 320)))
        if artifact_path is None and pretrained:
            artifact_path = ensure_local_model_artifact(name_lower)
        if artifact_path is not None and artifact_path.is_file():
            checkpoint = _load_tinynext_checkpoint(artifact_path, device="cpu")
            checkpoint_state_dict = _extract_tinynext_checkpoint_state_dict(checkpoint)
            is_official_tinynext = _looks_like_tinynext_official_detector_state_dict(
                checkpoint_state_dict
            )
            is_internal_tinynext = _looks_like_tinynext_internal_detector_state_dict(
                checkpoint_state_dict
            )
            inferred_class_count = infer_tinynext_state_dict_num_classes(checkpoint_state_dict)
            if is_official_tinynext:
                if inferred_class_count is not None and inferred_class_count != 81:
                    num_classes = inferred_class_count
                    tinynext_source_num_classes = inferred_class_count
                    tinynext_label_schema = "zero_based"
                    logger.info(
                        "[ModelZoo] Inferred {} TinyNeXt MMDetection class logits "
                        "building native zero-based TinyNeXt detector.",
                        inferred_class_count,
                    )
            elif is_internal_tinynext:
                if inferred_class_count is not None and inferred_class_count != int(num_classes):
                    num_classes = inferred_class_count
                    logger.info(
                        "[ModelZoo] Inferred {} TinyNeXt SSD class logits; building "
                        "matching detector head.",
                        inferred_class_count,
                    )
                tinynext_label_schema = "coco_91" if int(num_classes) >= 91 else "zero_based"
            else:
                backbone_weights_path = str(artifact_path)
        model = build_tinynext_detector(
            name_lower,
            num_classes=num_classes,
            device=device,
            backbone_weights_path=backbone_weights_path,
            image_size=tinynext_input_size,
        )
        _configure_tinynext_label_schema(model, num_classes=num_classes)
        ensure_detection_threshold_state(model, name_lower)
        if checkpoint_state_dict is not None and backbone_weights_path is None:
            if _looks_like_tinynext_official_detector_state_dict(checkpoint_state_dict):
                checkpoint_state_dict = _convert_tinynext_official_detector_state_dict(
                    checkpoint_state_dict,
                    target_num_classes=num_classes,
                    source_num_classes=tinynext_source_num_classes,
                    label_schema=tinynext_label_schema,
                )
            model.load_state_dict(checkpoint_state_dict, strict=False)
            logger.info("[ModelZoo] Loaded weights for {}.", name)
        model.eval()
        logger.info("[ModelZoo] Built TinyNeXt detector: {}", name)
        return model

    if name_lower in _RTDETR_MODELS:
        if artifact_path is None:
            artifact_path = ensure_local_model_artifact(name_lower)
        num_classes = _maybe_infer_ultralytics_num_classes(
            artifact_path,
            device=device,
            model_label="RT-DETR",
            current_num_classes=num_classes,
        )
        pt_name = str(artifact_path) if artifact_path is not None else _RTDETR_MODELS[name_lower]
        model = RTDETRDetectionModel(
            model_name=pt_name,
            confidence=confidence,
            device=device,
            num_classes=num_classes,
        )
        logger.info("[ModelZoo] Built RT-DETR model: {}.", name)
        return model

    # ── 5. Fallback: try torchvision eval ──
    raise ValueError(f"Unknown detection model: '{name}'.  Available: {list_available_models()}")


def invalidate_wrapper_predictor(model: nn.Module) -> None:
    if isinstance(model, YOLODetectionModel):
        invalidate_predictor(model.yolo)
    elif isinstance(model, RTDETRDetectionModel):
        invalidate_predictor(model.rtdetr)


# ═══════════════════════════════════════════════════════════════════════
#  7. Helpers
# ═══════════════════════════════════════════════════════════════════════


def list_available_models() -> List[str]:
    """Return sorted list of all available model names."""
    models = list(_TORCHVISION_BUILTIN.keys())
    models += list(_YOLO_MODELS.keys())
    models += list(_DETR_MODELS.keys())
    models += list(_RFDETR_MODELS.keys())
    models += list(_RTDETR_MODELS.keys())
    models += list(_TINYNEXT_MODELS.keys())
    return sorted(set(models))


def build_model_sample_input(
    model_or_name,
    *,
    image_size: Tuple[int, int] = (224, 224),
    device: str | torch.device = "cpu",
):
    """Return a representative sample input that matches the model's public
    forward signature.

    Detection models in this repository use ``list[Tensor[C,H,W]]`` inputs,
    while generic backbones/classifiers use a batched ``Tensor[N,C,H,W]``.
    """
    height, width = image_size
    if isinstance(model_or_name, str):
        family = get_model_family(model_or_name)
        is_wrapper = family in ("retinanet", "fcos", "yolo", "detr", "rtdetr", "rfdetr", "tinynext")
    else:
        is_wrapper = (
            hasattr(model_or_name, "roi_heads")
            or hasattr(model_or_name, "transform")
            or is_wrapper_model(model_or_name)
        )
    sample = torch.rand(3, height, width, device=device)
    if is_wrapper:
        return [sample]
    return sample.unsqueeze(0)


def is_wrapper_model(model_or_name) -> bool:
    """Return True if model is a non-torchvision wrapper (YOLO/DETR/RF-DETR/RT-DETR).

    Accepts an ``nn.Module`` instance **or** a model name string.
    """
    if isinstance(model_or_name, str):
        return get_model_family(model_or_name) in ("yolo", "detr", "rtdetr", "rfdetr")
    return isinstance(
        model_or_name,
        (YOLODetectionModel, DETRDetectionModel, RFDETRDetectionModel, RTDETRDetectionModel),
    )


def model_has_roi_heads(model_or_name) -> bool:
    """Return True if the model has standard torchvision ``roi_heads``.

    Accepts an ``nn.Module`` instance **or** a model name string.
    """
    if isinstance(model_or_name, str):
        return False
    return hasattr(model_or_name, "roi_heads")


def get_model_family(name: str) -> str:
    """Return family string for the configured detector."""
    name_lower = name.lower().replace("-", "_")
    if "retinanet" in name_lower:
        return "retinanet"
    if "fcos" in name_lower:
        return "fcos"
    if name_lower in _YOLO_MODELS:
        return "yolo"
    if name_lower in _DETR_MODELS:
        return "detr"
    if name_lower in _RFDETR_MODELS:
        return "rfdetr"
    if name_lower in _RTDETR_MODELS:
        return "rtdetr"
    if name_lower in _TINYNEXT_MODELS:
        return "tinynext"
    return "unknown"


def get_detection_thresholds(model_name: str) -> tuple[float, float]:
    """Return model-family-aware low/high confidence cut-offs.

    The legacy 0.2/0.6 thresholds work well for YOLO-style detectors, but they
    are too aggressive for detector families whose public scores are naturally
    lower in this repository's wrapper pipeline, notably RF-DETR and TinyNeXt.
    """
    family = get_model_family(model_name)
    if family == "rfdetr":
        return 0.05, 0.20
    if family == "tinynext":
        return 0.02, 0.15
    return 0.2, 0.6


_DETECTION_THRESHOLD_LOW_BUFFER = "plank_threshold_low"
_DETECTION_THRESHOLD_HIGH_BUFFER = "plank_threshold_high"
_RFDETR_CACHE_VERSION_BUFFER = "plank_rfdetr_cache_version"
_RFDETR_CACHE_FORMAT_VERSION = 1.0
_RFDETR_NMS_IOU_THRESHOLD = 0.85


def ensure_detection_threshold_state(model: nn.Module, model_name: str) -> None:
    """Attach persistent threshold buffers used by serialized detector states."""
    if not hasattr(model, "register_buffer") or not hasattr(model, "_buffers"):
        return

    default_low, default_high = get_detection_thresholds(model_name)

    if _DETECTION_THRESHOLD_LOW_BUFFER not in model._buffers:
        model.register_buffer(
            _DETECTION_THRESHOLD_LOW_BUFFER,
            torch.tensor(float(default_low), dtype=torch.float32),
            persistent=True,
        )
    if _DETECTION_THRESHOLD_HIGH_BUFFER not in model._buffers:
        model.register_buffer(
            _DETECTION_THRESHOLD_HIGH_BUFFER,
            torch.tensor(float(default_high), dtype=torch.float32),
            persistent=True,
        )


def ensure_rfdetr_serialization_state(model: nn.Module) -> None:
    """Attach RF-DETR-specific metadata buffers used to validate cached weights."""
    if not hasattr(model, "register_buffer") or not hasattr(model, "_buffers"):
        return
    if _RFDETR_CACHE_VERSION_BUFFER not in model._buffers:
        model.register_buffer(
            _RFDETR_CACHE_VERSION_BUFFER,
            torch.tensor(float(_RFDETR_CACHE_FORMAT_VERSION), dtype=torch.float32),
            persistent=True,
        )


def has_compatible_rfdetr_cache_state(state_dict: object) -> bool:
    if not isinstance(state_dict, dict):
        return False
    version_value = state_dict.get(_RFDETR_CACHE_VERSION_BUFFER)
    if not torch.is_tensor(version_value):
        return False
    try:
        version = float(version_value.detach().cpu().item())
    except Exception:
        return False
    return version >= _RFDETR_CACHE_FORMAT_VERSION


def _split_rfdetr_serialized_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    wrapper_keys = {
        _DETECTION_THRESHOLD_LOW_BUFFER,
        _DETECTION_THRESHOLD_HIGH_BUFFER,
        _RFDETR_CACHE_VERSION_BUFFER,
    }
    wrapper_state: dict[str, torch.Tensor] = {}
    core_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in wrapper_keys:
            wrapper_state[key] = value
        else:
            core_state[key] = value
    return wrapper_state, core_state


def _postprocess_rfdetr_predictions(
    predictions: Mapping[str, torch.Tensor],
    *,
    target_sizes: torch.Tensor,
    threshold: float,
    num_classes: int,
    num_select: int,
    label_schema: str = "coco_91",
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Decode RF-DETR outputs into one label per query plus light NMS."""
    pred_logits = predictions["pred_logits"]
    pred_boxes = predictions["pred_boxes"]
    if pred_logits.ndim != 3 or pred_boxes.ndim != 3:
        raise RuntimeError("RF-DETR predictions must be [batch, queries, classes/4] tensors.")

    probabilities = pred_logits.sigmoid()
    label_offset = 0
    label_schema = str(label_schema or "coco_91").strip().lower()
    if label_schema == "zero_based" and probabilities.shape[-1] > 1:
        probabilities = probabilities[..., : max(1, probabilities.shape[-1] - 1)]
    elif probabilities.shape[-1] > 1:
        probabilities = probabilities[..., 1:]
        label_offset = 1

    scores_per_query, labels_per_query = probabilities.max(dim=-1)
    cx, cy, width, height = pred_boxes.unbind(-1)
    boxes_per_query = torch.stack(
        (
            cx - width * 0.5,
            cy - height * 0.5,
            cx + width * 0.5,
            cy + height * 0.5,
        ),
        dim=-1,
    )
    image_heights = target_sizes[:, 0].to(dtype=boxes_per_query.dtype)
    image_widths = target_sizes[:, 1].to(dtype=boxes_per_query.dtype)
    scale = torch.stack((image_widths, image_heights, image_widths, image_heights), dim=-1)
    boxes_per_query = boxes_per_query * scale.unsqueeze(1)
    boxes_per_query[..., 0::2] = boxes_per_query[..., 0::2].clamp(min=0)
    boxes_per_query[..., 1::2] = boxes_per_query[..., 1::2].clamp(min=0)
    boxes_per_query[..., 0] = torch.minimum(boxes_per_query[..., 0], image_widths.unsqueeze(1))
    boxes_per_query[..., 2] = torch.minimum(boxes_per_query[..., 2], image_widths.unsqueeze(1))
    boxes_per_query[..., 1] = torch.minimum(boxes_per_query[..., 1], image_heights.unsqueeze(1))
    boxes_per_query[..., 3] = torch.minimum(boxes_per_query[..., 3], image_heights.unsqueeze(1))
    detections: list[dict[str, torch.Tensor]] = []

    for batch_index in range(pred_logits.shape[0]):
        scores = scores_per_query[batch_index]
        labels = labels_per_query[batch_index]
        boxes = boxes_per_query[batch_index]

        if scores.numel():
            topk = min(int(num_select), int(scores.numel()))
            top_scores, top_indices = torch.topk(scores, topk, dim=0)
            scores = top_scores
            labels = labels.index_select(0, top_indices)
            boxes = boxes.index_select(0, top_indices)

        keep = scores > float(threshold)
        if not torch.any(keep):
            detections.append(
                {
                    "boxes": boxes.new_zeros((0, 4), dtype=torch.float32).cpu(),
                    "labels": labels.new_zeros((0,), dtype=torch.int64).cpu(),
                    "scores": scores.new_zeros((0,), dtype=torch.float32).cpu(),
                }
            )
            continue

        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        boxes = boxes.float()
        scores = scores.float()
        labels = labels.long()
        valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_geometry]
        labels = labels[valid_geometry]
        scores = scores[valid_geometry]

        if boxes.numel():
            nms_keep = batched_nms(boxes, scores, labels, _RFDETR_NMS_IOU_THRESHOLD)
            boxes = boxes.index_select(0, nms_keep)
            labels = labels.index_select(0, nms_keep)
            scores = scores.index_select(0, nms_keep)

        order = torch.argsort(scores, descending=True)
        boxes = boxes.index_select(0, order)
        labels = labels.index_select(0, order)
        scores = scores.index_select(0, order)

        if label_offset:
            labels = labels + int(label_offset)

        detections.append(
            {
                "boxes": boxes.detach().to(device="cpu", dtype=torch.float32),
                "labels": labels.detach().to(device="cpu", dtype=torch.int64),
                "scores": scores.detach().to(device="cpu", dtype=torch.float32),
            }
        )

    return detections


def get_model_detection_thresholds(
    model: nn.Module | None,
    model_name: str | None = None,
) -> tuple[float, float]:
    """Return model-specific thresholds, preferring serialized overrides."""
    default_low, default_high = get_detection_thresholds(str(model_name or ""))
    if model is None:
        return default_low, default_high

    low_buffer = getattr(model, _DETECTION_THRESHOLD_LOW_BUFFER, None)
    high_buffer = getattr(model, _DETECTION_THRESHOLD_HIGH_BUFFER, None)
    if torch.is_tensor(low_buffer) and torch.is_tensor(high_buffer):
        low_value = float(low_buffer.detach().cpu().item())
        high_value = float(high_buffer.detach().cpu().item())
        if np.isfinite(low_value) and np.isfinite(high_value) and 0.0 <= low_value <= high_value:
            return low_value, high_value
    return default_low, default_high


def set_model_detection_thresholds(
    model: nn.Module,
    *,
    threshold_low: float,
    threshold_high: float,
    model_name: str | None = None,
) -> None:
    """Persist calibrated detection thresholds inside the model state_dict."""
    ensure_detection_threshold_state(model, str(model_name or ""))
    getattr(model, _DETECTION_THRESHOLD_LOW_BUFFER).fill_(float(threshold_low))
    getattr(model, _DETECTION_THRESHOLD_HIGH_BUFFER).fill_(float(threshold_high))
