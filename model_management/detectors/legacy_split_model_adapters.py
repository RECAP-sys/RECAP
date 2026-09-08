from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection.fcos import FCOS
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.ssd import SSD
from torchvision.ops import boxes as box_ops
from ultralytics.models.utils.loss import RTDETRDetectionLoss

from model_management.detection_box_projection import (
    ORIGINAL_XYXY,
    project_original_xyxy_to_model_input_xyxy,
    require_coordinate_metadata,
)
from model_management.detectors.legacy_model_zoo import (
    COCO_80_TO_91,
    DETRDetectionModel,
    RFDETRDetectionModel,
    RTDETRDetectionModel,
    YOLODetectionModel,
    _postprocess_rfdetr_predictions,
    _remap_tinynext_public_detections,
)
from model_management.payload import BoundaryPayload
from model_management.ultralytics_parity import (
    postprocess_predictions,
    preprocess_bgr_images,
)

try:
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors
    from rfdetr.utilities.tensors import NestedTensor
except Exception:
    build_criterion_and_postprocessors = None
    NestedTensor = None


COCO_91_TO_80 = {label: idx for idx, label in enumerate(COCO_80_TO_91)}
_RFDETR_PACKED_AUX_OUTPUTS_MARKER = "__plank_rfdetr_packed_aux_outputs__"


class TorchvisionAnchorDetectorReplay(torch.nn.Module):
    """Replay-friendly anchor-detector core that operates on transformed tensors."""

    def __init__(self, detector: SSD) -> None:
        super().__init__()
        self.detector = detector
        self.backbone = detector.backbone
        self.head = detector.head

    def _as_transformed_batch(self, images: Any) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                return images
            if images.ndim == 3:
                return images.unsqueeze(0)
            raise TypeError(
                f"Unsupported anchor-detector replay tensor shape: {tuple(images.shape)!r}"
            )
        if isinstance(images, (list, tuple)):
            tensors: list[torch.Tensor] = []
            for image in images:
                if not isinstance(image, torch.Tensor):
                    raise TypeError(
                        f"Unsupported anchor-detector replay input type: {type(image)!r}"
                    )
                if image.ndim == 4:
                    tensors.extend(list(image))
                elif image.ndim == 3:
                    tensors.append(image)
                else:
                    raise TypeError(
                        f"Unsupported anchor-detector replay tensor shape: {tuple(image.shape)!r}"
                    )
            if not tensors:
                raise RuntimeError("Anchor-detector replay received an empty image batch.")
            return torch.stack(tensors, dim=0)
        raise TypeError(f"Unsupported anchor-detector replay input type: {type(images)!r}")

    def forward(self, images: Any) -> dict[str, torch.Tensor]:
        transformed_batch = self._as_transformed_batch(images)
        features = self.backbone(transformed_batch)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        if isinstance(features, dict):
            feature_list = list(features.values())
        elif isinstance(features, (list, tuple)):
            feature_list = list(features)
        else:
            feature_list = [features]
        outputs = self.head(feature_list)
        if isinstance(outputs, dict):
            return {
                str(key): value for key, value in outputs.items() if isinstance(value, torch.Tensor)
            }
        if isinstance(outputs, (list, tuple)):
            extracted = {}
            if len(outputs) >= 1 and isinstance(outputs[0], torch.Tensor):
                extracted["cls_logits"] = outputs[0]
            if len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor):
                extracted["bbox_regression"] = outputs[1]
            if len(outputs) >= 3 and isinstance(outputs[2], torch.Tensor):
                extracted["bbox_ctrness"] = outputs[2]
            if extracted:
                return extracted
        raise RuntimeError("Anchor-detector replay head did not return tensor outputs.")


class RFDETRReplay(torch.nn.Module):
    """Replay-friendly RF-DETR wrapper that preserves training-time auxiliaries."""

    def __init__(self, detector: RFDETRDetectionModel) -> None:
        super().__init__()
        self.detector = detector
        self.model = detector.rfdetr.model.model
        _patch_rfdetr_decoder_batch_polymorphism(self.model)

    @staticmethod
    def _normalize_images(images: torch.Tensor | Any) -> torch.Tensor | Any:
        if NestedTensor is None or not isinstance(images, torch.Tensor):
            return images
        if images.ndim != 4:
            return images
        batch_size, _, height, width = images.shape
        mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=images.device)
        return NestedTensor(images, mask)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(self._normalize_images(images))
        if isinstance(outputs, tuple):
            return {
                "pred_logits": _contiguous_tensor_tree(outputs[1]),
                "pred_boxes": _contiguous_tensor_tree(outputs[0]),
            }
        replayed = {
            "pred_logits": _contiguous_tensor_tree(outputs["pred_logits"]),
            "pred_boxes": _contiguous_tensor_tree(outputs["pred_boxes"]),
        }
        aux_outputs = outputs.get("aux_outputs")
        enc_outputs = outputs.get("enc_outputs")
        if isinstance(aux_outputs, (list, tuple, dict)):
            replayed["aux_outputs"] = _pack_rfdetr_aux_outputs(aux_outputs)
        if isinstance(enc_outputs, dict):
            replayed["enc_outputs"] = _contiguous_tensor_tree(enc_outputs)
        return replayed


def _patch_rfdetr_decoder_batch_polymorphism(model: torch.nn.Module) -> None:
    """Replace RF-DETR's training-only split(bs) path with tensor reshapes.

    The upstream decoder rebuilds grouped queries via ``tgt2.split(bs, dim=0)``.
    Split replay traces that list length from the canonical batch can reject
    larger batches. The equivalent reshape keeps the same tensor math while
    avoiding a Python list whose length depends on batch size.
    """

    if not isinstance(model, torch.nn.Module):
        return
    for module in model.modules():
        if getattr(module, "_plank_batch_polymorphic_rfdetr", False):
            continue
        if not _looks_like_rfdetr_decoder_layer(module):
            continue
        module.forward_post = MethodType(_rfdetr_decoder_forward_post_polymorphic, module)
        module._plank_batch_polymorphic_rfdetr = True


def _looks_like_rfdetr_decoder_layer(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "self_attn")
        and hasattr(module, "cross_attn")
        and hasattr(module, "linear1")
        and hasattr(module, "linear2")
        and hasattr(module, "norm1")
        and hasattr(module, "norm2")
        and hasattr(module, "norm3")
        and int(getattr(module, "group_detr", 1) or 1) > 1
    )


def _rfdetr_decoder_forward_post_polymorphic(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    tgt_key_padding_mask=None,
    memory_key_padding_mask=None,
    pos=None,
    query_pos=None,
    query_sine_embed=None,
    is_first=False,
    reference_points=None,
    spatial_shapes=None,
    level_start_index=None,
    spatial_shapes_hw=None,
):
    del memory_mask, query_sine_embed, is_first
    _batch_size, num_queries, _ = tgt.shape

    q = k = tgt + query_pos
    v = tgt
    group_queries = num_queries // int(self.group_detr)
    if self.training:
        q = torch.cat(q.split(group_queries, dim=1), dim=0)
        k = torch.cat(k.split(group_queries, dim=1), dim=0)
        v = torch.cat(v.split(group_queries, dim=1), dim=0)

    tgt2 = self.self_attn(
        q,
        k,
        v,
        attn_mask=tgt_mask,
        key_padding_mask=tgt_key_padding_mask,
        need_weights=False,
    )[0]

    if self.training:
        feature_dim = tgt2.shape[-1]
        tgt2 = (
            tgt2.reshape(int(self.group_detr), -1, group_queries, feature_dim)
            .transpose(0, 1)
            .reshape(-1, num_queries, feature_dim)
        )

    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    tgt2 = self.cross_attn(
        self.with_pos_embed(tgt, query_pos),
        reference_points,
        memory,
        spatial_shapes,
        level_start_index,
        memory_key_padding_mask,
        input_spatial_shapes_hw=spatial_shapes_hw,
    )
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt


def _is_anchor_detector(model: torch.nn.Module) -> bool:
    return isinstance(model, SSD) or (
        hasattr(model, "transform")
        and hasattr(model, "backbone")
        and hasattr(model, "head")
        and hasattr(model, "anchor_generator")
        and hasattr(model, "postprocess_detections")
        and hasattr(model, "compute_loss")
    )


def _is_ultralytics_detection_core(model: torch.nn.Module) -> bool:
    model_type = type(model)
    if not str(getattr(model_type, "__module__", "")).startswith("ultralytics."):
        return False
    if str(getattr(model, "task", "")).strip().lower() == "detect":
        return True
    yaml = getattr(model, "yaml", None)
    return (
        isinstance(yaml, Mapping)
        and "backbone" in yaml
        and "head" in yaml
        and ("nc" in yaml or "names" in yaml)
    )


def get_split_runtime_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, YOLODetectionModel):
        return model.yolo.model
    if isinstance(model, RTDETRDetectionModel):
        return model.rtdetr.model
    if isinstance(model, DETRDetectionModel):
        return model.detr
    if isinstance(model, RFDETRDetectionModel):
        return RFDETRReplay(model)
    if _is_anchor_detector(model):
        return TorchvisionAnchorDetectorReplay(model)
    return model


def build_split_runtime_sample_input(
    model: torch.nn.Module,
    *,
    image_size: tuple[int, int] = (224, 224),
    device: str | torch.device = "cpu",
):
    height, width = image_size
    device = torch.device(device)
    if isinstance(model, YOLODetectionModel):
        sample = (np.random.rand(height, width, 3) * 255).astype("uint8")
        _, tensor = preprocess_bgr_images(
            model.yolo,
            [sample],
            conf=model.confidence,
            imgsz=(height, width),
        )
        return tensor.to(device)
    if isinstance(model, RTDETRDetectionModel):
        sample = (np.random.rand(height, width, 3) * 255).astype("uint8")
        _, tensor = preprocess_bgr_images(
            model.rtdetr,
            [sample],
            conf=model.confidence,
            imgsz=(height, width),
        )
        return tensor.to(device)
    if isinstance(model, DETRDetectionModel):
        image = (np.random.rand(height, width, 3) * 255).astype("uint8")
        pixel_values = model.processor(
            images=Image.fromarray(image),
            return_tensors="pt",
        )["pixel_values"]
        return pixel_values.to(device)
    if isinstance(model, RFDETRDetectionModel):
        sample = torch.rand(3, height, width, device=device)
        batch_tensor, _ = model._prepare_batch([sample])
        return batch_tensor.to(device)
    if _is_anchor_detector(model):
        sample = torch.rand(3, height, width, device=device)
        transformed_images, _ = model.transform([sample], None)
        return transformed_images.tensors.to(device)
    return [torch.rand(3, height, width, device=device)]


def get_split_runtime_input_resize_mode(model: torch.nn.Module) -> str | None:
    if isinstance(model, (YOLODetectionModel, RTDETRDetectionModel)):
        return "letterbox"
    if _is_ultralytics_detection_core(model):
        return "letterbox"
    if isinstance(model, (RFDETRDetectionModel, RFDETRReplay)):
        return "direct_resize"
    if isinstance(model, TorchvisionAnchorDetectorReplay):
        model = model.detector
    if _is_anchor_detector(model):
        fixed_size = getattr(getattr(model, "transform", None), "fixed_size", None)
        if isinstance(fixed_size, (list, tuple)) and len(fixed_size) >= 2:
            return "direct_resize"
    return None


def _ultralytics_imgsz_from_input_shape(
    input_tensor_shape: tuple[int, ...] | list[int] | None,
) -> tuple[int, int] | None:
    if not input_tensor_shape:
        return None
    shape = [int(dim) for dim in list(input_tensor_shape)]
    if len(shape) < 4:
        return None
    height = int(shape[-2])
    width = int(shape[-1])
    if height <= 0 or width <= 0:
        return None
    return height, width


def prepare_split_runtime_input(
    model: torch.nn.Module,
    frame: np.ndarray,
    *,
    device: str | torch.device,
    input_tensor_shape: tuple[int, ...] | list[int] | None = None,
):
    device = torch.device(device)
    target_imgsz = _ultralytics_imgsz_from_input_shape(input_tensor_shape)
    if isinstance(model, DETRDetectionModel):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixel_values = model.processor(
            images=Image.fromarray(rgb),
            return_tensors="pt",
        )["pixel_values"]
        return pixel_values.to(device)
    if isinstance(model, RFDETRDetectionModel):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
        batch_tensor, _ = model._prepare_batch([tensor])
        return batch_tensor.to(device)
    if _is_anchor_detector(model):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
        transformed_images, _ = model.transform([tensor], None)
        return transformed_images.tensors.to(device)

    if isinstance(model, YOLODetectionModel):
        _, tensor = preprocess_bgr_images(
            model.yolo,
            [frame],
            conf=model.confidence,
            imgsz=target_imgsz,
        )
        return tensor.to(device)
    if isinstance(model, RTDETRDetectionModel):
        _, tensor = preprocess_bgr_images(
            model.rtdetr,
            [frame],
            conf=model.confidence,
            imgsz=target_imgsz,
        )
        return tensor.to(device)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
    return [tensor]


def postprocess_split_runtime_output(
    model: torch.nn.Module,
    outputs: Any,
    *,
    threshold: float,
    model_input: Any | None = None,
    orig_image: np.ndarray | None = None,
) -> list[dict[str, torch.Tensor]]:
    if isinstance(model, YOLODetectionModel):
        return _postprocess_yolo_output(
            model,
            outputs,
            model_input=model_input,
            orig_image=orig_image,
        )
    if isinstance(model, RTDETRDetectionModel):
        return _postprocess_rtdetr_output(
            model,
            outputs,
            model_input=model_input,
            orig_image=orig_image,
        )
    if isinstance(model, DETRDetectionModel):
        if orig_image is None:
            raise RuntimeError("DETR split postprocess requires the original image.")
        return _postprocess_detr_output(
            model,
            outputs,
            threshold=threshold,
            image_size=orig_image.shape[:2],
        )
    if isinstance(model, RFDETRDetectionModel):
        if orig_image is None:
            raise RuntimeError("RF-DETR split postprocess requires the original image.")
        return _postprocess_rfdetr_output(
            model,
            outputs,
            threshold=threshold,
            image_size=orig_image.shape[:2],
        )
    if _is_anchor_detector(model):
        return _postprocess_anchor_detector_output(
            model,
            outputs,
            model_input=model_input,
            orig_image=orig_image,
        )
    return outputs


def summarize_split_runtime_observables(
    model: torch.nn.Module,
    outputs: Any,
    split_payload: BoundaryPayload | torch.Tensor | dict[str, torch.Tensor] | None = None,
    *,
    include_feature_spectral_entropy: bool = True,
) -> dict[str, float | None]:
    observables: dict[str, float | None] = {
        "feature_spectral_entropy": None,
        "logit_entropy": None,
        "logit_margin": None,
        "logit_energy": None,
    }
    if include_feature_spectral_entropy:
        observables["feature_spectral_entropy"] = _summarize_payload_spectral_entropy(
            split_payload
        )
    if include_feature_spectral_entropy and observables["feature_spectral_entropy"] is None:
        observables["feature_spectral_entropy"] = _summarize_runtime_output_spectral_entropy(
            model,
            outputs,
        )
    logits_tensor, logits_mode = _extract_runtime_logits(model, outputs)
    if isinstance(logits_tensor, torch.Tensor):
        observables.update(_summarize_logits_statistics(logits_tensor, mode=logits_mode))
    return observables


def build_split_training_loss(model: torch.nn.Module):
    if isinstance(model, YOLODetectionModel):
        core_model = get_split_runtime_model(model)
        _ensure_ultralytics_loss_args(core_model)

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            device = _first_tensor_device(outputs, fallback=next(core_model.parameters()).device)
            batch = _build_ultralytics_training_batch(
                targets,
                device=device,
                num_classes=int(getattr(model, "num_classes", 80)),
                label_schema=getattr(model, "label_schema", "coco_91"),
            )
            loss = core_model.loss(batch, outputs)
            total = loss[0] if isinstance(loss, tuple) else loss
            return total.sum() if isinstance(total, torch.Tensor) and total.ndim > 0 else total

        return _loss_fn

    if isinstance(model, RTDETRDetectionModel):
        core_model = get_split_runtime_model(model)
        criterion = RTDETRDetectionLoss(nc=getattr(core_model, "nc", 80), use_vfl=True)

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            device = _first_tensor_device(outputs, fallback=next(core_model.parameters()).device)
            batch = _build_ultralytics_training_batch(
                targets,
                device=device,
                num_classes=int(getattr(model, "num_classes", 80)),
                label_schema=getattr(model, "label_schema", "coco_91"),
            )
            target_pack = {
                "cls": batch["cls"].to(device=device, dtype=torch.long).view(-1),
                "bboxes": batch["bboxes"].to(device=device),
                "batch_idx": batch["batch_idx"].to(device=device, dtype=torch.long).view(-1),
                "gt_groups": [int(batch["batch_idx"].numel())],
            }
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = _extract_rtdetr_loss_outputs(
                outputs
            )
            if dn_meta is None:
                dn_bboxes, dn_scores = None, None
            else:
                dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
                dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

            dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
            dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
            loss_dict = criterion(
                (dec_bboxes, dec_scores),
                target_pack,
                dn_bboxes=dn_bboxes,
                dn_scores=dn_scores,
                dn_meta=dn_meta,
            )
            return sum(loss_dict.values())

        return _loss_fn

    if isinstance(model, DETRDetectionModel):
        core_model = get_split_runtime_model(model)

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            logits, pred_boxes = _extract_detr_outputs(outputs)

            if isinstance(targets, list):
                labels = []
                for target_item in targets:
                    labels.extend(
                        _build_detr_training_labels(
                            target_item,
                            device=logits.device,
                            num_labels=int(
                                getattr(core_model.config, "num_labels", logits.shape[-1])
                            ),
                        )
                    )
            else:
                labels = _build_detr_training_labels(
                    targets,
                    device=logits.device,
                    num_labels=int(getattr(core_model.config, "num_labels", logits.shape[-1])),
                )

            loss, _, _ = core_model.loss_function(
                logits,
                labels,
                core_model.device,
                pred_boxes,
                core_model.config,
                None,
                None,
            )
            return loss

        return _loss_fn

    if isinstance(model, RFDETRDetectionModel):
        if build_criterion_and_postprocessors is None:
            raise RuntimeError(
                "rfdetr training extras are unavailable; cannot build RF-DETR split loss."
            )
        criterion, _ = build_criterion_and_postprocessors(model.rfdetr.model.args)
        criterion.train()

        def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
            predictions = _extract_rfdetr_outputs(outputs)
            device = _first_tensor_device(predictions, fallback=next(model.parameters()).device)
            criterion.to(device)

            if isinstance(targets, list):
                labels = []
                for target_item in targets:
                    labels.extend(
                        _build_rfdetr_training_labels(
                            target_item,
                            device=device,
                            num_classes=int(getattr(model, "num_classes", 0)),
                            label_schema=getattr(model, "label_schema", "coco_91"),
                        )
                    )
            else:
                labels = _build_rfdetr_training_labels(
                    targets,
                    device=device,
                    num_classes=int(getattr(model, "num_classes", 0)),
                    label_schema=getattr(model, "label_schema", "coco_91"),
                )

            loss_dict = criterion(predictions, labels)
            return sum(loss_dict.values())

        return _loss_fn

    if _is_anchor_detector(model):
        return build_anchor_detector_training_loss(model)

    if hasattr(model, "roi_heads"):

        def _loss_fn(
            outputs: Any,
            targets: Any,
            *,
            runtime=None,
            candidate=None,
        ) -> torch.Tensor:
            from model_management.split_runtime import reduce_output_to_loss

            loss = reduce_output_to_loss(outputs, targets)
            if _has_nonempty_floating_tensors(outputs) and _loss_has_signal(loss):
                return loss

            activation_loss = _tail_activation_probe_loss(runtime, candidate)
            if activation_loss is not None:
                return activation_loss
            return loss

        return _loss_fn

    return None


def build_anchor_detector_training_loss(model: torch.nn.Module):
    """Build a split-tail loss for torchvision anchor detectors.

    The split runtime returns suffix-replayed head outputs.  Anchors only need
    feature-map shapes, so this path reconstructs dummy feature tensors from the
    head output shape instead of running the detector backbone again.
    """

    def _loss_fn(outputs: Any, targets: Any) -> torch.Tensor:
        head_outputs = _extract_anchor_detector_outputs(outputs)
        batch_size = _anchor_head_output_batch_size(head_outputs)
        image_targets, transformed_images, feature_list = _build_anchor_loss_inputs(
            model,
            head_outputs,
            targets,
            batch_size=batch_size,
        )
        anchors = model.anchor_generator(transformed_images, feature_list)
        if isinstance(model, SSD):
            matched_idxs = _match_anchor_targets(model, anchors, image_targets)
            loss_dict = model.compute_loss(
                image_targets,
                head_outputs,
                anchors,
                matched_idxs,
            )
        elif isinstance(model, FCOS):
            num_anchors_per_level = [
                int(feature.shape[-2] * feature.shape[-1]) for feature in feature_list
            ]
            loss_dict = model.compute_loss(
                image_targets,
                head_outputs,
                anchors,
                num_anchors_per_level,
            )
        else:
            loss_dict = model.compute_loss(
                image_targets,
                head_outputs,
                anchors,
            )
        return _sum_anchor_loss_dict(loss_dict, head_outputs)

    return _loss_fn


def build_ssd_split_training_loss(model: torch.nn.Module):
    if not isinstance(model, SSD):
        raise TypeError("build_ssd_split_training_loss() requires a torchvision SSD model.")
    return build_anchor_detector_training_loss(model)


def _sum_anchor_loss_dict(
    loss_dict: Mapping[str, torch.Tensor],
    head_outputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    losses = [loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor)]
    if losses:
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total
    first = next(iter(head_outputs.values()))
    return first.sum() * 0.0


def _anchor_head_output_batch_size(head_outputs: Mapping[str, torch.Tensor]) -> int:
    first = next(iter(head_outputs.values()))
    if first.ndim == 0:
        raise RuntimeError("Anchor-detector head outputs must include a batch dimension.")
    return int(first.shape[0])


def _anchor_head_output_anchor_count(head_outputs: Mapping[str, torch.Tensor]) -> int:
    bbox_regression = head_outputs.get("bbox_regression")
    cls_logits = head_outputs.get("cls_logits")
    if not isinstance(bbox_regression, torch.Tensor) or bbox_regression.ndim < 2:
        raise RuntimeError(
            "Anchor-detector bbox_regression output must have shape [N, anchors, 4]."
        )
    if isinstance(cls_logits, torch.Tensor) and cls_logits.ndim >= 2:
        if int(cls_logits.shape[1]) != int(bbox_regression.shape[1]):
            raise RuntimeError(
                "Anchor-detector cls_logits and bbox_regression disagree on anchor count."
            )
    return int(bbox_regression.shape[1])


def _anchor_num_classes(head_outputs: Mapping[str, torch.Tensor]) -> int | None:
    cls_logits = head_outputs.get("cls_logits")
    if isinstance(cls_logits, torch.Tensor) and cls_logits.ndim >= 3:
        return int(cls_logits.shape[-1])
    return None


def _anchor_generator_num_anchors_per_location(model: torch.nn.Module) -> list[int]:
    generator = getattr(model, "anchor_generator", None)
    value = getattr(generator, "num_anchors_per_location", None)
    if callable(value):
        anchors = [int(item) for item in value()]
        if anchors:
            return anchors
    wh_pairs = getattr(generator, "_wh_pairs", None)
    if isinstance(wh_pairs, (list, tuple)) and wh_pairs:
        return [int(len(item)) for item in wh_pairs]
    out_channels = getattr(getattr(model, "backbone", None), "out_channels", None)
    if isinstance(out_channels, (list, tuple)) and out_channels:
        return [1 for _ in out_channels]
    raise RuntimeError("Unable to infer anchor counts per feature level.")


def _anchor_generator_steps(model: torch.nn.Module) -> list[int] | None:
    steps = getattr(getattr(model, "anchor_generator", None), "steps", None)
    if not isinstance(steps, (list, tuple)) or not steps:
        return None
    normalized = [int(step) for step in steps]
    return normalized if all(step > 0 for step in normalized) else None


def _infer_anchor_feature_shapes_from_head_outputs(
    model: torch.nn.Module,
    head_outputs: Mapping[str, torch.Tensor],
    *,
    model_input_size: tuple[int, int],
) -> list[tuple[int, int]]:
    total_anchors = _anchor_head_output_anchor_count(head_outputs)
    anchors_per_location = _anchor_generator_num_anchors_per_location(model)
    steps = _anchor_generator_steps(model)
    height, width = model_input_size

    if steps is not None and len(steps) == len(anchors_per_location):
        shapes = [
            (
                max(1, int(math.ceil(float(height) / float(step)))),
                max(1, int(math.ceil(float(width) / float(step)))),
            )
            for step in steps
        ]
        expected = sum(
            int(level_height * level_width * anchors)
            for (level_height, level_width), anchors in zip(
                shapes, anchors_per_location, strict=True
            )
        )
        if expected == total_anchors:
            return shapes

    out_channels = getattr(getattr(model, "backbone", None), "out_channels", None)
    level_count = (
        len(out_channels) if isinstance(out_channels, (list, tuple)) else len(anchors_per_location)
    )
    if level_count != len(anchors_per_location):
        level_count = len(anchors_per_location)
    if level_count <= 0:
        raise RuntimeError("Unable to infer anchor feature levels from detector metadata.")

    if steps is not None and len(steps) >= 2:
        shapes = [
            (
                max(1, int(math.ceil(float(height) / float(steps[0])))),
                max(1, int(math.ceil(float(width) / float(steps[0])))),
            ),
            (
                max(1, int(math.ceil(float(height) / float(steps[1])))),
                max(1, int(math.ceil(float(width) / float(steps[1])))),
            ),
        ][:level_count]
        while len(shapes) < level_count:
            previous_height, previous_width = shapes[-1]
            shapes.append(
                (
                    max(1, int(math.ceil(float(previous_height) / 2.0))),
                    max(1, int(math.ceil(float(previous_width) / 2.0))),
                )
            )
        expected = sum(
            int(level_height * level_width * anchors)
            for (level_height, level_width), anchors in zip(
                shapes, anchors_per_location, strict=True
            )
        )
        if expected == total_anchors:
            return shapes

    common_strides = [2 ** (index + 3) for index in range(level_count)]
    shapes = [
        (
            max(1, int(math.ceil(float(height) / float(stride)))),
            max(1, int(math.ceil(float(width) / float(stride)))),
        )
        for stride in common_strides
    ]
    expected = sum(
        int(level_height * level_width * anchors)
        for (level_height, level_width), anchors in zip(shapes, anchors_per_location, strict=True)
    )
    if expected == total_anchors:
        return shapes

    raise RuntimeError(
        "Unable to infer anchor feature-map shapes from split head outputs "
        f"(anchors={total_anchors}, input_size={model_input_size}, "
        f"anchors_per_location={anchors_per_location})."
    )


def _make_dummy_features_for_anchor_generator(
    head_outputs: Mapping[str, torch.Tensor],
    feature_shapes: list[tuple[int, int]],
    *,
    batch_size: int,
) -> list[torch.Tensor]:
    first = next(iter(head_outputs.values()))
    return [
        torch.zeros(
            (batch_size, 1, int(height), int(width)),
            dtype=first.dtype,
            device=first.device,
        )
        for height, width in feature_shapes
    ]


def _make_anchor_image_list(
    head_outputs: Mapping[str, torch.Tensor],
    *,
    model_input_size: tuple[int, int],
    batch_size: int,
) -> ImageList:
    first = next(iter(head_outputs.values()))
    height, width = model_input_size
    tensors = torch.zeros(
        (batch_size, 3, int(height), int(width)),
        dtype=first.dtype,
        device=first.device,
    )
    return ImageList(tensors, [model_input_size for _ in range(batch_size)])


def _target_value_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return int(value.numel())
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 0


def _target_has_anchor_labels(targets: Any) -> bool:
    if not isinstance(targets, Mapping):
        return False
    boxes = targets.get("boxes")
    labels = targets.get("labels")
    return _target_value_count(boxes) > 0 or _target_value_count(labels) > 0


def _fallback_anchor_model_input_size(model: torch.nn.Module) -> tuple[int, int] | None:
    fixed_size = getattr(getattr(model, "transform", None), "fixed_size", None)
    if isinstance(fixed_size, (list, tuple)) and len(fixed_size) >= 2:
        height = int(fixed_size[0])
        width = int(fixed_size[1])
        if height > 0 and width > 0:
            return height, width
    size = getattr(model, "size", None)
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        height = int(size[0])
        width = int(size[1])
        if height > 0 and width > 0:
            return height, width
    return None


def _anchor_target_sizes(
    model: torch.nn.Module,
    target_item: Any,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    if isinstance(target_item, Mapping):
        try:
            original_image_size, model_input_size = _infer_original_and_model_input_image_sizes(
                target_item
            )
            resize_mode = _resolve_anchor_resize_mode(model, target_item)
            return original_image_size, model_input_size, resize_mode
        except RuntimeError:
            if _target_has_anchor_labels(target_item):
                raise

    fallback = _fallback_anchor_model_input_size(model)
    if fallback is None:
        raise RuntimeError(
            "Anchor-detector split loss requires coordinate metadata or a fixed model input size."
        )
    return fallback, fallback, "direct_resize"


def _normalize_anchor_training_targets(targets: Any, *, batch_size: int) -> list[Any]:
    if targets is None:
        return [None for _ in range(batch_size)]
    if isinstance(targets, (list, tuple)):
        target_list = list(targets)
    else:
        target_list = [targets]
    if len(target_list) < batch_size:
        target_list.extend([None for _ in range(batch_size - len(target_list))])
    if len(target_list) > batch_size:
        target_list = target_list[:batch_size]
    return target_list


def _build_transformed_targets_for_anchor_loss(
    model: torch.nn.Module,
    targets: Any,
    head_outputs: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], tuple[int, int]]:
    device = next(iter(head_outputs.values())).device
    num_classes = _anchor_num_classes(head_outputs)
    target_list = _normalize_anchor_training_targets(targets, batch_size=batch_size)
    image_targets: list[dict[str, torch.Tensor]] = []
    model_input_size: tuple[int, int] | None = None

    for target_item in target_list:
        original_size, sample_model_input_size, resize_mode = _anchor_target_sizes(
            model, target_item
        )
        if model_input_size is None:
            model_input_size = sample_model_input_size
        elif model_input_size != sample_model_input_size:
            raise RuntimeError(
                "Anchor-detector split retraining expects a consistent model input "
                "size within a batch. "
                f"Got {model_input_size} and {sample_model_input_size}."
            )
        target_dict = dict(target_item) if isinstance(target_item, Mapping) else {}
        image_targets.append(
            _build_anchor_training_target(
                target_dict,
                device=device,
                original_image_size=original_size,
                model_input_size=sample_model_input_size,
                resize_mode=resize_mode,
                num_classes=num_classes,
                label_schema=getattr(model, "label_schema", "coco_91"),
            )
        )

    if model_input_size is None:
        fallback = _fallback_anchor_model_input_size(model)
        if fallback is None:
            raise RuntimeError("Unable to resolve anchor-detector model input size.")
        model_input_size = fallback
    return image_targets, model_input_size


def _build_anchor_loss_inputs(
    model: torch.nn.Module,
    head_outputs: Mapping[str, torch.Tensor],
    targets: Any,
    *,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], ImageList, list[torch.Tensor]]:
    image_targets, model_input_size = _build_transformed_targets_for_anchor_loss(
        model,
        targets,
        head_outputs,
        batch_size=batch_size,
    )
    feature_shapes = _infer_anchor_feature_shapes_from_head_outputs(
        model,
        head_outputs,
        model_input_size=model_input_size,
    )
    transformed_images = _make_anchor_image_list(
        head_outputs,
        model_input_size=model_input_size,
        batch_size=batch_size,
    )
    feature_list = _make_dummy_features_for_anchor_generator(
        head_outputs,
        feature_shapes,
        batch_size=batch_size,
    )
    return image_targets, transformed_images, feature_list


def _num_anchors_per_level_for_split(
    model: torch.nn.Module,
    head_outputs: Mapping[str, torch.Tensor],
    feature_list: list[torch.Tensor],
) -> list[int]:
    num_locations_per_level = [
        int(feature.shape[-2] * feature.shape[-1]) for feature in feature_list
    ]
    if isinstance(model, FCOS):
        return num_locations_per_level
    total_locations = sum(num_locations_per_level)
    total_anchors = _anchor_head_output_anchor_count(head_outputs)
    anchors_per_location = max(1, total_anchors // max(1, total_locations))
    return [locations * anchors_per_location for locations in num_locations_per_level]


def _empty_detection_result(device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
            "scores": torch.zeros((0,), dtype=torch.float32, device=device),
        }
    ]


def _map_wrapper_labels(model: torch.nn.Module, cls_ids: torch.Tensor) -> torch.Tensor:
    mapped: list[int] = []
    for cls_id in cls_ids.detach().cpu().tolist():
        value = int(cls_id)
        if getattr(model, "_map_labels", False):
            mapped.append(COCO_80_TO_91[value] if 0 <= value < len(COCO_80_TO_91) else value + 1)
        elif str(getattr(model, "label_schema", "")).strip().lower() == "zero_based":
            mapped.append(value)
        else:
            mapped.append(value + 1)
    return torch.as_tensor(mapped, dtype=torch.int64, device=cls_ids.device)


def _clamp_xyxy_boxes(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp_(0.0, float(width))
    boxes[..., 1::2] = boxes[..., 1::2].clamp_(0.0, float(height))
    return boxes


def _postprocess_yolo_output(
    model: YOLODetectionModel,
    outputs: Any,
    *,
    model_input: Any | None,
    orig_image: np.ndarray | None,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(model_input, torch.Tensor) or orig_image is None:
        raise RuntimeError(
            "YOLO split postprocess requires the model input tensor and original frame."
        )

    results = postprocess_predictions(
        model.yolo,
        outputs,
        model_input,
        [orig_image],
        conf=model.confidence,
    )
    result = results[0]
    if result.boxes is None or result.boxes.data.numel() == 0:
        return _empty_detection_result(model_input.device)

    boxes = result.boxes.xyxy.detach().to(model_input.device).float()
    scores = result.boxes.conf.detach().to(model_input.device).float()
    labels = _map_wrapper_labels(
        model,
        result.boxes.cls.detach().to(model_input.device).long(),
    )
    return [{"boxes": boxes, "labels": labels, "scores": scores}]


def _postprocess_rtdetr_output(
    model: RTDETRDetectionModel,
    outputs: Any,
    *,
    model_input: Any | None,
    orig_image: np.ndarray | None,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(model_input, torch.Tensor) or orig_image is None:
        raise RuntimeError(
            "RT-DETR split postprocess requires the model input tensor and original frame."
        )

    results = postprocess_predictions(
        model.rtdetr,
        outputs,
        model_input,
        [orig_image],
        conf=model.confidence,
    )
    result = results[0]
    if result.boxes is None or result.boxes.data.numel() == 0:
        return _empty_detection_result(model_input.device)

    boxes = result.boxes.xyxy.detach().to(model_input.device).float()
    scores = result.boxes.conf.detach().to(model_input.device).float()
    labels = _map_wrapper_labels(
        model,
        result.boxes.cls.detach().to(model_input.device).long(),
    )
    return [{"boxes": boxes, "labels": labels, "scores": scores}]


def _postprocess_detr_output(
    model: DETRDetectionModel,
    outputs: Any,
    *,
    threshold: float,
    image_size: tuple[int, int],
) -> list[dict[str, torch.Tensor]]:
    logits, pred_boxes = _extract_detr_outputs(outputs)
    target_sizes = torch.as_tensor([list(image_size)], dtype=torch.long, device=logits.device)
    detr_outputs = SimpleNamespace(logits=logits, pred_boxes=pred_boxes)
    post = model.processor.post_process_object_detection(
        detr_outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )[0]
    return [
        {
            "boxes": post["boxes"].float(),
            "labels": post["labels"].long(),
            "scores": post["scores"].float(),
        }
    ]


def _postprocess_rfdetr_output(
    model: RFDETRDetectionModel,
    outputs: Any,
    *,
    threshold: float,
    image_size: tuple[int, int],
) -> list[dict[str, torch.Tensor]]:
    predictions = _extract_rfdetr_outputs(outputs)
    target_sizes = torch.as_tensor([list(image_size)], dtype=torch.long, device=model._device)
    decoded = _postprocess_rfdetr_predictions(
        predictions,
        target_sizes=target_sizes,
        threshold=float(threshold),
        num_classes=getattr(model, "num_classes", 91),
        label_schema=getattr(model, "label_schema", "coco_91"),
        num_select=getattr(
            model.rfdetr.model.postprocess, "num_select", predictions["pred_logits"].shape[1]
        ),
        device=model._device,
    )[0]
    return [
        {
            "boxes": decoded["boxes"].detach().to(model._device).float(),
            "labels": decoded["labels"].detach().to(model._device).long(),
            "scores": decoded["scores"].detach().to(model._device).float(),
        }
    ]


def _postprocess_anchor_detector_output(
    model: torch.nn.Module,
    outputs: Any,
    *,
    model_input: Any | None,
    orig_image: np.ndarray | None = None,
) -> list[dict[str, torch.Tensor]]:
    head_outputs = _extract_anchor_detector_outputs(outputs)
    device = next(iter(head_outputs.values())).device

    if isinstance(model_input, torch.Tensor):
        if model_input.ndim == 3:
            transformed_batch = model_input.unsqueeze(0).to(device)
        elif model_input.ndim == 4:
            transformed_batch = model_input.to(device)
        else:
            raise RuntimeError(
                "Anchor-detector split postprocess received an unsupported tensor input shape."
            )
        transformed_sizes = [
            (int(transformed_batch.shape[-2]), int(transformed_batch.shape[-1]))
            for _ in range(int(transformed_batch.shape[0]))
        ]
        transformed_images = ImageList(transformed_batch, transformed_sizes)
        if orig_image is not None:
            original_image_sizes = [tuple(int(value) for value in orig_image.shape[:2])]
        else:
            original_image_sizes = list(transformed_sizes)
    elif isinstance(model_input, (list, tuple)):
        images = [image for image in model_input if isinstance(image, torch.Tensor)]
        if not images:
            raise RuntimeError(
                "Anchor-detector split postprocess requires the original image tensor input."
            )
        images = [image.to(device) for image in images]
        original_image_sizes = [tuple(int(dim) for dim in image.shape[-2:]) for image in images]
        transformed_images, _ = model.transform(images, None)
    else:
        raise RuntimeError("Anchor-detector split postprocess requires the runtime model input.")

    model_input_size = (
        int(transformed_images.tensors.shape[-2]),
        int(transformed_images.tensors.shape[-1]),
    )
    feature_shapes = _infer_anchor_feature_shapes_from_head_outputs(
        model,
        head_outputs,
        model_input_size=model_input_size,
    )
    feature_list = _make_dummy_features_for_anchor_generator(
        head_outputs,
        feature_shapes,
        batch_size=int(transformed_images.tensors.shape[0]),
    )
    anchors = model.anchor_generator(transformed_images, feature_list)
    if isinstance(model, (RetinaNet, FCOS)):
        num_anchors_per_level = _num_anchors_per_level_for_split(
            model,
            head_outputs,
            feature_list,
        )
        split_head_outputs = {
            key: list(value.split(num_anchors_per_level, dim=1))
            for key, value in head_outputs.items()
        }
        split_anchors = [list(anchor_set.split(num_anchors_per_level)) for anchor_set in anchors]
        detections = model.postprocess_detections(
            split_head_outputs,
            split_anchors,
            transformed_images.image_sizes,
        )
    else:
        detections = model.postprocess_detections(
            head_outputs,
            anchors,
            transformed_images.image_sizes,
        )
    detections = model.transform.postprocess(
        detections,
        transformed_images.image_sizes,
        original_image_sizes,
    )
    return _remap_tinynext_public_detections(model, detections)


def _iter_payload_tensors(
    split_payload: BoundaryPayload | torch.Tensor | dict[str, torch.Tensor] | None,
):
    if split_payload is None:
        return
    if isinstance(split_payload, BoundaryPayload):
        primary_label = getattr(split_payload, "primary_label", None)
        if primary_label and primary_label in split_payload.tensors:
            yield split_payload.tensors[primary_label]
        for label, tensor in split_payload.tensors.items():
            if label == primary_label:
                continue
            yield tensor
        return
    if isinstance(split_payload, torch.Tensor):
        yield split_payload
        return
    if isinstance(split_payload, dict):
        for value in split_payload.values():
            if isinstance(value, torch.Tensor):
                yield value


def _feature_matrix_from_tensor(
    tensor: torch.Tensor,
    *,
    max_spatial_size: int = 16,
    max_feature_dims: int = 128,
) -> torch.Tensor | None:
    if (
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_floating_point()
        or tensor.numel() == 0
    ):
        return None

    x = tensor.detach().float()
    if x.ndim == 4:
        height = min(max_spatial_size, int(x.shape[-2]))
        width = min(max_spatial_size, int(x.shape[-1]))
        x = F.adaptive_avg_pool2d(x, output_size=(height, width))
        matrix = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    elif x.ndim == 3:
        if x.shape[0] <= 8 and x.shape[-1] >= 8:
            matrix = x.reshape(-1, x.shape[-1]).transpose(0, 1)
        else:
            matrix = x.reshape(x.shape[0], -1)
    elif x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[1] >= 8:
            matrix = x.reshape(-1, x.shape[-1]).transpose(0, 1)
        else:
            matrix = x if x.shape[0] <= x.shape[1] else x.transpose(0, 1)
    elif x.ndim == 1:
        matrix = x.unsqueeze(0)
    else:
        flattened = (
            x.reshape(x.shape[0], -1)
            if x.shape[0] <= 64
            else x.reshape(-1, x.shape[-1]).transpose(0, 1)
        )
        matrix = flattened

    if matrix is None or matrix.numel() == 0:
        return None

    if matrix.shape[0] > max_feature_dims:
        energy = matrix.square().mean(dim=1)
        topk = min(max_feature_dims, int(energy.numel()))
        indices = torch.topk(energy, k=topk).indices
        matrix = matrix.index_select(0, indices)
    return matrix


def _spectral_entropy_from_matrix(matrix: torch.Tensor) -> float | None:
    if not isinstance(matrix, torch.Tensor) or matrix.numel() == 0:
        return None
    if matrix.shape[0] <= 1 or matrix.shape[1] <= 1:
        return 0.0

    centered = matrix - matrix.mean(dim=1, keepdim=True)
    covariance = centered @ centered.transpose(0, 1)
    covariance = covariance / float(max(1, centered.shape[1] - 1))
    eigvals = torch.linalg.eigvalsh(covariance).real.clamp_min_(0.0)
    total = eigvals.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return 0.0

    probs = eigvals / total
    nonzero = probs[probs > 0]
    if nonzero.numel() == 0:
        return 0.0
    entropy = -(nonzero * torch.log(nonzero)).sum()
    normaliser = torch.log(torch.tensor(float(nonzero.numel()), device=entropy.device))
    if float(normaliser.item()) <= 0.0:
        return 0.0
    return float((entropy / normaliser).clamp(0.0, 1.0).item())


def _summarize_payload_spectral_entropy(
    split_payload: BoundaryPayload | torch.Tensor | dict[str, torch.Tensor] | None,
) -> float | None:
    for tensor in _iter_payload_tensors(split_payload):
        matrix = _feature_matrix_from_tensor(tensor)
        if matrix is None:
            continue
        try:
            return _spectral_entropy_from_matrix(matrix)
        except Exception:
            continue
    return None


def _summarize_runtime_output_spectral_entropy(
    model: torch.nn.Module,
    outputs: Any,
) -> float | None:
    if isinstance(model, YOLODetectionModel):
        feats = _extract_yolo_runtime_feats(outputs)
        for tensor in feats:
            matrix = _feature_matrix_from_tensor(tensor)
            if matrix is None:
                continue
            try:
                return _spectral_entropy_from_matrix(matrix)
            except Exception:
                continue
    return None


def _reshape_logits_rows(logits: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        return None
    if logits.ndim == 2:
        return logits.detach().float()
    if logits.ndim == 3:
        if logits.shape[-1] > 4 and logits.shape[-1] <= 512:
            return logits.detach().float().reshape(-1, logits.shape[-1])
        if logits.shape[1] > 4 and logits.shape[1] <= 512 and logits.shape[2] > logits.shape[1]:
            permuted = logits.detach().float().permute(0, 2, 1)
            return permuted.reshape(-1, logits.shape[1])
        return logits.detach().float().reshape(-1, logits.shape[-1])
    if logits.ndim == 4:
        if logits.shape[-1] > 4 and logits.shape[-1] <= 512:
            return logits.detach().float().reshape(-1, logits.shape[-1])
        if logits.shape[1] > 4 and logits.shape[1] <= 512:
            permuted = logits.detach().float().permute(0, 2, 3, 1)
            return permuted.reshape(-1, logits.shape[1])
    return None


def _summarize_logits_statistics(
    logits: torch.Tensor,
    *,
    mode: str = "sigmoid",
    max_rows: int = 256,
) -> dict[str, float | None]:
    rows = _reshape_logits_rows(logits)
    if rows is None or rows.numel() == 0 or rows.shape[-1] <= 0:
        return {
            "logit_entropy": None,
            "logit_margin": None,
            "logit_energy": None,
        }

    work_rows = rows
    if mode == "softmax_bg_last" and work_rows.shape[-1] > 1:
        work_rows = work_rows[:, :-1]
    if work_rows.shape[-1] <= 0:
        return {
            "logit_entropy": None,
            "logit_margin": None,
            "logit_energy": None,
        }

    if mode.startswith("softmax"):
        row_priority = torch.softmax(work_rows, dim=-1).max(dim=-1).values
    else:
        row_priority = torch.sigmoid(work_rows).max(dim=-1).values
    if row_priority.numel() > max_rows:
        top_indices = torch.topk(row_priority, k=max_rows).indices
        work_rows = work_rows.index_select(0, top_indices)

    if mode.startswith("softmax"):
        probs = torch.softmax(work_rows, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] >= 2:
            margin = (top2[:, 0] - top2[:, 1]).mean()
        else:
            margin = top2[:, 0].mean()
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / max(float(np.log(max(2, probs.shape[-1]))), 1.0)
    else:
        probs = torch.sigmoid(work_rows)
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] >= 2:
            margin = (top2[:, 0] - top2[:, 1]).mean()
        else:
            margin = top2[:, 0].mean()
        p = top2[:, 0].clamp(1e-6, 1.0 - 1e-6)
        entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
        entropy = entropy / float(np.log(2.0))

    energy = torch.logsumexp(work_rows, dim=-1).mean()
    return {
        "logit_entropy": float(entropy.mean().clamp(0.0, 1.0).item()),
        "logit_margin": float(margin.clamp(0.0, 1.0).item()),
        "logit_energy": float(energy.item()),
    }


def _extract_runtime_logits(
    model: torch.nn.Module,
    outputs: Any,
) -> tuple[torch.Tensor | None, str]:
    if isinstance(model, YOLODetectionModel):
        try:
            logits = _extract_yolo_runtime_scores(outputs)
            if isinstance(logits, torch.Tensor):
                return logits, "sigmoid"
        except Exception:
            pass

    if isinstance(model, RTDETRDetectionModel):
        try:
            _, dec_scores, _, enc_scores, _ = _extract_rtdetr_loss_outputs(outputs)
            if isinstance(dec_scores, torch.Tensor):
                return (dec_scores[-1] if dec_scores.ndim >= 4 else dec_scores), "softmax"
            if isinstance(enc_scores, torch.Tensor):
                return enc_scores, "softmax"
        except Exception:
            pass

    if isinstance(model, (DETRDetectionModel, RFDETRDetectionModel)):
        try:
            if isinstance(model, RFDETRDetectionModel):
                extracted = _extract_rfdetr_outputs(outputs)
                logits = extracted.get("pred_logits")
            else:
                logits, _ = _extract_detr_outputs(outputs)
            if isinstance(logits, torch.Tensor):
                return logits, "softmax_bg_last"
        except Exception:
            pass

    if _is_anchor_detector(model):
        try:
            head_outputs = _extract_anchor_detector_outputs(outputs)
            logits = head_outputs.get("cls_logits")
            if isinstance(logits, torch.Tensor):
                return logits, "sigmoid"
        except Exception:
            pass

    if isinstance(outputs, dict):
        if isinstance(outputs.get("cls_logits"), torch.Tensor):
            return outputs["cls_logits"], "sigmoid"
        if isinstance(outputs.get("pred_logits"), torch.Tensor):
            return outputs["pred_logits"], "softmax_bg_last"
        if isinstance(outputs.get("logits"), torch.Tensor):
            return outputs["logits"], "softmax_bg_last"

    return None, "sigmoid"


def _extract_yolo_runtime_aux(outputs: Any) -> dict[str, Any] | None:
    if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], dict):
        return outputs[1]
    if isinstance(outputs, dict):
        if any(isinstance(outputs.get(branch), dict) for branch in ("one2many", "one2one")):
            return outputs
    return None


def _extract_yolo_runtime_scores(outputs: Any) -> torch.Tensor | None:
    aux = _extract_yolo_runtime_aux(outputs)
    if aux is None:
        return None
    for branch_name in ("one2many", "one2one"):
        branch = aux.get(branch_name)
        if isinstance(branch, dict) and isinstance(branch.get("scores"), torch.Tensor):
            return branch["scores"]
    return None


def _extract_yolo_runtime_feats(outputs: Any) -> list[torch.Tensor]:
    aux = _extract_yolo_runtime_aux(outputs)
    if aux is None:
        return []
    for branch_name in ("one2many", "one2one"):
        branch = aux.get(branch_name)
        if not isinstance(branch, dict):
            continue
        feats = branch.get("feats")
        if isinstance(feats, torch.Tensor):
            return [feats]
        if isinstance(feats, (list, tuple)):
            return [tensor for tensor in feats if isinstance(tensor, torch.Tensor)]
    return []


def _extract_detr_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
        return outputs.logits, outputs.pred_boxes
    if isinstance(outputs, dict):
        logits = outputs.get("logits", outputs.get("pred_logits"))
        pred_boxes = outputs.get("pred_boxes")
        if isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor):
            return logits, pred_boxes
    tensors = list(_iter_tensors(outputs))
    logits = next((tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] > 4), None)
    pred_boxes = next(
        (tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] == 4), None
    )
    if logits is None or pred_boxes is None:
        raise RuntimeError("Unable to extract DETR logits/pred_boxes from split replay output.")
    return logits, pred_boxes


def _extract_rfdetr_outputs(outputs: Any) -> dict[str, Any]:
    if isinstance(outputs, dict):
        logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        if isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor):
            extracted = {
                "pred_logits": _contiguous_tensor_tree(logits),
                "pred_boxes": _contiguous_tensor_tree(pred_boxes),
            }
            if isinstance(outputs.get("aux_outputs"), (list, tuple, dict)):
                extracted["aux_outputs"] = _unpack_rfdetr_aux_outputs(outputs["aux_outputs"])
            if isinstance(outputs.get("enc_outputs"), dict):
                extracted["enc_outputs"] = _contiguous_tensor_tree(outputs["enc_outputs"])
            return extracted
    logits, pred_boxes = _extract_detr_outputs(outputs)
    return {
        "pred_logits": _contiguous_tensor_tree(logits),
        "pred_boxes": _contiguous_tensor_tree(pred_boxes),
    }


def _pack_rfdetr_aux_outputs(value: Any) -> Any:
    if isinstance(value, dict):
        return _contiguous_tensor_tree(value)
    if not isinstance(value, (list, tuple)):
        return _contiguous_tensor_tree(value)
    if not value:
        return []
    if not all(isinstance(item, Mapping) for item in value):
        return _contiguous_tensor_tree(value)
    keys = sorted(
        set.intersection(
            *[
                {str(key) for key, item in dict(layer).items() if isinstance(item, torch.Tensor)}
                for layer in value
            ]
        )
    )
    packed: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [dict(layer)[key] for layer in value]
        if not all(
            isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(tensors[0].shape)
            for tensor in tensors
        ):
            continue
        packed[key] = torch.stack(
            [tensor if tensor.is_contiguous() else tensor.contiguous() for tensor in tensors],
            dim=0,
        )
    if not packed:
        return _contiguous_tensor_tree(value)
    marker_device = next(iter(packed.values())).device
    return {
        _RFDETR_PACKED_AUX_OUTPUTS_MARKER: torch.ones(
            (),
            dtype=torch.bool,
            device=marker_device,
        ),
        **packed,
    }


def _is_rfdetr_packed_aux_marker(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        try:
            return bool(value.detach().bool().cpu().item())
        except Exception:
            return False
    return False


def _unpack_rfdetr_aux_outputs(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return _contiguous_tensor_tree(list(value))
    if not isinstance(value, dict):
        return _contiguous_tensor_tree(value)
    if not _is_rfdetr_packed_aux_marker(value.get(_RFDETR_PACKED_AUX_OUTPUTS_MARKER)):
        return _contiguous_tensor_tree(value)
    tensor_items = {
        str(key): tensor
        for key, tensor in value.items()
        if key != _RFDETR_PACKED_AUX_OUTPUTS_MARKER
        and isinstance(tensor, torch.Tensor)
        and tensor.ndim > 0
    }
    if not tensor_items:
        return []
    layer_count = min(int(tensor.shape[0]) for tensor in tensor_items.values())
    if layer_count <= 0:
        return []
    return [
        {key: _contiguous_tensor_tree(tensor[layer_index]) for key, tensor in tensor_items.items()}
        for layer_index in range(layer_count)
    ]


def _extract_anchor_detector_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        extracted = {
            str(key): value for key, value in outputs.items() if isinstance(value, torch.Tensor)
        }
        cls_logits = extracted.get("cls_logits")
        bbox_regression = extracted.get("bbox_regression")
        if isinstance(cls_logits, torch.Tensor) and isinstance(bbox_regression, torch.Tensor):
            return extracted
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        cls_logits = outputs[0]
        bbox_regression = outputs[1]
        if isinstance(cls_logits, torch.Tensor) and isinstance(bbox_regression, torch.Tensor):
            extracted = {
                "cls_logits": cls_logits,
                "bbox_regression": bbox_regression,
            }
            if len(outputs) >= 3 and isinstance(outputs[2], torch.Tensor):
                extracted["bbox_ctrness"] = outputs[2]
            return extracted
    raise RuntimeError(
        "Unable to extract anchor-detector cls_logits/bbox_regression from split replay output."
    )


def _extract_rtdetr_loss_outputs(
    outputs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 2 and isinstance(outputs[1], (list, tuple)) and len(outputs[1]) == 5:
            candidate = outputs[1]
            if all(isinstance(item, (torch.Tensor, dict, type(None))) for item in candidate):
                return tuple(candidate)
        if len(outputs) == 5 and all(
            isinstance(item, (torch.Tensor, dict, type(None))) for item in outputs
        ):
            return tuple(outputs)
        for item in outputs:
            if isinstance(item, (list, tuple)):
                try:
                    return _extract_rtdetr_loss_outputs(item)
                except RuntimeError:
                    continue
    raise RuntimeError("Unable to extract RT-DETR decoder outputs from split replay output.")


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            yield from _iter_tensors(getattr(value, field_info.name))


def _contiguous_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value if value.is_contiguous() else value.contiguous()
    if isinstance(value, dict):
        return {key: _contiguous_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_contiguous_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_contiguous_tensor_tree(item) for item in value]
    return value


def _first_tensor_device(value: Any, *, fallback: torch.device) -> torch.device:
    first = next(_iter_tensors(value), None)
    return first.device if isinstance(first, torch.Tensor) else fallback


def _loss_has_signal(loss: Any) -> bool:
    return (
        isinstance(loss, torch.Tensor) and loss.requires_grad and bool(torch.isfinite(loss).item())
    )


def _has_nonempty_floating_tensors(value: Any) -> bool:
    for tensor in _iter_tensors(value):
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.numel() > 0:
            return True
    return False


def _tail_activation_probe_loss(runtime, candidate) -> torch.Tensor | None:
    trace_graph = getattr(runtime, "trace_graph", None)
    runtime_state = getattr(runtime, "runtime_state", None)
    state_values = getattr(runtime_state, "values", None)
    if (
        runtime is None
        or candidate is None
        or trace_graph is None
        or not isinstance(state_values, Mapping)
    ):
        return None

    selected_labels: list[str] = []
    for label in reversed(candidate.cloud_nodes):
        node = dict(getattr(trace_graph, "nodes", {}) or {}).get(str(label))
        layer = getattr(node, "layer", None)
        if node is None or not getattr(layer, "parent_param_logs", None):
            continue
        if label not in state_values:
            continue
        selected_labels.append(label)
        if len(selected_labels) >= 4:
            break

    total: torch.Tensor | None = None
    pieces = 0
    for label in selected_labels:
        value = state_values.get(label)
        for tensor in _iter_tensors(value):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                continue
            if tensor.numel() == 0:
                continue
            finite = tensor[torch.isfinite(tensor)]
            if finite.numel() == 0:
                continue
            partial = finite.square().mean()
            total = partial if total is None else total + partial
            pieces += 1
    if total is None or pieces == 0:
        return None
    return total / float(pieces)


def _ensure_ultralytics_loss_args(core_model: torch.nn.Module) -> None:
    defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
    args = getattr(core_model, "args", None)
    if isinstance(args, dict):
        merged = dict(args)
        for key, value in defaults.items():
            merged.setdefault(key, value)
        core_model.args = SimpleNamespace(**merged)
        return
    if args is None:
        core_model.args = SimpleNamespace(**defaults)
        return
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _infer_input_image_size(targets: Any) -> tuple[int, int]:
    if not isinstance(targets, dict):
        raise RuntimeError(
            "Split training targets must be a dict for wrapper-model loss computation."
        )
    split_meta = targets.get("_split_meta", {})
    input_tensor_shape = split_meta.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
        height = int(input_tensor_shape[-2])
        width = int(input_tensor_shape[-1])
        if height > 0 and width > 0:
            return height, width
    raise RuntimeError(
        "Missing input_tensor_shape metadata required for wrapper-model split retraining."
    )


def _infer_original_and_model_input_image_sizes(
    targets: Any,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(targets, dict):
        raise RuntimeError(
            "Split training targets must be a dict for wrapper-model loss computation."
        )
    split_meta = targets.get("_split_meta", {})
    original_image_size, model_input_size, _resize_mode = require_coordinate_metadata(split_meta)
    return original_image_size, model_input_size


def _infer_split_resize_mode(targets: Any) -> str:
    if not isinstance(targets, dict):
        raise RuntimeError(
            "Split training targets must be a dict for wrapper-model loss computation."
        )
    _original_image_size, _model_input_size, resize_mode = require_coordinate_metadata(
        targets.get("_split_meta", {})
    )
    return resize_mode


def _assert_original_xyxy_targets(targets: dict[str, Any]) -> None:
    coordinate_space = str(targets.get("label_coordinate_space") or "").strip()
    has_targets = _has_target_values(targets.get("boxes")) or _has_target_values(
        targets.get("labels")
    )
    if coordinate_space != ORIGINAL_XYXY and (coordinate_space or has_targets):
        raise RuntimeError(
            "Split training targets must use original_xyxy canonical labels before "
            "model-specific loss conversion."
        )


def _has_target_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return int(value.numel()) > 0
    try:
        return len(value) > 0
    except TypeError:
        return True


def _infer_ultralytics_image_sizes(targets: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    return _infer_original_and_model_input_image_sizes(targets)


def _build_anchor_training_target(
    targets: dict[str, Any],
    *,
    device: torch.device,
    original_image_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str = "letterbox",
    num_classes: int | None = None,
    label_schema: str = "coco_91",
) -> dict[str, torch.Tensor]:
    _assert_original_xyxy_targets(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    if original_image_size != model_input_size:
        boxes = _project_boxes_to_model_input(
            boxes,
            original_image_size=original_image_size,
            model_input_size=model_input_size,
            resize_mode=resize_mode,
        )
    boxes = _clamp_xyxy_boxes(boxes, model_input_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(int(boxes.shape[0]), int(labels.shape[0]))
        boxes = boxes[:count]
        labels = labels[:count]
    if labels.numel():
        is_zero_based = str(label_schema or "coco_91").strip().lower() == "zero_based"
        if is_zero_based:
            upper_bound = max(1, int(num_classes or 1) - 1)
            valid_labels = (labels >= 0) & (labels < upper_bound)
            labels = labels + 1
        else:
            valid_labels = labels > 0
            if num_classes is not None:
                valid_labels = valid_labels & (labels < int(num_classes))
        boxes = boxes[valid_labels]
        labels = labels[valid_labels]
    if boxes.numel():
        valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_geometry]
        labels = labels[valid_geometry]
    return {
        "boxes": boxes,
        "labels": labels,
    }


def _match_anchor_targets(
    model: torch.nn.Module,
    anchors: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> list[torch.Tensor]:
    matched_idxs: list[torch.Tensor] = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image["boxes"].numel() == 0:
            matched_idxs.append(
                torch.full(
                    (anchors_per_image.size(0),),
                    -1,
                    dtype=torch.int64,
                    device=anchors_per_image.device,
                )
            )
            continue
        match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
        matched_idxs.append(model.proposal_matcher(match_quality_matrix))
    return matched_idxs


def _as_boxes_tensor(boxes: Any, *, device: torch.device) -> torch.Tensor:
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)
    tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    return tensor.reshape(-1, 4)


def _as_labels_tensor(labels: Any, *, device: torch.device) -> torch.Tensor:
    if labels is None:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    tensor = torch.as_tensor(labels, dtype=torch.int64, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0)
    return tensor.reshape(-1)


def _xyxy_to_normalized_cxcywh(
    boxes_xyxy: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    height, width = image_size
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    return torch.stack(
        (
            ((x1 + x2) * 0.5) / float(width),
            ((y1 + y2) * 0.5) / float(height),
            (x2 - x1) / float(width),
            (y2 - y1) / float(height),
        ),
        dim=-1,
    ).clamp_(0.0, 1.0)


def _project_boxes_to_model_input(
    boxes_xyxy: torch.Tensor,
    *,
    original_image_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str = "letterbox",
) -> torch.Tensor:
    projected = project_original_xyxy_to_model_input_xyxy(
        boxes_xyxy,
        original_image_size,
        model_input_size,
        resize_mode,
    )
    if not isinstance(projected, torch.Tensor):
        projected = torch.as_tensor(projected, dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    return projected.to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype).reshape(-1, 4)


def _resolve_anchor_resize_mode(model: torch.nn.Module, targets: Any) -> str:
    return _infer_split_resize_mode(targets)


def _prepare_coco80_targets(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_classes: int = 80,
    label_schema: str = "coco_91",
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    original_image_size, model_input_size = _infer_ultralytics_image_sizes(targets)
    resize_mode = _infer_split_resize_mode(targets)
    _assert_original_xyxy_targets(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    mapped_labels: list[int] = []
    keep_indices: list[int] = []
    is_zero_based = str(label_schema).strip().lower() == "zero_based"
    max_label = max(int(num_classes), 0)
    for index, label in enumerate(labels.detach().cpu().tolist()):
        value = int(label)
        if is_zero_based:
            if 0 <= value < max_label:
                mapped_labels.append(value)
                keep_indices.append(index)
        elif value in COCO_91_TO_80:
            mapped_labels.append(COCO_91_TO_80[value])
            keep_indices.append(index)
        elif 0 <= value < 80:
            mapped_labels.append(value)
            keep_indices.append(index)
        elif 1 <= value <= 80:
            mapped_labels.append(value - 1)
            keep_indices.append(index)

    if not keep_indices:
        empty_boxes = boxes.new_zeros((0, 4))
        empty_labels = labels.new_zeros((0,), dtype=torch.int64)
        return empty_boxes, empty_labels, model_input_size

    keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
    boxes = _project_boxes_to_model_input(
        boxes.index_select(0, keep_tensor),
        original_image_size=original_image_size,
        model_input_size=model_input_size,
        resize_mode=resize_mode,
    )
    labels = torch.as_tensor(mapped_labels, dtype=torch.int64, device=device)
    valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid_geometry], labels[valid_geometry], model_input_size


def _build_ultralytics_training_batch(
    targets: Any,
    *,
    device: torch.device,
    num_classes: int = 80,
    label_schema: str = "coco_91",
) -> dict[str, torch.Tensor]:
    if isinstance(targets, dict):
        boxes, labels, image_size = _prepare_coco80_targets(
            targets,
            device=device,
            num_classes=num_classes,
            label_schema=label_schema,
        )
        normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
        height, width = image_size
        return {
            "img": torch.zeros((1, 3, height, width), dtype=torch.float32, device=device),
            "batch_idx": torch.zeros((labels.shape[0],), dtype=torch.long, device=device),
            "cls": labels.to(dtype=torch.float32).view(-1, 1),
            "bboxes": normalized_boxes.to(dtype=torch.float32),
        }

    if (
        isinstance(targets, (list, tuple))
        and targets
        and all(isinstance(item, dict) for item in targets)
    ):
        box_pieces: list[torch.Tensor] = []
        label_pieces: list[torch.Tensor] = []
        batch_idx_pieces: list[torch.Tensor] = []
        image_size: tuple[int, int] | None = None
        for batch_index, sample_targets in enumerate(targets):
            boxes, labels, sample_image_size = _prepare_coco80_targets(
                sample_targets,
                device=device,
                num_classes=num_classes,
                label_schema=label_schema,
            )
            if image_size is None:
                image_size = sample_image_size
            elif image_size != sample_image_size:
                raise RuntimeError(
                    "Wrapper-model split retraining expects a consistent model input "
                    "size within each batch. "
                    f"Got {image_size} and {sample_image_size}."
                )
            normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=sample_image_size)
            box_pieces.append(normalized_boxes.to(dtype=torch.float32))
            label_pieces.append(labels.to(dtype=torch.float32).view(-1, 1))
            if labels.numel() == 0:
                batch_idx_pieces.append(torch.zeros((0,), dtype=torch.long, device=device))
            else:
                batch_idx_pieces.append(
                    torch.full(
                        (labels.shape[0],), int(batch_index), dtype=torch.long, device=device
                    )
                )

        if image_size is None:
            raise RuntimeError(
                "Missing model input size metadata for wrapper-model split retraining batch."
            )
        height, width = image_size
        bboxes = (
            torch.cat(box_pieces, dim=0)
            if box_pieces
            else torch.zeros((0, 4), dtype=torch.float32, device=device)
        )
        cls = (
            torch.cat(label_pieces, dim=0)
            if label_pieces
            else torch.zeros((0, 1), dtype=torch.float32, device=device)
        )
        batch_idx = (
            torch.cat(batch_idx_pieces, dim=0)
            if batch_idx_pieces
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        return {
            "img": torch.zeros(
                (len(targets), 3, height, width), dtype=torch.float32, device=device
            ),
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }

    raise RuntimeError(
        "Split training targets must be a dict or a non-empty list of dicts for "
        "wrapper-model loss computation."
    )


def _build_detr_training_labels(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_labels: int,
) -> list[dict[str, torch.Tensor]]:
    original_image_size, image_size = _infer_original_and_model_input_image_sizes(targets)
    resize_mode = _infer_split_resize_mode(targets)
    _assert_original_xyxy_targets(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    if original_image_size != image_size:
        boxes = _project_boxes_to_model_input(
            boxes,
            original_image_size=original_image_size,
            model_input_size=image_size,
            resize_mode=resize_mode,
        )
    boxes = _clamp_xyxy_boxes(boxes, image_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    valid = (labels > 0) & (labels < int(num_labels))
    if boxes.numel():
        valid = valid & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
    return [
        {
            "class_labels": labels.to(dtype=torch.int64),
            "boxes": normalized_boxes.to(dtype=torch.float32),
        }
    ]


def _build_rfdetr_training_labels(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_classes: int,
    label_schema: str = "coco_91",
) -> list[dict[str, torch.Tensor]]:
    original_image_size, image_size = _infer_original_and_model_input_image_sizes(targets)
    resize_mode = _infer_split_resize_mode(targets)
    _assert_original_xyxy_targets(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    if original_image_size != image_size:
        boxes = _project_boxes_to_model_input(
            boxes,
            original_image_size=original_image_size,
            model_input_size=image_size,
            resize_mode=resize_mode,
        )
    boxes = _clamp_xyxy_boxes(boxes, image_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    if str(label_schema or "coco_91").strip().lower() == "zero_based":
        valid = (labels >= 0) & (labels < max(1, int(num_classes) - 1))
    else:
        # COCO-style RF-DETR keeps public 1-based label IDs and reserves 0 as
        # the background/dummy slot.
        valid = (labels > 0) & (labels < int(num_classes))
    if boxes.numel():
        valid = valid & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
    return [
        {
            "labels": labels.to(dtype=torch.int64),
            "boxes": normalized_boxes.to(dtype=torch.float32),
        }
    ]
