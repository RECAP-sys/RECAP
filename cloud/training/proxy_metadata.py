from __future__ import annotations

import re
from collections.abc import Mapping

import torch

from model_management.detection_box_projection import ORIGINAL_XYXY
from model_management.model_info import COCO_INSTANCE_CATEGORY_NAMES

POOL_LABEL_COORDINATE_SPACE = ORIGINAL_XYXY


def is_cuda_oom_error(exc: BaseException) -> bool:
    oom_error_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_error_type is not None and isinstance(exc, oom_error_type):
        return True
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


def looks_like_fused_ultralytics_state_dict(state: object) -> bool:
    """Detect BN-folded Ultralytics checkpoints saved as raw state-dicts."""
    if not isinstance(state, Mapping):
        return False

    string_keys = [key for key in state.keys() if isinstance(key, str)]
    if not string_keys:
        return False

    has_conv_bias = any(".conv.bias" in key for key in string_keys)
    has_batch_norm = any(".bn." in key for key in string_keys)
    return has_conv_bias and not has_batch_norm


def coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def normalise_shard_dtype(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "original", "preserve"}:
        return None
    return text


def normalise_label_schema(value: object, default: str = "coco_91") -> str:
    schema = str(value or default).strip().lower()
    return schema or default


def normalise_class_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def class_names_from_metadata(metadata: Mapping[str, object] | None) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    value = metadata.get("class_names")
    if isinstance(value, Mapping):
        ordered = sorted(
            ((int(key), item) for key, item in value.items() if str(key).lstrip("-").isdigit()),
            key=lambda item: item[0],
        )
        return [str(item) for _key, item in ordered]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def label_name_from_schema(
    label: object,
    *,
    label_schema: str,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    try:
        label_index = int(label)
    except (TypeError, ValueError):
        return None

    names = list(class_names or [])
    if names:
        if normalise_label_schema(label_schema) == "zero_based":
            if 0 <= label_index < len(names):
                return str(names[label_index])
        else:
            if 1 <= label_index <= len(names):
                return str(names[label_index - 1])
            if 0 <= label_index < len(names):
                return str(names[label_index])

    if normalise_label_schema(label_schema) != "zero_based":
        if 0 <= label_index < len(COCO_INSTANCE_CATEGORY_NAMES):
            name = COCO_INSTANCE_CATEGORY_NAMES[label_index]
            if name not in {"__background__", "N/A"}:
                return str(name)
    return None


def infer_yolo_state_dict_num_classes(state: object) -> int | None:
    if not isinstance(state, Mapping):
        return None

    class_counts: list[int] = []
    head_pattern = re.compile(r"(?:^|\.)(?:one2one_)?cv3\.\d+\.2\.(?:weight|bias)$")
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        if not head_pattern.search(key) or value.ndim < 1:
            continue
        count = int(value.shape[0])
        if count > 0:
            class_counts.append(count)

    unique_counts = set(class_counts)
    if len(unique_counts) != 1:
        return None
    return unique_counts.pop()


def infer_yolo_model_num_classes(model: torch.nn.Module) -> int | None:
    try:
        return infer_yolo_state_dict_num_classes(model.state_dict())
    except Exception:
        return None


def is_low_quality_trigger_sample(
    manifest: Mapping[str, object],
    sample: Mapping[str, object],
) -> bool:
    if str(sample.get("quality_bucket", "")).strip() == "low_quality":
        return True
    trigger_context = manifest.get("trigger_manifest")
    if isinstance(trigger_context, Mapping):
        return sample.get("raw_relpath") is not None
    return False


def runtime_image_size_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    input_tensor_shape = metadata.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
        height = int(input_tensor_shape[-2])
        width = int(input_tensor_shape[-1])
        if height > 0 and width > 0:
            return height, width
    input_image_size = metadata.get("input_image_size")
    if isinstance(input_image_size, (list, tuple)) and len(input_image_size) >= 2:
        height = int(input_image_size[0])
        width = int(input_image_size[1])
        if height > 0 and width > 0:
            return height, width
    return None


def original_image_size_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    input_image_size = metadata.get("input_image_size")
    if isinstance(input_image_size, (list, tuple)) and len(input_image_size) >= 2:
        height = int(input_image_size[0])
        width = int(input_image_size[1])
        if height > 0 and width > 0:
            return height, width
    return runtime_image_size_from_metadata(metadata)


def runtime_input_tensor_shape_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int, int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    input_tensor_shape = metadata.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 4:
        channels = int(input_tensor_shape[-3])
        height = int(input_tensor_shape[-2])
        width = int(input_tensor_shape[-1])
        if channels > 0 and height > 0 and width > 0:
            return (1, channels, height, width)
    runtime_image_size = runtime_image_size_from_metadata(metadata)
    if runtime_image_size is None:
        return None
    return (1, 3, runtime_image_size[0], runtime_image_size[1])


def pool_label_metadata_from_record(
    record: Mapping[str, object],
    *,
    model_input_size: tuple[int, int] | None,
    resize_mode: str,
) -> dict[str, object]:
    original_size = original_image_size_from_metadata(record)
    metadata: dict[str, object] = {
        "label_coordinate_space": POOL_LABEL_COORDINATE_SPACE,
        "label_resize_mode": str(resize_mode or "direct_resize"),
    }
    if original_size is not None:
        metadata["label_image_size"] = [
            int(original_size[0]),
            int(original_size[1]),
        ]
    return metadata


__all__ = [
    "POOL_LABEL_COORDINATE_SPACE",
    "class_names_from_metadata",
    "coerce_positive_int",
    "infer_yolo_model_num_classes",
    "infer_yolo_state_dict_num_classes",
    "is_cuda_oom_error",
    "is_low_quality_trigger_sample",
    "label_name_from_schema",
    "looks_like_fused_ultralytics_state_dict",
    "normalise_class_name",
    "normalise_label_schema",
    "normalise_shard_dtype",
    "original_image_size_from_metadata",
    "pool_label_metadata_from_record",
    "runtime_image_size_from_metadata",
    "runtime_input_tensor_shape_from_metadata",
]
