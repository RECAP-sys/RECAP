from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping

import torch

import model_management.model_zoo as model_zoo
from cloud.training.proxy_metadata import (
    coerce_positive_int as _coerce_positive_int,
)
from model_management.detection_box_projection import ORIGINAL_XYXY

POOL_LABEL_COORDINATE_SPACE = ORIGINAL_XYXY
POOL_LABEL_METADATA_FIELDS = (
    "label_coordinate_space",
    "label_image_size",
    "label_input_size",
    "label_resize_mode",
)
_CACHED_SPLIT_PROXY_EVAL_MODEL_FAMILIES = frozenset({"yolo", "rfdetr", "tinynext"})


class _TeacherAnnotationQueueState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.next_ticket = 0
        self.serving_ticket = 0
        self.ticket_states: dict[int, str] = {}
        self.ticket_local = threading.local()


_GLOBAL_TEACHER_ANNOTATION_QUEUE = _TeacherAnnotationQueueState()


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_fingerprint(payload: object) -> str:
    return hashlib.sha1(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_file(path: str) -> dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _manifest_model_metadata(manifest: Mapping[str, object]) -> dict[str, object]:
    model_meta = manifest.get("model")
    metadata = dict(model_meta) if isinstance(model_meta, Mapping) else {}
    for manifest_key, metadata_key in (
        ("model_id", "model_id"),
        ("model_version", "model_version"),
        ("model_num_classes", "num_classes"),
        ("model_label_schema", "label_schema"),
    ):
        value = manifest.get(manifest_key)
        if value is not None and metadata_key not in metadata:
            metadata[metadata_key] = value
    return metadata


def _rfdetr_num_classes_from_metadata(
    metadata: Mapping[str, object] | None,
) -> int | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in (
        "rfdetr_head_num_classes",
        "num_classes",
        "class_logits",
        "head_num_classes",
    ):
        value = _coerce_positive_int(metadata.get(key))
        if value is not None:
            return value
    return None


def _infer_rfdetr_checkpoint_num_classes(checkpoint: object) -> int | None:
    inferred = model_zoo.infer_rfdetr_state_dict_num_classes(checkpoint)
    if inferred is not None:
        return inferred
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model", "state_dict"):
        nested = checkpoint.get(key)
        inferred = model_zoo.infer_rfdetr_state_dict_num_classes(nested)
        if inferred is not None:
            return inferred
    return None


def _validate_rfdetr_weights_match_metadata(
    *,
    model_name: str,
    weights_path: str,
    model_metadata: Mapping[str, object] | None,
    device: torch.device | str,
) -> None:
    expected = _rfdetr_num_classes_from_metadata(model_metadata)
    if expected is None or model_zoo.get_model_family(str(model_name)) != "rfdetr":
        return
    if not weights_path or not os.path.exists(weights_path):
        return

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    actual = _infer_rfdetr_checkpoint_num_classes(checkpoint)
    if actual is None or actual == expected:
        return

    raise RuntimeError(
        "[FixedSplitCL] RF-DETR weights class head mismatch for "
        f"{model_name}: edge manifest expects {expected} logits, but weights "
        f"at {weights_path} contain {actual}. Configure server.weights_path to "
        "the same custom checkpoint as client.weights_path."
    )


def _iter_tensors(value: object):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _sanitize_cache_segment(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned or "unknown"


def _normalize_model_version(
    value: object,
    *,
    field_name: str,
) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "0"
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer string, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")
    return str(number)


def _increment_model_version(
    value: object,
    *,
    field_name: str,
) -> str:
    return str(int(_normalize_model_version(value, field_name=field_name)) + 1)


__all__ = [name for name in globals() if not name.startswith("__")]
