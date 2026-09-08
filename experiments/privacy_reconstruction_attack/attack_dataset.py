from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping in {config_path}, got {type(payload).__name__}.")
    return dict(payload)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected JSON object in {path}.")
    return dict(payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned or "unknown"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def prediction_to_json(
    boxes: Any,
    labels: Any,
    scores: Any,
    *,
    image_size: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "boxes": [list(map(float, box)) for box in _as_list(boxes)],
        "labels": [int(label) for label in _as_list(labels)],
        "scores": [float(score) for score in _as_list(scores)],
    }
    if image_size is not None:
        payload["image_size"] = [int(image_size[0]), int(image_size[1])]
    return payload


def prediction_from_artifacts(artifacts: Any) -> dict[str, list[Any]]:
    return prediction_to_json(
        getattr(artifacts, "final_detection_boxes", []) or [],
        getattr(artifacts, "final_detection_labels", []) or [],
        getattr(artifacts, "final_detection_scores", []) or [],
    )


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return bgr_to_rgb(image)


def save_rgb_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), rgb_to_bgr(clipped))
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3 or int(x.shape[0]) != 3:
        raise ValueError(f"Expected CHW or BCHW RGB tensor, got {tuple(x.shape)}.")
    x = x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def save_tensor_image(path: str | Path, tensor: torch.Tensor) -> None:
    save_rgb_image(path, tensor_to_rgb_image(tensor))


def load_tensor(path: str | Path, *, device: str | torch.device = "cpu") -> torch.Tensor:
    value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(value).__name__}.")
    return value


def _iter_metadata_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("metadata.json"))


@dataclass(frozen=True)
class AttackSample:
    split_name: str
    sample_id: str
    sample_dir: Path
    metadata: dict[str, Any]

    @property
    def privacy_leakage_score(self) -> float:
        return float(self.metadata.get("privacy_leakage_score", math.nan))

    @property
    def split_point(self) -> str:
        return str(self.metadata.get("split_point") or "")

    def path(self, key: str) -> Path:
        value = self.metadata.get(key)
        if value:
            path = Path(str(value))
            if path.is_absolute():
                return path
            return (self.sample_dir / path).resolve()
        default_name = {
            "raw_image_path": "raw_frame.png",
            "model_input_tensor_path": "model_input_tensor.pt",
            "boundary_payload_path": "boundary_payload.pt.gz",
            "boundary_feature_path": "boundary_feature.pt",
            "teacher_prediction_path": "teacher_prediction.json",
            "student_prediction_path": "student_prediction.json",
        }.get(key, key)
        return self.sample_dir / default_name


def load_attack_samples(targets_dir: str | Path) -> list[AttackSample]:
    root = Path(targets_dir)
    samples: list[AttackSample] = []
    for metadata_path in _iter_metadata_files(root):
        metadata = read_json(metadata_path)
        split_name = str(metadata.get("split_name") or metadata_path.parents[1].name)
        sample_id = str(metadata.get("sample_id") or metadata_path.parent.name)
        samples.append(
            AttackSample(
                split_name=split_name,
                sample_id=sample_id,
                sample_dir=metadata_path.parent,
                metadata=metadata,
            )
        )
    return samples


def group_samples_by_split(samples: Iterable[AttackSample]) -> dict[str, list[AttackSample]]:
    grouped: dict[str, list[AttackSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.split_name, []).append(sample)
    for split_samples in grouped.values():
        split_samples.sort(
            key=lambda item: (int(item.metadata.get("frame_index", 0)), item.sample_id)
        )
    return dict(sorted(grouped.items()))


def read_resolved_split_points(targets_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(targets_dir) / "resolved_split_points.json"
    if not path.exists():
        return []
    payload = read_json(path)
    split_points = payload.get("privacy_score_split_points")
    if not isinstance(split_points, list):
        return []
    return [dict(item) for item in split_points if isinstance(item, Mapping)]


def parse_score_from_split_name(name: str) -> float | None:
    match = re.search(r"(\d+)_(\d+)$", str(name))
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")
