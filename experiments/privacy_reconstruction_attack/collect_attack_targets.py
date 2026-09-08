from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
import torch
from loguru import logger

import model_management.object_detection as object_detection_runtime
from config import load_runtime_config
from experiments.privacy_reconstruction_attack.attack_dataset import (
    load_experiment_config,
    prediction_to_json,
    sanitize_segment,
    write_json,
)
from experiments.privacy_reconstruction_attack.edge_prefix_whitebox import (
    configure_edge_prefix_parameters,
)
from model_management.model_zoo import get_model_family
from model_management.object_detection import Object_Detection
from model_management.split_model_adapters import (
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
)
from model_management.split_runtime import BoundaryPayloadCacheCodec
from model_management.universal_model_split import UniversalModelSplitter


@dataclass(frozen=True)
class ResolvedSplitPoint:
    name: str
    privacy_leakage_score: float
    split_point: str
    requested_split_point: str
    actual_privacy_leakage_score: float | None
    score_error: float | None


def configure_object_detection_device(device: torch.device) -> None:
    object_detection_runtime.device = torch.device(device)


def _candidate_privacy_score(candidate: Any) -> float:
    ratio = float(getattr(candidate, "edge_parameter_ratio", 0.0) or 0.0)
    return max(0.0, min(1.0, 1.0 - ratio))


def _normalise_candidate_id(value: object) -> str:
    text = str(value or "").strip()
    if (
        text
        and text != "auto"
        and text != "first_compute"
        and not text.startswith("after:")
        and not text.startswith("percent:")
    ):
        return f"after:{text}"
    return text


def _candidate_label(candidate: Any) -> str:
    text = str(getattr(candidate, "candidate_id", "") or "")
    return text.removeprefix("after:")


def _candidate_label_order(label: str) -> int:
    for item in reversed(label.split("_")):
        try:
            return int(item)
        except ValueError:
            continue
    return 10**9


def _looks_like_compute_candidate(candidate: Any) -> bool:
    label = _candidate_label(candidate)
    op = label.split("_", maxsplit=1)[0].lower()
    return op in {
        "conv1d",
        "conv2d",
        "conv3d",
        "linear",
        "matmul",
        "bmm",
        "batchnorm",
        "layernorm",
        "groupnorm",
        "relu",
        "gelu",
        "silu",
        "softmax",
        "maxpool2d",
        "avgpool2d",
    }


def _first_compute_candidate(candidates: Sequence[Any]) -> Any:
    compute_candidates = [
        candidate for candidate in candidates if _looks_like_compute_candidate(candidate)
    ]
    if not compute_candidates:
        raise RuntimeError("No first compute split candidate could be resolved.")
    return min(
        compute_candidates,
        key=lambda candidate: (
            getattr(candidate, "legacy_layer_index", None)
            if getattr(candidate, "legacy_layer_index", None) is not None
            else _candidate_label_order(_candidate_label(candidate)),
            str(getattr(candidate, "candidate_id", "")),
        ),
    )


def _target_score(entry: Mapping[str, Any]) -> float | None:
    value = entry.get("privacy_leakage_score")
    if value is None or str(value).strip().lower() in {"", "auto", "actual"}:
        return None
    return float(value)


def _resolve_split_points(
    splitter: UniversalModelSplitter,
    config: Mapping[str, Any],
) -> list[ResolvedSplitPoint]:
    entries = list(config.get("privacy_score_split_points") or [])
    if not entries:
        raise RuntimeError("privacy_score_split_points must contain at least one entry.")
    split_resolution = dict(config.get("split_resolution") or {})
    require_unique = bool(split_resolution.get("require_unique", True))
    max_candidates = int(split_resolution.get("max_candidates", 0) or 0)
    candidate_limit = max_candidates if max_candidates > 0 else None
    candidates = splitter.enumerate_candidates(max_candidates=candidate_limit)
    if not candidates:
        raise RuntimeError("No TorchLens split candidates were enumerated.")
    score_by_id = {
        str(candidate.candidate_id): _candidate_privacy_score(candidate) for candidate in candidates
    }
    used: set[str] = set()
    resolved: list[ResolvedSplitPoint] = []

    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Each privacy_score_split_points entry must be a mapping.")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise RuntimeError("Every split entry must define a non-empty name.")
        target_score = _target_score(entry)
        requested = str(entry.get("split_point") or "auto").strip()
        requested_lower = requested.lower()
        if requested_lower == "first_compute":
            chosen = _first_compute_candidate(candidates)
            split_point = str(chosen.candidate_id)
            actual = _candidate_privacy_score(chosen)
            if target_score is None:
                target_score = actual
            error = abs(actual - target_score)
        elif requested_lower == "auto":
            if target_score is None:
                raise RuntimeError(
                    "privacy_leakage_score is required when split_point is auto."
                )
            pool = [
                candidate
                for candidate in candidates
                if not require_unique or candidate.candidate_id not in used
            ]
            if not pool:
                raise RuntimeError(
                    "Could not resolve unique split points for all requested privacy scores."
                )
            chosen = min(
                pool,
                key=lambda candidate: (
                    abs(_candidate_privacy_score(candidate) - target_score),
                    str(candidate.candidate_id),
                ),
            )
            split_point = str(chosen.candidate_id)
            actual = _candidate_privacy_score(chosen)
            error = abs(actual - target_score)
        else:
            split_point = _normalise_candidate_id(requested)
            actual = score_by_id.get(split_point)
            if target_score is None:
                target_score = actual
            if target_score is None:
                raise RuntimeError(
                    f"privacy_leakage_score is required for unknown split point {split_point!r}."
                )
            error = None if actual is None else abs(actual - target_score)
        if require_unique and split_point in used:
            raise RuntimeError(f"Resolved duplicate split point {split_point!r}.")
        used.add(split_point)
        resolved.append(
            ResolvedSplitPoint(
                name=name,
                privacy_leakage_score=target_score,
                split_point=split_point,
                requested_split_point=requested,
                actual_privacy_leakage_score=actual,
                score_error=error,
            )
        )
    return resolved


def _sample_video_frames(
    video_path: Path,
    *,
    num_frames: int,
    frame_stride: int,
) -> list[tuple[int, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    stride = max(1, int(frame_stride))
    target_count = max(1, int(num_frames))
    frames: list[tuple[int, Any]] = []
    index = 0
    try:
        while len(frames) < target_count:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append((index, frame))
            index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames sampled from {video_path}.")
    return frames


def _build_trace_splitter(
    detector: Object_Detection,
    sample_input: torch.Tensor,
    *,
    split_point: str,
    device: torch.device,
) -> UniversalModelSplitter:
    splitter = UniversalModelSplitter(device=str(device))
    split_model = detector.get_split_runtime_model()
    split_model.to(device)
    split_model.eval()
    splitter.trace(
        split_model,
        sample_input,
        boundary=split_point,
        model_name=detector.model_name,
        model_family=get_model_family(detector.model_name),
    )
    return splitter


def _prediction_from_split_output(
    detector: Object_Detection,
    frame: Any,
    model_input: torch.Tensor,
    outputs: Any,
) -> dict[str, Any]:
    postprocessed = postprocess_split_runtime_output(
        detector.model,
        outputs,
        threshold=float(detector.threshold_low),
        model_input=model_input,
        orig_image=frame,
    )
    boxes, labels, scores = detector._parse_prediction_output(postprocessed, detector.threshold_low)
    if boxes is None or labels is None or scores is None:
        return prediction_to_json([], [], [], image_size=frame.shape[:2])
    final_threshold = detector._resolve_final_detection_threshold()
    keep = [index for index, score in enumerate(scores) if float(score) > float(final_threshold)]
    boxes = [boxes[index] for index in keep]
    labels = [labels[index] for index in keep]
    scores = [scores[index] for index in keep]
    if boxes:
        boxes, labels, scores = detector._deduplicate_final_predictions(
            boxes,
            labels,
            scores,
            threshold=float(final_threshold),
        )
    return prediction_to_json(boxes, labels, scores, image_size=frame.shape[:2])


def _teacher_prediction(
    detector: Object_Detection, frame: Any, threshold: float | None
) -> dict[str, Any]:
    boxes, labels, scores = detector.large_inference(frame, threshold=threshold)
    return prediction_to_json(boxes or [], labels or [], scores or [], image_size=frame.shape[:2])


def _config_snapshot(value: Any) -> Any:
    if is_dataclass(value):
        return _config_snapshot(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _config_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_config_snapshot(item) for item in value]
    if isinstance(value, list):
        return [_config_snapshot(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_config_snapshot(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def collect_targets(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
    configure_object_detection_device(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = load_experiment_config(args.config)
    runtime_config = load_runtime_config(args.yaml_path)
    edge_prefix_parameters = configure_edge_prefix_parameters(
        runtime_config,
        args.edge_prefix_weights,
    )
    video_path = Path(args.video_path)

    logger.info(
        "[PrivacyAttack] loading models student={} teacher={} "
        "edge_prefix_source={} edge_prefix_sha256={}",
        runtime_config.client.lightweight,
        runtime_config.server.golden,
        edge_prefix_parameters.get("source"),
        edge_prefix_parameters.get("sha256"),
    )
    student = Object_Detection(runtime_config.client, "small inference")
    teacher = Object_Detection(runtime_config.server, "large inference")
    student.model.to(device).eval()
    teacher.model.to(device).eval()

    frames = _sample_video_frames(
        video_path,
        num_frames=int(args.num_frames),
        frame_stride=int(args.frame_stride),
    )
    first_input = prepare_split_runtime_input(student.model, frames[0][1], device=device)
    if not isinstance(first_input, torch.Tensor):
        raise TypeError("Privacy reconstruction attack currently expects tensor split inputs.")
    first_input = first_input.to(device)
    resolver_splitter = _build_trace_splitter(
        student,
        first_input,
        split_point="auto",
        device=device,
    )
    resolved = _resolve_split_points(resolver_splitter, experiment_config)

    resolved_payload = {
        "privacy_score_split_points": [asdict(item) for item in resolved],
        "score_definition": "privacy_leakage_score = 1 - edge_parameter_ratio",
    }
    write_json(output_dir / "resolved_split_points.json", resolved_payload)

    teacher_threshold = experiment_config.get("metrics", {}).get("teacher_threshold")
    teacher_threshold_value = None if teacher_threshold is None else float(teacher_threshold)
    sampled_indices = [int(index) for index, _frame in frames]
    splitters: dict[str, UniversalModelSplitter] = {}
    for split in resolved:
        splitters[split.name] = _build_trace_splitter(
            student,
            first_input,
            split_point=split.split_point,
            device=device,
        )

    for split in resolved:
        split_dir = output_dir / sanitize_segment(split.name)
        split_dir.mkdir(parents=True, exist_ok=True)
        splitter = splitters[split.name]
        codec = BoundaryPayloadCacheCodec(splitter)
        for frame_index, frame in frames:
            sample_id = f"{sanitize_segment(video_path.stem)}_f{int(frame_index):06d}"
            sample_dir = split_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            raw_path = sample_dir / "raw_frame.png"
            ok = cv2.imwrite(str(raw_path), frame)
            if not ok:
                raise RuntimeError(f"Could not write raw frame to {raw_path}.")

            model_input = prepare_split_runtime_input(student.model, frame, device=device)
            if not isinstance(model_input, torch.Tensor):
                raise TypeError(
                    "Privacy reconstruction attack currently expects tensor split inputs."
                )
            model_input = model_input.to(device)
            with torch.no_grad():
                boundary = splitter.edge_forward(model_input)
                replayed = splitter.cloud_forward(boundary)
            student_prediction = _prediction_from_split_output(
                student, frame, model_input, replayed
            )
            teacher_prediction = _teacher_prediction(teacher, frame, teacher_threshold_value)

            torch.save(model_input.detach().cpu(), sample_dir / "model_input_tensor.pt")
            codec.save(
                sample_dir / "boundary_payload.pt.gz",
                boundary,
                metadata={"sample_id": sample_id, "frame_index": int(frame_index)},
            )
            torch.save(
                {
                    str(label): tensor.detach().cpu()
                    for label, tensor in dict(boundary.tensors).items()
                    if isinstance(tensor, torch.Tensor)
                },
                sample_dir / "boundary_feature.pt",
            )
            write_json(sample_dir / "student_prediction.json", student_prediction)
            write_json(sample_dir / "teacher_prediction.json", teacher_prediction)

            metadata = {
                "sample_id": sample_id,
                "video_name": video_path.name,
                "frame_index": int(frame_index),
                "split_name": split.name,
                "split_point": split.split_point,
                "requested_split_point": split.requested_split_point,
                "privacy_leakage_score": float(split.privacy_leakage_score),
                "actual_privacy_leakage_score": split.actual_privacy_leakage_score,
                "input_size": [int(dim) for dim in model_input.shape],
                "raw_image_size": [int(frame.shape[0]), int(frame.shape[1])],
                "raw_image_path": "raw_frame.png",
                "model_input_tensor_path": "model_input_tensor.pt",
                "boundary_payload_path": "boundary_payload.pt.gz",
                "boundary_feature_path": "boundary_feature.pt",
                "teacher_prediction_path": "teacher_prediction.json",
                "student_prediction_path": "student_prediction.json",
                "edge_prefix_parameters": edge_prefix_parameters,
            }
            write_json(sample_dir / "metadata.json", metadata)
            logger.info(
                "[PrivacyAttack] collected target split={} score={} frame={}",
                split.name,
                split.privacy_leakage_score,
                frame_index,
            )

    write_json(
        output_dir / "manifest.json",
        {
            "yaml_path": str(Path(args.yaml_path).resolve()),
            "experiment_config": str(Path(args.config).resolve()),
            "experiment_config_snapshot": _config_snapshot(experiment_config),
            "runtime_config_snapshot": _config_snapshot(runtime_config),
            "video_path": str(video_path.resolve()),
            "video_name": video_path.name,
            "sampled_frame_indices": sampled_indices,
            "num_frames": len(frames),
            "frame_stride": int(args.frame_stride),
            "device": str(device),
            "git_head": _git_head(),
            "student_model": str(runtime_config.client.lightweight),
            "teacher_model": str(runtime_config.server.golden),
            "edge_prefix_parameters": edge_prefix_parameters,
            "privacy_score_split_points": [asdict(item) for item in resolved],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect privacy reconstruction attack targets.")
    parser.add_argument("--yaml_path", default="./config/config.yaml")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--edge-prefix-weights",
        default=None,
        help=(
            "Path to the exact edge-side lightweight checkpoint whose split prefix "
            "produces the boundary payloads. Overrides client.weights_path and is "
            "recorded with sha256 for white-box reconstruction."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    collect_targets(args)


if __name__ == "__main__":
    main(sys.argv[1:])
