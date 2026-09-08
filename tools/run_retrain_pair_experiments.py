from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_server import CloudContinualLearner
from config import load_runtime_config
from edge.sample_quality import LOW_QUALITY, EntropyQualityClassifier
from edge.sample_store import EdgeSampleStore
from edge.transmit import pack_low_quality_trigger_bundle
from model_management.fixed_split import (
    SplitConstraints,
    SplitPlan,
    load_or_compute_fixed_split_plan,
)
from model_management.model_zoo import ensure_local_model_artifact, get_model_artifact_path
from model_management.object_detection import Object_Detection
from model_management.split_model_adapters import build_split_training_loss
from model_management.universal_model_split import UniversalModelSplitter

PROXY_MAP_PATTERN = re.compile(
    r"proxy_mAP@0\.5\s+([0-9.]+)\s+->\s+([0-9.]+)\s+\(delta=([+-][0-9.]+)"
)


@dataclass
class ExperimentResult:
    edge_model: str
    golden_model: str
    success: bool
    accepted_updated_weights: bool
    before_metric_map50: float | None
    after_metric_map50: float | None
    absolute_delta: float | None
    relative_delta_percent: float | None
    sampled_frame_indices: list[int]
    split_config_id: str | None
    split_index: int | None
    payload_bytes: int | None
    sample_stats: dict[str, Any]
    message: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple edge/golden continual-learning retraining experiments.",
    )
    parser.add_argument(
        "--yaml-path",
        default="./config/config.yaml",
        help="Base config path.",
    )
    parser.add_argument(
        "--video-path",
        default="./video_data/road.mp4",
        help="Video used to build the edge training bundle.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Model pairs in edge_model:golden_model format.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="How many video frames to sample for each experiment.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=240,
        help="Keep every Nth frame from the input video.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs per experiment.",
    )
    parser.add_argument(
        "--teacher-threshold",
        type=float,
        default=None,
        help="Optional override for server.continual_learning.teacher_annotation_threshold.",
    )
    parser.add_argument(
        "--send-low-conf-features",
        action="store_true",
        help="Upload teacher-needed features in addition to raw frames.",
    )
    parser.add_argument(
        "--refresh-weights",
        action="store_true",
        help="Delete existing local artifacts for the selected models and download fresh copies.",
    )
    parser.add_argument(
        "--output-root",
        default="./tmp/retrain_pair_experiments",
        help="Directory for per-pair caches and final summary.",
    )
    return parser.parse_args()


def _load_base_config(yaml_path: Path):
    return load_runtime_config(yaml_path)


def _resolve_local_weights_path(model_name: str, *, refresh: bool = False) -> str:
    artifact_path = get_model_artifact_path(model_name)
    if refresh:
        if artifact_path.is_file():
            logger.info("Removing existing artifact for {}: {}", model_name, artifact_path)
            artifact_path.unlink()
        elif artifact_path.is_dir():
            logger.info(
                "Removing existing artifact directory for {}: {}", model_name, artifact_path
            )
            shutil.rmtree(artifact_path, ignore_errors=True)
        tmp_download = artifact_path.with_name(artifact_path.name + ".tmp")
        if tmp_download.exists():
            logger.info("Removing stale temporary artifact for {}: {}", model_name, tmp_download)
            if tmp_download.is_dir():
                shutil.rmtree(tmp_download, ignore_errors=True)
            else:
                tmp_download.unlink()
        partial_download = artifact_path.with_name(artifact_path.name + ".download")
        if partial_download.exists():
            logger.info("Removing stale partial artifact for {}: {}", model_name, partial_download)
            if partial_download.is_dir():
                shutil.rmtree(partial_download, ignore_errors=True)
            else:
                partial_download.unlink()

    artifact_path = ensure_local_model_artifact(model_name)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Failed to prepare local weights for {model_name}: {artifact_path}"
        )
    return str(artifact_path)


def _sample_video_frames(
    video_path: Path,
    *,
    max_samples: int,
    frame_stride: int,
) -> tuple[list[int], list[Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_indices: list[int] = []
    frames: list[Any] = []
    frame_index = 0
    try:
        while len(frames) < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % frame_stride != 0:
                continue
            frame_indices.append(frame_index)
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(
            f"No frames collected from {video_path} with frame_stride={frame_stride}."
        )
    return frame_indices, frames


def _build_split_runtime(
    small_detector: Object_Detection,
    first_frame,
    *,
    model_name: str,
    fixed_split_cfg,
    cache_path: Path,
    configured_training_batch: int | None = None,
) -> tuple[UniversalModelSplitter, SplitPlan]:
    split_model = small_detector.get_split_runtime_model()
    splitter = UniversalModelSplitter(device=next(split_model.parameters()).device)
    splitter.trainability_loss_fn = build_split_training_loss(small_detector.model)
    sample_input = small_detector.prepare_splitter_input(first_frame)
    splitter.trace(split_model, sample_input)
    constraints = SplitConstraints.from_config(fixed_split_cfg)
    validation_batches = None
    if not tuple(getattr(constraints, "validation_batches", ()) or ()):
        try:
            batch = int(configured_training_batch)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            batch = 0
        if batch > 0:
            validation_batches = (1, batch)
    plan = load_or_compute_fixed_split_plan(
        split_model,
        constraints,
        sample_input=sample_input,
        device=next(split_model.parameters()).device,
        model_name=model_name,
        cache_path=str(cache_path),
        splitter=splitter,
        validation_batches=validation_batches,
    )
    return splitter, plan


def _collect_edge_samples(
    *,
    small_detector: Object_Detection,
    splitter: UniversalModelSplitter,
    plan: SplitPlan,
    frame_indices: list[int],
    frames: list[Any],
    sample_store: EdgeSampleStore,
    client_cfg: Any,
    model_name: str,
) -> dict[str, Any]:
    quality_cfg = getattr(client_cfg, "sample_quality", None)
    quality_classifier = EntropyQualityClassifier.from_config(quality_cfg)
    runtime_contract = dict(getattr(plan, "runtime_contract", {}) or {})
    split_key = str(
        getattr(plan, "canonical_split_key", "")
        or runtime_contract.get("logical_split_id")
        or plan.split_config_id
    )
    feature_abi_id = str(
        runtime_contract.get("feature_abi_id") or runtime_contract.get("feature_layout_id") or ""
    )
    for frame_index, frame in zip(frame_indices, frames):
        inference = small_detector.infer_sample(frame, splitter=splitter)
        if inference.intermediate is None:
            raise RuntimeError(
                f"Split runtime did not produce an intermediate payload for frame {frame_index}."
            )

        quality = quality_classifier.classify(
            inference,
            inference.intermediate,
            model_name=model_name,
            split_key=split_key,
            feature_abi_id=feature_abi_id,
        )
        sample_store.store_sample(
            sample_id=str(frame_index),
            frame_index=frame_index,
            confidence=float(inference.confidence),
            split_config_id=plan.split_config_id,
            model_id=model_name,
            model_version="0",
            quality_bucket=quality.quality_bucket,
            quality_metadata=quality.quality_metadata(
                persist_debug_stats=quality_classifier.persist_debug_stats
            ),
            inference_result=inference.to_inference_result(),
            intermediate=inference.intermediate,
            in_drift_window=False,
            raw_frame=frame if quality.quality_bucket == LOW_QUALITY else None,
            input_image_size=list(frame.shape[:2]),
            input_tensor_shape=inference.input_tensor_shape,
        )
    return sample_store.stats()


def _parse_proxy_summary(
    message: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    match = PROXY_MAP_PATTERN.search(message or "")
    if not match:
        return None, None, None, None
    before_value = float(match.group(1))
    after_value = float(match.group(2))
    absolute_delta = float(match.group(3))
    if before_value <= 0:
        relative_delta_percent = None
    else:
        relative_delta_percent = (absolute_delta / before_value) * 100.0
    return before_value, after_value, absolute_delta, relative_delta_percent


def _run_pair_experiment(
    *,
    base_config,
    video_path: Path,
    output_root: Path,
    edge_model: str,
    golden_model: str,
    max_samples: int,
    frame_stride: int,
    epochs: int,
    teacher_threshold: float | None,
    send_low_conf_features: bool,
    refresh_weights: bool,
) -> ExperimentResult:
    pair_root = output_root / f"{edge_model}__{golden_model}"
    if pair_root.exists():
        shutil.rmtree(pair_root)
    pair_root.mkdir(parents=True, exist_ok=True)

    client_cfg = copy.deepcopy(base_config.client)
    server_cfg = copy.deepcopy(base_config.server)
    client_cfg.lightweight = edge_model
    client_cfg.weights_path = _resolve_local_weights_path(edge_model, refresh=refresh_weights)
    server_cfg.edge_model_name = edge_model
    server_cfg.golden = golden_model
    server_cfg.weights_path = _resolve_local_weights_path(edge_model, refresh=refresh_weights)
    server_cfg.continual_learning.num_epoch = int(epochs)
    if teacher_threshold is not None:
        server_cfg.continual_learning.teacher_annotation_threshold = float(teacher_threshold)

    logger.info(
        "Running experiment edge={} golden={} samples={} stride={} epochs={} "
        "send_low_conf_features={}",
        edge_model,
        golden_model,
        max_samples,
        frame_stride,
        epochs,
        send_low_conf_features,
    )

    frame_indices, frames = _sample_video_frames(
        video_path,
        max_samples=max_samples,
        frame_stride=frame_stride,
    )
    logger.info("Sampled frames: {}", frame_indices)

    small_detector = Object_Detection(client_cfg, type="small inference")
    large_detector = Object_Detection(server_cfg, type="large inference")

    learner = CloudContinualLearner(config=server_cfg, large_object_detection=large_detector)
    learner.weight_folder = str(pair_root / "weights")
    os.makedirs(learner.weight_folder, exist_ok=True)
    split_plan: SplitPlan | None = None
    split_runtime_failure: str | None = None
    sample_stats: dict[str, Any] = {}
    before_map = after_map = absolute_delta = relative_delta_percent = None
    accepted_updated_weights = False

    try:
        splitter, split_plan = _build_split_runtime(
            small_detector,
            frames[0],
            model_name=edge_model,
            fixed_split_cfg=client_cfg.split_learning.fixed_split,
            cache_path=pair_root / "fixed_split_plan.json",
            configured_training_batch=server_cfg.continual_learning.batch_size,
        )

        sample_store = EdgeSampleStore(str(pair_root / "sample_store"))
        sample_stats = _collect_edge_samples(
            small_detector=small_detector,
            splitter=splitter,
            plan=split_plan,
            frame_indices=frame_indices,
            frames=frames,
            sample_store=sample_store,
            client_cfg=client_cfg,
            model_name=edge_model,
        )

        payload_zip, _ = pack_low_quality_trigger_bundle(
            sample_store,
            edge_id=1,
            send_low_conf_features=send_low_conf_features,
            split_plan=split_plan,
            model_id=edge_model,
            model_version="0",
        )
        bundle_root = pair_root / "bundle"
        bundle_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload_zip)) as archive:
            archive.extractall(bundle_root)

        success, _, message = learner.get_ground_truth_and_fixed_split_retrain(
            edge_id=1,
            bundle_cache_path=str(bundle_root),
            num_epoch=int(epochs),
        )

        before_map, after_map, absolute_delta, relative_delta_percent = _parse_proxy_summary(
            message
        )
        accepted_updated_weights = bool(success) and "Kept " not in (message or "")
    except RuntimeError as exc:
        split_runtime_failure = str(exc)
        logger.warning(
            "Fixed split retraining failed for edge={} golden={}: {}",
            edge_model,
            golden_model,
            split_runtime_failure,
        )
        success = False
        accepted_updated_weights = False
        before_map = None
        after_map = None
        absolute_delta = None
        relative_delta_percent = None
        message = split_runtime_failure

    result = ExperimentResult(
        edge_model=edge_model,
        golden_model=golden_model,
        success=bool(success),
        accepted_updated_weights=accepted_updated_weights,
        before_metric_map50=before_map,
        after_metric_map50=after_map,
        absolute_delta=absolute_delta,
        relative_delta_percent=relative_delta_percent,
        sampled_frame_indices=frame_indices,
        split_config_id=split_plan.split_config_id if split_plan is not None else None,
        split_index=split_plan.split_index if split_plan is not None else None,
        payload_bytes=split_plan.payload_bytes if split_plan is not None else None,
        sample_stats=sample_stats,
        message=message,
    )

    with (pair_root / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, ensure_ascii=False)
    logger.info(
        "Finished edge={} golden={} success={} accepted={} before={} after={} delta={}",
        edge_model,
        golden_model,
        result.success,
        result.accepted_updated_weights,
        result.before_metric_map50,
        result.after_metric_map50,
        result.absolute_delta,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = _parse_args()
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    base_config = _load_base_config(Path(args.yaml_path))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []
    for pair in args.pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid pair {pair!r}; expected edge_model:golden_model.")
        edge_model, golden_model = pair.split(":", 1)
        edge_model = edge_model.strip()
        golden_model = golden_model.strip()
        try:
            results.append(
                _run_pair_experiment(
                    base_config=base_config,
                    video_path=Path(args.video_path),
                    output_root=output_root,
                    edge_model=edge_model,
                    golden_model=golden_model,
                    max_samples=int(args.max_samples),
                    frame_stride=int(args.frame_stride),
                    epochs=int(args.epochs),
                    teacher_threshold=args.teacher_threshold,
                    send_low_conf_features=bool(args.send_low_conf_features),
                    refresh_weights=bool(args.refresh_weights),
                )
            )
        except Exception as exc:
            logger.exception(
                "Experiment failed for edge={} golden={}: {}",
                edge_model,
                golden_model,
                exc,
            )
            results.append(
                ExperimentResult(
                    edge_model=edge_model,
                    golden_model=golden_model,
                    success=False,
                    accepted_updated_weights=False,
                    before_metric_map50=None,
                    after_metric_map50=None,
                    absolute_delta=None,
                    relative_delta_percent=None,
                    sampled_frame_indices=[],
                    split_config_id=None,
                    split_index=None,
                    payload_bytes=None,
                    sample_stats={},
                    message=str(exc),
                )
            )

    summary = [asdict(result) for result in results]
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    for result in results:
        print(
            json.dumps(
                {
                    "edge_model": result.edge_model,
                    "golden_model": result.golden_model,
                    "success": result.success,
                    "accepted_updated_weights": result.accepted_updated_weights,
                    "before_metric_map50": result.before_metric_map50,
                    "after_metric_map50": result.after_metric_map50,
                    "absolute_delta": result.absolute_delta,
                    "relative_delta_percent": result.relative_delta_percent,
                    "low_quality_count": result.sample_stats.get("low_quality_count", 0),
                    "message": result.message,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
