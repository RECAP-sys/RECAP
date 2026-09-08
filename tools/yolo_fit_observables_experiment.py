from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_server import CloudContinualLearner
from config.runtime import load_runtime_config
from model_management.fixed_split import SplitPlan, apply_split_plan
from model_management.model_zoo import (
    build_detection_model,
    ensure_local_model_artifact,
    get_model_detection_thresholds,
)
from model_management.object_detection import Object_Detection
from model_management.split_model_adapters import (
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
    summarize_split_runtime_observables,
)
from model_management.universal_model_split import UniversalModelSplitter


def _load_trigger_eval_manifest(bundle_root: Path) -> dict[str, Any]:
    with (bundle_root / "trigger_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples: list[dict[str, str]] = []
    staging_root = bundle_root / ".observables_trigger_raw"
    staging_root.mkdir(parents=True, exist_ok=True)
    for shard in manifest.get("raw_shards", []) or []:
        shard_file = shard.get("file") if isinstance(shard, dict) else None
        if not shard_file:
            continue
        with tarfile.open(bundle_root / str(shard_file).replace("/", "\\"), "r") as archive:
            manifest_member = archive.extractfile("manifest.jsonl")
            if manifest_member is None:
                continue
            for line in manifest_member:
                if not line.strip():
                    continue
                entry = json.loads(line.decode("utf-8"))
                sample_id = str(entry.get("sample_id", "") or "")
                raw_file = str(entry.get("raw_file", "") or "")
                if not sample_id or not raw_file:
                    continue
                suffix = Path(raw_file).suffix or ".jpg"
                target = staging_root / f"{sample_id}{suffix}"
                source = archive.extractfile(raw_file)
                if source is None:
                    continue
                if not target.exists():
                    target.write_bytes(source.read())
                samples.append(
                    {
                        "sample_id": sample_id,
                        "raw_relpath": target.relative_to(bundle_root).as_posix(),
                    }
                )
    manifest["samples"] = samples
    return manifest


def _collect_eval_items(bundle_root: Path, holdout_dir: Path) -> list[dict[str, str]]:
    manifest = _load_trigger_eval_manifest(bundle_root)
    items: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for sample in manifest.get("samples", []):
        raw_relpath = sample.get("raw_relpath")
        if not raw_relpath:
            continue
        raw_path = bundle_root / str(raw_relpath).replace("/", "\\")
        if not raw_path.exists():
            continue
        key = str(raw_path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        items.append(
            {
                "sample_id": str(sample["sample_id"]),
                "subset": "train_bundle",
                "path": key,
            }
        )

    for image_path in sorted(holdout_dir.glob("*.jpg")):
        key = str(image_path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        items.append(
            {
                "sample_id": image_path.stem,
                "subset": "holdout",
                "path": key,
            }
        )
    return items


def _decode_state_dict(encoded: str) -> dict[str, torch.Tensor]:
    return torch.load(
        io.BytesIO(base64.b64decode(encoded)),
        map_location="cpu",
        weights_only=False,
    )


def _boxes_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_fit_metrics(
    teacher: dict[str, Any],
    prediction: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    gt_boxes = list(teacher.get("boxes") or [])
    gt_labels = list(teacher.get("labels") or [])
    pred_boxes = list(prediction.get("boxes") or [])
    pred_labels = list(prediction.get("labels") or [])
    pred_scores = list(prediction.get("scores") or [])

    order = sorted(
        range(len(pred_scores)),
        key=lambda index: float(pred_scores[index]),
        reverse=True,
    )
    matched_gt: set[int] = set()
    true_positive = 0

    for pred_index in order:
        best_match = None
        best_iou = 0.0
        for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_index in matched_gt:
                continue
            if int(pred_labels[pred_index]) != int(gt_label):
                continue
            iou = _boxes_iou(np.asarray(pred_boxes[pred_index]), np.asarray(gt_box))
            if iou >= iou_threshold and iou > best_iou:
                best_match = gt_index
                best_iou = iou
        if best_match is not None:
            matched_gt.add(best_match)
            true_positive += 1

    false_positive = max(0, len(pred_boxes) - true_positive)
    false_negative = max(0, len(gt_boxes) - true_positive)
    precision = true_positive / float(max(1, true_positive + false_positive))
    recall = true_positive / float(max(1, true_positive + false_negative))
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "teacher_count": int(len(gt_boxes)),
        "prediction_count": int(len(pred_boxes)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        avg_rank = 0.5 * (index + end) + 1.0
        ranks[order[index : end + 1]] = avg_rank
        index = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have the same length.")
    if len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return None
    ranked_x = _rankdata(x_arr[mask])
    ranked_y = _rankdata(y_arr[mask])
    x_std = float(ranked_x.std())
    y_std = float(ranked_y.std())
    if x_std <= 0.0 or y_std <= 0.0:
        return None
    corr = np.corrcoef(ranked_x, ranked_y)[0, 1]
    return float(corr)


def _trace_split_runtime(
    model: torch.nn.Module,
    *,
    bundle_root: Path,
    manifest: dict[str, Any],
    device: str = "cpu",
) -> tuple[UniversalModelSplitter, Any]:
    splitter = UniversalModelSplitter(device=device)
    runtime_model = get_split_runtime_model(model)
    trace_frame = None
    for sample in manifest.get("samples", []):
        raw_relpath = sample.get("raw_relpath")
        if not raw_relpath:
            continue
        candidate_path = bundle_root / str(raw_relpath).replace("/", "\\")
        if candidate_path.exists():
            trace_frame = cv2.imread(str(candidate_path))
            if trace_frame is not None:
                break
    if trace_frame is None:
        raise RuntimeError("Unable to locate a trace frame for yolo26n split runtime.")
    sample_input = prepare_split_runtime_input(model, trace_frame, device=device)
    splitter.trace(runtime_model, sample_input)
    candidate = apply_split_plan(splitter, SplitPlan.from_dict(manifest["split_plan"]))
    splitter.prepare_inference_replay(sample_input)
    return splitter, candidate


def _evaluate_state(
    *,
    state_name: str,
    model: torch.nn.Module,
    splitter: UniversalModelSplitter,
    candidate: Any,
    items: list[dict[str, str]],
    teacher_targets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    _, threshold_high = get_model_detection_thresholds(model, "yolo26n")
    records: list[dict[str, Any]] = []
    for item in items:
        frame = cv2.imread(item["path"])
        if frame is None:
            continue
        runtime_input = prepare_split_runtime_input(model, frame, device="cpu")
        with torch.no_grad():
            outputs, split_payload = splitter.replay_inference(
                runtime_input,
                candidate=candidate,
                return_split_output=True,
            )
            observables = summarize_split_runtime_observables(model, outputs, split_payload)
            detections = postprocess_split_runtime_output(
                model,
                outputs,
                threshold=threshold_high,
                model_input=runtime_input,
                orig_image=frame,
            )
        detection = detections[0]
        prediction = {
            "boxes": detection["boxes"].detach().cpu().tolist(),
            "labels": detection["labels"].detach().cpu().tolist(),
            "scores": detection["scores"].detach().cpu().tolist(),
        }
        fit_metrics = _match_fit_metrics(teacher_targets[item["sample_id"]], prediction)
        records.append(
            {
                "state": state_name,
                "sample_id": item["sample_id"],
                "subset": item["subset"],
                "fit_precision": fit_metrics["precision"],
                "fit_recall": fit_metrics["recall"],
                "fit_f1": fit_metrics["f1"],
                "teacher_count": fit_metrics["teacher_count"],
                "prediction_count": fit_metrics["prediction_count"],
                "tp": fit_metrics["tp"],
                "fp": fit_metrics["fp"],
                "fn": fit_metrics["fn"],
                "feature_spectral_entropy": observables["feature_spectral_entropy"],
                "logit_entropy": observables["logit_entropy"],
                "logit_margin": observables["logit_margin"],
                "logit_energy": observables["logit_energy"],
            }
        )
    return records


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "fit_precision",
        "fit_recall",
        "fit_f1",
        "feature_spectral_entropy",
        "logit_entropy",
        "logit_margin",
        "logit_energy",
        "teacher_count",
        "prediction_count",
    ]
    summary: dict[str, Any] = {}
    for subset_name in ("all", "train_bundle", "holdout"):
        subset_records = (
            records
            if subset_name == "all"
            else [row for row in records if row["subset"] == subset_name]
        )
        if not subset_records:
            continue
        subset_summary: dict[str, Any] = {"count": len(subset_records)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset_records if row.get(metric) is not None]
            if values:
                subset_summary[metric] = float(np.mean(values))
        summary[subset_name] = subset_summary
    return summary


def _cross_sample_correlations(records: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    observables = [
        "feature_spectral_entropy",
        "logit_entropy",
        "logit_margin",
        "logit_energy",
    ]
    correlations: dict[str, dict[str, float | None]] = {}
    for subset_name in ("all", "train_bundle", "holdout"):
        subset_records = (
            records
            if subset_name == "all"
            else [row for row in records if row["subset"] == subset_name]
        )
        fit_scores = [float(row["fit_f1"]) for row in subset_records]
        subset_corr: dict[str, float | None] = {}
        for observable in observables:
            values = [float(row[observable]) for row in subset_records]
            subset_corr[observable] = _spearman(values, fit_scores)
        correlations[subset_name] = subset_corr
    return correlations


def _within_sample_state_correlations(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    observables = [
        "feature_spectral_entropy",
        "logit_entropy",
        "logit_margin",
        "logit_energy",
    ]
    by_sample: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        by_sample.setdefault((row["subset"], row["sample_id"]), []).append(row)

    results: dict[str, dict[str, Any]] = {}
    for subset_name in ("all", "train_bundle", "holdout"):
        subset_groups = [
            rows
            for (group_subset, _), rows in by_sample.items()
            if subset_name == "all" or group_subset == subset_name
        ]
        subset_report: dict[str, Any] = {}
        for observable in observables:
            scores: list[float] = []
            for rows in subset_groups:
                if len(rows) < 2:
                    continue
                fit_scores = [float(row["fit_f1"]) for row in rows]
                observable_values = [float(row[observable]) for row in rows]
                corr = _spearman(observable_values, fit_scores)
                if corr is not None:
                    scores.append(float(corr))
            subset_report[observable] = {
                "count": len(scores),
                "mean_spearman": float(np.mean(scores)) if scores else None,
                "median_spearman": float(np.median(scores)) if scores else None,
                "positive_fraction": (
                    float(np.mean([score > 0 for score in scores])) if scores else None
                ),
                "negative_fraction": (
                    float(np.mean([score < 0 for score in scores])) if scores else None
                ),
            }
        results[subset_name] = subset_report
    return results


def _make_model(state_name: str, trained_state: dict[str, torch.Tensor] | None) -> torch.nn.Module:
    if state_name == "pretrained":
        return build_detection_model("yolo26n", pretrained=True, device="cpu")
    model = build_detection_model("yolo26n", pretrained=False, device="cpu")
    if state_name == "cloud_trained":
        if trained_state is None:
            raise RuntimeError("cloud_trained state requested without a state_dict.")
        model.load_state_dict(trained_state, strict=False)
    return model


def run_experiment(
    *,
    config_path: Path,
    bundle_root: Path,
    holdout_dir: Path,
    report_root: Path,
) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)

    config = load_runtime_config(config_path)
    config.client.lightweight = "yolo26n"
    config.server.edge_model_name = "yolo26n"

    large_od = Object_Detection(config.server, "large inference")
    learner = CloudContinualLearner(config=config.server, large_object_detection=large_od)
    learner.weight_folder = str((report_root / "weights").resolve())
    Path(learner.weight_folder).mkdir(parents=True, exist_ok=True)

    success, model_data, train_message = learner.get_ground_truth_and_fixed_split_retrain(
        edge_id=99,
        bundle_cache_path=str(bundle_root),
        num_epoch=int(config.server.continual_learning.num_epoch),
    )
    if not success:
        raise RuntimeError(f"Cloud retrain failed: {train_message}")

    trained_state = _decode_state_dict(model_data)
    trained_state_path = report_root / "cloud_trained_yolo26n_state.pth"
    torch.save(trained_state, trained_state_path)

    items = _collect_eval_items(bundle_root, holdout_dir)
    teacher_targets: dict[str, dict[str, Any]] = {}
    for item in items:
        frame = cv2.imread(item["path"])
        if frame is None:
            continue
        targets = learner._build_teacher_targets(frame)
        if not targets:
            continue
        teacher_targets[item["sample_id"]] = targets
    items = [item for item in items if item["sample_id"] in teacher_targets]

    manifest = _load_trigger_eval_manifest(bundle_root)
    all_records: list[dict[str, Any]] = []
    state_reports: dict[str, Any] = {}
    for state_name in ("random_init", "pretrained", "cloud_trained"):
        model = _make_model(state_name, trained_state)
        splitter, candidate = _trace_split_runtime(
            model,
            bundle_root=bundle_root,
            manifest=manifest,
        )
        state_records = _evaluate_state(
            state_name=state_name,
            model=model,
            splitter=splitter,
            candidate=candidate,
            items=items,
            teacher_targets=teacher_targets,
        )
        all_records.extend(state_records)
        state_reports[state_name] = {
            "summary": _summarize_records(state_records),
            "cross_sample_spearman": _cross_sample_correlations(state_records),
        }

    report = {
        "config_path": str(config_path.resolve()),
        "bundle_root": str(bundle_root.resolve()),
        "holdout_dir": str(holdout_dir.resolve()),
        "weights": {
            "pretrained_artifact": str(ensure_local_model_artifact("yolo26n")),
            "cloud_trained_state": str(trained_state_path.resolve()),
        },
        "cloud_train_message": train_message,
        "sample_count": len(items),
        "state_reports": state_reports,
        "within_sample_state_spearman": _within_sample_state_correlations(all_records),
        "records": all_records,
    }
    with (report_root / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate yolo26n fit observables across weight regimes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("cache/e2e_full_flow/server_bundle"),
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=Path("cache/e2e_full_flow/holdout_frames"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("cache/tmp_debug/yolo26n_fit_observables"),
    )
    args = parser.parse_args()
    report = run_experiment(
        config_path=args.config,
        bundle_root=args.bundle_root,
        holdout_dir=args.holdout_dir,
        report_root=args.report_root,
    )
    print(
        json.dumps(
            {
                "cloud_train_message": report["cloud_train_message"],
                "sample_count": report["sample_count"],
                "report_path": str((args.report_root / "report.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
