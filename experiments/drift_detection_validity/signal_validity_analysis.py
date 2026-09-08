#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.drift_detection_validity.experiment_io import (  # noqa: E402
    load_config,
    output_dir,
)

LOGGER = logging.getLogger("drift_validity.signal_analysis")

SIGNAL_COLUMNS = {
    "confidence_drop_z": "mean_confidence_drop_z",
    "output_entropy": "mean_output_entropy",
    "ema_output_entropy_z": "mean_ema_output_entropy_z",
    "boundary_feature_deviation": "mean_boundary_feature_deviation",
    "ema_boundary_feature_deviation_z": "mean_ema_boundary_feature_deviation_z",
    "full_drift_score_z": "mean_full_drift_score_z",
}

SUMMARY_FIELDS = [
    "signal",
    "pearson",
    "spearman",
    "roc_auc",
    "pr_auc",
    "best_threshold",
    "best_f1",
    "best_precision",
    "best_recall",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_pairs(
    x_values: Iterable[Any],
    y_values: Iterable[Any],
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(_float(x), _float(y)) for x, y in zip(x_values, y_values)]
    finite = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if not finite:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    x, y = zip(*finite)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def pearson_correlation(x_values: Iterable[Any], y_values: Iterable[Any]) -> float:
    x, y = _finite_pairs(x_values, y_values)
    if x.size < 2:
        return math.nan
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(x_centered, y_centered) / denom)


def average_ranks(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([_float(value) for value in values], dtype=np.float64)
    ranks = np.full(arr.shape, math.nan, dtype=np.float64)
    finite_indices = np.where(np.isfinite(arr))[0]
    if finite_indices.size == 0:
        return ranks
    order = finite_indices[np.argsort(arr[finite_indices], kind="mergesort")]
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and arr[order[end]] == arr[order[start]]:
            end += 1
        # One-based average rank for deterministic tie handling.
        rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(x_values: Iterable[Any], y_values: Iterable[Any]) -> float:
    x_ranks = average_ranks(x_values)
    y_ranks = average_ranks(y_values)
    return pearson_correlation(x_ranks, y_ranks)


def _labels_scores(
    labels: Iterable[Any],
    scores: Iterable[Any],
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([1 if _bool(value) else 0 for value in labels], dtype=np.int64)
    s = np.asarray([_float(value) for value in scores], dtype=np.float64)
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    mask = np.isfinite(s)
    return y[mask], s[mask]


def _warn_degenerate(metric_name: str) -> float:
    LOGGER.warning("%s is undefined because only one class is present.", metric_name)
    return math.nan


def roc_auc_score(labels: Iterable[Any], scores: Iterable[Any]) -> float:
    y, s = _labels_scores(labels, scores)
    positives = s[y == 1]
    negatives = s[y == 0]
    if positives.size == 0 or negatives.size == 0:
        return _warn_degenerate("ROC-AUC")
    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return float(wins / float(positives.size * negatives.size))


def pr_auc_score(labels: Iterable[Any], scores: Iterable[Any]) -> float:
    y, s = _labels_scores(labels, scores)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return _warn_degenerate("PR-AUC")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for label in y_sorted.tolist():
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / float(positives)
        precision = tp / float(max(tp + fp, 1))
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return float(area)


def best_f1_threshold(labels: Iterable[Any], scores: Iterable[Any]) -> dict[str, float]:
    y, s = _labels_scores(labels, scores)
    if y.size == 0:
        return {
            "threshold": math.nan,
            "f1": math.nan,
            "precision": math.nan,
            "recall": math.nan,
        }
    thresholds = sorted(set(float(value) for value in s.tolist()), reverse=True)
    best = {"threshold": math.nan, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in thresholds:
        pred = s >= threshold
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        precision = tp / float(max(tp + fp, 1))
        recall = tp / float(max(tp + fn, 1))
        f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (
            precision + recall
        )
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def analyze_signal_validity(config: Mapping[str, Any]) -> Path:
    root = output_dir(config)
    input_path = root / "records" / "window_metrics.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing window metrics: {input_path}")
    rows = _read_csv(input_path)
    labels = [_bool(row.get("is_harmful_drift_window")) for row in rows]
    f1_drop = [_float(row.get("f1_drop")) for row in rows]
    summary_rows: list[dict[str, Any]] = []
    for signal_name, column in SIGNAL_COLUMNS.items():
        scores = [_float(row.get(column)) for row in rows]
        best = best_f1_threshold(labels, scores)
        summary_rows.append(
            {
                "signal": signal_name,
                "pearson": pearson_correlation(scores, f1_drop),
                "spearman": spearman_correlation(scores, f1_drop),
                "roc_auc": roc_auc_score(labels, scores),
                "pr_auc": pr_auc_score(labels, scores),
                "best_threshold": best["threshold"],
                "best_f1": best["f1"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
            }
        )
    output_path = root / "analysis" / "signal_validity_summary.csv"
    _write_csv(output_path, summary_rows, SUMMARY_FIELDS)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze drift signal validity.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = build_parser().parse_args(argv)
    analyze_signal_validity(load_config(args.config))


if __name__ == "__main__":
    main()
