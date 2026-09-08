from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

from experiments.privacy_reconstruction_attack.attack_dataset import (
    load_experiment_config,
    read_json,
    write_json,
)

PER_SAMPLE_FIELDS = [
    "method",
    "split_name",
    "split_point",
    "sample_id",
    "frame_index",
    "privacy_leakage_score",
    "reconstruction_mode",
    "MSE",
    "PSNR",
    "SSIM",
    "LPIPS",
    "FeatureDistanceInitial",
    "FeatureDistanceFinal",
    "drag_latent_init",
    "drag_strength",
    "init_label",
    "init_feature_loss",
    "linear_decoder_label",
    "linear_init_feature_loss",
    "feature_inversion_init",
    "feature_inversion_feature_loss",
    "feature_inversion_total_loss_final",
    "num_iterations",
    "metric_reference",
    "ObjectPrecision",
    "ObjectRecall",
    "ObjectF1",
    "L_actual",
    "LPIPSAvailable",
    "ObjectF1Valid",
    "metrics_path",
]

SUMMARY_FIELDS = [
    "method",
    "privacy_leakage_score",
    "num_samples",
    "valid_object_f1_samples",
    "MSE_mean",
    "MSE_std",
    "PSNR_mean",
    "PSNR_std",
    "SSIM_mean",
    "SSIM_std",
    "LPIPS_mean",
    "LPIPS_std",
    "ObjectF1_mean",
    "ObjectF1_std",
    "L_actual_mean",
    "L_actual_std",
    "FeatureDistanceFinal_mean",
    "FeatureDistanceFinal_std",
]


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _valid(value: Any) -> bool:
    parsed = _to_float(value)
    return not math.isnan(parsed)


def _mean(values: Iterable[Any]) -> float:
    numeric = [_to_float(value) for value in values]
    numeric = [value for value in numeric if not math.isnan(value)]
    return float(sum(numeric) / len(numeric)) if numeric else float("nan")


def _std(values: Iterable[Any]) -> float:
    numeric = [_to_float(value) for value in values]
    numeric = [value for value in numeric if not math.isnan(value)]
    if len(numeric) <= 1:
        return 0.0 if len(numeric) == 1 else float("nan")
    avg = sum(numeric) / len(numeric)
    return float(math.sqrt(sum((value - avg) ** 2 for value in numeric) / (len(numeric) - 1)))


def _metrics_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("metrics.json"))


def _read_method_rows(method: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _metrics_files(root):
        payload = read_json(path)
        row = {
            field: payload.get(field, "") for field in PER_SAMPLE_FIELDS if field != "metrics_path"
        }
        row["method"] = str(payload.get("method") or method)
        row["metrics_path"] = str(path)
        rows.append(row)
        logger.info(
            "[PrivacyEval] method={} score={} ObjectF1={} SSIM={} L_actual={}",
            row["method"],
            row.get("privacy_leakage_score"),
            row.get("ObjectF1"),
            row.get("SSIM"),
            row.get("L_actual"),
        )
    return rows


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        score = _to_float(row.get("privacy_leakage_score"))
        if math.isnan(score):
            continue
        grouped.setdefault((str(row.get("method") or ""), score), []).append(row)
    summary: list[dict[str, Any]] = []
    for (method, score), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], -item[0][1])
    ):
        item: dict[str, Any] = {
            "method": method,
            "privacy_leakage_score": score,
            "num_samples": len(group),
            "valid_object_f1_samples": sum(1 for row in group if _valid(row.get("ObjectF1"))),
        }
        for metric in (
            "MSE",
            "PSNR",
            "SSIM",
            "LPIPS",
            "ObjectF1",
            "L_actual",
            "FeatureDistanceFinal",
        ):
            item[f"{metric}_mean"] = _mean(row.get(metric) for row in group)
            item[f"{metric}_std"] = _std(row.get(metric) for row in group)
        summary.append(item)
    return summary


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return float(numerator / (dx * dy)) if dx > 0.0 and dy > 0.0 else float("nan")


def _spearman(x: list[float], y: list[float]) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore

        return float(spearmanr(x, y, nan_policy="omit").statistic)
    except Exception:
        return _pearson(_rank(x), _rank(y))


def _kendall(x: list[float], y: list[float]) -> float:
    try:
        from scipy.stats import kendalltau  # type: ignore

        return float(kendalltau(x, y, nan_policy="omit").statistic)
    except Exception:
        concordant = discordant = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                product = dx * dy
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
        denom = concordant + discordant
        return float((concordant - discordant) / denom) if denom else float("nan")


def _series_for(
    summary: list[dict[str, Any]], method: str, metric: str
) -> tuple[list[float], list[float]]:
    pairs: list[tuple[float, float]] = []
    for row in summary:
        if row.get("method") != method:
            continue
        score = _to_float(row.get("privacy_leakage_score"))
        value = _to_float(row.get(metric))
        if not math.isnan(score) and not math.isnan(value):
            pairs.append((score, value))
    pairs.sort(key=lambda item: item[0])
    return [score for score, _value in pairs], [value for _score, value in pairs]


def _delta(summary: list[dict[str, Any]], method: str, metric: str) -> float:
    values = {
        round(_to_float(row.get("privacy_leakage_score")), 6): _to_float(row.get(metric))
        for row in summary
        if row.get("method") == method
    }
    high = values.get(0.8)
    low = values.get(0.2)
    if high is None or low is None or math.isnan(high) or math.isnan(low):
        return float("nan")
    return float(high - low)


def _correlations(summary: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    methods = sorted({str(row.get("method") or "") for row in summary if row.get("method")})
    for method in methods:
        scores_l, l_actual = _series_for(summary, method, "L_actual_mean")
        scores_f1, object_f1 = _series_for(summary, method, "ObjectF1_mean")
        result[method] = {
            "spearman_score_vs_L_actual": _spearman(scores_l, l_actual)
            if len(scores_l) >= 2
            else float("nan"),
            "kendall_score_vs_L_actual": _kendall(scores_l, l_actual)
            if len(scores_l) >= 2
            else float("nan"),
            "spearman_score_vs_ObjectF1": _spearman(scores_f1, object_f1)
            if len(scores_f1) >= 2
            else float("nan"),
            "delta_L_actual_0_8_minus_0_2": _delta(summary, method, "L_actual_mean"),
        }
    return result


def evaluate(args: argparse.Namespace) -> None:
    load_experiment_config(args.config)
    output_dir = Path(args.output_dir)
    attack_root = Path(args.attack_dir or args.drag_dir)
    manifest_path = attack_root / "manifest.json"
    default_method = "drag_linear_clean"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        default_method = str(manifest.get("method") or default_method)
    rows = _read_method_rows(default_method, attack_root)
    _write_csv(output_dir / "per_sample.csv", rows, PER_SAMPLE_FIELDS)
    _write_csv(output_dir / "drag_per_sample.csv", rows, PER_SAMPLE_FIELDS)
    summary = _summary_rows(rows)
    _write_csv(output_dir / "summary_by_score.csv", summary, SUMMARY_FIELDS)
    write_json(output_dir / "score_correlation.json", _correlations(summary))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate privacy reconstruction scores.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--attack_dir", default=None)
    parser.add_argument("--drag_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.attack_dir and not args.drag_dir:
        raise SystemExit("Either --attack_dir or --drag_dir is required.\n")
    evaluate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
