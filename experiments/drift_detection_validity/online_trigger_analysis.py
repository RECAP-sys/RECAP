#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.drift_detection_validity.experiment_io import (  # noqa: E402
    load_config,
    output_dir,
    require_bool,
    require_float,
    require_int,
    require_mapping,
)

METHOD_COLUMNS = {
    "confidence_only": "mean_confidence_drop_z",
    "ema_entropy": "mean_ema_output_entropy_z",
    "ema_feature_deviation": "mean_ema_boundary_feature_deviation_z",
    "recap_full": "mean_full_drift_score_z",
}

SEQUENCE_SUMMARY_FIELDS = [
    "method",
    "sequence_name",
    "true_drift_events",
    "detected",
    "missed",
    "false_triggers",
    "avg_detection_delay_frames",
    "median_detection_delay_frames",
    "precision",
    "recall",
    "trigger_f1",
    "false_triggers_per_1000_frames",
    "early_tolerance_frames",
    "late_tolerance_frames",
]

METHOD_SUMMARY_FIELDS = [
    "method",
    "num_sequences",
    "true_drift_events",
    "detected",
    "missed",
    "false_triggers",
    "avg_detection_delay_frames",
    "median_detection_delay_frames",
    "precision",
    "recall",
    "trigger_f1",
    "false_triggers_per_1000_frames",
    "early_tolerance_frames",
    "late_tolerance_frames",
]

TRIGGER_EVENT_FIELDS = [
    "method",
    "sequence_name",
    "kind",
    "frame",
    "matched_event_frame",
    "delay_frames",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _group_by_sequence(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sequence_name") or ""), []).append(dict(row))
    for sequence_rows in grouped.values():
        sequence_rows.sort(key=lambda item: _int(item.get("window_start_frame")))
    return dict(sorted(grouped.items()))


def extract_harmful_drift_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    harmful_f1_drop_threshold: float,
    harmful_consecutive_windows: int,
    harmful_merge_gap_windows: int = 0,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: _int(item.get("window_start_frame")))
    if not ordered:
        return []
    consecutive = max(1, int(harmful_consecutive_windows))
    merge_gap = max(0, int(harmful_merge_gap_windows))
    events: list[dict[str, Any]] = []
    run_start: int | None = None
    run_length = 0
    in_episode = False
    clear_length = 0
    current_event: dict[str, Any] | None = None

    for index, row in enumerate(ordered):
        harmful = _float(row.get("f1_drop")) >= float(harmful_f1_drop_threshold)
        if harmful:
            clear_length = 0
            if in_episode:
                if current_event is not None:
                    current_event["end_frame"] = _int(row.get("window_end_frame"))
                continue
            if run_start is None:
                run_start = index
            run_length += 1
            if run_length >= consecutive:
                event_row = ordered[run_start]
                current_event = {
                    "frame": _int(event_row.get("window_start_frame")),
                    "end_frame": _int(row.get("window_end_frame")),
                    "domain": str(event_row.get("domain_majority")),
                    "transition_frame": _int(event_row.get("window_start_frame")),
                }
                events.append(current_event)
                in_episode = True
            continue

        run_start = None
        run_length = 0
        if in_episode:
            clear_length += 1
            if clear_length > merge_gap:
                in_episode = False
                clear_length = 0
                current_event = None
    return events


def replay_triggers(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    signal_column: str,
    threshold: float,
    trigger_consecutive_windows: int,
    cooldown_windows: int,
    rearm_requires_below_threshold: bool = True,
) -> list[int]:
    del method
    ordered = sorted(rows, key=lambda item: _int(item.get("window_start_frame")))
    consecutive_needed = max(1, int(trigger_consecutive_windows))
    cooldown_config = max(0, int(cooldown_windows))
    active_run = 0
    cooldown_remaining = 0
    armed = True
    triggers: list[int] = []
    for row in ordered:
        value = _float(row.get(signal_column), default=-math.inf)
        above_threshold = value >= float(threshold)
        if rearm_requires_below_threshold and not above_threshold:
            armed = True
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            active_run = 0
            continue
        active_run = active_run + 1 if above_threshold and armed else 0
        if active_run >= consecutive_needed and cooldown_remaining <= 0 and armed:
            triggers.append(_int(row.get("window_start_frame")))
            cooldown_remaining = cooldown_config
            active_run = 0
            if rearm_requires_below_threshold:
                armed = False
    return triggers


def match_triggers_to_events(
    events: Sequence[Mapping[str, Any]],
    triggers: Sequence[int],
    *,
    tolerance_frames: int,
    early_tolerance_frames: int = 0,
    total_frames: int,
) -> dict[str, Any]:
    sorted_events = sorted(
        (
            _int(event.get("frame")),
            _int(event.get("end_frame"), _int(event.get("frame"))),
        )
        for event in events
    )
    sorted_triggers = sorted(int(trigger) for trigger in triggers)
    matched_triggers: set[int] = set()
    matched_pairs: list[dict[str, int]] = []
    delays: list[int] = []
    detected = 0
    early_tolerance = max(0, int(early_tolerance_frames))
    late_tolerance = max(0, int(tolerance_frames))
    for event_frame, event_end_frame in sorted_events:
        chosen_index = None
        for index, trigger_frame in enumerate(sorted_triggers):
            if index in matched_triggers:
                continue
            if event_frame - early_tolerance <= trigger_frame <= event_end_frame + late_tolerance:
                chosen_index = index
                break
        if chosen_index is not None:
            matched_triggers.add(chosen_index)
            detected += 1
            delay = int(sorted_triggers[chosen_index] - event_frame)
            delays.append(delay)
            matched_pairs.append(
                {
                    "trigger_index": int(chosen_index),
                    "trigger_frame": int(sorted_triggers[chosen_index]),
                    "event_frame": int(event_frame),
                    "delay_frames": delay,
                }
            )
    false_triggers = len(sorted_triggers) - len(matched_triggers)
    missed = len(sorted_events) - detected
    precision = detected / float(max(detected + false_triggers, 1))
    recall = detected / float(max(len(sorted_events), 1))
    trigger_f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (
        precision + recall
    )
    return {
        "true_drift_events": int(len(sorted_events)),
        "detected": int(detected),
        "missed": int(missed),
        "false_triggers": int(false_triggers),
        "delays": delays,
        "avg_detection_delay_frames": float(np.mean(delays)) if delays else math.nan,
        "median_detection_delay_frames": float(np.median(delays)) if delays else math.nan,
        "precision": float(precision),
        "recall": float(recall),
        "trigger_f1": float(trigger_f1),
        "false_triggers_per_1000_frames": float(false_triggers)
        / max(float(total_frames), 1.0)
        * 1000.0,
        "early_tolerance_frames": int(early_tolerance),
        "late_tolerance_frames": int(late_tolerance),
        "matched_trigger_indices": sorted(matched_triggers),
        "matched_pairs": matched_pairs,
    }


def _method_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    trigger_cfg = require_mapping(config, "trigger")
    thresholds = require_mapping(trigger_cfg, "thresholds", context="trigger")
    return {
        "confidence_only": require_float(
            thresholds,
            "confidence_only_z",
            context="trigger.thresholds",
        ),
        "ema_entropy": require_float(
            thresholds,
            "ema_entropy_z",
            context="trigger.thresholds",
        ),
        "ema_feature_deviation": require_float(
            thresholds,
            "ema_feature_deviation_z",
            context="trigger.thresholds",
        ),
        "recap_full": require_float(
            thresholds,
            "full_score_z",
            context="trigger.thresholds",
        ),
    }


def _total_frames(rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 0
    starts = [_int(row.get("window_start_frame")) for row in rows]
    ends = [_int(row.get("window_end_frame")) for row in rows]
    return max(1, max(ends) - min(starts) + 1)


def _event_rows(
    *,
    method: str,
    sequence_name: str,
    events: Sequence[Mapping[str, Any]],
    triggers: Sequence[int],
    match_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matched_indices = set(match_result.get("matched_trigger_indices") or [])
    matched_pairs = {
        int(pair["trigger_index"]): dict(pair)
        for pair in list(match_result.get("matched_pairs") or [])
        if isinstance(pair, Mapping)
    }
    event_frames = sorted(_int(event.get("frame")) for event in events)
    rows: list[dict[str, Any]] = []
    for index, trigger in enumerate(sorted(int(item) for item in triggers)):
        matched_event = ""
        delay = ""
        kind = "false_trigger"
        if index in matched_indices:
            pair = matched_pairs.get(index, {})
            matched_event = pair.get("event_frame", "")
            delay = pair.get("delay_frames", "")
            kind = "detected"
        rows.append(
            {
                "method": method,
                "sequence_name": sequence_name,
                "kind": kind,
                "frame": trigger,
                "matched_event_frame": matched_event,
                "delay_frames": delay,
            }
        )
    detected_event_frames = {
        _int(row.get("matched_event_frame"))
        for row in rows
        if row.get("kind") == "detected" and row.get("matched_event_frame") != ""
    }
    for event_frame in event_frames:
        if event_frame not in detected_event_frames:
            rows.append(
                {
                    "method": method,
                    "sequence_name": sequence_name,
                    "kind": "missed",
                    "frame": event_frame,
                    "matched_event_frame": "",
                    "delay_frames": "",
                }
            )
    return rows


def analyze_online_triggers(config: Mapping[str, Any]) -> tuple[Path, Path]:
    root = output_dir(config)
    window_path = root / "records" / "window_metrics.csv"
    if not window_path.exists():
        raise FileNotFoundError(f"Missing window metrics: {window_path}")
    grouped = _group_by_sequence(_read_csv(window_path))
    window_cfg = require_mapping(config, "window")
    trigger_cfg = require_mapping(config, "trigger")
    thresholds = _method_thresholds(config)
    tolerance = require_int(trigger_cfg, "tolerance_frames", context="trigger")
    early_tolerance = require_int(trigger_cfg, "early_tolerance_frames", context="trigger")
    trigger_consecutive = require_int(
        trigger_cfg,
        "trigger_consecutive_windows",
        context="trigger",
    )
    cooldown = require_int(trigger_cfg, "cooldown_windows", context="trigger")
    rearm_requires_below = require_bool(
        trigger_cfg,
        "rearm_requires_below_threshold",
        context="trigger",
    )
    harmful_threshold = require_float(
        window_cfg,
        "harmful_f1_drop_threshold",
        context="window",
    )
    harmful_consecutive = require_int(
        window_cfg,
        "harmful_consecutive_windows",
        context="window",
    )
    harmful_merge_gap = require_int(
        window_cfg,
        "harmful_merge_gap_windows",
        context="window",
    )

    sequence_rows: list[dict[str, Any]] = []
    trigger_event_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {}
    for sequence_name, rows in grouped.items():
        events = extract_harmful_drift_events(
            rows,
            harmful_f1_drop_threshold=harmful_threshold,
            harmful_consecutive_windows=harmful_consecutive,
            harmful_merge_gap_windows=harmful_merge_gap,
        )
        total_frames = _total_frames(rows)
        for method, column in METHOD_COLUMNS.items():
            triggers = replay_triggers(
                rows,
                method=method,
                signal_column=column,
                threshold=thresholds[method],
                trigger_consecutive_windows=trigger_consecutive,
                cooldown_windows=cooldown,
                rearm_requires_below_threshold=rearm_requires_below,
            )
            matched = match_triggers_to_events(
                events,
                triggers,
                tolerance_frames=tolerance,
                early_tolerance_frames=early_tolerance,
                total_frames=total_frames,
            )
            sequence_row = {
                "method": method,
                "sequence_name": sequence_name,
                **{key: matched[key] for key in SEQUENCE_SUMMARY_FIELDS if key in matched},
            }
            sequence_rows.append(sequence_row)
            trigger_event_rows.extend(
                _event_rows(
                    method=method,
                    sequence_name=sequence_name,
                    events=events,
                    triggers=triggers,
                    match_result=matched,
                )
            )
            state = aggregate.setdefault(
                method,
                {
                    "method": method,
                    "num_sequences": 0,
                    "true_drift_events": 0,
                    "detected": 0,
                    "missed": 0,
                    "false_triggers": 0,
                    "total_frames": 0,
                    "delays": [],
                    "early_tolerance_frames": early_tolerance,
                    "late_tolerance_frames": tolerance,
                },
            )
            state["num_sequences"] += 1
            state["true_drift_events"] += matched["true_drift_events"]
            state["detected"] += matched["detected"]
            state["missed"] += matched["missed"]
            state["false_triggers"] += matched["false_triggers"]
            state["total_frames"] += total_frames
            state["delays"].extend(matched["delays"])

    method_rows: list[dict[str, Any]] = []
    for state in aggregate.values():
        detected = int(state["detected"])
        false_triggers = int(state["false_triggers"])
        true_events = int(state["true_drift_events"])
        precision = detected / float(max(detected + false_triggers, 1))
        recall = detected / float(max(true_events, 1))
        trigger_f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (
            precision + recall
        )
        delays = list(state["delays"])
        method_rows.append(
            {
                "method": state["method"],
                "num_sequences": int(state["num_sequences"]),
                "true_drift_events": true_events,
                "detected": detected,
                "missed": int(state["missed"]),
                "false_triggers": false_triggers,
                "avg_detection_delay_frames": float(np.mean(delays)) if delays else math.nan,
                "median_detection_delay_frames": float(np.median(delays))
                if delays
                else math.nan,
                "precision": float(precision),
                "recall": float(recall),
                "trigger_f1": float(trigger_f1),
                "false_triggers_per_1000_frames": false_triggers
                / max(float(state["total_frames"]), 1.0)
                * 1000.0,
                "early_tolerance_frames": int(state["early_tolerance_frames"]),
                "late_tolerance_frames": int(state["late_tolerance_frames"]),
            }
        )

    analysis_dir = root / "analysis"
    sequence_path = analysis_dir / "online_trigger_sequence_summary.csv"
    method_path = analysis_dir / "online_trigger_method_summary.csv"
    _write_csv(sequence_path, sequence_rows, SEQUENCE_SUMMARY_FIELDS)
    _write_csv(method_path, method_rows, METHOD_SUMMARY_FIELDS)
    _write_csv(analysis_dir / "online_trigger_events.csv", trigger_event_rows, TRIGGER_EVENT_FIELDS)
    return sequence_path, method_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze online drift trigger replay.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    analyze_online_triggers(load_config(args.config))


if __name__ == "__main__":
    main()
