#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.drift_detection_validity.experiment_io import (  # noqa: E402
    load_config,
    output_dir,
    require_mapping,
)
from experiments.drift_detection_validity.evaluate_real_weather_scenes import (  # noqa: E402
    write_real_weather_report,
)

def _load_run_dir(config_path: Path) -> Path:
    return output_dir(load_config(config_path))


def _run(script_name: str, config_path: Path) -> None:
    script = PROJECT_ROOT / "experiments" / "drift_detection_validity" / script_name
    subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run drift detection validity pipeline.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    data_cfg = require_mapping(config, "data")
    scene_videos = data_cfg.get("scene_videos")
    if not isinstance(scene_videos, list) or len(scene_videos) != 3:
        raise ValueError("data.scene_videos must contain the three real Suwon weather scenes.")

    print("[DriftValidity] evaluate real weather scenes ...", flush=True)
    _run("evaluate_real_weather_scenes.py", config_path)
    print("[DriftValidity] plot signal validity summary ...", flush=True)
    _run("plot_signal_validity.py", config_path)
    print("[DriftValidity] plot online trigger summary ...", flush=True)
    _run("plot_online_trigger.py", config_path)
    print("[DriftValidity] refresh HTML report ...", flush=True)
    write_real_weather_report(config)
    print(f"[DriftValidity] done: {_load_run_dir(config_path)}", flush=True)


if __name__ == "__main__":
    main()
