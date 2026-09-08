from __future__ import annotations

import json
import subprocess
import sys


def test_feature_shard_benchmark_outputs_files(tmp_path) -> None:
    output_dir = tmp_path / "bench"
    subprocess.run(
        [
            sys.executable,
            "tools/benchmark_feature_shard_formats.py",
            "--sample-count",
            "4",
            "--batch-size",
            "2",
            "--formats",
            "npy_memmap_shard",
            "--repeat",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    raw_path = output_dir / "feature_shard_benchmark_raw.json"
    csv_path = output_dir / "feature_shard_benchmark_summary.csv"
    md_path = output_dir / "feature_shard_benchmark_summary.md"
    assert raw_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["sample_order"] == [f"sample_{index:06d}" for index in range(4)]
    text = csv_path.read_text(encoding="utf-8")
    assert "storage_bytes" in text
    assert "epoch_train_time_sec" in text
    assert "npy_memmap_shard" in md_path.read_text(encoding="utf-8")
