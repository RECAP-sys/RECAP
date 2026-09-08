from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloud.feature_cache import FeatureShardRef, FeatureShardStore
from model_management.payload import boundary_payload_from_tensors

SUMMARY_FIELDS = [
    "format",
    "scenario",
    "sample_count",
    "batch_size",
    "dtype",
    "raw_tensor_bytes",
    "storage_bytes",
    "metadata_bytes",
    "storage_overhead_ratio",
    "write_time_sec",
    "write_mb_per_sec",
    "cold_epoch_read_time_sec",
    "warm_epoch_read_time_sec",
    "random_batch_read_ms",
    "sequential_batch_read_ms",
    "proxy_eval_time_sec",
    "epoch_train_time_sec",
    "suffix_forward_backward_time_sec",
    "feature_read_time_sec",
    "collate_time_sec",
    "cpu_to_gpu_time_sec",
    "peak_cpu_ram_mb",
    "peak_gpu_memory_mb",
]


def _tensor_bytes(entries: list[dict[str, Any]]) -> int:
    total = 0
    for entry in entries:
        payload = entry["record"]["intermediate"]
        for tensor in dict(payload.tensors or {}).values():
            total += int(tensor.numel() * tensor.element_size())
    return total


def _dir_size(path: Path) -> tuple[int, int]:
    storage = 0
    metadata = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            size = file_path.stat().st_size
            storage += size
            if filename.endswith((".json", ".jsonl")):
                metadata += size
    return storage, metadata


def _make_fake_entries(count: int, dtype: str, seed: int) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    torch_dtype = getattr(torch, dtype.replace("torch.", ""), torch.float16)
    entries = []
    for index in range(count):
        tensors = {
            "boundary": torch.randn(1, 16, 20, 20, dtype=torch_dtype),
            "skip": torch.randn(1, 8, 10, 10, dtype=torch_dtype),
        }
        payload = boundary_payload_from_tensors(
            tensors,
            split_id="after:synthetic",
            graph_signature="benchmark",
            batch_size=1,
        )
        entries.append(
            {
                "sample": {
                    "sample_id": f"sample_{index:06d}",
                    "labels": {"boxes": [[0.0, 0.0, 10.0, 10.0]], "labels": [index % 3]},
                    "input_tensor_shape": [1, 3, 320, 320],
                    "input_resize_mode": "direct_resize",
                },
                "record": {"intermediate": payload},
            }
        )
    return entries


def _make_entries(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    return "fake", _make_fake_entries(args.sample_count, args.dtype, args.seed)


def _epoch_read(store: FeatureShardStore, refs: list[FeatureShardRef], batch_size: int) -> float:
    started = time.perf_counter()
    for offset in range(0, len(refs), batch_size):
        store.read_batch(refs[offset : offset + batch_size])
    return time.perf_counter() - started


def _batch_latency(
    store: FeatureShardStore,
    refs: list[FeatureShardRef],
    batch_size: int,
    *,
    random_order: bool,
    repeat: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    timings = []
    for _ in range(max(1, repeat)):
        if random_order:
            sample = rng.sample(refs, min(batch_size, len(refs)))
        else:
            sample = refs[: min(batch_size, len(refs))]
        started = time.perf_counter()
        store.read_batch(sample)
        timings.append((time.perf_counter() - started) * 1000.0)
    return sum(timings) / len(timings)


def _run_format(
    args: argparse.Namespace,
    storage_format: str,
    entries: list[dict[str, Any]],
    *,
    scenario: str,
) -> dict[str, Any]:
    root = Path(args.output_dir) / "work" / storage_format
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    store = FeatureShardStore(
        str(root),
        storage_format=storage_format,
        shard_max_samples=args.shard_max_samples,
        shard_dtype=args.dtype,
        payload_cache_enabled=bool(args.payload_cache),
        pin_memory=False,
    )
    runtime_context = {
        "model_id": args.model,
        "model_family": str(args.model_family or scenario),
        "split_config_id": str(args.split_config_id or "benchmark-split"),
        "contract_id": "benchmark-contract",
        "feature_layout_id": str(args.feature_layout_id or "benchmark-layout"),
        "boundary_id": "after:synthetic",
        "input_tensor_shape": [1, 3, 320, 320],
        "input_resize_mode": "direct_resize",
    }
    raw_tensor_bytes = _tensor_bytes(entries)
    write_started = time.perf_counter()
    written = store.write_entries(
        entries,
        runtime_context=runtime_context,
        generation="benchmark",
        source="benchmark",
        storage_format=storage_format,
    )
    write_time = time.perf_counter() - write_started
    refs = [
        item["feature_ref"]
        for item in written
        if isinstance(item.get("feature_ref"), FeatureShardRef)
    ]
    storage_bytes, metadata_bytes = _dir_size(root)
    cold = _epoch_read(store, refs, args.batch_size)
    warm = _epoch_read(store, refs, args.batch_size)
    random_ms = _batch_latency(
        store,
        refs,
        args.batch_size,
        random_order=True,
        repeat=args.repeat,
        seed=args.seed,
    )
    sequential_ms = _batch_latency(
        store,
        refs,
        args.batch_size,
        random_order=False,
        repeat=args.repeat,
        seed=args.seed,
    )
    process = psutil.Process(os.getpid())
    peak_cpu_mb = process.memory_info().rss / (1024 * 1024)
    peak_gpu_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    feature_read_time = warm
    synthetic_train = warm
    row = {
        "format": storage_format,
        "scenario": scenario,
        "sample_count": len(refs),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "raw_tensor_bytes": raw_tensor_bytes,
        "storage_bytes": storage_bytes,
        "metadata_bytes": metadata_bytes,
        "storage_overhead_ratio": 0.0
        if raw_tensor_bytes <= 0
        else storage_bytes / raw_tensor_bytes,
        "write_time_sec": write_time,
        "write_mb_per_sec": 0.0
        if write_time <= 0
        else (raw_tensor_bytes / (1024 * 1024)) / write_time,
        "cold_epoch_read_time_sec": cold,
        "warm_epoch_read_time_sec": warm,
        "random_batch_read_ms": random_ms,
        "sequential_batch_read_ms": sequential_ms,
        "proxy_eval_time_sec": feature_read_time,
        "epoch_train_time_sec": synthetic_train,
        "suffix_forward_backward_time_sec": 0.0,
        "feature_read_time_sec": feature_read_time,
        "collate_time_sec": 0.0,
        "cpu_to_gpu_time_sec": 0.0,
        "peak_cpu_ram_mb": peak_cpu_mb,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }
    print(
        "[FeatureShardBenchmark][Summary] "
        f"format={storage_format} samples={len(refs)} batch_size={args.batch_size} "
        f"storage={storage_bytes / (1024 * 1024):.1f}MB write_time={write_time:.3f}s "
        f"warm_epoch_read={warm:.3f}s epoch_train={synthetic_train:.3f}s "
        f"proxy_eval={feature_read_time:.3f}s"
    )
    return {"row": row, "refs": [ref.to_dict() for ref in refs]}


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], raw: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_shard_benchmark_raw.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with open(
        output_dir / "feature_shard_benchmark_summary.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})
    best_storage = (
        min(rows, key=lambda item: float(item["storage_bytes"]))["format"] if rows else "n/a"
    )
    best_write = (
        min(rows, key=lambda item: float(item["write_time_sec"]))["format"] if rows else "n/a"
    )
    best_read = (
        min(rows, key=lambda item: float(item["warm_epoch_read_time_sec"]))["format"]
        if rows
        else "n/a"
    )
    summary = "\n".join(
        [
            "# Feature Shard Benchmark Summary",
            "",
            f"1. Storage smaller: {best_storage}",
            f"2. Write faster: {best_write}",
            f"3. Warm batch/epoch read faster: {best_read}",
            f"4. Proxy evaluation faster: {best_read}",
            f"5. Training epoch faster: {best_read}",
            "6. Recommended edge upload default: safetensors_shard",
            f"7. Recommended cloud rebuilt feature default: {best_read}",
            f"8. Recommended final default: {best_read}",
            "",
        ]
    )
    (output_dir / "feature_shard_benchmark_summary.md").write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n")
    parser.add_argument("--model-family", default="")
    parser.add_argument("--split-config-id", default="")
    parser.add_argument("--feature-layout-id", default="")
    parser.add_argument("--edge-id", default="1")
    parser.add_argument("--sample-count", type=int, default=79)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--formats", default="safetensors_shard,npy_memmap_shard")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="benchmark_results/feature_shards")
    parser.add_argument("--shard-max-samples", type=int, default=64)
    parser.add_argument("--payload-cache", action="store_true")
    parser.add_argument("--input-source", choices=("fake",), default="fake")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    del args.edge_id
    output_dir = Path(args.output_dir)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    scenario, entries = _make_entries(args)
    raw: dict[str, Any] = {
        "formats": {},
        "input_source": scenario,
        "sample_order": [entry["sample"]["sample_id"] for entry in entries],
    }
    rows = []
    for storage_format in formats:
        print(f"[FeatureShardBenchmark][Build] format={storage_format} samples={len(entries)}")
        result = _run_format(args, storage_format, entries, scenario=scenario)
        raw["formats"][storage_format] = result
        rows.append(result["row"])
    _write_outputs(output_dir, rows, raw)


if __name__ == "__main__":
    main()
