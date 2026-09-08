from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from cloud.feature_cache.shard_writer import FeatureShardWriter
from cloud.feature_cache.types import FeatureShardRef


def write_feature_label_shards(
    *,
    output_root: str | None,
    storage_format: str,
    shard_max_samples: int,
    shard_dtype: str | None,
    runtime_context: Mapping[str, Any],
    generation: str,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Write edge upload feature shards and per-shard JSONL labels.

    Returns ``(root_dir, shard_manifest_entries, labels_by_shard)``.
    """
    if output_root is not None:
        os.makedirs(output_root, exist_ok=True)
    root_dir = tempfile.mkdtemp(prefix="edge_feature_shard_", dir=output_root)
    writer = FeatureShardWriter(
        root_dir=root_dir,
        storage_format=storage_format,
        shard_max_samples=shard_max_samples,
        shard_dtype=shard_dtype,
    )
    written = writer.write_entries(
        entries,
        runtime_context=runtime_context,
        generation=generation,
        source="edge_uploaded",
    )
    refs_by_shard: dict[str, list[FeatureShardRef]] = {}
    labels_by_shard: dict[str, list[dict[str, Any]]] = {}
    samples_by_id = {
        str(dict(entry.get("sample") or {}).get("sample_id") or ""): dict(entry.get("sample") or {})
        for entry in entries
    }
    for entry in written:
        ref = entry.get("feature_ref")
        if not isinstance(ref, FeatureShardRef):
            continue
        refs_by_shard.setdefault(ref.shard_id, []).append(ref)
        sample = samples_by_id.get(ref.sample_id, {})
        labels = dict(sample.get("labels") or {})
        labels_by_shard.setdefault(ref.shard_id, []).append(
            {
                "sample_id": ref.sample_id,
                "boxes": list(labels.get("boxes") or []),
                "labels": list(labels.get("labels") or []),
                **(
                    {"scores": list(labels.get("scores") or [])}
                    if labels.get("scores") is not None
                    else {}
                ),
                **{
                    key: value
                    for key, value in labels.items()
                    if str(key).startswith("label_") or str(key).startswith("input_")
                },
            }
        )
    manifest_entries: list[dict[str, Any]] = []
    for shard_id, refs in sorted(refs_by_shard.items()):
        first = refs[0]
        label_rel = f"label_shards/{shard_id}.jsonl"
        label_path = os.path.join(root_dir, label_rel)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as handle:
            for label in labels_by_shard.get(shard_id, []):
                handle.write(json.dumps(label, sort_keys=True, separators=(",", ":")) + "\n")
        entry: dict[str, Any] = {
            "shard_id": shard_id,
            "storage_format": first.storage_format,
            "label_file": label_rel,
            "sample_count": len(refs),
            "sample_ids": [ref.sample_id for ref in refs],
        }
        if first.shard_path:
            entry["shard_file"] = os.path.relpath(first.shard_path, root_dir).replace("\\", "/")
            entry["index_file"] = os.path.relpath(first.index_path, root_dir).replace("\\", "/")
            entry["meta_file"] = entry["index_file"].replace(".index.json", ".meta.json")
        if first.shard_dir:
            entry["shard_dir"] = os.path.relpath(first.shard_dir, root_dir).replace("\\", "/")
            entry["index_file_name"] = os.path.basename(first.index_path)
            entry["meta_file_name"] = os.path.basename(first.index_path).replace(
                ".index.json", ".meta.json"
            )
        manifest_entries.append(entry)
    return root_dir, manifest_entries, labels_by_shard
