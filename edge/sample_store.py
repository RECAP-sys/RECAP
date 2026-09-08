from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import cv2
import torch

from cloud.feature_cache import FeatureShardRef, FeatureShardStore
from edge.sample_quality import HIGH_QUALITY, LOW_QUALITY, QUALITY_METHOD
from model_management.payload import BoundaryPayload


def _atomic_json_dump(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    os.replace(tmp_path, path)


def _atomic_cv2_imwrite(path: str, image: Any, params: list[int]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}.jpg"
    try:
        ok = cv2.imwrite(tmp_path, image, params)
        if not ok:
            raise OSError(f"cv2.imwrite failed for {path}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _to_relpath(root_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.relpath(path, root_dir).replace("\\", "/")


def _from_relpath(root_dir: str, relpath: str | None) -> str | None:
    if relpath is None:
        return None
    return os.path.join(root_dir, relpath.replace("/", os.sep))


def _normalise_payload(
    intermediate: BoundaryPayload | torch.Tensor | dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if isinstance(intermediate, BoundaryPayload):
        source = dict(getattr(intermediate, "tensors", {}) or {})
    elif isinstance(intermediate, torch.Tensor):
        source = {"payload": intermediate}
    elif isinstance(intermediate, dict):
        source = dict(intermediate.get("tensors") or intermediate)
    else:
        raise TypeError(f"Unsupported split feature type: {type(intermediate)!r}")
    tensors = {
        str(key): value.detach().cpu()
        for key, value in source.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValueError("Cached split feature did not contain tensor values.")
    return tensors


def _feature_ref_paths(ref: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(ref, Mapping):
        return []
    paths: list[str] = []
    for key in ("shard_path", "index_path", "meta_path"):
        value = ref.get(key)
        if value:
            paths.append(str(value))
    shard_dir = ref.get("shard_dir")
    if shard_dir and os.path.isdir(str(shard_dir)):
        for root, _dirs, files in os.walk(str(shard_dir)):
            for filename in files:
                paths.append(os.path.join(root, filename))
    return sorted(set(paths))


def _feature_ref_bytes(ref: Mapping[str, Any] | None) -> int:
    total = 0
    for path in _feature_ref_paths(ref):
        if os.path.exists(path):
            total += os.path.getsize(path)
    return int(total)


def _normalise_quality_metadata(
    quality: object,
    *,
    quality_bucket: str | None = None,
) -> dict[str, Any]:
    payload = dict(quality) if isinstance(quality, Mapping) else {}
    bucket = str(payload.get("quality") or quality_bucket or LOW_QUALITY)
    if bucket not in {HIGH_QUALITY, LOW_QUALITY}:
        bucket = LOW_QUALITY
    result: dict[str, Any] = {
        "method": str(payload.get("method") or QUALITY_METHOD),
        "quality": bucket,
    }
    debug = payload.get("debug")
    if isinstance(debug, Mapping):
        result["debug"] = dict(debug)
    return result


@dataclass
class StoredSampleRecord:
    sample_id: str
    frame_index: int | None
    timestamp: str
    confidence: float
    split_config_id: str
    model_id: str
    model_version: str
    front_version: str
    quality_bucket: str
    quality: dict[str, Any] = field(default_factory=dict)
    window_id: str | None = None
    in_drift_window: bool = False
    has_raw_sample: bool = False
    has_feature: bool = True
    input_image_size: list[int] | None = None
    input_tensor_shape: list[int] | None = None
    input_resize_mode: str | None = None
    feature_ref: dict[str, Any] | None = None
    result_relpath: str = ""
    metadata_relpath: str = ""
    raw_relpath: str | None = None
    feature_bytes: int = 0
    raw_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        quality_payload = _normalise_quality_metadata(
            self.quality,
            quality_bucket=self.quality_bucket,
        )
        return {
            "sample_id": self.sample_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "split_config_id": self.split_config_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "front_version": self.front_version,
            "quality": quality_payload,
            "quality_bucket": str(quality_payload["quality"]),
            "window_id": self.window_id,
            "in_drift_window": self.in_drift_window,
            "has_raw_sample": self.has_raw_sample,
            "has_feature": self.has_feature,
            "input_image_size": self.input_image_size,
            "input_tensor_shape": self.input_tensor_shape,
            "input_resize_mode": self.input_resize_mode,
            "feature_ref": dict(self.feature_ref)
            if isinstance(self.feature_ref, Mapping)
            else None,
            "result_relpath": self.result_relpath,
            "metadata_relpath": self.metadata_relpath,
            "raw_relpath": self.raw_relpath,
            "feature_bytes": self.feature_bytes,
            "raw_bytes": self.raw_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StoredSampleRecord":
        quality_payload = _normalise_quality_metadata(
            payload.get("quality"),
            quality_bucket=str(payload.get("quality_bucket", LOW_QUALITY)),
        )
        return cls(
            sample_id=str(payload["sample_id"]),
            frame_index=payload.get("frame_index"),
            timestamp=str(payload["timestamp"]),
            confidence=float(payload.get("confidence", 0.0)),
            split_config_id=str(payload.get("split_config_id", "")),
            model_id=str(payload.get("model_id", "")),
            model_version=str(payload.get("model_version", "")),
            front_version=str(payload.get("front_version", "0") or "0"),
            quality_bucket=str(quality_payload["quality"]),
            quality=dict(quality_payload),
            window_id=(None if payload.get("window_id") is None else str(payload.get("window_id"))),
            in_drift_window=bool(payload.get("in_drift_window", False)),
            has_raw_sample=bool(payload.get("has_raw_sample", False)),
            has_feature=isinstance(payload.get("feature_ref"), Mapping),
            input_image_size=list(payload["input_image_size"])
            if payload.get("input_image_size") is not None
            else None,
            input_tensor_shape=list(payload["input_tensor_shape"])
            if payload.get("input_tensor_shape") is not None
            else None,
            input_resize_mode=(
                str(payload["input_resize_mode"])
                if payload.get("input_resize_mode") is not None
                else None
            ),
            feature_ref=(
                dict(payload["feature_ref"])
                if isinstance(payload.get("feature_ref"), Mapping)
                else None
            ),
            result_relpath=str(payload["result_relpath"]),
            metadata_relpath=str(payload["metadata_relpath"]),
            raw_relpath=payload.get("raw_relpath"),
            feature_bytes=int(payload.get("feature_bytes", 0)),
            raw_bytes=int(payload.get("raw_bytes", 0)),
        )


@dataclass
class _SampleStoreCounters:
    total_samples: int = 0
    high_quality_count: int = 0
    low_quality_count: int = 0
    drift_window_sample_count: int = 0
    high_quality_feature_bytes: int = 0
    low_quality_feature_bytes: int = 0
    low_quality_raw_bytes: int = 0

    def add(self, record: StoredSampleRecord, *, sign: int = 1) -> None:
        factor = 1 if sign >= 0 else -1
        self.total_samples += factor
        if record.quality_bucket == HIGH_QUALITY:
            self.high_quality_count += factor
            self.high_quality_feature_bytes += factor * int(record.feature_bytes)
        elif record.quality_bucket == LOW_QUALITY:
            self.low_quality_count += factor
            self.low_quality_feature_bytes += factor * int(record.feature_bytes)
            self.low_quality_raw_bytes += factor * int(record.raw_bytes)
        if record.in_drift_window:
            self.drift_window_sample_count += factor

    def clamp(self) -> None:
        self.total_samples = max(0, int(self.total_samples))
        self.high_quality_count = max(0, int(self.high_quality_count))
        self.low_quality_count = max(0, int(self.low_quality_count))
        self.drift_window_sample_count = max(0, int(self.drift_window_sample_count))
        self.high_quality_feature_bytes = max(0, int(self.high_quality_feature_bytes))
        self.low_quality_feature_bytes = max(0, int(self.low_quality_feature_bytes))
        self.low_quality_raw_bytes = max(0, int(self.low_quality_raw_bytes))

    def to_stats(self) -> dict[str, Any]:
        total = max(0, int(self.total_samples))
        low = max(0, int(self.low_quality_count))
        return {
            "total_samples": total,
            "high_quality_count": max(0, int(self.high_quality_count)),
            "low_quality_count": low,
            "low_quality_rate": (low / float(total)) if total else 0.0,
            "drift_window_sample_count": max(0, int(self.drift_window_sample_count)),
            "high_quality_feature_bytes": max(0, int(self.high_quality_feature_bytes)),
            "low_quality_feature_bytes": max(0, int(self.low_quality_feature_bytes)),
            "low_quality_raw_bytes": max(0, int(self.low_quality_raw_bytes)),
        }


class EdgeSampleStore:
    def __init__(
        self,
        root_dir: str,
        *,
        feature_storage_format: str = "npy_memmap_shard",
        feature_shard_dtype: str | None = None,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.feature_shard_dir = os.path.join(self.root_dir, "feature_shards")
        self.feature_store = FeatureShardStore(
            self.feature_shard_dir,
            storage_format=str(feature_storage_format or "npy_memmap_shard"),
            shard_max_samples=1,
            shard_dtype=feature_shard_dtype,
        )
        self.results_dir = os.path.join(self.root_dir, "results")
        self.metadata_dir = os.path.join(self.root_dir, "metadata")
        self.raw_dir = os.path.join(self.root_dir, "raw")
        self.index_dir = os.path.join(self.root_dir, "indexes")
        self.manifest_path = os.path.join(self.root_dir, "manifest.json")
        self._lock = threading.RLock()
        self._counters = _SampleStoreCounters()
        self._records: dict[str, StoredSampleRecord] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._ensure_layout()
        self._recover_counters()

    def _ensure_layout(self) -> None:
        os.makedirs(self.feature_shard_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        if not os.path.exists(self.manifest_path):
            _atomic_json_dump(
                self.manifest_path,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    def _recover_counters(self) -> None:
        counters = _SampleStoreCounters()
        records: dict[str, StoredSampleRecord] = {}
        if os.path.isdir(self.metadata_dir):
            for filename in sorted(os.listdir(self.metadata_dir)):
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(self.metadata_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        record = StoredSampleRecord.from_dict(json.load(handle))
                except Exception:
                    continue
                records[record.sample_id] = record
                counters.add(record)
        counters.clamp()
        with self._lock:
            self._counters = counters
            self._records = records

    def clear(self) -> None:
        with self._lock:
            if os.path.isdir(self.root_dir):
                shutil.rmtree(self.root_dir, ignore_errors=True)
            self._ensure_layout()
            self._counters = _SampleStoreCounters()
            self._records = {}
            self._results = {}

    def delete_samples(
        self,
        sample_ids: Iterable[str],
        *,
        quality_bucket: str | None = None,
    ) -> int:
        requested_ids = {str(sample_id) for sample_id in sample_ids if str(sample_id)}
        if not requested_ids:
            return 0
        with self._lock:
            removed: list[StoredSampleRecord] = []
            for sample_id in sorted(requested_ids):
                record = self._records.get(sample_id)
                if record is None:
                    continue
                if quality_bucket is not None and record.quality_bucket != quality_bucket:
                    continue
                removed.append(record)
                self._records.pop(sample_id, None)
                self._results.pop(sample_id, None)
                self._counters.add(record, sign=-1)
            if not removed:
                return 0
            self._counters.clamp()

            protected_paths: set[str] = set()
            for record in self._records.values():
                protected_paths.update(self.iter_existing_paths(record))
            for record in removed:
                for path in self.iter_existing_paths(record):
                    if path in protected_paths:
                        continue
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
            self._rewrite_indexes_unlocked()
            return len(removed)

    def _index_path(self, bucket: str) -> str:
        return os.path.join(self.index_dir, f"{bucket}.jsonl")

    def _append_index(self, bucket: str, record: StoredSampleRecord) -> None:
        with open(self._index_path(bucket), "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

    def _rewrite_indexes_unlocked(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        for filename in os.listdir(self.index_dir):
            if filename.endswith(".jsonl"):
                try:
                    os.remove(os.path.join(self.index_dir, filename))
                except OSError:
                    pass
        records = sorted(
            self._records.values(),
            key=lambda item: (item.timestamp, item.sample_id),
        )
        for record in records:
            self._append_index("all", record)
            self._append_index(record.quality_bucket, record)

    def _existing_record_unlocked(self, sample_id: str) -> StoredSampleRecord | None:
        return self._records.get(str(sample_id))

    def store_sample(
        self,
        *,
        sample_id: str,
        frame_index: int | None,
        confidence: float,
        split_config_id: str,
        model_id: str,
        model_version: str,
        front_version: str = "0",
        quality_bucket: str | None = None,
        quality: Mapping[str, Any] | None = None,
        quality_metadata: Mapping[str, Any] | None = None,
        window_id: str | None = None,
        in_drift_window: bool = False,
        inference_result: dict[str, Any],
        intermediate: BoundaryPayload | torch.Tensor | dict[str, torch.Tensor],
        raw_frame: Any | None = None,
        raw_jpeg_quality: int = 82,
        timestamp: str | None = None,
        input_image_size: list[int] | tuple[int, int] | None = None,
        input_tensor_shape: list[int] | tuple[int, ...] | None = None,
        input_resize_mode: str | None = None,
        runtime_contract: Mapping[str, Any] | None = None,
        model_family: str | None = None,
    ) -> StoredSampleRecord:
        sample_key = str(sample_id)
        with self._lock:
            previous_record = self._existing_record_unlocked(sample_key)

        quality_payload = _normalise_quality_metadata(
            quality_metadata if quality_metadata is not None else quality,
            quality_bucket=quality_bucket,
        )
        quality_bucket = str(quality_payload["quality"])
        if quality_bucket not in {HIGH_QUALITY, LOW_QUALITY}:
            raise ValueError(f"Unsupported quality bucket: {quality_bucket!r}")
        if quality_bucket != LOW_QUALITY:
            raw_frame = None

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        result_path = os.path.join(self.results_dir, f"{sample_key}.json")
        metadata_path = os.path.join(self.metadata_dir, f"{sample_key}.json")
        raw_path = (
            os.path.join(self.raw_dir, f"{sample_key}.jpg") if raw_frame is not None else None
        )

        runtime_contract_payload = dict(runtime_contract or {})
        written_features = self.feature_store.write_entries(
            [
                {
                    "sample": {"sample_id": sample_key},
                    "record": {
                        "intermediate": (
                            intermediate
                            if isinstance(intermediate, BoundaryPayload)
                            else _normalise_payload(intermediate)
                        )
                    },
                }
            ],
            runtime_context={
                "model_id": str(model_id),
                "model_family": str(model_family or ""),
                "split_config_id": str(split_config_id),
                "contract_id": (
                    None
                    if runtime_contract_payload.get("contract_id") in (None, "")
                    else str(runtime_contract_payload.get("contract_id"))
                ),
                "feature_layout_id": str(runtime_contract_payload.get("feature_layout_id") or ""),
                "boundary_id": str(
                    runtime_contract_payload.get("logical_split_id")
                    or getattr(intermediate, "split_id", "")
                    or ""
                ),
                "input_tensor_shape": list(input_tensor_shape or []),
                "input_resize_mode": str(input_resize_mode or ""),
                "runtime_contract": runtime_contract_payload,
            },
            generation="edge_sample_store",
            source="edge_sample_store",
        )
        feature_ref = written_features[0]["feature_ref"]
        if not isinstance(feature_ref, FeatureShardRef):
            raise RuntimeError("Edge sample feature shard write did not return FeatureShardRef.")
        feature_ref_payload = feature_ref.to_dict()
        _atomic_json_dump(result_path, inference_result)
        if raw_frame is not None:
            quality = max(1, min(100, int(raw_jpeg_quality)))
            _atomic_cv2_imwrite(raw_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        record = StoredSampleRecord(
            sample_id=sample_key,
            frame_index=frame_index,
            timestamp=ts,
            confidence=float(confidence),
            split_config_id=str(split_config_id),
            model_id=str(model_id),
            model_version=str(model_version),
            front_version=str(front_version or "0"),
            quality_bucket=quality_bucket,
            quality=dict(quality_payload),
            window_id=window_id,
            in_drift_window=bool(in_drift_window),
            has_raw_sample=raw_path is not None,
            has_feature=True,
            input_image_size=list(input_image_size) if input_image_size is not None else None,
            input_tensor_shape=list(input_tensor_shape) if input_tensor_shape is not None else None,
            input_resize_mode=str(input_resize_mode) if input_resize_mode is not None else None,
            feature_ref=feature_ref_payload,
            result_relpath=_to_relpath(self.root_dir, result_path),
            metadata_relpath=_to_relpath(self.root_dir, metadata_path),
            raw_relpath=_to_relpath(self.root_dir, raw_path),
            feature_bytes=_feature_ref_bytes(feature_ref_payload),
            raw_bytes=os.path.getsize(raw_path) if raw_path is not None else 0,
        )

        _atomic_json_dump(metadata_path, record.to_dict())

        self._append_index("all", record)
        self._append_index(quality_bucket, record)
        with self._lock:
            if previous_record is not None:
                self._counters.add(previous_record, sign=-1)
            self._counters.add(record)
            self._counters.clamp()
            self._records[sample_key] = record
            self._results[sample_key] = dict(inference_result)
        return record

    def load_record(self, sample_id: str) -> StoredSampleRecord:
        with self._lock:
            cached = self._records.get(str(sample_id))
            if cached is not None:
                return cached
            metadata_path = os.path.join(self.metadata_dir, f"{sample_id}.json")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                return StoredSampleRecord.from_dict(json.load(handle))

    def list_records(
        self,
        *,
        quality_bucket: str | None = None,
    ) -> list[StoredSampleRecord]:
        with self._lock:
            records_by_id: dict[str, StoredSampleRecord] = dict(self._records)

            if not records_by_id:
                index_path = self._index_path("all")
                if os.path.exists(index_path):
                    with open(index_path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = StoredSampleRecord.from_dict(json.loads(line))
                            except Exception:
                                records_by_id.clear()
                                break
                            records_by_id[record.sample_id] = record

            if not records_by_id:
                if not os.path.isdir(self.metadata_dir):
                    return []
                for filename in sorted(os.listdir(self.metadata_dir)):
                    if not filename.endswith(".json"):
                        continue
                    with open(
                        os.path.join(self.metadata_dir, filename), "r", encoding="utf-8"
                    ) as handle:
                        record = StoredSampleRecord.from_dict(json.load(handle))
                    records_by_id[record.sample_id] = record

            records = [
                record
                for record in records_by_id.values()
                if quality_bucket is None or record.quality_bucket == quality_bucket
            ]
            records.sort(key=lambda item: (item.timestamp, item.sample_id))
            return records

    def low_quality_count(self) -> int:
        with self._lock:
            return int(self._counters.low_quality_count)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return self._counters.to_stats()

    def load_inference_result(self, record: StoredSampleRecord) -> dict[str, Any]:
        with self._lock:
            cached = self._results.get(str(record.sample_id))
            if cached is not None:
                return dict(cached)
            result_path = _from_relpath(self.root_dir, record.result_relpath)
            with open(result_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

    def load_intermediate(self, record: StoredSampleRecord) -> BoundaryPayload:
        with self._lock:
            if not isinstance(record.feature_ref, Mapping):
                raise FileNotFoundError(
                    f"Edge sample {record.sample_id!r} does not have a shard feature_ref."
                )
            ref = FeatureShardRef.from_dict(dict(record.feature_ref))
            return self.feature_store.read_batch([ref])

    def iter_existing_paths(self, record: StoredSampleRecord) -> Iterable[str]:
        for path in _feature_ref_paths(record.feature_ref):
            if os.path.exists(path):
                yield path
        for relpath in (record.result_relpath, record.metadata_relpath, record.raw_relpath):
            path = _from_relpath(self.root_dir, relpath)
            if path is not None and os.path.exists(path):
                yield path
