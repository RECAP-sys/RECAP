from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from cloud.feature_cache.path_utils import fs_path
from cloud.feature_cache.shard_reader import FeatureShardPayloadCache, ShardFeatureBatchReader
from cloud.feature_cache.shard_writer import FeatureShardWriter
from cloud.feature_cache.types import (
    NPY_MEMMAP_SHARD,
    SAFETENSORS_SHARD,
    SUPPORTED_STORAGE_FORMATS,
    FeatureShardRef,
)
from common.logging_sanitizer import log_diagnostic_debug


def _read_json(path: str) -> dict[str, Any]:
    with open(fs_path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    import threading

    os.makedirs(fs_path(os.path.dirname(path)), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}-{int(time.time() * 1000000)}"
    with open(fs_path(tmp_path), "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(fs_path(tmp_path), fs_path(path))


class FeatureShardStore:
    def __init__(
        self,
        root_dir: str,
        *,
        storage_format: str = SAFETENSORS_SHARD,
        accepted_storage_formats: Sequence[str] | None = None,
        shard_max_samples: int = 64,
        shard_dtype: str | None = None,
        payload_cache_enabled: bool = True,
        payload_cache_max_cpu_bytes: int = 4 * 1024 * 1024 * 1024,
        pin_memory: bool = True,
        non_blocking_transfer: bool = True,
        log_internal_ids: bool = False,
    ) -> None:
        self.root_dir = os.path.abspath(str(root_dir))
        self.storage_format = str(storage_format or SAFETENSORS_SHARD)
        self.accepted_storage_formats = {
            str(item)
            for item in list(accepted_storage_formats or [SAFETENSORS_SHARD, NPY_MEMMAP_SHARD])
        }
        unknown = self.accepted_storage_formats - SUPPORTED_STORAGE_FORMATS
        if self.storage_format not in SUPPORTED_STORAGE_FORMATS or unknown:
            raise ValueError(
                "Unsupported feature shard storage format "
                f"storage_format={self.storage_format!r} accepted_unknown={sorted(unknown)}."
            )
        self.shard_max_samples = max(1, int(shard_max_samples or 64))
        self.shard_dtype = None if shard_dtype in (None, "") else str(shard_dtype)
        self.log_internal_ids = bool(log_internal_ids)
        self.payload_cache = FeatureShardPayloadCache(
            enabled=payload_cache_enabled,
            max_cpu_bytes=payload_cache_max_cpu_bytes,
        )
        self.reader = ShardFeatureBatchReader(
            payload_cache=self.payload_cache,
            pin_memory=pin_memory,
            non_blocking_transfer=non_blocking_transfer,
        )
        os.makedirs(fs_path(self.root_dir), exist_ok=True)

    def writer(self, *, storage_format: str | None = None) -> FeatureShardWriter:
        return FeatureShardWriter(
            root_dir=self.root_dir,
            storage_format=storage_format or self.storage_format,
            shard_max_samples=self.shard_max_samples,
            shard_dtype=self.shard_dtype,
        )

    def write_entries(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        runtime_context: Mapping[str, Any],
        generation: str,
        source: str,
        storage_format: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.writer(storage_format=storage_format).write_entries(
            entries,
            runtime_context=runtime_context,
            generation=generation,
            source=source,
        )

    def read_batch(
        self,
        refs: Sequence[FeatureShardRef | Mapping[str, object]],
        *,
        device=None,
        runtime=None,
    ):
        return self.reader.read_batch(refs, device=device, runtime=runtime)

    def validate_ref(self, ref: FeatureShardRef | Mapping[str, object]) -> FeatureShardRef:
        parsed = ref if isinstance(ref, FeatureShardRef) else FeatureShardRef.from_dict(ref)
        if parsed.storage_format not in self.accepted_storage_formats:
            raise ValueError(f"Rejected feature shard storage_format={parsed.storage_format!r}.")
        if not parsed.index_path or not os.path.exists(fs_path(parsed.index_path)):
            raise FileNotFoundError(parsed.index_path)
        if parsed.storage_format == SAFETENSORS_SHARD:
            if not parsed.shard_path or not os.path.exists(fs_path(parsed.shard_path)):
                raise FileNotFoundError(parsed.shard_path or "")
        if parsed.storage_format == NPY_MEMMAP_SHARD:
            if not parsed.shard_dir or not os.path.isdir(fs_path(parsed.shard_dir)):
                raise FileNotFoundError(parsed.shard_dir or "")
        return parsed

    def import_shard_bundle(
        self,
        *,
        bundle_root: str,
        manifest: Mapping[str, Any],
        shard_entries: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Copy edge-created shard artifacts into the cloud store and return ref entries."""
        started = time.perf_counter()
        registered: list[dict[str, Any]] = []
        model_id = str(manifest.get("model_id") or "")
        feature_layout_id = str(
            dict(manifest.get("runtime_contract") or {}).get("feature_layout_id")
            or manifest.get("feature_layout_id")
            or ""
        )
        feature_abi_id = str(
            dict(manifest.get("runtime_contract") or {}).get("feature_abi_id")
            or manifest.get("feature_abi_id")
            or ""
        )
        runtime_identity_id = str(
            dict(manifest.get("runtime_contract") or {}).get("runtime_identity_id")
            or manifest.get("runtime_identity_id")
            or ""
        )
        generation = str(manifest.get("request_id") or manifest.get("generation") or "edge_upload")
        target_dir = os.path.join(
            self.root_dir,
            model_id or "unknown",
            feature_layout_id or "unknown",
            generation,
        )
        os.makedirs(fs_path(target_dir), exist_ok=True)
        for shard in list(shard_entries or []):
            storage_format = str(
                shard.get("storage_format") or manifest.get("storage_format") or ""
            )
            if storage_format not in self.accepted_storage_formats:
                raise ValueError(f"Rejected uploaded shard storage_format={storage_format!r}.")
            shard_id = str(shard.get("shard_id") or "")
            if not shard_id:
                raise ValueError("Uploaded feature shard is missing shard_id.")
            if storage_format == SAFETENSORS_SHARD:
                source_file = os.path.join(
                    bundle_root, str(shard.get("shard_file") or "").replace("/", os.sep)
                )
                source_index = os.path.join(
                    bundle_root, str(shard.get("index_file") or "").replace("/", os.sep)
                )
                source_meta = os.path.join(
                    bundle_root, str(shard.get("meta_file") or "").replace("/", os.sep)
                )
                for path in (source_file, source_index, source_meta):
                    if not os.path.exists(fs_path(path)):
                        raise FileNotFoundError(path)
                shard_path = os.path.join(target_dir, os.path.basename(source_file))
                index_path = os.path.join(target_dir, os.path.basename(source_index))
                meta_path = os.path.join(target_dir, os.path.basename(source_meta))
                shutil.copyfile(fs_path(source_file), fs_path(shard_path))
                shutil.copyfile(fs_path(source_meta), fs_path(meta_path))
                index_payload = _read_json(source_index)
                index_payload["shard_path"] = shard_path
                index_payload["index_path"] = index_path
                index_payload["metadata_path"] = meta_path
                _atomic_json_dump(index_path, index_payload)
                meta_payload = _read_json(meta_path)
                meta_payload["shard_path"] = shard_path
                meta_payload["index_path"] = index_path
                _atomic_json_dump(meta_path, meta_payload)
                shard_dir = None
            elif storage_format == NPY_MEMMAP_SHARD:
                source_dir = os.path.join(
                    bundle_root, str(shard.get("shard_dir") or "").replace("/", os.sep)
                )
                if not os.path.isdir(fs_path(source_dir)):
                    raise FileNotFoundError(source_dir)
                shard_dir = os.path.join(target_dir, os.path.basename(source_dir.rstrip(os.sep)))
                if os.path.exists(fs_path(shard_dir)):
                    shutil.rmtree(fs_path(shard_dir))
                shutil.copytree(fs_path(source_dir), fs_path(shard_dir))
                index_path = os.path.join(
                    shard_dir, str(shard.get("index_file_name") or f"{shard_id}.index.json")
                )
                meta_path = os.path.join(
                    shard_dir, str(shard.get("meta_file_name") or f"{shard_id}.meta.json")
                )
                for path in (index_path, meta_path):
                    if not os.path.exists(fs_path(path)):
                        raise FileNotFoundError(path)
                index_payload = _read_json(index_path)
                index_payload["shard_dir"] = shard_dir
                index_payload["index_path"] = index_path
                index_payload["metadata_path"] = meta_path
                _atomic_json_dump(index_path, index_payload)
                meta_payload = _read_json(meta_path)
                meta_payload["shard_dir"] = shard_dir
                meta_payload["index_path"] = index_path
                _atomic_json_dump(meta_path, meta_payload)
                shard_path = None
            else:
                raise ValueError(f"Unsupported uploaded shard storage_format={storage_format!r}.")
            index_payload = _read_json(index_path)
            sample_to_row = dict(index_payload.get("sample_to_row") or {})
            leaf_keys = list(dict(index_payload.get("leaf_specs") or {}).keys())
            for sample_id, row_id in sample_to_row.items():
                ref = FeatureShardRef(
                    storage_format=storage_format,
                    shard_id=shard_id,
                    shard_path=shard_path,
                    shard_dir=shard_dir,
                    index_path=index_path,
                    row_id=int(row_id),
                    sample_id=str(sample_id),
                    feature_layout_id=str(
                        index_payload.get("feature_layout_id") or feature_layout_id
                    ),
                    contract_id=(
                        None
                        if index_payload.get("contract_id") in (None, "")
                        else str(index_payload.get("contract_id"))
                    ),
                    boundary_id=str(index_payload.get("boundary_id") or ""),
                    payload_kind=str(index_payload.get("payload_kind") or "boundary_payload"),
                    dtype=str(index_payload.get("dtype") or ""),
                    shape_bucket=str(index_payload.get("shape_bucket") or ""),
                    leaf_keys=[str(key) for key in leaf_keys],
                    passthrough_keys=[],
                    metadata={
                        "model_id": str(index_payload.get("model_id") or model_id),
                        "split_config_id": str(index_payload.get("split_config_id") or ""),
                        "leaf_specs": dict(index_payload.get("leaf_specs") or {}),
                    },
                    feature_abi_id=str(index_payload.get("feature_abi_id") or feature_abi_id),
                    runtime_identity_id=str(
                        index_payload.get("runtime_identity_id") or runtime_identity_id
                    ),
                )
                registered.append({"sample_id": str(sample_id), "feature_ref": ref})
            logger.info(
                "[FeatureShard][Register] format={} samples={} dtype={}.",
                storage_format,
                len(sample_to_row),
                index_payload.get("dtype") or "",
            )
            log_diagnostic_debug(
                self,
                "[FeatureShard][Register] diagnostics",
                lambda: {
                    "shard_id": shard_id,
                    "shard_path": shard_path,
                    "index_path": index_path,
                    "feature_layout_id": index_payload.get("feature_layout_id")
                    or feature_layout_id,
                    "feature_abi_id": index_payload.get("feature_abi_id")
                    or feature_abi_id,
                    "contract_id": index_payload.get("contract_id"),
                },
            )
        logger.info(
            "[FeatureShard][Receive] shards={} samples={} format={} time={:.3f}s.",
            len(list(shard_entries or [])),
            len(registered),
            ",".join(sorted({str(item.get("storage_format") or "") for item in shard_entries})),
            time.perf_counter() - started,
        )
        return registered
