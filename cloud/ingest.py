from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from collections.abc import Mapping

import cv2
from loguru import logger

from cloud.contracts import validate_fixed_split_plan, validate_low_quality_manifest
from cloud.feature_cache import FeatureShardRef, FeatureShardStore
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary


def read_json_file(path: str) -> dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned or "unknown"


def materialize_low_quality_trigger_bundle(
    bundle_cache_path: str,
    *,
    feature_store: FeatureShardStore | None = None,
) -> dict[str, object] | None:
    trigger_manifest_path = os.path.join(bundle_cache_path, "trigger_manifest.json")
    if not os.path.exists(trigger_manifest_path):
        return None
    trigger_manifest = validate_low_quality_manifest(read_json_file(trigger_manifest_path))
    staging_root = os.path.join(bundle_cache_path, "low_quality_staging")
    shutil.rmtree(staging_root, ignore_errors=True)
    raw_root = os.path.join(staging_root, "raw")
    feature_root = os.path.join(staging_root, "features")
    os.makedirs(raw_root, exist_ok=True)
    os.makedirs(feature_root, exist_ok=True)

    samples: list[dict[str, object]] = []
    for shard in list(trigger_manifest.get("raw_shards", []) or []):
        if not isinstance(shard, Mapping):
            continue
        relpath = shard.get("file") or shard.get("path")
        if not relpath:
            continue
        tar_path = os.path.join(bundle_cache_path, str(relpath).replace("/", os.sep))
        if not os.path.exists(tar_path):
            continue
        with tarfile.open(tar_path, "r") as archive:
            manifest_member = archive.extractfile("manifest.jsonl")
            if manifest_member is None:
                continue
            raw_entries = [
                json.loads(line.decode("utf-8"))
                for line in manifest_member.readlines()
                if line.strip()
            ]
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, Mapping):
                    continue
                sample_id = str(raw_entry.get("sample_id", "") or "")
                raw_file = raw_entry.get("raw_file") or raw_entry.get("raw_path")
                if not sample_id or not raw_file:
                    continue
                member_name = str(raw_file).replace("\\", "/")
                if member_name.startswith("/") or ".." in member_name.split("/"):
                    raise RuntimeError(f"Unsafe raw shard member path: {member_name!r}")
                member = archive.getmember(member_name)
                source = archive.extractfile(member)
                if source is None:
                    continue
                suffix = os.path.splitext(member_name)[1] or ".jpg"
                safe_sample_id = _sanitize_segment(sample_id)
                raw_relpath = f"low_quality_staging/raw/{safe_sample_id}{suffix}"
                raw_path = os.path.join(bundle_cache_path, raw_relpath.replace("/", os.sep))
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                with open(raw_path, "wb") as handle:
                    shutil.copyfileobj(source, handle)

                frame = cv2.imread(raw_path)
                input_image_size = (
                    [int(frame.shape[0]), int(frame.shape[1])]
                    if frame is not None and frame.ndim >= 2
                    else None
                )
                samples.append(
                    {
                        "sample_id": sample_id,
                        "raw_relpath": raw_relpath,
                        "raw_bytes": os.path.getsize(raw_path),
                        "has_raw_sample": True,
                        "model_id": trigger_manifest.get("model_id", ""),
                        "model_version": trigger_manifest.get("model_version", ""),
                        "front_version": str(trigger_manifest.get("front_version", "0") or "0"),
                        **(
                            {"input_image_size": input_image_size}
                            if input_image_size is not None
                            else {}
                        ),
                        "input_tensor_shape": list(
                            trigger_manifest.get("input_tensor_shape", []) or []
                        ),
                        "input_resize_mode": str(
                            trigger_manifest.get("input_resize_mode", "") or "direct_resize"
                        ),
                    }
                )

    split_plan_payload = dict(trigger_manifest.get("split_plan", {}) or {})
    runtime_contract_payload = validate_fixed_split_plan(split_plan_payload)
    feature_refs_by_sample_id: dict[str, dict[str, object]] = {}
    feature_shards = [
        dict(shard)
        for shard in list(trigger_manifest.get("feature_shards", []) or [])
        if isinstance(shard, Mapping)
    ]
    if feature_shards and feature_store is not None:
        try:
            for registered_entry in feature_store.import_shard_bundle(
                bundle_root=bundle_cache_path,
                manifest=trigger_manifest,
                shard_entries=feature_shards,
            ):
                sample_id = str(registered_entry.get("sample_id") or "")
                feature_ref = registered_entry.get("feature_ref")
                if sample_id and isinstance(feature_ref, FeatureShardRef):
                    feature_refs_by_sample_id[sample_id] = feature_ref.to_dict()
        except Exception as exc:
            logger.warning(
                "[ShardCL][CloudUnpack] uploaded low-quality feature shard import failed; "
                "raw samples will be rebuilt where available: {}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                feature_store,
                "[ShardCL][CloudUnpack] feature shard import diagnostics",
                lambda error=exc: {
                    "bundle_path": bundle_cache_path,
                    "error": repr(error),
                },
            )
    if feature_refs_by_sample_id:
        for sample in samples:
            sample_id = str(sample.get("sample_id") or "")
            feature_ref = feature_refs_by_sample_id.get(sample_id)
            if not feature_ref:
                continue
            sample["feature_ref"] = feature_ref
            sample["has_feature"] = True
            sample["feature_layout_id"] = str(feature_ref.get("feature_layout_id") or "")
            sample["feature_abi_id"] = str(
                feature_ref.get("feature_abi_id")
                or runtime_contract_payload.get("feature_abi_id")
                or ""
            )
            sample["runtime_identity_id"] = str(feature_ref.get("runtime_identity_id") or "")
            sample["runtime_contract"] = runtime_contract_payload
    normalized_manifest = dict(trigger_manifest)
    normalized_manifest.update(
        {
            "edge_id": trigger_manifest.get("edge_id"),
            "model_id": str(trigger_manifest.get("model_id", "") or ""),
            "front_version": str(trigger_manifest.get("front_version", "0") or "0"),
            "split_config_id": str(trigger_manifest.get("split_config_id", "") or ""),
            "canonical_split_key": str(trigger_manifest.get("canonical_split_key", "") or ""),
            "edge_split_id": str(trigger_manifest.get("edge_split_id", "") or ""),
            "input_tensor_shape": list(trigger_manifest.get("input_tensor_shape", []) or []),
            "input_resize_mode": str(
                trigger_manifest.get("input_resize_mode", "") or "direct_resize"
            ),
            "model": {
                "model_id": str(trigger_manifest.get("model_id", "") or ""),
                "model_version": str(trigger_manifest.get("model_version", "") or "0"),
            },
            "runtime_contract": runtime_contract_payload,
            "split_plan": split_plan_payload,
            "training_mode": {
                "send_low_conf_features": bool(trigger_manifest.get("feature_shards")),
                "low_quality_mode": str(trigger_manifest.get("upload_mode", "raw-only")),
            },
            "selection_policy": {
                "policy": "low_quality_trigger_shards",
                "selected_sample_count": len(samples),
                "zip_payload_bytes": 0,
            },
            "samples": samples,
            "trigger_manifest": {
                "shard_size": trigger_manifest.get("shard_size"),
                "raw_shard_count": len(trigger_manifest.get("raw_shards", []) or []),
                "feature_shard_count": len(trigger_manifest.get("feature_shards", []) or []),
            },
        }
    )
    logger.info(
        "[ShardCL][CloudUnpack] materialized low-quality trigger shards samples={} "
        "raw_shards={} feature_shards={}",
        len(samples),
        len(trigger_manifest.get("raw_shards", []) or []),
        len(trigger_manifest.get("feature_shards", []) or []),
    )
    return normalized_manifest


def load_high_quality_shard_candidates(
    *,
    manifest: Mapping[str, object],
    bundle_cache_path: str,
    feature_store: FeatureShardStore,
    label_coordinate_space: str,
) -> tuple[list[dict[str, object]], list[str]]:
    candidates: list[dict[str, object]] = []
    unreadable_ids: list[str] = []
    manifest_input_tensor_shape = list(manifest.get("input_tensor_shape", []) or [])
    manifest_resize_mode = str(manifest.get("input_resize_mode", "") or "direct_resize")
    manifest_model_id = str(manifest.get("model_id", "") or "")
    manifest_split_config_id = str(manifest.get("split_config_id", "") or "")
    manifest_front_version = str(manifest.get("front_version", "0") or "0")
    manifest_runtime_contract = dict(
        manifest.get("runtime_contract")
        if isinstance(manifest.get("runtime_contract"), Mapping)
        else {}
    )
    manifest_feature_layout_id = str(
        manifest_runtime_contract.get("feature_layout_id")
        or manifest.get("feature_layout_id")
        or ""
    )
    labels_by_id: dict[str, dict[str, object]] = {}
    for shard in list(manifest.get("shards", []) or []):
        if not isinstance(shard, Mapping):
            continue
        label_file = shard.get("label_file") or shard.get("label_shard")
        if not label_file:
            continue
        label_path = os.path.join(bundle_cache_path, str(label_file).replace("/", os.sep))
        try:
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    label_payload = json.loads(line)
                    if isinstance(label_payload, Mapping) and label_payload.get("sample_id"):
                        labels_by_id[str(label_payload["sample_id"])] = dict(label_payload)
        except Exception:
            unreadable_ids.append(str(shard.get("shard_id") or label_file))
    try:
        registered = feature_store.import_shard_bundle(
            bundle_root=bundle_cache_path,
            manifest=manifest,
            shard_entries=[
                dict(shard)
                for shard in list(manifest.get("shards", []) or [])
                if isinstance(shard, Mapping)
            ],
        )
    except Exception as exc:
        logger.warning(
            "[FeatureShard][Register] high-quality upload failed: {}.",
            safe_error_summary(exc),
        )
        log_diagnostic_debug(
            feature_store,
            "[FeatureShard][Register] upload diagnostics",
            lambda error=exc: {
                "bundle_path": bundle_cache_path,
                "request_id": manifest.get("request_id"),
                "error": repr(error),
            },
        )
        return [], [str(manifest.get("request_id") or "uploaded_feature_shards")]
    for registered_entry in registered:
        sample_key = str(registered_entry.get("sample_id") or "")
        feature_ref = registered_entry.get("feature_ref")
        if not sample_key or not isinstance(feature_ref, FeatureShardRef):
            unreadable_ids.append(sample_key)
            continue
        if sample_key not in labels_by_id:
            unreadable_ids.append(sample_key)
            continue
        label_payload = dict(labels_by_id[sample_key])
        sample_input_image_size = label_payload.get("input_image_size")
        sample_input_tensor_shape = list(
            label_payload.get("input_tensor_shape") or manifest_input_tensor_shape or []
        )
        sample_resize_mode = str(
            label_payload.get("input_resize_mode") or manifest_resize_mode or ""
        )
        candidates.append(
            {
                "sample_id": sample_key,
                "labels": {
                    "boxes": list(label_payload.get("boxes") or []),
                    "labels": list(label_payload.get("labels") or []),
                    **(
                        {"scores": list(label_payload.get("scores") or [])}
                        if label_payload.get("scores") is not None
                        else {}
                    ),
                    "label_coordinate_space": str(
                        label_payload.get("label_coordinate_space") or label_coordinate_space
                    ),
                    **(
                        {"label_image_size": list(label_payload.get("label_image_size") or [])}
                        if label_payload.get("label_image_size") is not None
                        else {}
                    ),
                    **(
                        {"label_input_size": list(label_payload.get("label_input_size") or [])}
                        if label_payload.get("label_input_size") is not None
                        else {}
                    ),
                    "label_resize_mode": str(
                        label_payload.get("label_resize_mode") or sample_resize_mode
                    ),
                },
                "sample_source": "high_quality",
                "label_source": "edge_pseudo",
                "feature_ref": feature_ref.to_dict(),
                "model_id": manifest_model_id,
                "split_config_id": manifest_split_config_id,
                "front_version": manifest_front_version,
                "runtime_contract": manifest_runtime_contract,
                "feature_layout_id": str(
                    manifest_feature_layout_id or feature_ref.feature_layout_id
                ),
                "feature_abi_id": str(
                    feature_ref.feature_abi_id
                    or manifest_runtime_contract.get("feature_abi_id")
                    or ""
                ),
                "runtime_identity_id": str(feature_ref.runtime_identity_id or ""),
                "source_feature_abi_id": str(feature_ref.feature_abi_id or ""),
                "source_feature_layout_id": str(feature_ref.feature_layout_id),
                "source_feature_schema_hash": "",
                "source_feature_value_schema_hash": "",
                "source_feature_split_id": str(feature_ref.boundary_id or ""),
                "source_feature_graph_signature": str(
                    dict(feature_ref.metadata or {}).get("graph_signature") or ""
                ),
                "input_image_size": (
                    [int(dim) for dim in list(sample_input_image_size)]
                    if sample_input_image_size is not None
                    else None
                ),
                "input_tensor_shape": [int(dim) for dim in list(sample_input_tensor_shape)],
                "input_resize_mode": sample_resize_mode,
                "created_at": time.time(),
            }
        )
    return candidates, unreadable_ids


__all__ = [
    "load_high_quality_shard_candidates",
    "materialize_low_quality_trigger_bundle",
    "read_json_file",
]
