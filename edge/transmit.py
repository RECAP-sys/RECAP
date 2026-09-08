import io
import json
import os
import tarfile
import tempfile
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable

import grpc
from loguru import logger

from cloud.feature_cache.path_utils import fs_path
from cloud.feature_cache.types import NPY_MEMMAP_SHARD, SAFETENSORS_SHARD
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from edge.sample_quality import LOW_QUALITY
from edge.sample_store import EdgeSampleStore
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from model_management.fixed_split import SplitPlan
from tools.grpc_options import grpc_message_options


def _server_workspace_hint(edge_id: int, request_kind: str) -> str:
    return f"edge_{int(edge_id)}/{request_kind}"


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GiB"


def measure_trigger_bundle_payload(payload_zip: bytes) -> dict[str, int]:
    raw_frame_bytes = 0
    feature_bytes = 0
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as archive:
        for item in archive.infolist():
            name = str(item.filename)
            if name.startswith("raw_shards/"):
                raw_frame_bytes += int(item.compress_size)
            elif name.startswith("feature_shards/"):
                feature_bytes += int(item.compress_size)
    total_upload_bytes = len(payload_zip)
    prediction_metadata_bytes = max(
        0,
        total_upload_bytes - raw_frame_bytes - feature_bytes,
    )
    return {
        "raw_frame_bytes": raw_frame_bytes,
        "feature_bytes": feature_bytes,
        "prediction_metadata_bytes": prediction_metadata_bytes,
        "total_upload_bytes": total_upload_bytes,
    }


def _quality_sort_key(record) -> tuple[float, str, str]:
    return (
        0.0 if bool(getattr(record, "in_drift_window", False)) else 1.0,
        str(record.timestamp),
        str(record.sample_id),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_sample_filename(sample_id: str, suffix: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(sample_id)
    ).strip("._")
    return f"{safe or uuid.uuid4().hex}{suffix}"


def _record_abs_path(sample_store: EdgeSampleStore, relpath: str | None) -> str | None:
    if relpath is None:
        return None
    return os.path.join(sample_store.root_dir, str(relpath).replace("/", os.sep))


def _read_json_payload(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _metadata_path_from_feature_ref(ref: Mapping[str, Any]) -> str | None:
    index_path = str(ref.get("index_path") or "")
    index_payload = _read_json_payload(index_path)
    metadata_path = index_payload.get("metadata_path") or index_payload.get("meta_path")
    if metadata_path:
        path = str(metadata_path)
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(index_path), path)
        return path
    if index_path.endswith(".index.json"):
        return index_path[: -len(".index.json")] + ".meta.json"
    return f"{index_path}.meta.json" if index_path else None


def _feature_ref_paths(ref: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(ref, Mapping):
        return []
    paths: list[str] = []
    for key in ("shard_path", "index_path"):
        value = ref.get(key)
        if value:
            paths.append(str(value))
    metadata_path = _metadata_path_from_feature_ref(ref)
    if metadata_path:
        paths.append(metadata_path)
    shard_dir = ref.get("shard_dir")
    if shard_dir and os.path.isdir(str(shard_dir)):
        for root, _dirs, files in os.walk(str(shard_dir)):
            for filename in files:
                paths.append(os.path.join(root, filename))
    return sorted(set(paths))


def _feature_ref_matches_split(ref: Mapping[str, Any], split_plan: SplitPlan) -> bool:
    runtime_contract = dict(getattr(split_plan, "runtime_contract", {}) or {})
    expected_layout = str(runtime_contract.get("feature_layout_id") or "")
    expected_contract = str(runtime_contract.get("contract_id") or "")
    if expected_layout and str(ref.get("feature_layout_id") or "") != expected_layout:
        return False
    if expected_contract and str(ref.get("contract_id") or "") != expected_contract:
        return False
    return True


def _low_quality_trigger_feature_paths(
    record,
    *,
    split_plan: SplitPlan | None = None,
) -> list[str]:
    feature_ref = getattr(record, "feature_ref", None)
    if (
        isinstance(feature_ref, Mapping)
        and split_plan is not None
        and not _feature_ref_matches_split(feature_ref, split_plan)
    ):
        return []
    return [path for path in _feature_ref_paths(feature_ref) if os.path.exists(path)]


def _select_low_quality_trigger_records(
    sample_store: EdgeSampleStore,
    records: Sequence,
    *,
    send_low_conf_features: bool,
    bundle_cap_bytes: int | None,
    max_samples: int | None = None,
    split_plan: SplitPlan | None = None,
) -> tuple[list, dict[str, Any]]:
    cap = None if bundle_cap_bytes is None else max(1, int(bundle_cap_bytes))
    sample_cap = None if max_samples in (None, 0) else max(1, int(max_samples))
    selected = []
    selected_bytes = 0
    selected_feature_paths: set[str] = set()
    eligible = [record for record in records if record.quality_bucket == LOW_QUALITY]
    ordered = sorted(eligible, key=_quality_sort_key)
    if sample_cap is not None and len(ordered) > sample_cap:
        drift_records = [record for record in ordered if record.in_drift_window]
        regular_records = [record for record in ordered if not record.in_drift_window]
        chosen = drift_records[-sample_cap:]
        remaining = sample_cap - len(chosen)
        if remaining > 0:
            chosen = regular_records[-remaining:] + chosen
        ordered = sorted(chosen, key=_quality_sort_key)
    for record in ordered:
        raw_path = _record_abs_path(sample_store, record.raw_relpath)
        if raw_path is None or not os.path.exists(raw_path):
            continue
        feature_paths = (
            _low_quality_trigger_feature_paths(record, split_plan=split_plan)
            if send_low_conf_features
            else []
        )
        new_feature_paths = [path for path in feature_paths if path not in selected_feature_paths]
        source_bytes = os.path.getsize(raw_path) + sum(
            os.path.getsize(path) for path in new_feature_paths
        )
        protected = bool(getattr(record, "in_drift_window", False))
        if cap is not None and not protected and selected and selected_bytes + source_bytes > cap:
            continue
        selected.append(record)
        selected_bytes += source_bytes
        selected_feature_paths.update(new_feature_paths)
    return selected, {
        "policy": (
            "low_quality_trigger_raw_feature"
            if send_low_conf_features
            else "low_quality_trigger_raw_only"
        ),
        "bundle_cap_bytes": 0 if cap is None else int(cap),
        "max_samples": 0 if sample_cap is None else int(sample_cap),
        "selected_sample_count": len(selected),
        "omitted_sample_count": max(0, len(eligible) - len(selected)),
        "source_total_bytes": int(selected_bytes),
        "zip_payload_bytes": 0,
    }


def _chunks(items: Sequence, size: int) -> list[list]:
    shard_size = max(1, int(size))
    return [list(items[index : index + shard_size]) for index in range(0, len(items), shard_size)]


def _write_low_quality_raw_tar(
    sample_store: EdgeSampleStore,
    records: Sequence,
) -> tuple[bytes, list[dict[str, str]]]:
    tar_buffer = io.BytesIO()
    manifest_entries: list[dict[str, str]] = []
    with tarfile.open(fileobj=tar_buffer, mode="w") as tf:
        for record in records:
            sample_id = str(record.sample_id)
            raw_path = _record_abs_path(sample_store, record.raw_relpath)
            if raw_path is None or not os.path.exists(raw_path):
                raise FileNotFoundError(raw_path or record.raw_relpath or sample_id)
            suffix = os.path.splitext(raw_path)[1] or ".jpg"
            raw_name = f"raw/{_safe_sample_filename(sample_id, suffix)}"
            tf.add(raw_path, arcname=raw_name, recursive=False)
            manifest_entries.append(
                {
                    "sample_id": sample_id,
                    "raw_file": raw_name,
                }
            )
        manifest_bytes = (
            "\n".join(
                json.dumps(entry, sort_keys=True, separators=(",", ":"))
                for entry in manifest_entries
            )
            + ("\n" if manifest_entries else "")
        ).encode("utf-8")
        manifest_info = tarfile.TarInfo("manifest.jsonl")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = int(time.time())
        tf.addfile(manifest_info, io.BytesIO(manifest_bytes))
    return tar_buffer.getvalue(), manifest_entries


def _build_low_quality_feature_shard_uploads(
    records: Sequence,
    *,
    split_plan: SplitPlan,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    manifest_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    artifacts_by_arcname: dict[str, str] = {}

    def add_artifact(path: str, arcname: str) -> None:
        if os.path.exists(fs_path(path)):
            artifacts_by_arcname.setdefault(arcname.replace("\\", "/"), path)

    for record in records:
        ref = getattr(record, "feature_ref", None)
        if not isinstance(ref, Mapping) or not _feature_ref_matches_split(ref, split_plan):
            continue
        storage_format = str(ref.get("storage_format") or "")
        shard_id = str(ref.get("shard_id") or "")
        if storage_format not in {SAFETENSORS_SHARD, NPY_MEMMAP_SHARD} or not shard_id:
            continue
        safe_shard_id = _safe_sample_filename(shard_id, "").strip("._") or uuid.uuid4().hex
        sample_id = str(getattr(record, "sample_id", "") or ref.get("sample_id") or "")
        if storage_format == SAFETENSORS_SHARD:
            shard_path = str(ref.get("shard_path") or "")
            index_path = str(ref.get("index_path") or "")
            meta_path = _metadata_path_from_feature_ref(ref)
            if not shard_path or not index_path or not meta_path:
                continue
            if not all(
                os.path.exists(fs_path(path)) for path in (shard_path, index_path, meta_path)
            ):
                continue
            key = (storage_format, shard_id, index_path)
            base = f"feature_shards/{safe_shard_id}"
            entry = manifest_by_key.setdefault(
                key,
                {
                    "shard_id": shard_id,
                    "storage_format": storage_format,
                    "shard_file": f"{base}/{os.path.basename(shard_path)}",
                    "index_file": f"{base}/{os.path.basename(index_path)}",
                    "meta_file": f"{base}/{os.path.basename(meta_path)}",
                    "sample_ids": [],
                },
            )
            add_artifact(shard_path, str(entry["shard_file"]))
            add_artifact(index_path, str(entry["index_file"]))
            add_artifact(meta_path, str(entry["meta_file"]))
        elif storage_format == NPY_MEMMAP_SHARD:
            shard_dir = str(ref.get("shard_dir") or "")
            index_path = str(ref.get("index_path") or "")
            if not shard_dir or not index_path or not os.path.isdir(fs_path(shard_dir)):
                continue
            key = (storage_format, shard_id, shard_dir)
            base = f"feature_shards/{safe_shard_id}/{os.path.basename(shard_dir.rstrip(os.sep))}"
            entry = manifest_by_key.setdefault(
                key,
                {
                    "shard_id": shard_id,
                    "storage_format": storage_format,
                    "shard_dir": base,
                    "index_file_name": os.path.basename(index_path),
                    "meta_file_name": os.path.basename(index_path).replace(
                        ".index.json",
                        ".meta.json",
                    ),
                    "sample_ids": [],
                },
            )
            walk_root = fs_path(shard_dir)
            for root, _dirs, files in os.walk(walk_root):
                for filename in files:
                    path = os.path.join(root, filename)
                    relpath = os.path.relpath(path, walk_root).replace("\\", "/")
                    add_artifact(path, f"{base}/{relpath}")
        else:
            continue
        sample_ids = entry.setdefault("sample_ids", [])
        if sample_id and sample_id not in sample_ids:
            sample_ids.append(sample_id)

    manifest_entries: list[dict[str, Any]] = []
    for entry in manifest_by_key.values():
        sample_ids = [str(sample_id) for sample_id in list(entry.get("sample_ids") or [])]
        entry["sample_ids"] = sample_ids
        entry["sample_count"] = len(sample_ids)
        manifest_entries.append(dict(entry))
    artifacts = [(path, arcname) for arcname, path in sorted(artifacts_by_arcname.items())]
    return manifest_entries, artifacts


def pack_low_quality_trigger_bundle(
    sample_store: EdgeSampleStore,
    *,
    edge_id: int,
    send_low_conf_features: bool,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    edge_session_id: str | None = None,
    model_metadata: Mapping[str, object] | None = None,
    bundle_cap_bytes: int | None = None,
    max_samples: int | None = None,
    shard_size: int | None = None,
) -> tuple[bytes, dict]:
    zip_path, manifest, _ = pack_low_quality_trigger_bundle_to_file(
        sample_store,
        edge_id=edge_id,
        send_low_conf_features=send_low_conf_features,
        split_plan=split_plan,
        model_id=model_id,
        model_version=model_version,
        edge_session_id=edge_session_id,
        model_metadata=model_metadata,
        bundle_cap_bytes=bundle_cap_bytes,
        max_samples=max_samples,
        shard_size=shard_size,
    )
    try:
        with open(zip_path, "rb") as handle:
            return handle.read(), manifest
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass


def pack_low_quality_trigger_bundle_to_file(
    sample_store: EdgeSampleStore,
    *,
    edge_id: int,
    send_low_conf_features: bool,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    edge_session_id: str | None = None,
    model_metadata: Mapping[str, object] | None = None,
    bundle_cap_bytes: int | None = None,
    max_samples: int | None = None,
    shard_size: int | None = None,
    output_dir: str | None = None,
) -> tuple[str, dict, dict]:
    pack_started = time.perf_counter()
    records = [
        record
        for record in sample_store.list_records()
        if record.split_config_id == split_plan.split_config_id
        and record.model_id == str(model_id)
        and str(getattr(record, "front_version", "0") or "0")
        == str(getattr(split_plan, "front_version", "0") or "0")
        and record.quality_bucket == LOW_QUALITY
    ]
    send_features_requested = bool(send_low_conf_features)
    selected, selection_policy = _select_low_quality_trigger_records(
        sample_store,
        records,
        send_low_conf_features=send_low_conf_features,
        split_plan=split_plan,
        bundle_cap_bytes=bundle_cap_bytes,
        max_samples=max_samples,
    )
    resolved_shard_size = max(1, int(shard_size or 64))
    model_meta = {
        "model_id": str(model_id),
        "model_version": str(model_version),
    }
    for key, value in dict(model_metadata or {}).items():
        if value is None:
            continue
        model_meta[str(key)] = value

    manifest = {
        "edge_id": int(edge_id),
        "edge_session_id": str(edge_session_id or ""),
        "model_id": str(model_id),
        "model_version": str(model_version),
        "front_version": str(getattr(split_plan, "front_version", "0") or "0"),
        "split_config_id": str(split_plan.split_config_id),
        "canonical_split_key": str(getattr(split_plan, "canonical_split_key", "") or ""),
        "edge_split_id": str(getattr(split_plan, "edge_split_id", "") or ""),
        "input_tensor_shape": [
            int(dim) for dim in list(getattr(split_plan, "input_tensor_shape", []) or [])
        ],
        "input_resize_mode": str(getattr(split_plan, "input_resize_mode", "") or "direct_resize"),
        "runtime_contract": dict(getattr(split_plan, "runtime_contract", {}) or {}),
        "upload_mode": "raw+feature" if send_low_conf_features else "raw-only",
        "created_at": _utc_now(),
        "model": model_meta,
        "split_plan": split_plan.to_dict(),
        "training_mode": {
            "send_low_conf_features": bool(send_low_conf_features),
            "send_low_conf_features_requested": send_features_requested,
            "low_quality_mode": "raw+feature" if send_low_conf_features else "raw-only",
        },
        "selection_policy": selection_policy,
        "shard_size": resolved_shard_size,
        "sample_count": len(selected),
        "sample_ids": [str(record.sample_id) for record in selected],
        "raw_shards": [],
        "feature_shards": [],
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"low_trigger_edge_{int(edge_id)}_",
        suffix=".zip",
        dir=output_dir,
        delete=False,
    )
    zip_path = handle.name
    handle.close()
    try:
        raw_shard_payloads: list[tuple[str, bytes, list[str]]] = []
        for shard_index, shard_records in enumerate(_chunks(selected, resolved_shard_size), 1):
            sample_ids = [str(record.sample_id) for record in shard_records]
            raw_shard_name = f"raw_shards/low_raw_shard_{shard_index:06d}.tar"
            tar_bytes, _manifest_entries = _write_low_quality_raw_tar(
                sample_store,
                shard_records,
            )
            raw_shard_payloads.append((raw_shard_name, tar_bytes, sample_ids))
        manifest["raw_shards"] = [
            {
                "shard_id": f"edge{int(edge_id)}_low_raw_{index:06d}",
                "file": name,
                "sample_count": len(sample_ids),
                "sample_ids": list(sample_ids),
            }
            for index, (name, _payload, sample_ids) in enumerate(raw_shard_payloads, 1)
        ]
        feature_shard_entries, feature_artifacts = (
            _build_low_quality_feature_shard_uploads(
                selected,
                split_plan=split_plan,
            )
            if send_low_conf_features
            else ([], [])
        )
        manifest["feature_shards"] = feature_shard_entries
        if not feature_shard_entries:
            manifest["upload_mode"] = "raw-only"
            manifest["training_mode"]["send_low_conf_features"] = False
            manifest["training_mode"]["low_quality_mode"] = "raw-only"

        def _write_zip() -> None:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for name, payload, _sample_ids in raw_shard_payloads:
                    zf.writestr(name, payload, compress_type=zipfile.ZIP_STORED)
                for source_path, arcname in feature_artifacts:
                    zf.write(fs_path(source_path), arcname, compress_type=zipfile.ZIP_STORED)
                zf.writestr(
                    "trigger_manifest.json",
                    json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
                    compress_type=zipfile.ZIP_STORED,
                )

        for _ in range(3):
            _write_zip()
            zip_payload_bytes = os.path.getsize(zip_path)
            if int(manifest["selection_policy"]["zip_payload_bytes"]) == int(zip_payload_bytes):
                break
            manifest["selection_policy"]["zip_payload_bytes"] = int(zip_payload_bytes)
        _write_zip()
        zip_payload_bytes = os.path.getsize(zip_path)
        manifest["selection_policy"]["zip_payload_bytes"] = int(zip_payload_bytes)
        stats = dict(manifest["selection_policy"])
        stats.update(
            {
                "pack_elapsed_sec": float(time.perf_counter() - pack_started),
                "zip_path": zip_path,
                "zip_payload_bytes": int(zip_payload_bytes),
                "sample_count": len(selected),
                "raw_shard_count": len(manifest["raw_shards"]),
                "feature_shard_count": len(manifest["feature_shards"]),
            }
        )
        return zip_path, manifest, stats
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise


def submit_training_job(
    server_ip: str,
    *,
    edge_id: int,
    request_id: str,
    job_type: int,
    cache_path: str,
    send_low_conf_features: bool = False,
    frame_indices: list[int] | None = None,
    payload_zip: bytes = b"",
    channel=None,
    log_internal_ids: bool = False,
):
    owned_channel = channel is None
    request_started = time.perf_counter()
    try:
        if channel is None:
            channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        payload_size = len(payload_zip or b"")
        logger.info(
            "[EdgeCL] submitting training request: edge={} type={} size={}.",
            edge_id,
            job_type,
            _format_bytes(payload_size),
        )
        log_diagnostic_debug(
            log_internal_ids,
            "[EdgeCL] training request diagnostics",
            lambda: {
                "request_id": request_id,
                "cache_path": cache_path,
                "server": server_ip,
                "payload_zip_bytes": payload_size,
            },
        )
        req = message_transmission_pb2.SubmitTrainingJobRequest(
            edge_id=int(edge_id),
            request_id=str(request_id or ""),
            job_type=int(job_type),
            cache_path=str(cache_path or ""),
            send_low_conf_features=bool(send_low_conf_features),
            frame_indices=[int(index) for index in (frame_indices or [])],
            payload_zip=payload_zip,
        )
        reply = stub.submit_training_job(req)
        logger.info(
            "[EdgeCL] training request reply: accepted={} status={} "
            "queue_position={} elapsed={:.3f}s.",
            bool(reply.accepted),
            reply.status,
            int(getattr(reply, "queue_position", -1)),
            time.perf_counter() - request_started,
        )
        log_diagnostic_debug(
            log_internal_ids,
            "[EdgeCL] training reply diagnostics",
            lambda: {"request_id": request_id, "job_id": reply.job_id},
        )
        return reply
    except Exception as exc:
        logger.error(
            "[EdgeCL] training request failed: elapsed={:.3f}s reason={}.",
            time.perf_counter() - request_started,
            safe_error_summary(exc),
        )
        log_diagnostic_debug(
            log_internal_ids,
            "[EdgeCL] training request failure diagnostics",
            lambda error=exc: {"request_id": request_id, "error": repr(error)},
        )
        return None
    finally:
        if owned_channel and channel is not None:
            channel.close()


def _build_proto_or_namespace(
    names: Sequence[str],
    fields: Mapping[str, Any],
):
    for name in names:
        message_cls = getattr(message_transmission_pb2, name, None)
        if message_cls is None:
            continue
        descriptor = getattr(message_cls, "DESCRIPTOR", None)
        if descriptor is not None:
            allowed = set(descriptor.fields_by_name.keys())
            return message_cls(**{key: value for key, value in fields.items() if key in allowed})
        return message_cls(**dict(fields))
    return SimpleNamespace(**dict(fields))


def submit_sample_sync(
    server_ip: str,
    *,
    edge_id: int,
    request_id: str,
    sync_type: str,
    model_id: str,
    model_version: str,
    split_config_id: str,
    payload_zip: bytes,
    cache_path: str | None = None,
    channel=None,
    log_internal_ids: bool = False,
):
    owned_channel = channel is None
    request_started = time.perf_counter()
    try:
        if channel is None:
            channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = _build_proto_or_namespace(
            (
                "SampleSyncRequest",
                "SyncSamplesRequest",
                "SampleSyncBundleRequest",
            ),
            {
                "edge_id": int(edge_id),
                "model_id": str(model_id or ""),
                "model_version": str(model_version or ""),
                "split_config_id": str(split_config_id or ""),
                "sync_type": str(sync_type or ""),
                "payload_zip": payload_zip or b"",
            },
        )
        reply = stub.sync_samples(req)
        logger.info(
            "[EdgeUpload] sample shard uploaded: edge={} quality={} model={} "
            "version={} size={} elapsed={:.3f}s.",
            edge_id,
            sync_type,
            model_id,
            model_version,
            _format_bytes(len(payload_zip or b"")),
            time.perf_counter() - request_started,
        )
        log_diagnostic_debug(
            log_internal_ids,
            "[EdgeUpload] sample sync diagnostics",
            lambda: {
                "request_id": request_id,
                "split_config_id": split_config_id,
                "cache_path": cache_path,
            },
        )
        return reply
    except Exception as exc:
        logger.error(
            "[EdgeUpload] sample sync failed: elapsed={:.3f}s reason={}.",
            time.perf_counter() - request_started,
            safe_error_summary(exc),
        )
        log_diagnostic_debug(
            log_internal_ids,
            "[EdgeUpload] sample sync failure diagnostics",
            lambda error=exc: {"request_id": request_id, "error": repr(error)},
        )
        return None
    finally:
        if owned_channel and channel is not None:
            channel.close()


def get_training_job_status(
    server_ip: str,
    *,
    edge_id: int,
    job_id: str,
    channel=None,
):
    owned_channel = channel is None
    try:
        if channel is None:
            channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.TrainingJobStatusRequest(
            edge_id=int(edge_id),
            job_id=str(job_id or ""),
        )
        return stub.get_training_job_status(req)
    except Exception as exc:
        logger.error("[EdgeCL] training status poll failed: {}.", safe_error_summary(exc))
        return None
    finally:
        if owned_channel and channel is not None:
            channel.close()


def download_trained_model(
    server_ip: str,
    *,
    edge_id: int,
    job_id: str,
    channel=None,
):
    owned_channel = channel is None
    try:
        if channel is None:
            channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.DownloadTrainedModelRequest(
            edge_id=int(edge_id),
            job_id=str(job_id or ""),
        )
        reply = stub.download_trained_model(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.error("[EdgeCL] model update download failed: {}.", safe_error_summary(exc))
        return False, "", str(exc)
    finally:
        if owned_channel and channel is not None:
            channel.close()


def report_edge_model_version(
    server_ip: str,
    *,
    edge_id: int,
    model_id: str,
    model_version: str,
    channel=None,
) -> tuple[bool, str]:
    owned_channel = channel is None
    try:
        if channel is None:
            channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.ReportEdgeModelVersionRequest(
            edge_id=int(edge_id),
            model_id=str(model_id or ""),
            model_version=str(model_version or ""),
        )
        reply = stub.report_edge_model_version(req)
        return bool(reply.success), str(reply.message)
    except Exception as exc:
        logger.warning("[EdgeCL] model version report failed: {}.", safe_error_summary(exc))
        return False, str(exc)
    finally:
        if owned_channel and channel is not None:
            channel.close()


def submit_continual_learning_job(
    server_ip: str,
    *,
    edge_id: int,
    sample_store: EdgeSampleStore,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    send_low_conf_features: bool,
    model_metadata: Mapping[str, object] | None = None,
    edge_session_id: str | None = None,
    bundle_cap_bytes: int | None = None,
    max_samples: int | None = None,
    trigger_shard_size: int | None = None,
    bandwidth_mbps: float = 0.0,
    request_id: str | None = None,
    channel=None,
    log_internal_ids: bool = False,
    metrics_callback: Callable[[Mapping[str, Any]], None] | None = None,
):
    resolved_request_id: str | None = None
    try:
        pack_started = time.perf_counter()
        stats = sample_store.stats()
        logger.info(
            "[EdgeUpload] packing low-quality trigger: edge={} samples={} high={} "
            "low={} include_features={} model={} version={}.",
            edge_id,
            int(stats.get("total_samples", 0)),
            int(stats.get("high_quality_count", 0)),
            int(stats.get("low_quality_count", 0)),
            bool(send_low_conf_features),
            model_id,
            model_version,
        )
        payload_zip, manifest = pack_low_quality_trigger_bundle(
            sample_store,
            edge_id=edge_id,
            send_low_conf_features=send_low_conf_features,
            split_plan=split_plan,
            model_id=model_id,
            model_version=model_version,
            edge_session_id=edge_session_id,
            model_metadata=model_metadata,
            bundle_cap_bytes=bundle_cap_bytes,
            max_samples=max_samples,
            shard_size=trigger_shard_size,
        )
        uploaded_sample_ids = [str(value) for value in list(manifest.get("sample_ids", []) or [])]
        zip_payload_bytes = len(payload_zip)
        payload_metrics = measure_trigger_bundle_payload(payload_zip)
        selection_policy = dict(manifest.get("selection_policy", {}) or {})
        estimated_upload_sec = None
        if bandwidth_mbps > 0.0 and zip_payload_bytes > 0:
            estimated_upload_sec = zip_payload_bytes * 8.0 / (float(bandwidth_mbps) * 1_000_000.0)
        logger.info(
            "[EdgeUpload] low-quality trigger packed: edge={} elapsed={:.3f}s "
            "samples={} raw_shards={} feature_shards={} source_size={} cap={} "
            "size={} estimated_upload={}.",
            edge_id,
            time.perf_counter() - pack_started,
            int(manifest.get("sample_count", 0) or 0),
            len(manifest.get("raw_shards", []) or []),
            len(manifest.get("feature_shards", []) or []),
            _format_bytes(int(selection_policy.get("source_total_bytes", 0))),
            _format_bytes(int(selection_policy.get("bundle_cap_bytes", 0))),
            _format_bytes(zip_payload_bytes),
            (f"{estimated_upload_sec:.3f}s" if estimated_upload_sec is not None else "unknown"),
        )
        upload_started_at_ms = int(time.time() * 1000)
        upload_started = time.perf_counter()
        resolved_request_id = str(request_id or uuid.uuid4().hex)
        reply = submit_training_job(
            server_ip,
            edge_id=edge_id,
            request_id=resolved_request_id,
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            cache_path=_server_workspace_hint(edge_id, "continual_learning"),
            send_low_conf_features=bool(send_low_conf_features),
            payload_zip=payload_zip,
            channel=channel,
            log_internal_ids=log_internal_ids,
        )
        upload_elapsed = time.perf_counter() - upload_started
        upload_done_at_ms = int(time.time() * 1000)
        if metrics_callback is not None:
            metrics_callback(
                {
                    **payload_metrics,
                    "upload_ms": upload_elapsed * 1000.0,
                    "upload_started_at_ms": upload_started_at_ms,
                    "upload_done_at_ms": upload_done_at_ms,
                    "raw_sample_count": sum(
                        int(item.get("sample_count", 0) or 0)
                        for item in list(manifest.get("raw_shards", []) or [])
                    ),
                    "feature_sample_count": sum(
                        int(item.get("sample_count", 0) or 0)
                        for item in list(manifest.get("feature_shards", []) or [])
                    ),
                }
            )
        upload_mbps = (
            zip_payload_bytes * 8.0 / upload_elapsed / 1_000_000.0
            if upload_elapsed > 0.0 and zip_payload_bytes > 0
            else 0.0
        )
        if reply is None:
            logger.error(
                "[EdgeUpload] low-quality trigger upload failed: edge={} "
                "elapsed={:.3f}s speed={:.3f}Mbps size={}.",
                edge_id,
                upload_elapsed,
                upload_mbps,
                _format_bytes(zip_payload_bytes),
            )
            return False, "", "submit_training_job failed", uploaded_sample_ids
        logger.info(
            "[EdgeUpload] low-quality trigger uploaded: edge={} samples={} version={} "
            "size={} elapsed={:.3f}s speed={:.3f}Mbps.",
            edge_id,
            int(manifest.get("sample_count", 0) or 0),
            model_version,
            _format_bytes(zip_payload_bytes),
            upload_elapsed,
            upload_mbps,
        )
        log_diagnostic_debug(
            log_internal_ids,
            "[EdgeUpload] continual-learning submission diagnostics",
            lambda: {
                "request_id": resolved_request_id,
                "job_id": reply.job_id,
                "split_config_id": getattr(split_plan, "split_config_id", ""),
                "session_id": edge_session_id,
            },
        )
        return bool(reply.accepted), str(reply.job_id), str(reply.message), uploaded_sample_ids
    except Exception as exc:
        logger.error(
            "[EdgeUpload] continual-learning submission failed: {}.",
            safe_error_summary(exc),
        )
        log_diagnostic_debug(
            log_internal_ids,
            "[EdgeUpload] continual-learning failure diagnostics",
            lambda error=exc: {
                "request_id": resolved_request_id or request_id,
                "error": repr(error),
            },
        )
        return False, "", str(exc), []
