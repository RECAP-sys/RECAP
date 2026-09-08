from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Callable

import torch
from loguru import logger

import edge.transmit as transmit
from cloud.feature_cache.path_utils import fs_path
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from edge.feature_shard import write_feature_label_shards
from edge.sample_quality import HIGH_QUALITY
from edge.sample_store import EdgeSampleStore, StoredSampleRecord
from model_management.detection_box_projection import ORIGINAL_XYXY
from model_management.payload import BoundaryPayload
from model_management.split_contract import feature_layout_from_tensors

UPLOAD_LEDGER_FILENAME = "upload_ledger.json"
UPLOAD_PENDING = "pending"
UPLOAD_UPLOADED = "uploaded"
UPLOAD_COMMITTED = "committed"
UPLOAD_FAILED = "failed"
UPLOAD_STALE_SPLIT = "stale_split"
_RETRYABLE_STATES = {UPLOAD_PENDING, UPLOAD_UPLOADED, UPLOAD_FAILED}
_ATOMIC_WRITE_RETRIES = 8
_ATOMIC_WRITE_RETRY_DELAY_SEC = 0.02


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalise_shard_dtype(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "original", "preserve"}:
        return None
    return text


def _stable_json(payload: object) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def _hash_payload(payload: object) -> str:
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(
        directory,
        f".{os.path.basename(path)}.tmp-{threading.get_ident()}-{uuid.uuid4().hex}",
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(_ATOMIC_WRITE_RETRIES):
            try:
                os.replace(tmp_path, path)
                return
            except OSError as exc:
                if (
                    not _is_retryable_atomic_replace_error(exc)
                    or attempt + 1 >= _ATOMIC_WRITE_RETRIES
                ):
                    raise
                time.sleep(_ATOMIC_WRITE_RETRY_DELAY_SEC * (attempt + 1))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _is_retryable_atomic_replace_error(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    return (
        isinstance(exc, PermissionError)
        or exc.errno in {errno.EACCES, errno.EPERM}
        or winerror in {5, 32}
    )


def _chunks(items: Sequence[StoredSampleRecord], size: int) -> list[list[StoredSampleRecord]]:
    shard_size = max(1, int(size))
    return [list(items[index : index + shard_size]) for index in range(0, len(items), shard_size)]


def _training_labels(result: Mapping[str, Any], record: StoredSampleRecord) -> dict[str, Any]:
    labels: dict[str, Any] = {
        "boxes": list(result.get("boxes") or []),
        "labels": list(result.get("labels") or []),
        "label_coordinate_space": ORIGINAL_XYXY,
    }
    if getattr(record, "input_image_size", None) is not None:
        labels["label_image_size"] = list(record.input_image_size or [])
    if getattr(record, "input_resize_mode", None):
        labels["label_resize_mode"] = str(record.input_resize_mode)
    return labels


def _feature_layout_metadata(intermediate: Any) -> dict[str, Any]:
    if not isinstance(intermediate, BoundaryPayload):
        return {}
    tensors = {
        str(label): tensor.detach().cpu()
        for label, tensor in dict(intermediate.tensors or {}).items()
        if isinstance(tensor, torch.Tensor)
    }
    layout = feature_layout_from_tensors(tensors) if tensors else {}
    schema_payload = {}
    for label, spec in dict(getattr(intermediate, "spec", {}) or {}).items():
        schema_payload[str(label)] = {
            "label": str(getattr(spec, "label", label)),
            "canonical_id": str(getattr(spec, "canonical_id", label)),
            "torchlens_label": str(getattr(spec, "torchlens_label", label)),
            "module_path": str(getattr(spec, "module_path", "")),
            "op_type": str(getattr(spec, "op_type", "")),
            "symbolic_shape": [str(dim) for dim in list(getattr(spec, "shape", ()) or ())],
            "dtype": str(getattr(spec, "dtype", "")),
            "requires_grad": bool(getattr(spec, "requires_grad", False)),
            "role": str(getattr(spec, "role", "")),
            "output_index": getattr(spec, "output_index", None),
            "device_policy": str(getattr(spec, "device_policy", "runtime")),
        }
    return {
        "feature_layout_id": _hash_payload(layout) if layout else "",
        "feature_layout": layout,
        "feature_schema_hash": _hash_payload(schema_payload) if schema_payload else "",
        "feature_value_schema_hash": "",
        "feature_split_id": str(getattr(intermediate, "split_id", "") or ""),
        "feature_graph_signature": str(
            intermediate.metadata.get("graph_shape_hash")
            or intermediate.metadata.get("graph_signature")
            or ""
        ),
    }


def pack_high_quality_sync_bundle(
    sample_store: EdgeSampleStore,
    records: Sequence[StoredSampleRecord],
    *,
    edge_id: int,
    shard_size: int,
    storage_format: str = "safetensors_shard",
    shard_dtype: str | None = None,
    request_id: str | None = None,
    split_context: Mapping[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    zip_path, manifest, _stats = pack_high_quality_sync_bundle_to_file(
        sample_store,
        records,
        edge_id=edge_id,
        shard_size=shard_size,
        storage_format=storage_format,
        shard_dtype=shard_dtype,
        request_id=request_id,
        split_context=split_context,
    )
    try:
        with open(zip_path, "rb") as handle:
            return handle.read(), manifest
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass


def pack_high_quality_sync_bundle_to_file(
    sample_store: EdgeSampleStore,
    records: Sequence[StoredSampleRecord],
    *,
    edge_id: int,
    shard_size: int,
    storage_format: str = "safetensors_shard",
    shard_dtype: str | None = None,
    request_id: str | None = None,
    split_context: Mapping[str, Any] | None = None,
    output_dir: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    selected = [
        record
        for record in records
        if record.quality_bucket == HIGH_QUALITY and record.feature_ref is not None
    ]
    resolved_request_id = str(request_id or uuid.uuid4().hex)
    resolved_shard_size = max(1, int(shard_size))
    context = dict(split_context or {})
    first_record = selected[0] if selected else None
    model_id = str(context.get("model_id") or getattr(first_record, "model_id", "") or "")
    model_version = str(
        context.get("model_version") or getattr(first_record, "model_version", "") or ""
    )
    edge_session_id = str(context.get("edge_session_id") or "").strip()
    front_version = str(
        context.get("front_version") or getattr(first_record, "front_version", "") or "0"
    )
    split_config_id = str(
        context.get("split_config_id") or getattr(first_record, "split_config_id", "") or ""
    )
    canonical_split_key = str(context.get("canonical_split_key") or "").strip()
    edge_split_id = str(context.get("edge_split_id") or canonical_split_key or "").strip()
    input_tensor_shape = list(
        context.get("input_tensor_shape") or getattr(first_record, "input_tensor_shape", []) or []
    )
    input_resize_mode = str(
        context.get("input_resize_mode")
        or getattr(first_record, "input_resize_mode", "")
        or "direct_resize"
    )
    runtime_contract = dict(context.get("runtime_contract") or {})
    inferred_feature_layout = {
        str(label): dict(spec)
        for label, spec in dict(runtime_contract.get("feature_layout") or {}).items()
        if isinstance(spec, Mapping)
    }
    inferred_boundary_labels = [
        str(label) for label in list(runtime_contract.get("boundary_tensor_labels") or [])
    ]
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"sample_sync_edge_{int(edge_id)}_",
        suffix=".zip",
        dir=output_dir,
        delete=False,
    )
    zip_path = handle.name
    handle.close()

    manifest: dict[str, Any] = {
        "edge_id": int(edge_id),
        "edge_session_id": edge_session_id,
        "model_id": model_id,
        "model_version": model_version,
        "front_version": front_version,
        "split_config_id": split_config_id,
        "canonical_split_key": canonical_split_key,
        "edge_split_id": edge_split_id,
        "input_tensor_shape": [int(dim) for dim in input_tensor_shape],
        "input_resize_mode": input_resize_mode,
        "label_coordinate_space": ORIGINAL_XYXY,
        "runtime_contract": runtime_contract,
        "request_id": resolved_request_id,
        "created_at": _utc_now(),
        "shard_size": resolved_shard_size,
        "sample_count": len(selected),
        "shards": [],
    }

    shard_tmp_root = ""
    try:
        feature_entries: list[dict[str, Any]] = []
        inferred_layout_id = str(runtime_contract.get("feature_layout_id") or "")
        for record in selected:
            sample_id = str(record.sample_id)
            intermediate = sample_store.load_intermediate(record)
            result = sample_store.load_inference_result(record)
            feature_layout_meta = _feature_layout_metadata(intermediate)
            inferred_layout_id = inferred_layout_id or str(
                feature_layout_meta.get("feature_layout_id") or ""
            )
            if not inferred_feature_layout and isinstance(
                feature_layout_meta.get("feature_layout"), Mapping
            ):
                inferred_feature_layout = {
                    str(label): dict(spec)
                    for label, spec in dict(feature_layout_meta.get("feature_layout") or {}).items()
                    if isinstance(spec, Mapping)
                }
            if not inferred_boundary_labels and inferred_feature_layout:
                inferred_boundary_labels = list(inferred_feature_layout)
            labels = _training_labels(result, record)
            label_payload = {
                "boxes": labels["boxes"],
                "labels": labels["labels"],
                "scores": list(result.get("scores") or []),
                "label_coordinate_space": labels["label_coordinate_space"],
                **(
                    {"label_image_size": labels["label_image_size"]}
                    if labels.get("label_image_size") is not None
                    else {}
                ),
                **(
                    {"label_resize_mode": labels["label_resize_mode"]}
                    if labels.get("label_resize_mode") is not None
                    else {}
                ),
                **(
                    {"input_image_size": list(record.input_image_size or [])}
                    if record.input_image_size is not None
                    else {}
                ),
                **(
                    {"input_tensor_shape": list(record.input_tensor_shape or [])}
                    if record.input_tensor_shape is not None
                    else {}
                ),
                **(
                    {"input_resize_mode": str(record.input_resize_mode)}
                    if record.input_resize_mode is not None
                    else {}
                ),
            }
            feature_entries.append(
                {
                    "sample": {
                        "sample_id": sample_id,
                        "labels": label_payload,
                        "input_image_size": list(record.input_image_size or []),
                        "input_tensor_shape": list(record.input_tensor_shape or []),
                        "input_resize_mode": str(record.input_resize_mode or input_resize_mode),
                    },
                    "record": (
                        {"intermediate": intermediate}
                        if isinstance(intermediate, BoundaryPayload)
                        else {"feature": intermediate}
                    ),
                }
            )
        feature_abi_spec = dict(runtime_contract.get("feature_abi_spec") or {})
        feature_abi_value = str(runtime_contract.get("feature_abi_id") or "")
        if not feature_abi_value or not feature_abi_spec:
            raise RuntimeError(
                "High-quality feature sync requires the current feature ABI contract."
            )
        runtime_context = {
            "model_id": model_id,
            "model_family": str(context.get("model_family") or ""),
            "split_config_id": split_config_id,
            "contract_id": None
            if runtime_contract.get("contract_id") in (None, "")
            else str(runtime_contract.get("contract_id")),
            "feature_layout_id": inferred_layout_id,
            "feature_abi_id": feature_abi_value,
            "feature_abi_spec": dict(feature_abi_spec),
            "runtime_identity_id": str(runtime_contract.get("runtime_identity_id") or ""),
            "feature_layout": dict(inferred_feature_layout),
            "boundary_tensor_labels": list(inferred_boundary_labels),
            "canonical_split_key": canonical_split_key,
            "boundary_id": edge_split_id or canonical_split_key,
            "input_tensor_shape": [int(dim) for dim in input_tensor_shape],
            "input_resize_mode": input_resize_mode,
            "front_version": front_version,
            "runtime_contract": runtime_contract,
        }
        shard_tmp_root, shard_manifest_entries, _labels_by_shard = write_feature_label_shards(
            output_root=output_dir,
            storage_format=storage_format,
            shard_max_samples=resolved_shard_size,
            shard_dtype=_normalise_shard_dtype(shard_dtype),
            runtime_context=runtime_context,
            generation=resolved_request_id,
            entries=feature_entries,
        )
        manifest["storage_format"] = str(storage_format)
        manifest["feature_layout_id"] = inferred_layout_id
        manifest["feature_abi_id"] = feature_abi_value
        manifest["shards"] = shard_manifest_entries
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for root, _dirs, files in os.walk(shard_tmp_root):
                for filename in files:
                    path = os.path.join(root, filename)
                    relpath = os.path.relpath(path, shard_tmp_root).replace("\\", "/")
                    zf.write(fs_path(path), relpath, compress_type=zipfile.ZIP_STORED)
            zf.writestr(
                "bundle_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
                compress_type=zipfile.ZIP_STORED,
            )
        stats = {
            "sample_count": len(selected),
            "shard_count": len(manifest["shards"]),
            "zip_path": zip_path,
            "zip_payload_bytes": os.path.getsize(zip_path),
        }
        return zip_path, manifest, stats
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise
    finally:
        if shard_tmp_root:
            shutil.rmtree(shard_tmp_root, ignore_errors=True)


class HighQualitySampleSyncer:
    def __init__(
        self,
        sample_store: EdgeSampleStore,
        *,
        server_ip: str,
        edge_id: int,
        feature_upload_config: object | None = None,
        shard_size: int | None = None,
        sync_interval_sec: float | None = None,
        enabled: bool = True,
        context_provider: Callable[[], Mapping[str, Any]] | None = None,
        log_internal_ids: bool = False,
    ) -> None:
        self.sample_store = sample_store
        self.server_ip = str(server_ip)
        self.edge_id = int(edge_id)
        self.shard_size = max(
            1,
            int(64 if shard_size is None else shard_size),
        )
        self.sync_interval_sec = max(
            0.1,
            float(30.0 if sync_interval_sec is None else sync_interval_sec),
        )
        self.enabled = bool(enabled)
        self.storage_format = str(
            getattr(feature_upload_config, "storage_format", "safetensors_shard")
            or "safetensors_shard"
        )
        self.shard_dtype = _normalise_shard_dtype(
            getattr(feature_upload_config, "shard_dtype", None)
        )
        configured_max = getattr(feature_upload_config, "shard_max_samples", None)
        if configured_max not in (None, ""):
            self.shard_size = max(1, int(configured_max))
        self.ledger_path = os.path.join(self.sample_store.root_dir, UPLOAD_LEDGER_FILENAME)
        self._ledger_lock = threading.RLock()
        self._condition = threading.Condition()
        self._flush_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._context_provider = context_provider
        self.log_internal_ids = bool(log_internal_ids)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"edge-high-quality-sync-{self.edge_id}",
            daemon=False,
        )
        self._thread.start()
        self.enqueue_existing_high_quality()

    def notify_sample(self, record: StoredSampleRecord) -> None:
        if not self.enabled or record.quality_bucket != HIGH_QUALITY:
            return
        self._mark_samples([str(record.sample_id)], UPLOAD_PENDING)
        with self._condition:
            self._condition.notify()

    def enqueue_existing_high_quality(self) -> None:
        if not self.enabled:
            return
        sample_ids = []
        for record in self.sample_store.list_records(quality_bucket=HIGH_QUALITY):
            state = self._sample_state(str(record.sample_id))
            if state != UPLOAD_COMMITTED:
                sample_ids.append(str(record.sample_id))
        if sample_ids:
            self._mark_samples(sample_ids, UPLOAD_PENDING, preserve_attempts=True)
            with self._condition:
                self._condition.notify()

    def flush(self, *, timeout: float | None = None, include_partial: bool = True) -> bool:
        if not self.enabled:
            return True
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        if not self._acquire_flush_lock(deadline):
            return False
        try:
            record_groups = self._select_retryable_record_groups(include_partial=include_partial)
            if not record_groups:
                return True
            all_ok = True
            for records in record_groups:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                if not self._flush_record_group(records, deadline=deadline):
                    all_ok = False
            return all_ok
        finally:
            self._flush_lock.release()

    def _flush_record_group(
        self,
        records: Sequence[StoredSampleRecord],
        *,
        deadline: float | None,
    ) -> bool:
        if not records:
            return True
        request_id = uuid.uuid4().hex
        sample_ids = [str(record.sample_id) for record in records]
        shard_by_sample: dict[str, str] = {}
        zip_path = ""
        try:
            split_context = self._split_context_for_records(records)
            zip_path, manifest, stats = pack_high_quality_sync_bundle_to_file(
                self.sample_store,
                records,
                edge_id=self.edge_id,
                shard_size=self.shard_size,
                storage_format=self.storage_format,
                shard_dtype=self.shard_dtype,
                request_id=request_id,
                split_context=split_context,
                output_dir=os.path.join(self.sample_store.root_dir, "sync_tmp"),
            )
            for shard in list(manifest.get("shards", []) or []):
                if not isinstance(shard, Mapping):
                    continue
                shard_id = str(shard.get("shard_id") or "")
                for sample_id in list(shard.get("sample_ids") or []):
                    if shard_id and sample_id:
                        shard_by_sample[str(sample_id)] = shard_id
            if deadline is not None and time.monotonic() >= deadline:
                return False
            with open(zip_path, "rb") as handle:
                payload_zip = handle.read()
            reply = transmit.submit_sample_sync(
                self.server_ip,
                edge_id=self.edge_id,
                request_id=request_id,
                sync_type="HIGH_QUALITY_FEATURE_LABEL_SHARD",
                model_id=str(manifest.get("model_id", "")),
                model_version=str(manifest.get("model_version", "")),
                split_config_id=str(manifest.get("split_config_id", "")),
                payload_zip=payload_zip,
                log_internal_ids=self.log_internal_ids,
            )
            if not _reply_succeeded(reply):
                self._mark_samples(sample_ids, UPLOAD_FAILED, error=_reply_message(reply))
                return False
            # Cloud sync only stages samples into the pending area; the active
            # canonical generation is committed later during the training-job
            # canonical rebuild. From the edge's perspective the sample has
            # been durably uploaded and no retry is required: the commit state
            # is therefore the terminal "uploaded to cloud pending" marker.
            self._mark_samples(
                sample_ids,
                UPLOAD_COMMITTED,
                shard_by_sample=shard_by_sample,
            )
            partial = any(
                int(shard.get("sample_count", 0) or 0) < self.shard_size
                for shard in list(manifest.get("shards", []) or [])
            )
            logger.info(
                "[EdgeUpload] high-quality shard uploaded: edge={} samples={} shards={} "
                "partial={} size={} version={}.",
                self.edge_id,
                len(sample_ids),
                int(stats.get("shard_count", 0)),
                partial,
                transmit._format_bytes(int(stats.get("zip_payload_bytes", 0))),
                str(manifest.get("model_version", "")),
            )
            log_diagnostic_debug(
                self,
                "[EdgeUpload] high-quality shard diagnostics",
                lambda: {
                    "request_id": request_id,
                    "split_config_id": manifest.get("split_config_id"),
                    "shard_ids": sorted(set(shard_by_sample.values())),
                    "zip_path": zip_path,
                },
            )
            return True
        except Exception as exc:
            self._mark_samples(sample_ids, UPLOAD_FAILED, error=str(exc))
            logger.error(
                "[EdgeUpload] high-quality sample sync failed: {}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "[EdgeUpload] high-quality sync failure diagnostics",
                lambda error=exc: {
                    "request_id": request_id,
                    "sample_ids": sample_ids,
                    "zip_path": zip_path,
                    "error": repr(error),
                },
            )
            return False
        finally:
            if zip_path:
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

    def close(self, *, timeout: float | None = None) -> bool:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        thread = self._thread
        if thread is not None and thread.is_alive():
            join_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=join_timeout)
        flush_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        flushed = self.flush(timeout=flush_timeout, include_partial=True)
        thread_stopped = thread is None or not thread.is_alive()
        return flushed and thread_stopped

    def _run(self) -> None:
        next_partial_flush = time.monotonic() + self.sync_interval_sec
        while not self._stop_event.is_set():
            action: str | None = None
            with self._condition:
                pending_count = self._retryable_count()
                full_group_count = self._retryable_full_group_count()
                now = time.monotonic()
                if full_group_count >= self.shard_size:
                    action = "full"
                elif pending_count > 0 and now >= next_partial_flush:
                    action = "partial"
                else:
                    if pending_count == 0:
                        next_partial_flush = now + self.sync_interval_sec
                    wait_for = max(0.1, next_partial_flush - now)
                    self._condition.wait(timeout=wait_for)
                    continue

            if action == "full":
                self.flush(timeout=self.sync_interval_sec, include_partial=False)
            elif action == "partial":
                self.flush(timeout=self.sync_interval_sec, include_partial=True)
            next_partial_flush = time.monotonic() + self.sync_interval_sec

    def _acquire_flush_lock(self, deadline: float | None) -> bool:
        if deadline is None:
            self._flush_lock.acquire()
            return True
        timeout = max(0.0, deadline - time.monotonic())
        return self._flush_lock.acquire(timeout=timeout)

    def _empty_ledger(self) -> dict[str, Any]:
        return {
            "created_at": _utc_now(),
            "samples": {},
        }

    def _load_ledger_unlocked(self) -> dict[str, Any]:
        if not os.path.exists(self.ledger_path):
            return self._empty_ledger()
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            logger.warning("Ignoring unreadable sample upload ledger.")
            log_diagnostic_debug(
                self,
                "unreadable sample upload ledger diagnostics",
                lambda: {"ledger_path": self.ledger_path},
            )
            return self._empty_ledger()
        if not isinstance(payload, dict):
            return self._empty_ledger()
        samples = payload.get("samples")
        if not isinstance(samples, dict):
            payload["samples"] = {}
        return payload

    def _write_ledger_unlocked(self, payload: Mapping[str, Any]) -> None:
        _atomic_json_dump(self.ledger_path, payload)

    def _sample_state(self, sample_id: str) -> str | None:
        with self._ledger_lock:
            entry = self._load_ledger_unlocked().get("samples", {}).get(str(sample_id))
        if isinstance(entry, Mapping):
            return str(entry.get("sync_state") or entry.get("state") or "")
        return None

    def _split_context_for_records(self, records: Sequence[StoredSampleRecord]) -> dict[str, Any]:
        provider_context: dict[str, Any] = {}
        if callable(self._context_provider):
            try:
                provider_context.update(dict(self._context_provider() or {}))
            except Exception as exc:
                logger.warning(
                    "High-quality sync context provider failed: {}.",
                    safe_error_summary(exc),
                )
        first_record = records[0] if records else None
        context = {
            "model_id": _first_nonempty(
                getattr(first_record, "model_id", ""),
                provider_context.get("model_id"),
            ),
            "model_version": _first_nonempty(
                getattr(first_record, "model_version", ""),
                provider_context.get("model_version"),
            ),
            "edge_session_id": str(provider_context.get("edge_session_id") or "").strip(),
            "front_version": _first_nonempty(
                getattr(first_record, "front_version", ""),
                provider_context.get("front_version"),
                "0",
            ),
            "split_config_id": _first_nonempty(
                getattr(first_record, "split_config_id", ""),
                provider_context.get("split_config_id"),
            ),
        }
        provider_split = str(provider_context.get("split_config_id") or "").strip()
        if provider_split and provider_split == context["split_config_id"]:
            context["canonical_split_key"] = provider_context.get("canonical_split_key")
            context["edge_split_id"] = provider_context.get("edge_split_id")
            context["input_tensor_shape"] = list(
                provider_context.get("input_tensor_shape", []) or []
            )
            context["input_resize_mode"] = provider_context.get("input_resize_mode")
            context["runtime_contract"] = dict(provider_context.get("runtime_contract") or {})
        return context

    def _mark_samples(
        self,
        sample_ids: Sequence[str],
        state: str,
        *,
        error: str | None = None,
        preserve_attempts: bool = False,
        shard_by_sample: Mapping[str, str] | None = None,
    ) -> None:
        if not sample_ids:
            return
        now = _utc_now()
        with self._ledger_lock:
            ledger = self._load_ledger_unlocked()
            samples = ledger.setdefault("samples", {})
            for sample_id in sample_ids:
                key = str(sample_id)
                previous = samples.get(key) if isinstance(samples.get(key), Mapping) else {}
                attempts = int(previous.get("attempts", 0) or 0)
                if state == UPLOAD_PENDING and not preserve_attempts:
                    attempts += 1
                entry = {
                    "sample_id": key,
                    "sync_state": state,
                    "attempts": attempts,
                    "updated_at": now,
                }
                shard_id = (
                    str(shard_by_sample[key])
                    if shard_by_sample and key in shard_by_sample
                    else previous.get("shard_id")
                )
                if shard_id:
                    entry["shard_id"] = shard_id
                if state in {UPLOAD_UPLOADED, UPLOAD_COMMITTED}:
                    entry["uploaded_at"] = previous.get("uploaded_at") or now
                elif previous.get("uploaded_at"):
                    entry["uploaded_at"] = previous.get("uploaded_at")
                if state == UPLOAD_COMMITTED:
                    entry["committed_at"] = now
                elif previous.get("committed_at"):
                    entry["committed_at"] = previous.get("committed_at")
                if error:
                    entry["last_error"] = str(error)
                elif previous.get("last_error") and state != UPLOAD_COMMITTED:
                    entry["last_error"] = previous.get("last_error")
                samples[key] = entry
            ledger["updated_at"] = now
            self._write_ledger_unlocked(ledger)

    def _retryable_count(self) -> int:
        with self._ledger_lock:
            samples = self._load_ledger_unlocked().get("samples", {})
            return sum(
                1
                for entry in samples.values()
                if isinstance(entry, Mapping)
                and str(entry.get("sync_state") or entry.get("state")) in _RETRYABLE_STATES
            )

    @staticmethod
    def _record_context_key(record: StoredSampleRecord) -> tuple[str, str, str]:
        return (
            str(getattr(record, "model_id", "") or ""),
            str(getattr(record, "front_version", "") or "0"),
            str(getattr(record, "split_config_id", "") or ""),
        )

    def _retryable_records_by_context(self) -> dict[tuple[str, str, str], list[StoredSampleRecord]]:
        provider_split = ""
        if callable(self._context_provider):
            try:
                provider_split = str(
                    dict(self._context_provider() or {}).get("split_config_id") or ""
                ).strip()
            except Exception as exc:
                logger.warning(
                    "High-quality sync context provider failed: {}.",
                    safe_error_summary(exc),
                )
        with self._ledger_lock:
            samples = dict(self._load_ledger_unlocked().get("samples", {}))
        groups: dict[tuple[str, str, str], list[StoredSampleRecord]] = {}
        stale_sample_ids: list[str] = []
        for record in self.sample_store.list_records(quality_bucket=HIGH_QUALITY):
            entry = samples.get(str(record.sample_id)) or {}
            state = str(entry.get("sync_state") or entry.get("state") or "")
            if state not in _RETRYABLE_STATES:
                continue
            record_split = str(getattr(record, "split_config_id", "") or "").strip()
            if provider_split and record_split and record_split != provider_split:
                stale_sample_ids.append(str(record.sample_id))
                continue
            groups.setdefault(self._record_context_key(record), []).append(record)
        if stale_sample_ids:
            self._mark_samples(
                stale_sample_ids,
                UPLOAD_STALE_SPLIT,
                error=(
                    "sample feature split_config_id no longer matches the active "
                    f"fixed split plan {provider_split!r}"
                ),
            )
        return groups

    def _retryable_full_group_count(self) -> int:
        groups = self._retryable_records_by_context()
        return max(
            ((len(records) // self.shard_size) * self.shard_size for records in groups.values()),
            default=0,
        )

    def _select_retryable_record_groups(
        self,
        *,
        include_partial: bool,
    ) -> list[list[StoredSampleRecord]]:
        groups = self._retryable_records_by_context()
        if include_partial:
            return [records for records in groups.values() if records]
        record_groups: list[list[StoredSampleRecord]] = []
        for records in groups.values():
            full_count = (len(records) // self.shard_size) * self.shard_size
            if full_count:
                record_groups.append(records[:full_count])
        return record_groups


def _reply_succeeded(reply: object | None) -> bool:
    if reply is None:
        return False
    for field_name in ("success", "accepted", "committed"):
        if hasattr(reply, field_name):
            return bool(getattr(reply, field_name))
    status = str(getattr(reply, "status", "") or "").upper()
    return status not in {"FAILED", "ERROR", "REJECTED"}


def _reply_message(reply: object | None) -> str:
    if reply is None:
        return "sync_samples returned no reply"
    return str(getattr(reply, "message", "") or getattr(reply, "status", "") or "")


def _reply_sample_ids(
    reply: object | None,
    field_name: str,
    *,
    default: Sequence[str],
) -> list[str]:
    if reply is None or not hasattr(reply, field_name):
        return [str(sample_id) for sample_id in default]
    value = getattr(reply, field_name)
    return [str(sample_id) for sample_id in (value or [])]
