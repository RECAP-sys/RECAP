from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from cloud.orchestration.fixed_split_dependencies import (
    _json_fingerprint,
    _normalize_model_version,
    _sanitize_cache_segment,
    _stable_json_dumps,
)
from cloud.orchestration.recent_training_window import RecentTrainingWindowStore
from common.logging_sanitizer import log_diagnostic_debug
from model_management.split_contract import contract_path


def stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_fingerprint(payload: object) -> str:
    return hashlib.sha1(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def sanitize_cache_segment(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned or "unknown"


def read_json_file(path: str) -> dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def manifest_model_metadata(manifest: Mapping[str, object]) -> dict[str, object]:
    model_meta = manifest.get("model")
    metadata = dict(model_meta) if isinstance(model_meta, Mapping) else {}
    for manifest_key, metadata_key in (
        ("model_id", "model_id"),
        ("model_version", "model_version"),
        ("model_num_classes", "num_classes"),
        ("model_label_schema", "label_schema"),
    ):
        value = manifest.get(manifest_key)
        if value is not None and metadata_key not in metadata:
            metadata[metadata_key] = value
    return metadata


def training_window_manifest_context(manifest: Mapping[str, object]) -> dict[str, object]:
    model_meta = dict(manifest.get("model", {}) or {})
    split_plan = dict(manifest.get("split_plan", {}) or {})
    runtime_contract = dict(
        manifest.get("runtime_contract")
        if isinstance(manifest.get("runtime_contract"), Mapping)
        else split_plan.get("runtime_contract")
        if isinstance(split_plan.get("runtime_contract"), Mapping)
        else {}
    )
    return {
        "model_id": str(manifest.get("model_id") or model_meta.get("model_id", "") or ""),
        "front_version": str(
            manifest.get("front_version") or split_plan.get("front_version") or "0"
        ),
        "split_config_id": str(
            manifest.get("split_config_id") or split_plan.get("split_config_id", "") or ""
        ),
        "feature_layout_id": str(runtime_contract.get("feature_layout_id") or ""),
        "boundary_tensor_labels": list(runtime_contract.get("boundary_tensor_labels", []) or []),
        "canonical_split_key": str(
            manifest.get("canonical_split_key")
            or split_plan.get("canonical_split_key")
            or runtime_contract.get("logical_split_id")
            or ""
        ),
        "edge_split_id": str(
            manifest.get("edge_split_id")
            or split_plan.get("edge_split_id")
            or runtime_contract.get("logical_split_id")
            or ""
        ),
        "input_tensor_shape": list(
            runtime_contract.get("input_tensor_shape")
            or manifest.get("input_tensor_shape")
            or split_plan.get("input_tensor_shape", [])
            or []
        ),
        "input_resize_mode": str(
            runtime_contract.get("input_resize_mode")
            or manifest.get("input_resize_mode")
            or split_plan.get("input_resize_mode")
            or "direct_resize"
        ),
        "runtime_contract": runtime_contract,
    }


def manifest_edge_session_id(manifest: Mapping[str, object]) -> str:
    return str(
        manifest.get("edge_session_id")
        or manifest.get("client_session_id")
        or manifest.get("session_id")
        or ""
    ).strip()


def manifest_model_version(
    manifest: Mapping[str, object],
    *,
    fallback: object = "",
) -> str:
    model_meta = manifest.get("model")
    model_meta = dict(model_meta) if isinstance(model_meta, Mapping) else {}
    return str(
        manifest.get("model_version") or model_meta.get("model_version") or fallback or ""
    ).strip()


def normalize_model_version(value: object, *, field_name: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "0"
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer string, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")
    return str(number)


def increment_model_version(value: object, *, field_name: str) -> str:
    return str(int(normalize_model_version(value, field_name=field_name)) + 1)


@dataclass(frozen=True)
class RequestContext:
    edge_id: int | str
    model_id: str
    model_version: str
    request_id: str
    workspace: str
    manifest_metadata: dict[str, object]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RequestContextMixin:
    @staticmethod
    def _training_window_manifest_context(
        manifest: Mapping[str, object],
    ) -> dict[str, object]:
        model_meta = dict(manifest.get("model", {}) or {})
        split_plan = dict(manifest.get("split_plan", {}) or {})
        runtime_contract = dict(
            manifest.get("runtime_contract")
            if isinstance(manifest.get("runtime_contract"), Mapping)
            else split_plan.get("runtime_contract")
            if isinstance(split_plan.get("runtime_contract"), Mapping)
            else {}
        )
        return {
            "model_id": str(manifest.get("model_id") or model_meta.get("model_id", "") or ""),
            "front_version": str(
                manifest.get("front_version") or split_plan.get("front_version") or "0"
            ),
            "split_config_id": str(
                manifest.get("split_config_id") or split_plan.get("split_config_id", "") or ""
            ),
            "feature_layout_id": str(runtime_contract.get("feature_layout_id") or ""),
            "boundary_tensor_labels": list(
                runtime_contract.get("boundary_tensor_labels", []) or []
            ),
            "canonical_split_key": str(
                manifest.get("canonical_split_key")
                or split_plan.get("canonical_split_key")
                or runtime_contract.get("logical_split_id")
                or ""
            ),
            "edge_split_id": str(
                manifest.get("edge_split_id")
                or split_plan.get("edge_split_id")
                or runtime_contract.get("logical_split_id")
                or ""
            ),
            "input_tensor_shape": list(
                runtime_contract.get("input_tensor_shape")
                or manifest.get("input_tensor_shape")
                or split_plan.get("input_tensor_shape", [])
                or []
            ),
            "input_resize_mode": str(
                runtime_contract.get("input_resize_mode")
                or manifest.get("input_resize_mode")
                or split_plan.get("input_resize_mode")
                or "direct_resize"
            ),
            "runtime_contract": runtime_contract,
        }

    def _recent_training_window_path(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> str:
        context = self._training_window_manifest_context(manifest)
        layout_key = str(context.get("feature_layout_id", "") or "").strip()
        split_key = (
            f"feature_layout_{layout_key}"
            if layout_key
            else str(context.get("split_config_id", "") or "").strip()
        )
        if not split_key:
            split_key = _json_fingerprint(
                {
                    "canonical_split_key": context.get("canonical_split_key"),
                    "boundary_tensor_labels": list(context.get("boundary_tensor_labels", []) or []),
                }
            )[:16]
        return os.path.join(
            self.recent_training_window_root,
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(context.get("model_id") or "unknown_model"),
            f"front_version_{_sanitize_cache_segment(context.get('front_version') or '0')}",
            _sanitize_cache_segment(split_key),
        )

    def _recent_training_window_model_root(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> str:
        context = self._training_window_manifest_context(manifest)
        return os.path.join(
            self.recent_training_window_root,
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(context.get("model_id") or "unknown_model"),
            f"front_version_{_sanitize_cache_segment(context.get('front_version') or '0')}",
        )

    def _recent_training_window_for_manifest(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> RecentTrainingWindowStore:
        return RecentTrainingWindowStore(
            self._recent_training_window_path(edge_id=edge_id, manifest=manifest),
            max_samples=(
                int(self.training_frame_count)
                * max(1, int(getattr(self, "training_replay_retention_multiplier", 1)))
            ),
        )

    @staticmethod
    def _manifest_edge_session_id(manifest: Mapping[str, object]) -> str:
        return str(
            manifest.get("edge_session_id")
            or manifest.get("client_session_id")
            or manifest.get("session_id")
            or ""
        ).strip()

    @staticmethod
    def _manifest_model_version(
        manifest: Mapping[str, object],
        *,
        fallback: object = "",
    ) -> str:
        model_meta = manifest.get("model")
        model_meta = dict(model_meta) if isinstance(model_meta, Mapping) else {}
        return str(
            manifest.get("model_version") or model_meta.get("model_version") or fallback or ""
        ).strip()

    def _remove_reset_path_if_safe(
        self,
        *,
        path: str,
        root: str,
        label: str,
    ) -> bool:
        abs_path = os.path.abspath(str(path or ""))
        abs_root = os.path.abspath(str(root or ""))
        if not abs_path or not abs_root:
            return False
        if abs_path == abs_root or not abs_path.startswith(abs_root + os.sep):
            logger.warning(
                "[FixedSplitCL][InitialReset] Skipping unsafe reset target: kind={}.",
                label,
            )
            log_diagnostic_debug(
                self,
                "[FixedSplitCL][InitialReset] unsafe reset diagnostics",
                lambda: {"root_path": abs_root, "target_path": abs_path},
            )
            return False
        if not os.path.exists(abs_path):
            return False
        if os.path.isdir(abs_path):
            shutil.rmtree(abs_path, ignore_errors=True)
        else:
            os.remove(abs_path)
        return True

    def _reset_initial_cloud_state_if_needed(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
        model_name: str,
        fallback_model_version: object = "",
        allow_without_session: bool = False,
    ) -> None:
        model_version = self._manifest_model_version(
            manifest,
            fallback=fallback_model_version,
        )
        if not model_version:
            return
        try:
            is_initial_model = (
                _normalize_model_version(
                    model_version,
                    field_name="initial reset model_version",
                )
                == "0"
            )
        except Exception:
            is_initial_model = False
        if not is_initial_model:
            return

        context = self._training_window_manifest_context(manifest)
        model_id = str(context.get("model_id") or model_name or self.edge_model_name)
        split_config_id = str(context.get("split_config_id") or "").strip()
        front_version = str(context.get("front_version") or "0")
        edge_session_id = self._manifest_edge_session_id(manifest)
        if not edge_session_id and not allow_without_session:
            return

        base_reset_key = _stable_json_dumps(
            {
                "edge_id": str(edge_id),
                "model_id": model_id,
                "front_version": front_version,
                "split_config_id": split_config_id,
            }
        )
        edge_segment = f"edge_{_sanitize_cache_segment(edge_id)}"
        model_segment = _sanitize_cache_segment(model_id)
        front_segment = f"front_version_{_sanitize_cache_segment(front_version)}"
        window_front_dir = os.path.join(
            self.recent_training_window_root,
            edge_segment,
            model_segment,
            front_segment,
        )
        stale_contract_dir = os.path.join(
            self.split_contract_root,
            "stale",
            edge_segment,
            model_segment,
        )
        deleted_labels: list[str] = []

        with self._initial_state_reset_lock:
            reset_sessions = self._initial_state_reset_sessions
            if not isinstance(reset_sessions, dict):
                reset_sessions = {str(key): "" for key in reset_sessions if str(key)}
                self._initial_state_reset_sessions = reset_sessions
            previous_session = str(reset_sessions.get(base_reset_key, ""))
            if previous_session:
                if not edge_session_id or previous_session == edge_session_id:
                    return
            elif base_reset_key in reset_sessions:
                if edge_session_id:
                    reset_sessions[base_reset_key] = edge_session_id
                return
            reset_paths = [
                (
                    window_front_dir,
                    self.recent_training_window_root,
                    "recent_training_window",
                ),
                (stale_contract_dir, self.split_contract_root, "stale_contracts"),
            ]
            if split_config_id:
                reset_paths.append(
                    (
                        contract_path(
                            self.split_contract_root,
                            edge_id=edge_id,
                            model_id=model_id,
                            split_config_id=split_config_id,
                        ),
                        self.split_contract_root,
                        "split_contract",
                    )
                )
            for path, root, label in reset_paths:
                if self._remove_reset_path_if_safe(path=path, root=root, label=label):
                    deleted_labels.append(label)
            reset_sessions[base_reset_key] = edge_session_id

        logger.info(
            "[FixedSplitCL][InitialReset] edge={} model={} front_version={} cleared={}.",
            edge_id,
            model_id,
            front_version,
            deleted_labels,
        )
        log_diagnostic_debug(
            self,
            "[FixedSplitCL][InitialReset] diagnostics",
            lambda: {
                "split_config_id": split_config_id,
                "session_id": edge_session_id,
                "base_reset_key": base_reset_key,
                "window_front_dir": window_front_dir,
                "stale_contract_dir": stale_contract_dir,
            },
        )
