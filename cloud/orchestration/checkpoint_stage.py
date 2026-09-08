from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from collections.abc import Mapping, Sequence

import torch
from loguru import logger

import model_management.model_zoo as model_zoo
from cloud.model_update import serialize_model_update
from cloud.orchestration.fixed_split_dependencies import (
    _normalize_model_version,
    _read_json_file,
    _rfdetr_num_classes_from_metadata,
    _validate_rfdetr_weights_match_metadata,
)
from cloud.training.proxy_metadata import (
    coerce_positive_int as _coerce_positive_int,
)
from cloud.training.proxy_metadata import (
    infer_yolo_model_num_classes as _infer_yolo_model_num_classes,
)
from cloud.training.proxy_metadata import (
    looks_like_fused_ultralytics_state_dict as _looks_like_fused_ultralytics_state_dict,
)
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from model_management.model_info import model_lib
from model_management.split_model_adapters import get_split_runtime_model


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointStage:
    def serialize_encoded_update(
        self,
        model: torch.nn.Module,
        *,
        model_name: str,
        checkpoint_path: str,
        weights_metadata: Mapping[str, object] | None,
        metadata_path: str | None,
    ) -> str:
        return base64.b64encode(
            serialize_model_update(
                model,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                weights_metadata=weights_metadata,
                metadata_path=metadata_path,
            )
        ).decode("utf-8")


class CheckpointStageMixin:
    def _edge_weights_path(self, model_name: str, *, edge_id: int | str | None = None) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name).strip())
        if edge_id is None:
            return os.path.join(self.weight_folder, f"tmp_edge_model_{safe_name}.pth")
        safe_edge = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(edge_id).strip()) or "unknown"
        return os.path.join(
            self.weight_folder,
            f"tmp_edge_model_{safe_name}_edge_{safe_edge}.pth",
        )

    @staticmethod
    def _normalize_model_name_for_lookup(model_name: str) -> str:
        return str(model_name).strip().lower().replace("-", "_")

    @classmethod
    def _known_model_name_for_weights_path(cls, weights_path: str) -> str | None:
        artifact_name = os.path.basename(str(weights_path)).strip().lower()
        if not artifact_name:
            return None
        for model_name, model_info in model_lib.items():
            known_artifact = os.path.basename(str(model_info.get("model_path", ""))).strip().lower()
            if artifact_name == known_artifact:
                return cls._normalize_model_name_for_lookup(model_name)
        return None

    def _configured_weights_path_for_model(
        self,
        model_name: str,
        *,
        warn: bool = True,
    ) -> str:
        configured_weights = str(getattr(self.config, "weights_path", "") or "").strip()
        if not configured_weights:
            return ""

        configured_model = self._known_model_name_for_weights_path(configured_weights)
        requested_model = self._normalize_model_name_for_lookup(model_name)
        if configured_model is not None and configured_model != requested_model:
            if warn:
                logger.warning(
                    "[CL] Ignoring configured weights for model={} because the requested "
                    "edge model is={}; using native weights.",
                    configured_model,
                    requested_model,
                )
                log_diagnostic_debug(
                    self,
                    "[CL] ignored configured weights diagnostics",
                    lambda: {"weights_path": configured_weights},
                )
            return ""
        return configured_weights

    def _edge_weights_metadata_path(
        self,
        model_name: str,
        *,
        edge_id: int | str | None = None,
    ) -> str:
        weights_path = self._edge_weights_path(model_name, edge_id=edge_id)
        return f"{os.path.splitext(weights_path)[0]}.meta.json"

    def _resolve_fixed_split_model_name(self, manifest: Mapping[str, object]) -> str:
        model_meta = dict(manifest.get("model", {}) or {})
        bundle_model_id = str(
            model_meta.get("model_id") or manifest.get("model_id", "") or ""
        ).strip()
        if bundle_model_id and bundle_model_id != self.edge_model_name:
            logger.warning(
                "[FixedSplitCL] Using bundle model {} instead of configured "
                "server.edge_model_name {} for this retrain round.",
                bundle_model_id,
                self.edge_model_name,
            )
            return bundle_model_id
        return bundle_model_id or self.edge_model_name

    def _native_training_source_label(self, model_name: str) -> str:
        configured_weights = self._configured_weights_path_for_model(
            model_name,
            warn=False,
        )
        if configured_weights:
            return "configured"
        return "pretrained" if model_name in model_lib else "randomly initialised"

    def _detection_model_build_kwargs(
        self,
        model_name: str,
        *,
        runtime_input_tensor_shape: tuple[int, ...] | list[int] | None = None,
        model_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        model_family = model_zoo.get_model_family(str(model_name))
        build_kwargs: dict[str, object] = {}

        if model_family == "rfdetr":
            manifest_num_classes = _rfdetr_num_classes_from_metadata(model_metadata)
            if manifest_num_classes is not None:
                build_kwargs["num_classes"] = manifest_num_classes
                logger.info(
                    "[CL] Using {} RF-DETR logits from edge model metadata for {}.",
                    manifest_num_classes,
                    model_name,
                )

        if model_family != "tinynext":
            return build_kwargs

        input_size = int(getattr(self.config, "tinynext_input_size", 320))
        shape = list(runtime_input_tensor_shape or [])
        if len(shape) >= 4:
            height = int(shape[-2])
            width = int(shape[-1])
            if height <= 0 or width <= 0:
                raise RuntimeError(
                    f"Invalid TinyNeXt runtime input shape: {runtime_input_tensor_shape!r}"
                )
            if height != width:
                raise RuntimeError(
                    "TinyNeXt SSD split runtime expects a square transformed input, "
                    f"got {runtime_input_tensor_shape!r}."
                )
            input_size = height
        build_kwargs["tinynext_input_size"] = input_size
        return build_kwargs

    def _build_native_training_model(
        self,
        model_name: str,
        *,
        runtime_input_tensor_shape: tuple[int, ...] | list[int] | None = None,
        model_metadata: Mapping[str, object] | None = None,
    ) -> torch.nn.Module:
        source_label = self._native_training_source_label(model_name)
        configured_weights = self._configured_weights_path_for_model(model_name)
        build_kwargs = {
            "pretrained": source_label == "pretrained",
            "device": self.device,
        }
        build_kwargs.update(
            self._detection_model_build_kwargs(
                model_name,
                runtime_input_tensor_shape=runtime_input_tensor_shape,
                model_metadata=model_metadata,
            )
        )
        if configured_weights:
            # Validate and use configured weights path
            if not os.path.exists(configured_weights):
                logger.error(
                    "[CL] Configured weights artifact is unavailable for model={}.",
                    model_name,
                )
                log_diagnostic_debug(
                    self,
                    "[CL] unavailable configured weights diagnostics",
                    lambda: {"weights_path": configured_weights},
                )
            else:
                _validate_rfdetr_weights_match_metadata(
                    model_name=model_name,
                    weights_path=configured_weights,
                    model_metadata=model_metadata,
                    device=self.device,
                )
                build_kwargs["weights_path"] = configured_weights
                logger.info(
                    "[CL] Building model={} with configured weights.",
                    model_name,
                )
                log_diagnostic_debug(
                    self,
                    "[CL] configured build weights diagnostics",
                    lambda: {"weights_path": configured_weights},
                )
        elif source_label == "pretrained":
            try:
                artifact_path = model_zoo.ensure_local_model_artifact(model_name)
            except Exception as exc:
                logger.warning(
                    "[CL] Failed to resolve native weights for model={}: {}.",
                    model_name,
                    safe_error_summary(exc),
                )
                log_diagnostic_debug(
                    self,
                    "[CL] native weights resolution diagnostics",
                    lambda error=exc: {"error": repr(error)},
                )
            else:
                if artifact_path.exists():
                    _validate_rfdetr_weights_match_metadata(
                        model_name=model_name,
                        weights_path=str(artifact_path),
                        model_metadata=model_metadata,
                        device=self.device,
                    )
                    build_kwargs["weights_path"] = str(artifact_path)
        return model_zoo.build_detection_model(model_name, **build_kwargs)

    def _read_edge_weights_metadata(
        self,
        model_name: str,
        *,
        edge_id: int | str | None = None,
    ) -> dict[str, object]:
        return _read_json_file(self._edge_weights_metadata_path(model_name, edge_id=edge_id))

    def _require_matching_edge_weights_metadata(
        self,
        *,
        model_name: str,
        edge_id: int | str,
        bundle_model_version: str,
    ) -> dict[str, object]:
        metadata_path = self._edge_weights_metadata_path(model_name, edge_id=edge_id)
        metadata = self._read_edge_weights_metadata(model_name, edge_id=edge_id)
        if not metadata:
            raise RuntimeError(
                "[FixedSplitCL] Missing persisted edge checkpoint metadata for "
                f"edge {edge_id} model {model_name} at {metadata_path}; "
                f"cannot resume bundle model_version={bundle_model_version}."
            )
        metadata_edge_id = str(metadata.get("edge_id", "")).strip()
        if metadata_edge_id and metadata_edge_id != str(edge_id):
            raise RuntimeError(
                "[FixedSplitCL] Edge checkpoint metadata mismatch for "
                f"edge {edge_id} model {model_name}: metadata edge_id={metadata_edge_id!r}."
            )
        metadata_model_name = str(metadata.get("model_name", "")).strip()
        if metadata_model_name and metadata_model_name != str(model_name):
            raise RuntimeError(
                "[FixedSplitCL] Edge checkpoint metadata mismatch for "
                f"edge {edge_id}: expected model {model_name} but found {metadata_model_name!r}."
            )
        checkpoint_model_version = _normalize_model_version(
            metadata.get("checkpoint_model_version", "0"),
            field_name="checkpoint model version",
        )
        if checkpoint_model_version != str(bundle_model_version):
            raise RuntimeError(
                "[FixedSplitCL] Persisted edge checkpoint version mismatch for "
                f"edge {edge_id} model {model_name}: checkpoint_model_version="
                f"{checkpoint_model_version}, bundle model_version={bundle_model_version}."
            )
        return metadata

    def _load_edge_training_model(
        self,
        *,
        model_name: str | None = None,
        edge_id: int | str | None = None,
        cache_policy: str = "native_only",
        runtime_input_tensor_shape: tuple[int, ...] | list[int] | None = None,
        model_metadata: Mapping[str, object] | None = None,
    ) -> torch.nn.Module:
        model_name = str(model_name or self.edge_model_name)
        cache_policy = str(cache_policy or "native_only").strip().lower()
        if cache_policy not in {"native_only", "edge_only"}:
            raise ValueError(f"Unsupported cache policy: {cache_policy!r}")
        edge_weights = self._edge_weights_path(model_name, edge_id=edge_id)

        if cache_policy == "native_only":
            tmp_model = self._build_native_training_model(
                model_name,
                runtime_input_tensor_shape=runtime_input_tensor_shape,
                model_metadata=model_metadata,
            )
            tmp_model.to(self.device)
            get_split_runtime_model(tmp_model).eval()
            return tmp_model

        if not os.path.exists(edge_weights):
            raise RuntimeError(
                f"[CL] Required edge-scoped cache for {model_name} is missing at {edge_weights}."
            )

        try:
            state = torch.load(edge_weights, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                "[CL] Failed to read required edge-scoped cache for "
                f"{model_name} from {edge_weights}: {exc}"
            ) from exc

        if str(model_name).lower().startswith(
            "rfdetr_"
        ) and not model_zoo.has_compatible_rfdetr_cache_state(state):
            raise RuntimeError(
                "[CL] Required edge-scoped RF-DETR weights use an unsupported cache format "
                f"at {edge_weights}."
            )
        if model_zoo.is_wrapper_model(model_name) and _looks_like_fused_ultralytics_state_dict(
            state
        ):
            raise RuntimeError(
                "[CL] Required edge-scoped wrapper weights look like a fused Ultralytics "
                f"state_dict at {edge_weights}."
            )

        build_kwargs = self._detection_model_build_kwargs(
            model_name,
            runtime_input_tensor_shape=runtime_input_tensor_shape,
            model_metadata=model_metadata,
        )
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family in {"yolo", "rtdetr"}:
            cache_num_classes = model_zoo.infer_ultralytics_state_dict_num_classes(state)
            cache_metadata = self._read_edge_weights_metadata(
                model_name,
                edge_id=edge_id,
            )
            if cache_num_classes is None:
                cache_num_classes = _coerce_positive_int(
                    cache_metadata.get("ultralytics_head_num_classes")
                )
                if cache_num_classes is None and model_family == "yolo":
                    cache_num_classes = _coerce_positive_int(
                        cache_metadata.get("yolo_head_num_classes")
                    )
            if cache_num_classes is not None and cache_num_classes != 80:
                build_kwargs["num_classes"] = cache_num_classes
                logger.info(
                    "[CL] Inferred {} {} class(es) from edge-scoped weights.",
                    cache_num_classes,
                    model_name,
                )
        elif model_family == "rfdetr":
            cache_num_classes = model_zoo.infer_rfdetr_state_dict_num_classes(state)
            if cache_num_classes is None:
                cache_num_classes = _rfdetr_num_classes_from_metadata(
                    self._read_edge_weights_metadata(model_name, edge_id=edge_id)
                )
            if cache_num_classes is not None and cache_num_classes != 91:
                build_kwargs["num_classes"] = cache_num_classes
                logger.info(
                    "[CL] Inferred {} RF-DETR logits from edge-scoped {} weights.",
                    cache_num_classes,
                    model_name,
                )
        elif model_family == "tinynext":
            cache_num_classes = model_zoo.infer_tinynext_state_dict_num_classes(state)
            if cache_num_classes is not None and cache_num_classes != 91:
                build_kwargs["num_classes"] = cache_num_classes
                logger.info(
                    "[CL] Inferred {} TinyNeXt SSD class logits from edge-scoped {} weights.",
                    cache_num_classes,
                    model_name,
                )

        tmp_model = model_zoo.build_detection_model(
            model_name,
            pretrained=False,
            device=self.device,
            **build_kwargs,
        )
        try:
            load_result = tmp_model.load_state_dict(state, strict=False)
        except Exception as exc:
            raise RuntimeError(
                "[CL] Failed to load required edge-scoped cache for "
                f"{model_name} from {edge_weights}: {exc}"
            ) from exc
        missing_keys = list(getattr(load_result, "missing_keys", ()) or ())
        unexpected_keys = list(getattr(load_result, "unexpected_keys", ()) or ())
        logger.info(
            "[CL] Loaded edge-scoped {} weights: missing_keys={} unexpected_keys={}.",
            model_name,
            len(missing_keys),
            len(unexpected_keys),
        )
        log_diagnostic_debug(
            self,
            "[CL] edge-scoped weights diagnostics",
            lambda: {"weights_path": edge_weights},
        )
        tmp_model.to(self.device)
        get_split_runtime_model(tmp_model).eval()
        return tmp_model

    def _build_checkpoint_weights_metadata(
        self,
        *,
        edge_id: int | str,
        model_name: str,
        model: torch.nn.Module,
        checkpoint_model_version: str,
        source_base_model_version: str,
        runtime_input_tensor_shape: Sequence[int] | None,
    ) -> dict[str, object]:
        weights_metadata: dict[str, object] = {
            "edge_id": int(edge_id),
            "model_name": str(model_name),
            "checkpoint_model_version": str(checkpoint_model_version),
            "source_base_model_version": str(source_base_model_version),
            "updated_at_ms": int(time.time() * 1000),
        }
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family in {"yolo", "rtdetr"}:
            yolo_num_classes = _infer_yolo_model_num_classes(model)
            if yolo_num_classes is None:
                yolo_num_classes = model_zoo.infer_ultralytics_state_dict_num_classes(
                    model.state_dict()
                )
            if yolo_num_classes is not None:
                weights_metadata["ultralytics_head_num_classes"] = int(yolo_num_classes)
                if model_family == "yolo":
                    weights_metadata["yolo_head_num_classes"] = int(yolo_num_classes)
        if model_family == "rfdetr":
            rfdetr_num_classes = model_zoo.infer_rfdetr_state_dict_num_classes(model.state_dict())
            if rfdetr_num_classes is None:
                rfdetr_num_classes = _coerce_positive_int(getattr(model, "num_classes", None))
            if rfdetr_num_classes is not None:
                weights_metadata["rfdetr_head_num_classes"] = int(rfdetr_num_classes)
                weights_metadata["num_classes"] = int(rfdetr_num_classes)
                logger.info(
                    "[FixedSplitCL] Serializing RF-DETR model with {} classes for edge {}.",
                    rfdetr_num_classes,
                    edge_id,
                )
            else:
                logger.warning(
                    "[FixedSplitCL] Could not infer RF-DETR num_classes from model for edge {}!",
                    edge_id,
                )
        if (
            runtime_input_tensor_shape
            and len(runtime_input_tensor_shape) >= 4
            and model_family == "tinynext"
        ):
            weights_metadata["tinynext_input_size"] = int(runtime_input_tensor_shape[-1])
            tinynext_num_classes = model_zoo.infer_tinynext_state_dict_num_classes(
                model.state_dict()
            )
            if tinynext_num_classes is not None:
                weights_metadata["tinynext_head_num_classes"] = int(tinynext_num_classes)
        return weights_metadata

    def _serialise_model_bytes(
        self,
        model: torch.nn.Module,
        *,
        model_name: str | None = None,
        edge_id: int | str | None = None,
        weights_metadata: Mapping[str, object] | None = None,
    ) -> bytes:
        resolved_model_name = model_name or self.edge_model_name
        if weights_metadata is not None and edge_id is None:
            raise ValueError("weights metadata requires an edge_id")
        checkpoint_path = self._edge_weights_path(
            resolved_model_name,
            edge_id=edge_id,
        )
        metadata_path = (
            self._edge_weights_metadata_path(
                resolved_model_name,
                edge_id=edge_id,
            )
            if weights_metadata is not None
            else None
        )
        payload = serialize_model_update(
            model,
            model_name=str(resolved_model_name),
            checkpoint_path=checkpoint_path,
            weights_metadata=weights_metadata,
            metadata_path=metadata_path,
        )
        source_version = (
            dict(weights_metadata or {}).get("source_base_model_version")
            if weights_metadata is not None
            else ""
        )
        checkpoint_version = (
            dict(weights_metadata or {}).get("checkpoint_model_version")
            if weights_metadata is not None
            else ""
        )
        logger.info(
            "[FixedSplitCL] Checkpoint serialized: model={} source_version={} "
            "checkpoint_version={} size={:.1f}MB.",
            resolved_model_name,
            source_version,
            checkpoint_version,
            len(payload or b"") / (1024.0 * 1024.0),
        )
        log_diagnostic_debug(
            self,
            "[FixedSplitCL] checkpoint serialization diagnostics",
            lambda: {
                "checkpoint_path": checkpoint_path,
                "metadata_path": metadata_path,
                "sha1": file_sha1(checkpoint_path),
            },
        )
        return payload
