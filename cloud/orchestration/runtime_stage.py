from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from loguru import logger

from cloud.contracts import validate_fixed_split_plan
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from model_management.universal_model_split import load_cached_split_batches

FIXED_SPLIT_DYNAMIC_BATCH = (1, 64)
FIXED_SPLIT_DYNAMIC_BATCH_MIN = FIXED_SPLIT_DYNAMIC_BATCH[0]
FIXED_SPLIT_DYNAMIC_BATCH_MAX = FIXED_SPLIT_DYNAMIC_BATCH[1]


def _json_fingerprint(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def fixed_split_plan_runtime_contract(split_plan: Mapping[str, object]) -> dict[str, object]:
    return validate_fixed_split_plan(split_plan)


def fixed_split_boundary_from_plan(split_plan: Mapping[str, object]) -> str:
    boundary = fixed_split_plan_runtime_contract(split_plan).get("logical_split_id") or "auto"
    boundary = str(boundary)
    if boundary != "auto" and not boundary.startswith("after:"):
        boundary = f"after:{boundary}"
    return boundary


def fixed_split_dynamic_batch_from_plan(
    split_plan: Mapping[str, object],
    default: tuple[int, int] | None,
) -> tuple[int, int] | None:
    raw = split_plan.get("dynamic_batch")
    if raw is None:
        return default
    try:
        lower, upper = list(raw)[:2]
    except (TypeError, ValueError):
        return default
    lower_int = max(1, int(lower))
    upper_int = max(lower_int, int(upper))
    return lower_int, upper_int


def fixed_split_trace_batch_mode_from_plan(split_plan: Mapping[str, object]) -> str:
    mode = str(split_plan.get("trace_batch_mode") or "").strip()
    return mode if mode in {"batch_1", "batch_gt1"} else "batch_1"


def fixed_split_trace_batch_size_from_plan(
    split_plan: Mapping[str, object],
    default: int,
) -> int:
    raw = split_plan.get("trace_batch_size")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def cloud_fixed_split_dynamic_batch(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
) -> tuple[int, int] | None:
    family = str(model_family or "").lower()
    default = (
        (1, FIXED_SPLIT_DYNAMIC_BATCH_MAX) if family == "rfdetr" else FIXED_SPLIT_DYNAMIC_BATCH
    )
    return fixed_split_dynamic_batch_from_plan(split_plan, default)


def cloud_fixed_split_trace_batch_mode(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
) -> str:
    del model_family
    return fixed_split_trace_batch_mode_from_plan(split_plan)


def cloud_fixed_split_trace_batch_size(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
    default: int,
) -> int:
    del model_family
    return fixed_split_trace_batch_size_from_plan(split_plan, default)


def fixed_split_validation_batches(
    *,
    model_family: str | None,
    trace_batch_size: int,
    runtime_batch_size: int | None,
    dynamic_batch: tuple[int, int] | None,
) -> list[int]:
    del model_family, trace_batch_size
    upper = (dynamic_batch or FIXED_SPLIT_DYNAMIC_BATCH)[1]
    training_batch = max(1, int(runtime_batch_size or 1))
    candidates = [1, min(int(upper), training_batch)]
    batches: list[int] = []
    for batch_size in candidates:
        batch = max(1, int(batch_size))
        if batch not in batches:
            batches.append(batch)
    return batches


def fixed_split_manifest_has_rebuildable_raw_samples(
    manifest: Mapping[str, object],
) -> bool:
    samples = [
        sample for sample in list(manifest.get("samples", []) or []) if isinstance(sample, Mapping)
    ]
    if not samples:
        return False
    return all(sample.get("raw_relpath") is not None for sample in samples)


def splitter_dynamic_batch_range(
    splitter: object | None,
) -> tuple[int, int] | None:
    sources = [
        getattr(splitter, "split_spec", None),
        getattr(getattr(splitter, "runtime", None), "split_spec", None),
    ]
    for split_spec in sources:
        dynamic_batch = getattr(split_spec, "dynamic_batch", None)
        if dynamic_batch is None:
            continue
        try:
            lower, upper = list(dynamic_batch)[:2]
            lower_int = max(1, int(lower))
            upper_int = max(lower_int, int(upper))
        except (TypeError, ValueError):
            continue
        return lower_int, upper_int
    return None


def splitter_dynamic_batch_min(splitter: object | None) -> int:
    dynamic_batch = splitter_dynamic_batch_range(splitter)
    return int(dynamic_batch[0]) if dynamic_batch is not None else 1


def cached_split_runtime_batch_size_candidates(
    *,
    configured_batch_size: int,
    trace_batch_size: int,
    dynamic_batch_min: int,
) -> list[int]:
    candidates: list[int] = []
    for value in (configured_batch_size, trace_batch_size, dynamic_batch_min, 1):
        value = max(1, int(value))
        if value not in candidates:
            candidates.append(value)
    return candidates


def negotiate_cached_split_runtime_batch_size(
    *,
    model_name: str,
    training_cache_path: str,
    all_sample_ids: Sequence[object],
    gt_annotations: Mapping[object, object],
    splitter: Any | None,
    candidate: Any,
    configured_batch_size: int,
    trace_batch_size: int,
    preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
) -> int:
    if splitter is None or not training_cache_path or not all_sample_ids:
        return max(1, int(configured_batch_size))

    candidates = cached_split_runtime_batch_size_candidates(
        configured_batch_size=int(configured_batch_size),
        trace_batch_size=int(trace_batch_size),
        dynamic_batch_min=splitter_dynamic_batch_min(splitter),
    )
    sample_ids = list(all_sample_ids)
    errors: dict[int, str] = {}
    for candidate_batch_size in candidates:
        smoke_indices = sample_ids[: max(1, min(len(sample_ids), int(candidate_batch_size)))]
        try:
            batches = load_cached_split_batches(
                cache_path=training_cache_path,
                all_indices=smoke_indices,
                annotations=gt_annotations,
                batch_size=int(candidate_batch_size),
                runtime=splitter,
                preloaded_records=preloaded_records,
            )
            if not batches:
                raise RuntimeError("cached split smoke validation prepared no batches")
            _batch_indices, boundary, _targets = batches[0]
            with torch.no_grad():
                splitter.cloud_forward(boundary, candidate=candidate)
        except Exception as exc:  # noqa: BLE001 - negotiation probes candidate sizes.
            errors[int(candidate_batch_size)] = str(exc) or type(exc).__name__
            logger.debug(
                "[FixedSplitCL] cached split runtime batch-size candidate failed "
                "(model_name={}, batch_size={}, error={}).",
                model_name,
                int(candidate_batch_size),
                errors[int(candidate_batch_size)],
            )
            continue

        logger.info(
            "[FixedSplitCL] negotiated cached split runtime batch size "
            "(model_name={}, configured_batch_size={}, selected_batch_size={}, candidates={}).",
            model_name,
            int(configured_batch_size),
            int(candidate_batch_size),
            candidates,
        )
        return int(candidate_batch_size)

    error_summary = ", ".join(
        f"batch_size={batch_size}: {safe_error_summary(error)}"
        for batch_size, error in errors.items()
    )
    logger.warning(
        "[FixedSplitCL] cached split runtime batch-size negotiation failed "
        "(model_name={}, candidates={}, errors={}).",
        model_name,
        candidates,
        error_summary,
    )
    raise RuntimeError(
        "Fixed-split cached runtime batch-size negotiation failed for all "
        f"candidate batch sizes ({error_summary})."
    )


__all__ = [
    "FIXED_SPLIT_DYNAMIC_BATCH",
    "FIXED_SPLIT_DYNAMIC_BATCH_MAX",
    "FIXED_SPLIT_DYNAMIC_BATCH_MIN",
    "cloud_fixed_split_dynamic_batch",
    "cloud_fixed_split_trace_batch_mode",
    "cloud_fixed_split_trace_batch_size",
    "fixed_split_boundary_from_plan",
    "fixed_split_dynamic_batch_from_plan",
    "fixed_split_manifest_has_rebuildable_raw_samples",
    "fixed_split_plan_runtime_contract",
    "fixed_split_trace_batch_mode_from_plan",
    "fixed_split_trace_batch_size_from_plan",
    "fixed_split_validation_batches",
    "cached_split_runtime_batch_size_candidates",
    "negotiate_cached_split_runtime_batch_size",
    "splitter_dynamic_batch_min",
    "splitter_dynamic_batch_range",
]

import os
import shutil
from datetime import datetime, timezone

import model_management.model_zoo as model_zoo
from cloud.feature_cache import ShardFeatureRefValidator
from cloud.orchestration.fixed_split_dependencies import (
    _sanitize_cache_segment,
    _stable_json_dumps,
)
from model_management.payload import BoundaryPayload
from model_management.split_contract import (
    SplitRuntimeContract,
    classify_contract_compatibility,
    contract_path,
    feature_layout_from_tensors,
)
from model_management.split_contract import (
    feature_layout_id as make_feature_layout_id,
)
from model_management.split_runtime import BoundaryPayloadCacheCodec
from model_management.universal_model_split import (
    UniversalModelSplitter,
)


class FixedSplitRuntimeContractMixin:
    @staticmethod
    def _feature_layout_summary_from_candidate(
        candidate: Mapping[str, object],
    ) -> dict[str, object]:
        feature_ref = candidate.get("feature_ref")
        ref_payload = dict(feature_ref) if isinstance(feature_ref, Mapping) else {}
        layout = dict(candidate.get("feature_layout") or {})
        shard_validation = None
        if not layout and ref_payload.get("storage_format"):
            shard_validation = ShardFeatureRefValidator().validate_feature_ref(
                ref_payload,
                {},
                allow_abi_compatible_migration=False,
                deep_validate_payload=False,
            )
            layout = dict(shard_validation.feature_layout or {})
        metadata = shard_validation.metadata if shard_validation is not None else None
        return {
            "sample_id": str(candidate.get("sample_id") or ""),
            "feature_layout_id": str(
                candidate.get("feature_layout_id")
                or ref_payload.get("feature_layout_id")
                or (metadata.feature_layout_id if metadata is not None else "")
                or (make_feature_layout_id(layout) if layout else "")
            ),
            "feature_layout": layout,
            "source_feature_layout_id": str(
                candidate.get("source_feature_layout_id")
                or ref_payload.get("feature_layout_id")
                or (metadata.feature_layout_id if metadata is not None else "")
                or ""
            ),
            "source_feature_abi_id": str(
                candidate.get("source_feature_abi_id")
                or ref_payload.get("feature_abi_id")
                or (metadata.feature_abi_id if metadata is not None else "")
                or ""
            ),
            "source_feature_schema_hash": str(
                candidate.get("source_feature_schema_hash")
                or (metadata.boundary_schema_hash if metadata is not None else "")
                or ""
            ),
            "source_feature_value_schema_hash": str(
                candidate.get("source_feature_value_schema_hash") or ""
            ),
            "source_feature_split_id": str(
                candidate.get("source_feature_split_id")
                or ref_payload.get("boundary_id")
                or (metadata.boundary_id if metadata is not None else "")
                or ""
            ),
            "source_feature_graph_signature": str(
                candidate.get("source_feature_graph_signature")
                or dict(ref_payload.get("metadata") or {}).get("graph_signature")
                or (
                    dict(metadata.metadata or {}).get("graph_signature")
                    if metadata is not None
                    else ""
                )
                or ""
            ),
        }

    @staticmethod
    def _layout_specs_match_ignoring_labels(
        actual: Mapping[str, Mapping[str, object]],
        expected: Mapping[str, Mapping[str, object]],
    ) -> bool:
        def normalised_specs(layout: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
            specs: list[dict[str, object]] = []
            for spec in dict(layout or {}).values():
                if not isinstance(spec, Mapping):
                    continue
                specs.append(
                    {
                        "dtype": str(spec.get("dtype") or ""),
                        "shape_without_batch": [
                            int(dim) for dim in list(spec.get("shape_without_batch") or [])
                        ],
                    }
                )
            return sorted(specs, key=lambda item: _stable_json_dumps(item))

        return normalised_specs(actual) == normalised_specs(expected)

    def _contract_layout_tensors_from_runtime(
        self,
        *,
        splitter: UniversalModelSplitter,
        candidate: object | None,
        input_tensor_shape: list[int],
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if len(input_tensor_shape) < 4:
            return None
        batch_shape = [2, *[int(dim) for dim in input_tensor_shape[1:]]]
        example = torch.zeros(
            batch_shape,
            dtype=torch.float32,
            device=self.device if device is None else device,
        )
        try:
            with torch.no_grad():
                payload = splitter.edge_forward(example, candidate=candidate)
        except Exception as exc:
            logger.warning(
                "[FixedSplitCL] Could not sample cloud batch feature layout from "
                "runtime; using uploaded feature layout. reason={}.",
                safe_error_summary(exc),
            )
            log_diagnostic_debug(
                self,
                "[FixedSplitCL] feature-layout sampling failure",
                lambda error=exc: {"error": repr(error)},
                runtime=True,
            )
            return None
        if not isinstance(payload, BoundaryPayload):
            return None
        sample_payload = BoundaryPayloadCacheCodec(splitter).split_batch(
            payload,
            actual_batch_size=1,
        )[0]
        return {
            str(label): tensor.detach().cpu()
            for label, tensor in dict(sample_payload.tensors or {}).items()
            if isinstance(tensor, torch.Tensor)
        } or None

    def _load_split_runtime_contract(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
    ) -> SplitRuntimeContract | None:
        context = self._training_window_manifest_context(manifest)
        model_id = str(context.get("model_id") or self.edge_model_name)
        split_config_id = str(context.get("split_config_id") or "").strip()
        if not split_config_id:
            raise RuntimeError("SplitRuntimeContract requires split_config_id.")
        existing = SplitRuntimeContract.load(
            self.split_contract_root,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        if existing is None:
            return None
        canonical_from_context = str(context.get("canonical_split_key") or "").strip()
        if canonical_from_context and canonical_from_context != existing.canonical_split_key:
            logger.info(
                "[FixedSplitCL] Ignoring stale SplitRuntimeContract for sync/read: "
                "existing canonical_split_key={!r}, incoming={!r}.",
                existing.canonical_split_key,
                canonical_from_context,
            )
            return None
        front_from_context = str(context.get("front_version") or "0")
        if front_from_context != existing.front_version:
            logger.info(
                "[FixedSplitCL] Ignoring stale SplitRuntimeContract for sync/read: "
                "existing front_version={!r}, incoming={!r}.",
                existing.front_version,
                front_from_context,
            )
            return None
        return existing

    @staticmethod
    def _candidate_cloud_batch_split_id(
        *,
        splitter: UniversalModelSplitter,
        candidate: object | None,
        canonical_split_key: str,
    ) -> str:
        runtime = getattr(splitter, "runtime", None)
        return str(
            getattr(candidate, "candidate_id", None)
            or getattr(candidate, "split_id", None)
            or getattr(runtime, "split_id", "")
            or canonical_split_key
        )

    def _stale_split_contract_path(
        self,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
    ) -> str:
        source_path = contract_path(
            self.split_contract_root,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        stale_dir = os.path.join(
            self.split_contract_root,
            "stale",
            f"edge_{_sanitize_cache_segment(edge_id)}",
            _sanitize_cache_segment(model_id),
        )
        os.makedirs(stale_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return source_path, os.path.join(
            stale_dir,
            f"{_sanitize_cache_segment(split_config_id)}.{stamp}.json",
        )

    def _move_stale_split_runtime_contract(
        self,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
        reason: str,
    ) -> bool:
        source_path, stale_path = self._stale_split_contract_path(
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        if not os.path.exists(source_path):
            return False
        shutil.move(source_path, stale_path)
        logger.info(
            "[FixedSplitCL] Stale split contract archived: reason={}.",
            reason,
        )
        log_diagnostic_debug(
            self,
            "[FixedSplitCL] stale split contract archive diagnostics",
            lambda: {
                "split_config_id": split_config_id,
                "source_path": source_path,
                "stale_path": stale_path,
            },
            runtime=True,
        )
        return True

    def _runtime_identity_for_contract(
        self,
        *,
        manifest: Mapping[str, object],
        splitter: UniversalModelSplitter,
        cloud_batch_split_id: str,
        feature_layout_id: str,
    ) -> dict[str, object]:
        context = self._training_window_manifest_context(manifest)
        split_plan = dict(manifest.get("split_plan", {}) or {})
        cloud_runtime_contract = dict(manifest.get("_cloud_runtime_contract") or {})
        runtime = getattr(splitter, "runtime", splitter)
        dynamic_batch = splitter_dynamic_batch_range(splitter)
        symbolic_schema = getattr(runtime, "symbolic_input_schema", None)
        model_id = str(context.get("model_id") or self.edge_model_name)
        return {
            "model_id": model_id,
            "model_family": model_zoo.get_model_family(model_id),
            "model_version": str(
                dict(manifest.get("model", {}) or {}).get("model_version", "") or "0"
            ),
            "front_version": str(context.get("front_version") or "0"),
            "split_config_id": str(context.get("split_config_id") or ""),
            "canonical_split_key": str(context.get("canonical_split_key") or ""),
            "cloud_batch_split_id": str(cloud_batch_split_id),
            "input_tensor_shape": [
                int(dim) for dim in list(context.get("input_tensor_shape", []) or [])
            ],
            "input_resize_mode": str(context.get("input_resize_mode") or "direct_resize"),
            "graph_signature": str(
                getattr(getattr(runtime, "trace_graph", None), "graph_shape_hash", "") or ""
            ),
            "split_plan_hash": _json_fingerprint(split_plan),
            "symbolic_input_schema_hash": _json_fingerprint(symbolic_schema or {}),
            "dynamic_batch": (
                [int(dynamic_batch[0]), int(dynamic_batch[1])]
                if dynamic_batch is not None
                else None
            ),
            "trace_batch_size": getattr(runtime, "trace_batch_size", None),
            "mode": str(
                getattr(getattr(runtime, "split_spec", None), "mode", "")
                or getattr(runtime, "mode", "")
                or ""
            ),
            "feature_layout_id": str(feature_layout_id),
            "runtime_contract": cloud_runtime_contract,
        }

    def _get_or_create_split_runtime_contract(
        self,
        *,
        edge_id: int | str,
        manifest: Mapping[str, object],
        feature_tensors: Mapping[str, torch.Tensor] | None = None,
        contract_layout_tensors: Mapping[str, torch.Tensor] | None = None,
        model: torch.nn.Module | None = None,
        splitter: UniversalModelSplitter | None = None,
        candidate: object | None = None,
        bundle_root: str | None = None,
        create_if_missing: bool = False,
    ) -> SplitRuntimeContract:
        context = self._training_window_manifest_context(manifest)
        model_id = str(context.get("model_id") or self.edge_model_name)
        split_config_id = str(context.get("split_config_id") or "").strip()
        if not split_config_id:
            raise RuntimeError("SplitRuntimeContract requires split_config_id.")
        existing = SplitRuntimeContract.load(
            self.split_contract_root,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        stale_replaced = False
        if existing is not None:
            stale_reason = None
            canonical_from_context = str(context.get("canonical_split_key") or "").strip()
            front_from_context = str(context.get("front_version") or "0")
            if canonical_from_context and canonical_from_context != existing.canonical_split_key:
                stale_reason = "canonical_split_key"
            elif front_from_context != existing.front_version:
                stale_reason = "front_version"
            if (
                stale_reason is None
                and create_if_missing
                and splitter is not None
                and candidate is not None
            ):
                runtime_split_id = self._candidate_cloud_batch_split_id(
                    splitter=splitter,
                    candidate=candidate,
                    canonical_split_key=existing.canonical_split_key,
                )
                if runtime_split_id != existing.cloud_batch_split_id:
                    stale_reason = "cloud_batch_split_id"
                layout_tensors_for_existing = {
                    str(label): tensor
                    for label, tensor in dict(contract_layout_tensors or {}).items()
                    if isinstance(tensor, torch.Tensor)
                } or self._contract_layout_tensors_from_runtime(
                    splitter=splitter,
                    candidate=candidate,
                    input_tensor_shape=[
                        int(dim) for dim in list(context.get("input_tensor_shape", []) or [])
                    ],
                )
                if stale_reason is None and layout_tensors_for_existing is not None:
                    cloud_runtime_contract = dict(manifest.get("_cloud_runtime_contract") or {})
                    contract_layout = feature_layout_from_tensors(layout_tensors_for_existing)
                    contract_layout_id = str(
                        cloud_runtime_contract.get("feature_layout_id")
                        or make_feature_layout_id(contract_layout)
                    )
                    runtime_identity = self._runtime_identity_for_contract(
                        manifest=manifest,
                        splitter=splitter,
                        cloud_batch_split_id=runtime_split_id,
                        feature_layout_id=contract_layout_id,
                    )
                    boundary_tensor_labels = list(
                        getattr(candidate, "boundary_tensor_labels", None)
                        or cloud_runtime_contract.get("boundary_tensor_labels")
                        or context.get("boundary_tensor_labels")
                        or []
                    )
                    proposed = SplitRuntimeContract.create(
                        edge_id=edge_id,
                        model_id=model_id,
                        split_config_id=split_config_id,
                        canonical_split_key=existing.canonical_split_key,
                        edge_split_id=str(context.get("edge_split_id") or existing.edge_split_id),
                        cloud_batch_split_id=runtime_split_id,
                        input_tensor_shape=list(context.get("input_tensor_shape", []) or []),
                        input_resize_mode=str(context.get("input_resize_mode") or "direct_resize"),
                        boundary_tensor_labels=boundary_tensor_labels,
                        front_version=front_from_context,
                        feature_tensors=layout_tensors_for_existing,
                        tail_version=str(
                            dict(manifest.get("model", {}) or {}).get("model_version", "") or ""
                        )
                        or None,
                        runtime_identity=runtime_identity,
                    )
                    compatibility = classify_contract_compatibility(existing, proposed)
                    if bool(compatibility.get("compatible")):
                        if (
                            proposed.runtime_identity_id != existing.runtime_identity_id
                            or proposed.contract_id != existing.contract_id
                            or proposed.feature_layout_id != existing.feature_layout_id
                        ):
                            aliases = [
                                dict(item)
                                for item in list(existing.contract_aliases or [])
                                if isinstance(item, Mapping)
                            ]
                            if not any(
                                str(alias.get("contract_id") or "") == existing.contract_id
                                for alias in aliases
                            ):
                                aliases.append(
                                    {
                                        "contract_id": existing.contract_id,
                                        "runtime_identity_id": existing.runtime_identity_id,
                                        "feature_layout_id": existing.feature_layout_id,
                                        "feature_abi_id": existing.feature_abi_id,
                                        "reason": str(
                                            compatibility.get("reason") or "compatible_rebind"
                                        ),
                                    }
                                )
                            proposed.contract_aliases = aliases
                            path = proposed.save(self.split_contract_root)
                            if proposed.runtime_identity_id != existing.runtime_identity_id:
                                logger.info(
                                    "[FixedSplitCL] Split contract rebound: feature ABI "
                                    "compatible; active samples preserved."
                                )
                            else:
                                logger.info(
                                    "[FixedSplitCL] Split contract rebound: feature ABI "
                                    "compatible; active samples preserved."
                                )
                            log_diagnostic_debug(
                                self,
                                "[FixedSplitCL] split contract rebound diagnostics",
                                lambda: {
                                    "edge_id": edge_id,
                                    "model_id": model_id,
                                    "split_config_id": split_config_id,
                                    "feature_abi_id": proposed.feature_abi_id,
                                    "source_contract_id": existing.contract_id,
                                    "current_contract_id": proposed.contract_id,
                                    "path": path,
                                },
                                runtime=True,
                            )
                            return proposed
                        return existing
                    stale_reason = str(compatibility.get("reason") or "feature_abi")
            if stale_reason is not None:
                stale_replaced = self._move_stale_split_runtime_contract(
                    edge_id=edge_id,
                    model_id=model_id,
                    split_config_id=split_config_id,
                    reason=stale_reason,
                )
                existing = None
            else:
                return existing

        if not create_if_missing:
            raise RuntimeError(
                "SplitRuntimeContract is not ready for this split_config_id; "
                "wait for a continual learning training job to create it."
            )
        front_version = str(context.get("front_version") or "0")
        bundle_model_version = str(
            dict(manifest.get("model", {}) or {}).get("model_version", "0") or "0"
        )
        if front_version == "0" and bundle_model_version != "0" and not stale_replaced:
            raise RuntimeError(
                "SplitRuntimeContract for front_version=0 must be created from "
                "native pretrained model_version=0, not tail checkpoint "
                f"model_version={bundle_model_version}."
            )

        split_plan = dict(manifest.get("split_plan", {}) or {})
        edge_runtime_contract = fixed_split_plan_runtime_contract(split_plan)
        cloud_runtime_contract = dict(manifest.get("_cloud_runtime_contract") or {})
        canonical_split_key = str(
            context.get("canonical_split_key")
            or context.get("edge_split_id")
            or edge_runtime_contract.get("logical_split_id")
            or fixed_split_boundary_from_plan(split_plan)
        ).strip()
        if (
            canonical_split_key
            and canonical_split_key != "auto"
            and not canonical_split_key.startswith("after:")
        ):
            canonical_split_key = f"after:{canonical_split_key}"
        if not canonical_split_key:
            raise RuntimeError("SplitRuntimeContract creation requires an exact split id.")
        if feature_tensors is None and contract_layout_tensors is None:
            raise RuntimeError(
                "SplitRuntimeContract creation requires a representative feature tensor."
            )
        if splitter is None or candidate is None:
            raise RuntimeError(
                "SplitRuntimeContract creation requires the training job's "
                "already-bound cloud batch runtime."
            )

        batch_candidate = candidate
        cloud_batch_split_id = self._candidate_cloud_batch_split_id(
            splitter=splitter,
            candidate=batch_candidate,
            canonical_split_key=canonical_split_key,
        )
        if cloud_batch_split_id != canonical_split_key:
            raise RuntimeError(
                "SplitRuntimeContract runtime split does not match the exact plan split "
                f"(expected={canonical_split_key!r}, actual={cloud_batch_split_id!r})."
            )
        layout_tensors = {
            str(label): tensor
            for label, tensor in dict(contract_layout_tensors or {}).items()
            if isinstance(tensor, torch.Tensor)
        } or self._contract_layout_tensors_from_runtime(
            splitter=splitter,
            candidate=batch_candidate,
            input_tensor_shape=[
                int(dim) for dim in list(context.get("input_tensor_shape", []) or [])
            ],
        )
        contract_feature_tensors = layout_tensors or feature_tensors
        if contract_feature_tensors is None:
            raise RuntimeError(
                "SplitRuntimeContract creation requires a representative feature tensor."
            )
        contract_layout = feature_layout_from_tensors(contract_feature_tensors)
        contract_layout_id = str(
            cloud_runtime_contract.get("feature_layout_id")
            or make_feature_layout_id(contract_layout)
        )
        runtime_identity = self._runtime_identity_for_contract(
            manifest=manifest,
            splitter=splitter,
            cloud_batch_split_id=cloud_batch_split_id,
            feature_layout_id=contract_layout_id,
        )
        edge_split_id = str(context.get("edge_split_id") or canonical_split_key)
        boundary_tensor_labels = list(
            getattr(batch_candidate, "boundary_tensor_labels", None)
            or cloud_runtime_contract.get("boundary_tensor_labels")
            or context.get("boundary_tensor_labels")
            or []
        )
        contract = SplitRuntimeContract.create(
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
            canonical_split_key=canonical_split_key,
            edge_split_id=edge_split_id,
            cloud_batch_split_id=cloud_batch_split_id,
            input_tensor_shape=list(context.get("input_tensor_shape", []) or []),
            input_resize_mode=str(context.get("input_resize_mode") or "direct_resize"),
            boundary_tensor_labels=boundary_tensor_labels,
            front_version=str(context.get("front_version") or "0"),
            feature_tensors=contract_feature_tensors,
            tail_version=str(dict(manifest.get("model", {}) or {}).get("model_version", "") or "")
            or None,
            runtime_identity=runtime_identity,
        )
        path = contract.save(self.split_contract_root)
        logger.info(
            "[FixedSplitCL] Split contract created: edge={} model={} split={}.",
            edge_id,
            model_id,
            canonical_split_key,
        )
        log_diagnostic_debug(
            self,
            "[FixedSplitCL] split contract creation diagnostics",
            lambda: {
                "split_config_id": split_config_id,
                "cloud_batch_split_id": cloud_batch_split_id,
                "contract_id": contract.contract_id,
                "feature_layout_id": contract.feature_layout_id,
                "feature_abi_id": contract.feature_abi_id,
                "path": path,
            },
            runtime=True,
        )
        return contract
