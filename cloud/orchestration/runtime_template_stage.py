from __future__ import annotations

import copy
import os
import time
from collections.abc import Mapping
from dataclasses import replace

import cv2
import numpy as np
import torch
from loguru import logger

import model_management.model_zoo as model_zoo
from cloud.orchestration.fixed_split_dependencies import (
    _iter_tensors,
    _json_fingerprint,
)
from cloud.orchestration.runtime_stage import (
    FIXED_SPLIT_DYNAMIC_BATCH_MAX as _FIXED_SPLIT_DYNAMIC_BATCH_MAX,
)
from cloud.orchestration.runtime_stage import (
    FIXED_SPLIT_DYNAMIC_BATCH_MIN as _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
)
from cloud.orchestration.runtime_stage import (
    cloud_fixed_split_dynamic_batch as _cloud_fixed_split_dynamic_batch,
)
from cloud.orchestration.runtime_stage import (
    cloud_fixed_split_trace_batch_mode as _cloud_fixed_split_trace_batch_mode,
)
from cloud.orchestration.runtime_stage import (
    cloud_fixed_split_trace_batch_size as _cloud_fixed_split_trace_batch_size,
)
from cloud.orchestration.runtime_stage import (
    fixed_split_boundary_from_plan as _fixed_split_boundary_from_plan,
)
from cloud.orchestration.runtime_stage import (
    fixed_split_manifest_has_rebuildable_raw_samples,
)
from cloud.orchestration.runtime_stage import (
    fixed_split_plan_runtime_contract as _fixed_split_plan_runtime_contract,
)
from cloud.orchestration.runtime_stage import (
    fixed_split_validation_batches as _fixed_split_validation_batches,
)
from common.logging_sanitizer import (
    log_diagnostic_debug,
    safe_error_summary,
    should_log_runtime_diagnostics,
)
from model_management.fixed_split_runtime_template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateKey,
    FixedSplitRuntimeTemplateLookup,
    bind_request_splitter_from_template,
    describe_split_candidate,
    fixed_split_runtime_template_key,
)
from model_management.payload import BoundaryPayload
from model_management.split_contract import (
    classify_feature_layout_compatibility,
    resolve_cloud_runtime_contract,
)
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    get_split_runtime_model,
)
from model_management.split_runtime import (
    BoundaryPayloadCacheCodec,
    compare_outputs,
    make_split_spec,
)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    prepare_exact_split_runtime,
)


class FixedSplitRuntimeTemplateMixin:
    def _infer_bundle_trace_image_size(
        self,
        manifest: dict[str, object],
    ) -> tuple[int, int]:
        runtime_image_size = self._runtime_image_size_from_metadata(manifest)
        if runtime_image_size is not None:
            return runtime_image_size
        for sample in manifest.get("samples", []):
            runtime_image_size = self._runtime_image_size_from_metadata(sample)
            if runtime_image_size is not None:
                return runtime_image_size
        raise RuntimeError(
            "Missing input_tensor_shape/input_image_size metadata required to build "
            "cloud split-runtime trace input."
        )

    def _normalize_bundle_runtime_tensor(
        self,
        runtime_input,
        *,
        context: str,
    ) -> torch.Tensor:
        if not isinstance(runtime_input, torch.Tensor):
            raise TypeError(
                f"{context} requires tensor split-runtime inputs, got "
                f"{type(runtime_input).__name__}."
            )
        if runtime_input.ndim == 3:
            runtime_input = runtime_input.unsqueeze(0)
        if runtime_input.ndim < 4:
            raise RuntimeError(
                f"{context} expected a batched image tensor, got shape "
                f"{tuple(runtime_input.shape)}."
            )
        if runtime_input.shape[0] != 1:
            raise RuntimeError(
                f"{context} expected a single-sample runtime tensor before batching, "
                f"got shape {tuple(runtime_input.shape)}."
            )
        return runtime_input

    def _prepare_bundle_runtime_tensor(
        self,
        model: torch.nn.Module,
        frame,
        *,
        sample_metadata: Mapping[str, object] | None = None,
        context: str,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        runtime_input = self._prepare_split_runtime_input(
            model,
            frame,
            sample_metadata=sample_metadata,
            device=device,
        )
        return self._normalize_bundle_runtime_tensor(
            runtime_input,
            context=context,
        )

    def _build_bundle_batch_trace_sample_input(
        self,
        model: torch.nn.Module,
        bundle_root: str,
        manifest: dict[str, object],
        *,
        runtime_batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        batch_target = max(
            1,
            int(self.batch_size if runtime_batch_size is None else runtime_batch_size),
        )
        prepared_inputs: list[torch.Tensor] = []

        for sample in manifest.get("samples", []):
            raw_relpath = sample.get("raw_relpath")
            if raw_relpath is None:
                continue
            raw_path = os.path.join(bundle_root, str(raw_relpath).replace("/", os.sep))
            if not os.path.exists(raw_path):
                continue
            frame = cv2.imread(raw_path)
            if frame is None:
                continue
            prepared_inputs.append(
                self._prepare_bundle_runtime_tensor(
                    model,
                    frame,
                    sample_metadata=sample,
                    context="Cloud fixed-split batch tracing",
                    device=device,
                )
            )
            if len(prepared_inputs) >= batch_target:
                break

        if not prepared_inputs:
            trace_image_size = self._infer_bundle_trace_image_size(manifest)
            prepared_inputs.append(
                self._normalize_bundle_runtime_tensor(
                    build_split_runtime_sample_input(
                        model,
                        image_size=trace_image_size,
                        device=self.device if device is None else device,
                    ),
                    context="Cloud fixed-split batch tracing",
                )
            )

        batch_input = self._pad_runtime_batch_inputs(
            prepared_inputs,
            target_batch_size=batch_target,
        )
        if self._fixed_split_runtime_diagnostics_enabled():
            logger.debug(
                "[FixedSplitCL][diagnostics] tracing split runtime with batch input "
                "(input_tensor_shape={}).",
                tuple(batch_input.shape),
            )
        return batch_input

    @staticmethod
    def _pad_runtime_batch_inputs(
        prepared_inputs: list[torch.Tensor],
        *,
        target_batch_size: int,
    ) -> torch.Tensor:
        if not prepared_inputs:
            raise ValueError("prepared_inputs must contain at least one tensor.")
        padded_inputs = list(prepared_inputs)
        while len(padded_inputs) < target_batch_size:
            padded_inputs.append(padded_inputs[-1].clone())
        return torch.cat(padded_inputs[:target_batch_size], dim=0)

    @staticmethod
    def _pad_batched_runtime_tensor(
        batch_input: torch.Tensor,
        *,
        target_batch_size: int,
    ) -> torch.Tensor:
        if batch_input.ndim < 1:
            raise RuntimeError(
                f"Expected batched runtime tensor, got shape {tuple(batch_input.shape)}."
            )
        current_batch_size = int(batch_input.shape[0])
        if current_batch_size == int(target_batch_size):
            return batch_input
        if current_batch_size > int(target_batch_size):
            return batch_input[: int(target_batch_size)]
        if current_batch_size <= 0:
            raise RuntimeError("Cannot pad an empty runtime tensor batch.")
        repeats = [int(target_batch_size) - current_batch_size, *([1] * (batch_input.ndim - 1))]
        padding = batch_input[-1:].repeat(*repeats)
        return torch.cat([batch_input, padding], dim=0)

    def _prepare_bundle_runtime_batch(
        self,
        model: torch.nn.Module,
        frames: list[np.ndarray],
        samples: list[Mapping[str, object]],
        *,
        target_batch_size: int,
        context: str,
    ) -> torch.Tensor:
        if not frames:
            raise ValueError("frames must contain at least one frame.")
        if len(frames) != len(samples):
            raise ValueError(f"{context} requires one sample metadata record per frame.")
        model_family = model_zoo.get_model_family(str(getattr(model, "model_name", "")))
        if model_family == "rfdetr" and hasattr(model, "_prepare_batch"):
            tensors: list[torch.Tensor] = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb))
                tensor = tensor.permute(2, 0, 1).float().div(255.0).to(self.device)
                tensors.append(tensor)
            batch_tensor, _ = model._prepare_batch(tensors)
            return self._pad_batched_runtime_tensor(
                batch_tensor.to(self.device),
                target_batch_size=target_batch_size,
            )

        prepared_inputs = [
            self._prepare_bundle_runtime_tensor(
                model,
                frame,
                sample_metadata=sample,
                context=context,
            )
            for frame, sample in zip(frames, samples)
        ]
        return self._pad_runtime_batch_inputs(
            prepared_inputs,
            target_batch_size=target_batch_size,
        )

    def _bundle_batch_feature_provider(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        splitter: UniversalModelSplitter | None = None,
        candidate=None,
        runtime_batch_size: int | None = None,
    ):
        if splitter is None or candidate is None:
            splitter, candidate = self._build_bundle_splitter(
                model,
                manifest,
                bundle_root=bundle_root,
                runtime_batch_size=runtime_batch_size,
            )

        def _batch_provider(
            raw_paths: list[str],
            samples: list[dict[str, object]],
            manifest_payload: dict[str, object],
        ):
            if not raw_paths:
                return []
            if len(raw_paths) != len(samples):
                raise ValueError(
                    "Cloud batch reconstruction expects one sample metadata record per raw path."
                )

            def _detach_payload_value(value: object):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, Mapping):
                    return {key: _detach_payload_value(item) for key, item in value.items()}
                if isinstance(value, tuple):
                    return tuple(_detach_payload_value(item) for item in value)
                if isinstance(value, list):
                    return [_detach_payload_value(item) for item in value]
                return value

            def _detach_payload(payload: BoundaryPayload) -> BoundaryPayload:
                changes = {
                    "tensors": {
                        str(label): tensor.detach().cpu()
                        for label, tensor in dict(payload.tensors or {}).items()
                        if isinstance(tensor, torch.Tensor)
                    },
                    "metadata": {
                        str(label): _detach_payload_value(value)
                        for label, value in dict(payload.metadata or {}).items()
                    },
                }
                return replace(payload, **changes)

            payloads: list[BoundaryPayload] = []
            codec = BoundaryPayloadCacheCodec(splitter)
            chunk_size = max(
                1,
                int(self.batch_size if runtime_batch_size is None else runtime_batch_size),
            )
            chunk_size = min(chunk_size, _FIXED_SPLIT_DYNAMIC_BATCH_MAX)
            for offset in range(0, len(raw_paths), chunk_size):
                chunk_paths = raw_paths[offset : offset + chunk_size]
                chunk_samples = samples[offset : offset + chunk_size]
                prepared_inputs: list[np.ndarray] = []
                for raw_path, sample in zip(chunk_paths, chunk_samples):
                    frame = cv2.imread(raw_path)
                    if frame is None:
                        raise FileNotFoundError(raw_path)
                    prepared_inputs.append(frame)

                actual_chunk_size = len(chunk_paths)
                execution_batch_size = max(
                    _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
                    actual_chunk_size,
                )
                inputs = self._prepare_bundle_runtime_batch(
                    model,
                    prepared_inputs,
                    chunk_samples,
                    target_batch_size=execution_batch_size,
                    context="Cloud fixed-split feature reconstruction",
                )
                batch_payload = splitter.edge_forward(inputs, candidate=candidate)
                if not isinstance(batch_payload, BoundaryPayload):
                    raise RuntimeError(
                        "Cloud feature reconstruction expected a TorchLens ReplayBoundary "
                        f"from prefix execution, got {type(batch_payload).__name__}."
                    )
                if int(getattr(batch_payload, "batch_size", 0)) != execution_batch_size:
                    raise RuntimeError(
                        "Cloud feature reconstruction produced a BoundaryPayload with the wrong "
                        f"batch size (payload_batch={getattr(batch_payload, 'batch_size', None)}, "
                        f"expected={execution_batch_size})."
                    )
                payloads.extend(
                    _detach_payload(sample_payload)
                    for sample_payload in codec.split_batch(
                        batch_payload,
                        actual_batch_size=actual_chunk_size,
                    )
                )
            return payloads

        return _batch_provider

    def _fixed_split_runtime_template_key(
        self,
        *,
        model_name: str,
        manifest: Mapping[str, object],
        runtime_batch_size: int | None = None,
    ) -> FixedSplitRuntimeTemplateKey:
        split_plan = dict(manifest.get("split_plan", {}))
        runtime_contract = _fixed_split_plan_runtime_contract(split_plan)
        trace_image_size = self._infer_bundle_trace_image_size(dict(manifest))
        image_size = trace_image_size or (640, 640)
        boundary = _fixed_split_boundary_from_plan(split_plan)
        model_family = model_zoo.get_model_family(str(model_name))
        dynamic_batch = _cloud_fixed_split_dynamic_batch(
            split_plan,
            model_family=model_family,
        )
        trace_batch_mode = _cloud_fixed_split_trace_batch_mode(
            split_plan,
            model_family=model_family,
        )
        trace_batch_size = _cloud_fixed_split_trace_batch_size(
            split_plan,
            model_family=model_family,
            default=self.trace_batch_size,
        )
        split_spec = make_split_spec(
            boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
        )
        symbolic_example = torch.empty(
            (trace_batch_size, 3, int(image_size[0]), int(image_size[1]))
        )
        return fixed_split_runtime_template_key(
            model_name=str(model_name),
            model_family=model_family,
            split_spec=split_spec,
            example_inputs=symbolic_example,
            graph_signature=str(runtime_contract.get("trace_signature") or "") or None,
            split_plan_hash=self._fixed_split_template_structural_plan_hash(split_plan),
            canonical_split_key=self._fixed_split_template_split_key(split_plan),
        )

    @staticmethod
    def _fixed_split_template_split_key(split_plan: Mapping[str, object]) -> str:
        runtime_contract = _fixed_split_plan_runtime_contract(split_plan)
        return str(
            split_plan.get("canonical_split_key")
            or split_plan.get("edge_split_id")
            or runtime_contract.get("logical_split_id")
            or _fixed_split_boundary_from_plan(split_plan)
        )

    @staticmethod
    def _fixed_split_template_structural_plan_hash(
        split_plan: Mapping[str, object],
    ) -> str:
        runtime_contract = _fixed_split_plan_runtime_contract(split_plan)
        canonical_split_key = (
            FixedSplitRuntimeTemplateMixin._fixed_split_template_split_key(split_plan)
        )
        return _json_fingerprint(
            {
                "canonical_split_key": canonical_split_key,
                "logical_split_id": str(runtime_contract.get("logical_split_id") or ""),
                "boundary_tensor_labels": [
                    str(label)
                    for label in list(runtime_contract.get("boundary_tensor_labels") or [])
                ],
                "boundary_schema": dict(runtime_contract.get("boundary_schema") or {}),
                "split_granularity": str(split_plan.get("split_granularity") or ""),
            }
        )

    def _fixed_split_runtime_diagnostics_enabled(self) -> bool:
        return should_log_runtime_diagnostics(self)

    def _fixed_split_runtime_smoke_validate_enabled(self) -> bool:
        return bool(getattr(self, "fixed_split_runtime_smoke_validate", False))

    @staticmethod
    def _fixed_split_runtime_template_log_label(
        model_name: str,
        split_key: str,
    ) -> str:
        return f"model={model_name} split={split_key}"

    @staticmethod
    def _runtime_example_args(sample_input):
        if isinstance(sample_input, tuple):
            return sample_input
        if isinstance(sample_input, list):
            return tuple(sample_input)
        return (sample_input,)

    @staticmethod
    def _tensor_shape_from_runtime_input(sample_input) -> tuple[int, ...] | None:
        if isinstance(sample_input, torch.Tensor):
            return tuple(int(dim) for dim in sample_input.shape)
        if isinstance(sample_input, (list, tuple)):
            for value in sample_input:
                if isinstance(value, torch.Tensor):
                    return tuple(int(dim) for dim in value.shape)
        return None

    def _infer_pool_runtime_input_tensor_shape(
        self,
        model: torch.nn.Module,
        *,
        bundle_root: str,
        manifest: dict[str, object],
        prepared_trace_sample_input,
    ) -> tuple[int, ...] | None:
        shape = self._tensor_shape_from_runtime_input(prepared_trace_sample_input)
        if shape is not None:
            return shape
        for sample in manifest.get("samples", []):
            if not isinstance(sample, Mapping):
                continue
            raw_relpath = sample.get("raw_relpath")
            if raw_relpath is None:
                continue
            raw_path = os.path.join(bundle_root, str(raw_relpath).replace("/", os.sep))
            if not os.path.exists(raw_path):
                continue
            frame = cv2.imread(raw_path)
            if frame is None:
                continue
            runtime_input = self._prepare_split_runtime_input(
                model,
                frame,
                sample_metadata=sample,
            )
            runtime_tensor = self._normalize_bundle_runtime_tensor(
                runtime_input,
                context="Cloud sample-pool runtime shape inference",
            )
            return tuple(int(dim) for dim in runtime_tensor.shape)
        trace_image_size = self._infer_bundle_trace_image_size(manifest)
        runtime_tensor = self._normalize_bundle_runtime_tensor(
            build_split_runtime_sample_input(
                model,
                image_size=trace_image_size,
                device=self.device,
            ),
            context="Cloud sample-pool runtime shape inference",
        )
        return tuple(int(dim) for dim in runtime_tensor.shape)

    @staticmethod
    def _preferred_fixed_split_runtime_mode(model_family: str | None) -> str:
        return "generated_eager"

    def _validate_prepared_split_runtime(
        self,
        runtime,
        model: torch.nn.Module,
        sample_input,
        *,
        model_name: str,
        mode: str,
    ) -> tuple[bool, str | None]:
        inputs = self._runtime_example_args(sample_input)
        try:
            with torch.no_grad():
                boundary_payload = runtime.run_prefix(*inputs)
                replayed = runtime.run_suffix(boundary_payload)
                expected = model(*inputs)
            ok, max_diff = compare_outputs(expected, replayed)
        except Exception as exc:  # noqa: BLE001 - report and possibly fall back.
            return False, str(exc)
        if not ok:
            return False, f"split replay output mismatch (max_diff={max_diff})"
        if self._fixed_split_runtime_diagnostics_enabled():
            logger.debug(
                "[FixedSplitCL][diagnostics] TorchLens runtime replay validation passed "
                "(mode={}, split_id={}).",
                mode,
                getattr(runtime, "split_id", None),
            )
        return True, None

    def _prepare_replayable_split_runtime(
        self,
        model: torch.nn.Module,
        sample_input,
        split_spec,
        *,
        model_name: str,
        preferred_mode: str = "generated_eager",
    ) -> tuple[object, str]:
        modes = []
        for mode in (preferred_mode, "generated_eager", "compiled"):
            mode = str(mode)
            if mode not in modes:
                modes.append(mode)

        errors: dict[str, str | None] = {}
        for index, mode in enumerate(modes):
            runtime = prepare_exact_split_runtime(
                model,
                sample_input,
                split_spec,
                mode=mode,
            )
            ok, error = self._validate_prepared_split_runtime(
                runtime,
                model,
                sample_input,
                model_name=model_name,
                mode=mode,
            )
            if ok:
                return runtime, mode
            errors[mode] = error
            if index + 1 < len(modes):
                logger.warning(
                    "[FixedSplitCL] TorchLens {} runtime failed replay validation "
                    "(model={}, split={}, reason={}); retrying with {}.",
                    mode,
                    model_name,
                    getattr(runtime, "split_id", None),
                    safe_error_summary(error),
                    modes[index + 1],
                )

        error_summary = ", ".join(f"{mode}_error={error}" for mode, error in errors.items())
        raise RuntimeError(
            "TorchLens fixed split runtime is not replayable in any supported mode "
            f"({error_summary})."
        )

    def _resolve_runtime_contract_trace_device(
        self,
        runtime_contract: Mapping[str, object],
    ) -> torch.device:
        requested = str(runtime_contract.get("trace_device_type") or "").strip().lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        return torch.device(self.device)

    @staticmethod
    def _module_device(module: torch.nn.Module) -> torch.device:
        for parameter in module.parameters(recurse=True):
            return parameter.device
        for buffer in module.buffers(recurse=True):
            return buffer.device
        return torch.device("cpu")

    def _trace_model_for_device(
        self,
        model: torch.nn.Module,
        trace_device: torch.device,
    ) -> torch.nn.Module:
        model_device = self._module_device(model)
        if model_device.type == trace_device.type:
            return model
        trace_model = copy.deepcopy(model)
        trace_model.to(trace_device)
        trace_model.eval()
        return trace_model

    @staticmethod
    def _move_runtime_input_to_device(value: object, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(
                FixedSplitRuntimeTemplateMixin._move_runtime_input_to_device(item, device)
                for item in value
            )
        if isinstance(value, list):
            return [
                FixedSplitRuntimeTemplateMixin._move_runtime_input_to_device(item, device)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: FixedSplitRuntimeTemplateMixin._move_runtime_input_to_device(item, device)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _batch_polymorphic_smoke_loss(outputs: object, _targets: object) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        for tensor in _iter_tensors(outputs):
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.is_floating_point()
                and tensor.requires_grad
                and tensor.numel() > 0
            ):
                terms.append(tensor.reshape(-1).mean())
        if not terms:
            raise RuntimeError(
                "Batch-polymorphic split validation could not find a differentiable "
                "floating output tensor."
            )
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total

    def _validate_dynamic_batch_trainability(
        self,
        runtime,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        model_family: str | None,
        trace_batch_size: int,
        runtime_batch_size: int | None,
        dynamic_batch: tuple[int, int] | None,
        runtime_device: torch.device | str | None = None,
    ) -> list[int]:
        batch_sizes = _fixed_split_validation_batches(
            model_family=model_family,
            trace_batch_size=trace_batch_size,
            runtime_batch_size=runtime_batch_size,
            dynamic_batch=dynamic_batch,
        )
        if not batch_sizes:
            return []
        suffix_segment = getattr(runtime, "suffix_segment", None)
        if isinstance(suffix_segment, torch.nn.Module):
            suffix_segment.train()

        for batch_size in batch_sizes:
            sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=batch_size,
                device=runtime_device,
            )
            try:
                boundary_payload = runtime.run_prefix(*self._runtime_example_args(sample_input))
                runtime.train_suffix(
                    boundary_payload,
                    None,
                    loss_fn=self._batch_polymorphic_smoke_loss,
                    optimizer=None,
                )
            except Exception as exc:
                raise RuntimeError(
                    "TorchLens fixed split runtime failed dynamic-batch trainability "
                    f"validation (model_family={model_family}, "
                    f"split_id={getattr(runtime, 'split_id', None)}, "
                    f"batch_size={batch_size}, trace_batch_size={trace_batch_size}): {exc}"
                ) from exc
            if isinstance(suffix_segment, torch.nn.Module):
                suffix_segment.zero_grad(set_to_none=True)
        logger.info(
            "[FixedSplitCL] dynamic-batch trainability validation passed "
            "(split_id={}, batches={}).",
            getattr(runtime, "split_id", None),
            batch_sizes,
        )
        return batch_sizes

    def _build_fixed_split_runtime_template(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        template_key: FixedSplitRuntimeTemplateKey,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ) -> FixedSplitRuntimeTemplate:
        split_plan_payload = dict(manifest.get("split_plan", {}))
        split_model = get_split_runtime_model(model)
        sample_input = trace_sample_input
        model_name = self._resolve_fixed_split_model_name(manifest)
        model_family = model_zoo.get_model_family(model_name)
        edge_runtime_contract = _fixed_split_plan_runtime_contract(split_plan_payload)
        trace_device = self._resolve_runtime_contract_trace_device(edge_runtime_contract)
        trace_model = self._trace_model_for_device(split_model, trace_device)
        if sample_input is None:
            trace_batch_size = _cloud_fixed_split_trace_batch_size(
                split_plan_payload,
                model_family=model_family,
                default=self.trace_batch_size,
            )
            sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=trace_batch_size,
                device=trace_device,
            )
        else:
            trace_batch_size = _cloud_fixed_split_trace_batch_size(
                split_plan_payload,
                model_family=model_family,
                default=self.trace_batch_size,
            )
            sample_input = self._move_runtime_input_to_device(sample_input, trace_device)
        boundary = _fixed_split_boundary_from_plan(split_plan_payload)
        dynamic_batch = _cloud_fixed_split_dynamic_batch(
            split_plan_payload,
            model_family=model_family,
        )
        trace_batch_mode = _cloud_fixed_split_trace_batch_mode(
            split_plan_payload,
            model_family=model_family,
        )
        split_spec = make_split_spec(
            boundary,
            dynamic_batch=dynamic_batch,
            trainable=True,
            trace_batch_mode=trace_batch_mode,
            model_family=model_family,
        )
        runtime, runtime_mode = self._prepare_replayable_split_runtime(
            trace_model,
            sample_input,
            split_spec,
            model_name=model_name,
            preferred_mode=self._preferred_fixed_split_runtime_mode(model_family),
        )
        model_meta = dict(manifest.get("model", {}) or {})
        context = self._training_window_manifest_context(manifest)
        runtime_splitter = UniversalModelSplitter(device=self.device).bind_runtime(
            runtime,
            model=trace_model,
        )
        runtime_candidate = getattr(runtime_splitter, "current_candidate", None)
        cloud_runtime_contract = resolve_cloud_runtime_contract(
            runtime,
            runtime_candidate,
            logical_split_id=boundary,
            model_id=str(model_meta.get("model_id") or model_name),
            model_version=str(model_meta.get("model_version", "") or "0"),
            input_tensor_shape=list(
                edge_runtime_contract.get("input_tensor_shape")
                or context.get("input_tensor_shape")
                or []
            ),
            input_resize_mode=str(
                edge_runtime_contract.get("input_resize_mode")
                or context.get("input_resize_mode")
                or "direct_resize"
            ),
            sample_input=sample_input,
            runtime_backend=runtime_mode,
        )
        compatibility = classify_feature_layout_compatibility(
            edge_runtime_contract,
            cloud_runtime_contract,
        )
        manifest["_cloud_runtime_contract"] = cloud_runtime_contract
        manifest["_feature_layout_compatibility"] = compatibility
        if not bool(compatibility.get("compatible")):
            if not fixed_split_manifest_has_rebuildable_raw_samples(manifest):
                raise RuntimeError(
                    "Fixed split feature layout mismatch and raw rebuild is unavailable: "
                    f"{compatibility}."
                )
            logger.info(
                "[FixedSplitCL] Edge/cloud feature layout differs; rebuilding "
                "low-quality trigger features from raw frames with the cloud runtime: "
                "model={} split={}.",
                model_name,
                boundary,
            )
            log_diagnostic_debug(
                self,
                "[FixedSplitCL] feature layout mismatch diagnostics",
                lambda: {"compatibility": compatibility},
                runtime=True,
            )
            manifest["_cloud_rebuild_features_for_runtime_contract_mismatch"] = True
        if trace_model is split_model:
            training_runtime = runtime
            training_runtime_mode = runtime_mode
        else:
            training_sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=trace_batch_size,
                device=self.device,
            )
            training_runtime, training_runtime_mode = self._prepare_replayable_split_runtime(
                split_model,
                training_sample_input,
                split_spec,
                model_name=model_name,
                preferred_mode=runtime_mode,
            )
            if training_runtime_mode != runtime_mode:
                logger.warning(
                    "[FixedSplitCL] request-local TorchLens runtime mode differed "
                    "from the template trace artifact; continuing with request-local runtime."
                )
                if self._fixed_split_runtime_diagnostics_enabled():
                    logger.debug(
                        "[FixedSplitCL][diagnostics] request-local runtime mode mismatch "
                        "details={}",
                        {
                            "request_mode": training_runtime_mode,
                            "template_mode": runtime_mode,
                        },
                    )
        if self._fixed_split_runtime_smoke_validate_enabled():
            self._validate_dynamic_batch_trainability(
                training_runtime,
                model,
                manifest,
                bundle_root=bundle_root,
                model_family=model_family,
                trace_batch_size=trace_batch_size,
                runtime_batch_size=runtime_batch_size,
                dynamic_batch=dynamic_batch,
                runtime_device=self.device,
            )
        trace_signature = str(
            getattr(getattr(runtime, "trace_graph", None), "graph_shape_hash", "") or ""
        )
        verifier = UniversalModelSplitter(device=self.device).bind_runtime(
            training_runtime,
            model=split_model,
            split_spec=split_spec,
        )
        current_candidate_id = str(
            getattr(getattr(verifier, "current_candidate", None), "candidate_id", "") or ""
        )
        if boundary != "auto" and current_candidate_id and current_candidate_id != boundary:
            raise RuntimeError(
                "TorchLens fixed split runtime resolved a different split candidate "
                f"(requested={boundary!r}, actual={current_candidate_id!r})."
            )
        if self._fixed_split_runtime_diagnostics_enabled():
            logger.debug(
                "[FixedSplitCL][diagnostics] runtime template prepared TorchLens split "
                "details={}",
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "split_id": getattr(runtime, "split_id", None),
                    "trace_signature": trace_signature,
                    "mode": runtime_mode,
                    "key": template_key.as_dict(),
                },
            )
        return FixedSplitRuntimeTemplate(
            cache_key=template_key,
            runtime=runtime,
            split_spec=split_spec,
            model_name=model_name,
            model_family=model_family,
            graph_signature=trace_signature,
            symbolic_input_schema_hash=template_key.symbolic_input_schema_hash,
            split_plan_hash=str(template_key.split_plan_hash),
            mode=runtime_mode,
            runtime_device=str(trace_device.type),
            candidate_descriptor=(
                describe_split_candidate(runtime_candidate)
                if runtime_candidate is not None
                else None
            ),
            runtime_contract=cloud_runtime_contract,
            boundary_tensor_labels=tuple(
                str(label)
                for label in list(
                    getattr(getattr(runtime, "plan", None), "boundary_nodes", ()) or ()
                )
            ),
            boundary_schema=dict(
                getattr(getattr(runtime, "plan", None), "boundary_specs", {}) or {}
            ),
        )

    def _get_or_create_fixed_split_runtime_template(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ) -> FixedSplitRuntimeTemplateLookup:
        model_name = self._resolve_fixed_split_model_name(manifest)
        template_key = self._fixed_split_runtime_template_key(
            model_name=model_name,
            manifest=manifest,
            runtime_batch_size=runtime_batch_size,
        )
        return self._fixed_split_runtime_template_cache.get_or_create_lookup(
            template_key,
            lambda: self._build_fixed_split_runtime_template(
                model,
                manifest,
                bundle_root=bundle_root,
                template_key=template_key,
                trace_sample_input=trace_sample_input,
                runtime_batch_size=runtime_batch_size,
            ),
            log_label=self._fixed_split_runtime_template_log_label(
                model_name,
                template_key.canonical_split_key,
            ),
            diagnostics=self._fixed_split_runtime_diagnostics_enabled(),
        )

    def _bind_bundle_splitter_from_template(
        self,
        model: torch.nn.Module,
        template: FixedSplitRuntimeTemplate,
        *,
        manifest: dict[str, object],
        bundle_root: str,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ) -> tuple[UniversalModelSplitter, object]:
        bind_started = time.perf_counter()
        split_model = get_split_runtime_model(model)
        split_plan_payload = dict(manifest.get("split_plan", {}) or {})
        model_family = model_zoo.get_model_family(str(template.model_name))
        trace_batch_size = _cloud_fixed_split_trace_batch_size(
            split_plan_payload,
            model_family=model_family,
            default=self.trace_batch_size,
        )
        if trace_sample_input is not None:
            request_sample_input = self._move_runtime_input_to_device(
                trace_sample_input,
                self.device,
            )
        else:
            request_sample_input = self._build_bundle_batch_trace_sample_input(
                model,
                bundle_root,
                manifest,
                runtime_batch_size=trace_batch_size,
                device=self.device,
            )
        splitter, candidate = bind_request_splitter_from_template(
            split_model,
            template,
            example_inputs=request_sample_input,
            device=self.device,
        )
        bind_elapsed = time.perf_counter() - bind_started
        logger.info(
            "[FixedSplitCL] Runtime bound: split={} elapsed={:.2f}s.",
            template.cache_key.canonical_split_key or getattr(splitter.runtime, "split_id", None),
            bind_elapsed,
        )
        if self._fixed_split_runtime_diagnostics_enabled():
            logger.debug(
                "[FixedSplitCL][diagnostics] request-local TorchLens runtime bind details={}",
                {
                    "split_id": getattr(splitter.runtime, "split_id", None),
                    "key": template.cache_key.as_dict(),
                },
            )
        return splitter, candidate

    def _build_bundle_splitter(
        self,
        model: torch.nn.Module,
        manifest: dict[str, object],
        *,
        bundle_root: str,
        trace_sample_input: torch.Tensor | None = None,
        runtime_batch_size: int | None = None,
    ):
        template_lookup = self._get_or_create_fixed_split_runtime_template(
            model,
            manifest,
            bundle_root=bundle_root,
            trace_sample_input=trace_sample_input,
            runtime_batch_size=runtime_batch_size,
        )
        if (
            template_lookup.cache_status in {"hit", "wait"}
            and self._fixed_split_runtime_diagnostics_enabled()
        ):
            logger.debug(
                "[FixedSplitCL][diagnostics] runtime template reused details={}",
                {
                    "cache_status": template_lookup.cache_status,
                    "key": template_lookup.template.cache_key.as_dict(),
                },
            )
        return self._bind_bundle_splitter_from_template(
            model,
            template_lookup.template,
            manifest=manifest,
            bundle_root=bundle_root,
            trace_sample_input=trace_sample_input,
            runtime_batch_size=runtime_batch_size,
        )
