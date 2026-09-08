from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from loguru import logger
from torch import nn

from cloud.orchestration.runtime_template_stage import FixedSplitRuntimeTemplateMixin
from model_management.fixed_split_runtime_template import fixed_split_runtime_template_key
from model_management.split_contract import build_runtime_contract
from model_management.split_runtime import make_split_spec, prepare_split_runtime
from model_management.split_runtime.template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateCache,
)
from model_management.split_runtime.torchlens_native_runtime import trace_signature

TINY_SPLIT = "after:conv2d_1_1"


class TinyImageRuntimeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


class RuntimeTemplateHarness(FixedSplitRuntimeTemplateMixin):
    def __init__(self, *, diagnostics: bool = False, smoke_validate: bool = False) -> None:
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.trace_batch_size = 1
        self.edge_model_name = "tiny"
        self.fixed_split_runtime_diagnostics = diagnostics
        self.fixed_split_runtime_smoke_validate = smoke_validate
        self._fixed_split_runtime_template_cache = FixedSplitRuntimeTemplateCache()

    def _resolve_fixed_split_model_name(self, manifest: Mapping[str, object]) -> str:
        model_meta = dict(manifest.get("model", {}) or {})
        return str(model_meta.get("model_id") or manifest.get("model_id") or self.edge_model_name)

    @staticmethod
    def _training_window_manifest_context(manifest: Mapping[str, object]) -> dict[str, object]:
        split_plan = dict(manifest.get("split_plan", {}) or {})
        runtime_contract = dict(split_plan.get("runtime_contract") or {})
        return {
            "model_id": str(manifest.get("model_id") or ""),
            "front_version": str(manifest.get("front_version") or "0"),
            "split_config_id": str(manifest.get("split_config_id") or ""),
            "canonical_split_key": str(manifest.get("canonical_split_key") or ""),
            "input_tensor_shape": list(runtime_contract.get("input_tensor_shape") or []),
            "input_resize_mode": str(runtime_contract.get("input_resize_mode") or "direct_resize"),
        }

    def _infer_bundle_trace_image_size(self, manifest: dict[str, object]) -> tuple[int, int]:
        shape = list(manifest.get("input_tensor_shape") or [])
        if len(shape) >= 4:
            return int(shape[-2]), int(shape[-1])
        return 8, 8


def _runtime_contract_for_model(
    model: nn.Module,
    example: torch.Tensor,
    *,
    model_id: str = "tiny",
    split: str = TINY_SPLIT,
    model_version: str = "0",
) -> dict[str, object]:
    split_spec = make_split_spec(split, dynamic_batch=(1, 64), trainable=True)
    runtime = prepare_split_runtime(model, example, split_spec)
    return build_runtime_contract(
        logical_split_id=split,
        trace_signature=trace_signature(runtime),
        trace_device_type="cpu",
        runtime_backend="torchlens_native",
        boundary_tensor_labels=[
            str(label) for label in list(getattr(runtime.plan, "boundary_nodes", ()) or [])
        ],
        boundary_schema=dict(getattr(runtime.plan, "boundary_specs", {}) or {}),
        model_id=model_id,
        model_version=model_version,
        input_tensor_shape=[int(dim) for dim in example.shape],
        input_resize_mode="direct_resize",
        feature_layout={},
    )


def _manifest(
    runtime_contract: Mapping[str, object],
    *,
    model_id: str = "tiny",
    split: str = TINY_SPLIT,
    split_config_id: str = "split-a",
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model": {"model_id": model_id, "model_version": "0"},
        "front_version": "0",
        "split_config_id": split_config_id,
        "canonical_split_key": split,
        "input_tensor_shape": list(runtime_contract.get("input_tensor_shape") or []),
        "input_resize_mode": "direct_resize",
        "samples": [],
        "split_plan": {
            "split_config_id": split_config_id,
            "canonical_split_key": split,
            "edge_split_id": split,
            "split_granularity": "operation",
            "trace_batch_size": 1,
            "trace_batch_mode": "batch_1",
            "dynamic_batch": [1, 64],
            "runtime_contract": dict(runtime_contract),
        },
    }


def _template_key(
    harness: RuntimeTemplateHarness,
    manifest: Mapping[str, object],
    example: torch.Tensor,
    *,
    model_name: str = "tiny",
    split: str = TINY_SPLIT,
):
    split_plan = dict(manifest.get("split_plan") or {})
    return fixed_split_runtime_template_key(
        model_name=model_name,
        model_family=model_name,
        split_spec=make_split_spec(split, dynamic_batch=(1, 64), trainable=True),
        example_inputs=example,
        graph_signature=str(dict(split_plan.get("runtime_contract") or {}).get("trace_signature")),
        split_plan_hash=harness._fixed_split_template_structural_plan_hash(split_plan),
        canonical_split_key=split,
    )


def test_default_runtime_template_build_does_not_call_dynamic_batch_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    torch.manual_seed(101)
    harness = RuntimeTemplateHarness()
    model = TinyImageRuntimeModel().eval()
    example = torch.randn(1, 3, 8, 8)
    manifest = _manifest(_runtime_contract_for_model(model, example))
    template_key = _template_key(harness, manifest, example)

    def fail_validation(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("default runtime template build must not smoke-validate")

    monkeypatch.setattr(harness, "_validate_dynamic_batch_trainability", fail_validation)

    template = harness._build_fixed_split_runtime_template(
        model,
        manifest,
        bundle_root=str(tmp_path),
        template_key=template_key,
        trace_sample_input=example,
        runtime_batch_size=8,
    )

    assert template.cache_key == template_key


def test_default_runtime_template_info_logs_exclude_diagnostic_fields(tmp_path: Path) -> None:
    torch.manual_seed(103)
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="INFO",
        format="{message}",
    )
    try:
        harness = RuntimeTemplateHarness()
        model = TinyImageRuntimeModel().eval()
        example = torch.randn(1, 3, 8, 8)
        manifest = _manifest(_runtime_contract_for_model(model, example))

        harness._build_bundle_splitter(
            model,
            manifest,
            bundle_root=str(tmp_path),
            trace_sample_input=example,
            runtime_batch_size=8,
        )
    finally:
        logger.remove(sink_id)

    text = "\n".join(messages)
    assert f"Runtime template miss: model=tiny split={TINY_SPLIT}." in text
    assert f"Runtime prepared: model=tiny split={TINY_SPLIT}" in text
    assert f"Runtime bound: split={TINY_SPLIT}" in text
    for forbidden in (
        "runtime template cache key",
        "runtime_version",
        "graph_signature",
        "split_plan_hash",
        "symbolic_input_schema_hash",
        "validated_batch_max",
        "runtime_batch_validation_signature",
        "FixedSplitRuntimeTemplateKey(",
    ):
        assert forbidden not in text


def test_same_rfdetr_split_template_key_hits_across_runtime_batches() -> None:
    harness = RuntimeTemplateHarness()
    contract = build_runtime_contract(
        logical_split_id="after:linear_4_32",
        trace_signature="rfdetr-graph",
        trace_device_type="cpu",
        runtime_backend="torchlens_native",
        boundary_tensor_labels=["linear_4_32"],
        boundary_schema={},
        model_id="rfdetr_nano",
        model_version="0",
        input_tensor_shape=[1, 3, 384, 384],
        input_resize_mode="direct_resize",
        feature_layout={},
    )
    manifest_a = _manifest(
        contract,
        model_id="rfdetr_nano",
        split="after:linear_4_32",
        split_config_id="rfdetr-split",
    )
    manifest_b = _manifest(
        contract,
        model_id="rfdetr_nano",
        split="after:linear_4_32",
        split_config_id="rfdetr-split",
    )
    manifest_b["split_plan"] = {
        **dict(manifest_b["split_plan"]),
        "trace_batch_size": 32,
        "validated_batch_max": 32,
        "runtime_batch_validation_signature": "round-specific-validation",
        "dynamic_batch": [1, 128],
    }
    key_a = harness._fixed_split_runtime_template_key(
        model_name="rfdetr_nano",
        manifest=manifest_a,
        runtime_batch_size=8,
    )
    key_b = harness._fixed_split_runtime_template_key(
        model_name="rfdetr_nano",
        manifest=manifest_b,
        runtime_batch_size=32,
    )
    cache = FixedSplitRuntimeTemplateCache()
    split_spec = make_split_spec("after:linear_4_32", dynamic_batch=(1, 64), trainable=True)
    build_calls = {"count": 0}

    def build_template() -> FixedSplitRuntimeTemplate:
        build_calls["count"] += 1
        return FixedSplitRuntimeTemplate(
            cache_key=key_a,
            runtime=object(),
            split_spec=split_spec,
            model_name="rfdetr_nano",
            model_family="rfdetr",
            graph_signature="rfdetr-graph",
            symbolic_input_schema_hash=key_a.symbolic_input_schema_hash,
            split_plan_hash=key_a.split_plan_hash,
        )

    lookup_a = cache.get_or_create_lookup(
        key_a,
        build_template,
        log_label="model=rfdetr_nano split=after:linear_4_32",
    )
    lookup_b = cache.get_or_create_lookup(
        key_b,
        lambda: (_ for _ in ()).throw(AssertionError("template should have hit cache")),
        log_label="model=rfdetr_nano split=after:linear_4_32",
    )

    assert key_a == key_b
    assert lookup_a.cache_status == "miss"
    assert lookup_b.cache_status == "hit"
    assert build_calls["count"] == 1
