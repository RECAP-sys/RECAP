from __future__ import annotations

import gzip
import inspect
import threading
import warnings
from dataclasses import replace

import pytest
import torch
from torch import nn

import model_management.fixed_split as fixed_split_module
import model_management.split_runtime.template as runtime_template_module
from model_management.fixed_split import (
    SplitConstraints,
    _enumerate_feasible_candidates,
    _select_candidate,
    compute_fixed_split_for_model,
)
from model_management.fixed_split_runtime_template import (
    FixedSplitRuntimeTemplate,
    bind_request_runtime_from_template,
    bind_request_splitter_from_template,
    fixed_split_runtime_template_key,
)
from model_management.payload import boundary_payload_from_tensors
from model_management.split_candidate import SplitCandidate
from model_management.split_model_adapters import (
    _RFDETR_PACKED_AUX_OUTPUTS_MARKER,
    _pack_rfdetr_aux_outputs,
    _unpack_rfdetr_aux_outputs,
)
from model_management.split_runtime import (
    BOUNDARY_CACHE_PROTOCOL,
    BoundaryPayloadCacheCodec,
    make_split_spec,
)
from model_management.split_runtime import torchlens_native_runtime as native_runtime
from model_management.split_runtime.torchlens_forward_guard import torchlens_forward_guard
from model_management.universal_model_split import (
    UniversalModelSplitter,
    collect_suffix_trainable_parameters,
)


class TinySplitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 5)
        self.act = nn.ReLU()
        self.head = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.stem(x)))


def _prepared_splitter() -> tuple[TinySplitModel, torch.Tensor, UniversalModelSplitter]:
    torch.manual_seed(7)
    model = TinySplitModel().eval()
    example = torch.randn(2, 4)
    splitter = UniversalModelSplitter(device="cpu").trace(model, example, model_name="tiny")
    candidates = splitter.enumerate_candidates()
    assert candidates
    splitter.split(candidate=candidates[0])
    return model, example, splitter


def _fixed_runtime_template(
    model: TinySplitModel,
    example: torch.Tensor,
    splitter: UniversalModelSplitter,
) -> FixedSplitRuntimeTemplate:
    del model
    runtime = splitter.runtime
    key = fixed_split_runtime_template_key(
        model_name="tiny",
        model_family="tiny",
        split_spec=runtime.split_spec,
        example_inputs=example,
        graph_signature=str(getattr(runtime.trace_graph, "graph_shape_hash", "") or "tiny"),
        split_plan_hash="tiny-plan",
    )
    return FixedSplitRuntimeTemplate(
        cache_key=key,
        runtime=runtime,
        split_spec=runtime.split_spec,
        model_name="tiny",
        model_family="tiny",
        graph_signature=str(getattr(runtime.trace_graph, "graph_shape_hash", "") or "tiny"),
        symbolic_input_schema_hash=key.symbolic_input_schema_hash,
        split_plan_hash=str(key.split_plan_hash),
        mode=getattr(runtime.split_spec, "mode", "generated_eager"),
        runtime_device="cpu",
    )


def test_template_cache_key_excludes_validation_version_and_dynamic_batch_fields() -> None:
    spec = make_split_spec("after:head", dynamic_batch=(1, 8), trainable=True)
    dynamic_spec = make_split_spec("after:head", dynamic_batch=(1, 64), trainable=True)
    example = torch.randn(1, 4)
    old_identity_fields = {
        "trace_batch_size",
        "validated_batch_max",
        "runtime_batch_validation_signature",
        "runtime_version",
        "dynamic_batch",
        "version",
        "mode",
    }

    key_parameters = inspect.signature(fixed_split_runtime_template_key).parameters
    assert old_identity_fields.isdisjoint(key_parameters)

    key_a = fixed_split_runtime_template_key(
        model_name="tiny",
        model_family="tiny",
        split_spec=spec,
        example_inputs=example,
        graph_signature="graph",
        split_plan_hash="plan",
        canonical_split_key="after:head",
    )
    key_b = fixed_split_runtime_template_key(
        model_name="tiny",
        model_family="tiny",
        split_spec=dynamic_spec,
        example_inputs=example,
        graph_signature="graph",
        split_plan_hash="plan",
        canonical_split_key="after:head",
    )

    assert key_a == key_b
    assert key_a.as_dict() == {
        "model_name": "tiny",
        "model_family": "tiny",
        "graph_signature": "graph",
        "split_plan_hash": "plan",
        "symbolic_input_schema_hash": key_b.symbolic_input_schema_hash,
        "canonical_split_key": "after:head",
    }
    assert "runtime_version" not in key_a.as_dict()
    assert "adapter_version" not in key_a.as_dict()
    assert "dynamic_batch" not in key_a.as_dict()
    assert "mode" not in key_a.as_dict()
    assert "trace_batch_size" not in key_a.as_dict()
    assert "validated_batch_max" not in key_a.as_dict()
    assert "runtime_batch_validation_signature" not in key_a.as_dict()


def _poison_trace_graph_deepcopy(runtime) -> None:
    node = next(iter(runtime.trace_graph.nodes.values()))
    setattr(
        node.layer,
        "_recap_nonleaf_deepcopy_probe",
        torch.ones(2, requires_grad=True) * 2,
    )


def test_template_binding_same_model_returns_template_runtime() -> None:
    model, example, splitter = _prepared_splitter()
    template = _fixed_runtime_template(model, example, splitter)

    assert bind_request_runtime_from_template(template, model=None) is splitter.runtime
    assert bind_request_runtime_from_template(template, model=model) is splitter.runtime


def test_template_binding_different_model_requires_example_inputs() -> None:
    model, example, splitter = _prepared_splitter()
    template = _fixed_runtime_template(model, example, splitter)

    with pytest.raises(RuntimeError, match="cannot be rebound by deepcopy"):
        bind_request_runtime_from_template(template, model=TinySplitModel().eval())


def test_template_binding_different_model_reprepares_runtime(monkeypatch) -> None:
    model, example, splitter = _prepared_splitter()
    _poison_trace_graph_deepcopy(splitter.runtime)
    template = _fixed_runtime_template(model, example, splitter)
    prepared_calls: list[tuple[nn.Module, object, object, object]] = []
    real_prepare = runtime_template_module.prepare_split_runtime

    def spy_prepare(model_arg, example_inputs_arg, split_spec_arg, mode=None):
        prepared_calls.append((model_arg, example_inputs_arg, split_spec_arg, mode))
        return real_prepare(model_arg, example_inputs_arg, split_spec_arg, mode=mode)

    monkeypatch.setattr(runtime_template_module, "prepare_split_runtime", spy_prepare)

    request_model = TinySplitModel().eval()
    request_runtime = bind_request_runtime_from_template(
        template,
        model=request_model,
        example_inputs=example,
    )

    assert prepared_calls
    assert prepared_calls[0][0] is request_model
    assert request_runtime is not splitter.runtime
    with torch.no_grad():
        replayed = request_runtime.run_suffix(request_runtime.run_prefix(example))
        expected = request_model(example)
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def test_template_splitter_binding_reprepares_for_different_model() -> None:
    model, example, splitter = _prepared_splitter()
    _poison_trace_graph_deepcopy(splitter.runtime)
    template = _fixed_runtime_template(model, example, splitter)
    request_model = TinySplitModel().eval()

    request_splitter, candidate = bind_request_splitter_from_template(
        request_model,
        template,
        example_inputs=example,
        device="cpu",
    )

    assert candidate is not None
    boundary = request_splitter.edge_forward(example)
    replayed = request_splitter.cloud_forward(boundary)
    with torch.no_grad():
        expected = request_model(example)
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def test_native_split_replays_and_enumerates_compute_boundaries() -> None:
    model, example, splitter = _prepared_splitter()

    candidates = splitter.enumerate_candidates()
    assert all(candidate.candidate_id.startswith("after:") for candidate in candidates)
    assert all(
        candidate.metadata.get("runtime_backend") == "torchlens_native" for candidate in candidates
    )

    boundary = splitter.edge_forward(example)
    replayed = splitter.cloud_forward(boundary)
    assert torch.allclose(replayed, model(example), atol=1e-5, rtol=1e-5)
    assert set(boundary.tensors)
    assert set(boundary.spec) == set(boundary.tensors)
    assert "graph_shape_hash" in boundary.metadata


def test_boundary_cache_uses_torchlens_protocol_and_rejects_old_protocol(tmp_path) -> None:
    _model, example, splitter = _prepared_splitter()
    boundary = splitter.edge_forward(example)
    codec = BoundaryPayloadCacheCodec(splitter.runtime)

    cache_path = tmp_path / "feature.pt.gz"
    record = codec.save(cache_path, boundary)
    assert record["cache_protocol"] == BOUNDARY_CACHE_PROTOCOL

    loaded = codec.load(cache_path)
    parts = codec.split_batch(loaded)
    assert len(parts) == int(example.shape[0])
    assert codec.collate(parts).batch_size == int(example.shape[0])

    old_path = tmp_path / "old.pt.gz"
    with gzip.open(old_path, "wb") as handle:
        torch.save({"cache_protocol": "old-boundary-cache", "intermediate": boundary}, handle)
    with pytest.raises(RuntimeError, match="rebuild feature cache"):
        codec.load(old_path)


def test_boundary_cache_restores_runtime_dtype_for_compressed_payloads() -> None:
    model, example, splitter = _prepared_splitter()
    boundary = splitter.edge_forward(example)
    compressed = replace(
        boundary,
        tensors={
            label: tensor.to(torch.float16) if tensor.is_floating_point() else tensor
            for label, tensor in dict(boundary.tensors).items()
        },
    )
    assert any(tensor.dtype == torch.float16 for tensor in compressed.tensors.values())

    codec = BoundaryPayloadCacheCodec(splitter.runtime)
    restored = codec.to_runtime_device(compressed)

    for label, tensor in restored.tensors.items():
        expected_dtype = restored.spec[label].dtype
        assert tensor.dtype == expected_dtype
    codec.validate(restored)
    replayed = splitter.cloud_forward(compressed)
    assert torch.allclose(replayed, model(example), atol=1e-2, rtol=1e-2)


def test_splitter_corrects_folded_boundary_batch_metadata() -> None:
    tensors = {"folded": torch.arange(8, dtype=torch.float32).reshape(8, 1)}
    boundary = boundary_payload_from_tensors(
        tensors,
        split_id="after:folded",
        graph_signature="folded-test",
        batch_size=8,
        schema={
            "folded": {
                "canonical_id": "folded",
                "torchlens_label": "folded",
                "module_path": "fake",
                "op_type": "reshape",
                "shape": ("B*4", 1),
                "dtype": torch.float32,
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            }
        },
    )

    class FoldedBoundaryRuntime:
        boundary_spec = boundary.spec

        def run_prefix(self, *inputs):
            del inputs
            return boundary

    runtime = FoldedBoundaryRuntime()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.runtime = runtime

    corrected = splitter.edge_forward(torch.randn(2, 3, 4, 4))

    assert corrected.batch_size == 2
    parts = BoundaryPayloadCacheCodec(runtime).split_batch(corrected)
    assert len(parts) == 2
    assert all(part.batch_size == 1 for part in parts)
    assert [tuple(part.tensors["folded"].shape) for part in parts] == [(4, 1), (4, 1)]
    assert torch.equal(parts[0].tensors["folded"], tensors["folded"][:4])
    assert torch.equal(parts[1].tensors["folded"], tensors["folded"][4:])


def test_suffix_trainable_parameters_come_from_trace_plan() -> None:
    _model, _example, splitter = _prepared_splitter()
    params = collect_suffix_trainable_parameters(splitter)
    assert params
    assert all(parameter.requires_grad for parameter in params)


def test_fixed_split_contract_is_torchlens_native() -> None:
    model, example, _splitter = _prepared_splitter()
    plan = compute_fixed_split_for_model(
        model,
        SplitConstraints(validate_candidates=True, max_payload_bytes=1 << 20),
        sample_input=example,
        model_name="tiny",
    )
    assert "plan_version" not in plan.to_dict()
    assert plan.runtime_contract["runtime_backend"] == "torchlens_native"

    required = {
        "canonical_id",
        "torchlens_label",
        "module_path",
        "op_type",
        "symbolic_shape",
        "dtype",
        "requires_grad",
        "role",
        "output_index",
        "device_policy",
    }
    assert plan.runtime_contract["boundary_schema"]
    for schema in plan.runtime_contract["boundary_schema"].values():
        assert required <= set(schema)
        assert "device" not in schema


def test_fixed_split_enumerates_full_pool_before_constraint_filtering() -> None:
    def candidate(candidate_id: str, *, edge_parameters: int) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=[],
            cloud_nodes=[],
            boundary_edges=[],
            boundary_tensor_labels=["x"],
            edge_input_labels=[],
            cloud_input_labels=["x"],
            cloud_output_labels=[],
            estimated_edge_flops=0.0,
            estimated_cloud_flops=0.0,
            estimated_payload_bytes=1024,
            estimated_privacy_risk=1.0 / max(1, edge_parameters),
            estimated_latency=1024.0,
            is_trainable_tail=True,
            is_validated=True,
            legacy_layer_index=0,
            boundary_count=1,
            edge_parameter_count=edge_parameters,
            total_parameter_count=1000,
            edge_parameter_ratio=float(edge_parameters) / 1000.0,
        )

    class FakeSplitter:
        def __init__(self) -> None:
            self.seen_kwargs = None
            self._candidates = [
                candidate("after:privacy_fail_0", edge_parameters=10),
                candidate("after:privacy_fail_1", edge_parameters=10),
                candidate("after:privacy_fail_2", edge_parameters=10),
                candidate("after:privacy_fail_3", edge_parameters=10),
                candidate("after:eligible", edge_parameters=200),
            ]

        def enumerate_candidates(self, **kwargs):
            self.seen_kwargs = dict(kwargs)
            return list(self._candidates)

    splitter = FakeSplitter()
    constraints = SplitConstraints(
        privacy_leakage_upper_bound=0.01,
        max_layer_freezing_ratio=0.75,
    )

    eligible, stats = _enumerate_feasible_candidates(splitter, constraints)

    assert splitter.seen_kwargs == {
        "max_boundary_count": constraints.max_boundary_count,
        "max_payload_bytes": constraints.max_payload_bytes,
    }
    assert [item[0].candidate_id for item in eligible] == ["after:eligible"]
    assert stats.total_candidates == 5
    assert stats.eligible_candidates == 1
    assert stats.rejected_privacy == 4


def test_fixed_split_validates_all_eligible_candidates_without_limit() -> None:
    def candidate(candidate_id: str, *, payload_bytes: int) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=[],
            cloud_nodes=[],
            boundary_edges=[],
            boundary_tensor_labels=["x"],
            edge_input_labels=[],
            cloud_input_labels=["x"],
            cloud_output_labels=[],
            estimated_edge_flops=0.0,
            estimated_cloud_flops=0.0,
            estimated_payload_bytes=payload_bytes,
            estimated_privacy_risk=0.0,
            estimated_latency=float(payload_bytes),
            is_trainable_tail=True,
            is_validated=True,
            legacy_layer_index=0,
            boundary_count=1,
            edge_parameter_count=1000,
            total_parameter_count=1000,
            edge_parameter_ratio=1.0,
        )

    class FakeSplitter:
        def __init__(self) -> None:
            self.validated: list[str] = []

        def split(self, *, candidate):
            return candidate

        def validate_candidate(self, candidate, **_kwargs):
            self.validated.append(candidate.candidate_id)
            return {
                "success": candidate.candidate_id == "after:success",
                "tail_trainability": True,
                "error": None if candidate.candidate_id == "after:success" else "replay failed",
            }

    runtime = FakeSplitter()
    constraints = SplitConstraints(validate_candidates=True)
    eligible = [
        (candidate("after:fail", payload_bytes=1), 0.0, 0.0),
        (candidate("after:success", payload_bytes=2), 0.0, 0.0),
    ]

    chosen, _privacy, _freezing, _profile, report = _select_candidate(
        runtime,
        eligible,
        constraints,
    )

    assert chosen.candidate_id == "after:success"
    assert runtime.validated == ["after:fail", "after:success"]
    assert report is not None
    assert report["success"] is True


def test_fixed_split_blacklists_batch_validation_failure_without_resorting() -> None:
    def candidate(candidate_id: str, *, payload_bytes: int) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=[],
            cloud_nodes=[],
            boundary_edges=[],
            boundary_tensor_labels=["x"],
            edge_input_labels=[],
            cloud_input_labels=["x"],
            cloud_output_labels=[],
            estimated_edge_flops=0.0,
            estimated_cloud_flops=0.0,
            estimated_payload_bytes=payload_bytes,
            estimated_privacy_risk=0.0,
            estimated_latency=float(payload_bytes),
            is_trainable_tail=True,
            is_validated=True,
            legacy_layer_index=0,
            boundary_count=1,
            edge_parameter_count=1000,
            total_parameter_count=1000,
            edge_parameter_ratio=1.0,
        )

    class FakeSplitter:
        def __init__(self) -> None:
            self.validated: list[tuple[str, tuple[int, ...]]] = []

        def split(self, *, candidate):
            return candidate

        def validate_candidate(self, candidate, *, validation_sample_inputs, **_kwargs):
            batches = tuple(int(sample.shape[0]) for sample in validation_sample_inputs)
            self.validated.append((candidate.candidate_id, batches))
            ok = candidate.candidate_id == "after:success"
            return {
                "success": ok,
                "tail_trainability": True,
                "validation_batches": list(batches),
                "error": None if ok else "batch=1 replay failed",
            }

    runtime = FakeSplitter()
    constraints = SplitConstraints(validate_candidates=True)
    eligible = [
        (candidate("after:first", payload_bytes=1), 0.0, 0.0),
        (candidate("after:success", payload_bytes=2), 0.0, 0.0),
    ]

    chosen, _privacy, _freezing, _profile, report = _select_candidate(
        runtime,
        eligible,
        constraints,
        validation_sample_inputs=[torch.zeros(1, 4), torch.zeros(4, 4)],
    )

    assert chosen.candidate_id == "after:success"
    assert runtime.validated == [
        ("after:first", (1, 4)),
        ("after:success", (1, 4)),
    ]
    assert report is not None
    assert report["validation_batches"] == [1, 4]


def test_fixed_split_uses_configured_training_batch_when_validation_batches_omitted() -> None:
    def candidate(candidate_id: str) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=[],
            cloud_nodes=[],
            boundary_edges=[],
            boundary_tensor_labels=["x"],
            edge_input_labels=[],
            cloud_input_labels=["x"],
            cloud_output_labels=[],
            estimated_edge_flops=0.0,
            estimated_cloud_flops=0.0,
            estimated_payload_bytes=1,
            estimated_privacy_risk=0.0,
            estimated_latency=1.0,
            is_trainable_tail=True,
            is_validated=True,
            legacy_layer_index=0,
            boundary_count=1,
            edge_parameter_count=1000,
            total_parameter_count=1000,
            edge_parameter_ratio=1.0,
        )

    class FixedSplitConfig:
        privacy_leakage_upper_bound = 0.15
        configured_training_batch = 4
        validate_candidates = True

    class TraceGraph:
        graph_shape_hash = "fake-graph"

    class RuntimeObject:
        trace_graph = TraceGraph()

    class FakeSplitter:
        def __init__(self, model: nn.Module) -> None:
            self.model = model
            self.runtime = RuntimeObject()
            self.validated: list[tuple[int, ...]] = []

        def enumerate_candidates(self, **_kwargs):
            return [candidate("after:success")]

        def split(self, *, candidate):
            return candidate

        def validate_candidate(self, _candidate, *, validation_sample_inputs, **_kwargs):
            batches = tuple(int(sample.shape[0]) for sample in validation_sample_inputs)
            self.validated.append(batches)
            return {
                "success": True,
                "tail_trainability": True,
                "validation_batches": list(batches),
                "error": None,
            }

    model = TinySplitModel()
    splitter = FakeSplitter(model)
    constraints = SplitConstraints.from_config(FixedSplitConfig())

    plan = compute_fixed_split_for_model(
        model,
        constraints,
        sample_input=torch.zeros(1, 4),
        model_name="tiny",
        splitter=splitter,
    )

    assert constraints.configured_training_batch == 4
    assert splitter.validated == [(1, 4)]
    assert plan.validation["validation_batches"] == [1, 4]


def test_fixed_split_recomputes_stale_cache_with_initially_untraced_splitter(
    tmp_path, monkeypatch
) -> None:
    class TraceGraph:
        graph_shape_hash = "new-trace"

    class RuntimeObject:
        trace_graph = TraceGraph()

    class FakeSplitter:
        def __init__(self) -> None:
            self.runtime = None
            self.model = None
            self.trace_calls = 0

        def trace(self, model, sample_input, **kwargs):
            del sample_input, kwargs
            self.model = model
            self.runtime = RuntimeObject()
            self.trace_calls += 1
            return self

    def split_plan(split_config_id: str, trace_signature: str) -> fixed_split_module.SplitPlan:
        constraints = SplitConstraints(validate_candidates=False)
        return fixed_split_module.SplitPlan(
            split_config_id=split_config_id,
            canonical_split_key="after:cached",
            edge_split_id="after:cached",
            model_name="tiny",
            candidate_id="after:cached",
            split_index=0,
            split_label="after:cached",
            boundary_tensor_labels=["x"],
            runtime_contract={
                "logical_split_id": "after:cached",
                "model_version": "0",
            },
            input_tensor_shape=[1, 4],
            input_resize_mode="direct_resize",
            front_version="0",
            payload_bytes=1,
            privacy_metric=0.0,
            privacy_risk=0.0,
            layer_freezing_ratio=0.0,
            constraints=fixed_split_module._constraints_payload(constraints),
            trace_signature=trace_signature,
        )

    cache_path = tmp_path / "fixed_split_plan.json"
    fixed_split_module.persist_split_plan(str(cache_path), split_plan("stale", "old-trace"))
    recomputed_plan = split_plan("recomputed", "new-trace")
    compute_calls = []

    def fake_compute_fixed_split_for_model(*args, **kwargs):
        compute_calls.append((args, kwargs))
        assert kwargs["splitter"].runtime is not None
        return recomputed_plan

    monkeypatch.setattr(
        fixed_split_module,
        "compute_fixed_split_for_model",
        fake_compute_fixed_split_for_model,
    )

    splitter = FakeSplitter()
    result = fixed_split_module.load_or_compute_fixed_split_plan(
        TinySplitModel(),
        SplitConstraints(validate_candidates=False),
        sample_input=torch.zeros(1, 4),
        model_name="tiny",
        cache_path=str(cache_path),
        splitter=splitter,
        validate_cached_plan=False,
    )

    cached_after_recompute = fixed_split_module.load_split_plan(str(cache_path))
    assert result.split_config_id == "recomputed"
    assert splitter.trace_calls == 1
    assert len(compute_calls) == 1
    assert cached_after_recompute is not None
    assert cached_after_recompute.split_config_id == "recomputed"


def test_rfdetr_aux_outputs_pack_uses_tensor_marker_for_split_replay() -> None:
    aux_outputs = [
        {
            "pred_logits": torch.randn(2, 3, 4),
            "pred_boxes": torch.randn(2, 3, 4),
        },
        {
            "pred_logits": torch.randn(2, 3, 4),
            "pred_boxes": torch.randn(2, 3, 4),
        },
    ]

    packed = _pack_rfdetr_aux_outputs(aux_outputs)

    assert isinstance(packed, dict)
    assert isinstance(packed[_RFDETR_PACKED_AUX_OUTPUTS_MARKER], torch.Tensor)
    assert packed[_RFDETR_PACKED_AUX_OUTPUTS_MARKER].dtype == torch.bool
    unpacked = _unpack_rfdetr_aux_outputs(packed)
    assert isinstance(unpacked, list)
    assert len(unpacked) == len(aux_outputs)
    for expected, actual in zip(aux_outputs, unpacked, strict=True):
        assert torch.equal(actual["pred_logits"], expected["pred_logits"])
        assert torch.equal(actual["pred_boxes"], expected["pred_boxes"])


def test_native_runtime_preserves_spec_mode_and_list_inputs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_split(model, example_inputs, spec):
        captured["model"] = model
        captured["example_inputs"] = example_inputs
        captured["spec"] = spec
        return object()

    monkeypatch.setattr(native_runtime.tl, "prepare_split", fake_prepare_split)

    model = TinySplitModel()
    list_input = [torch.randn(2, 4)]
    spec = make_split_spec("50%", mode="compiled")
    native_runtime.prepare_split_runtime(model, list_input, spec)

    example_inputs = captured["example_inputs"]
    assert isinstance(example_inputs, tuple)
    assert example_inputs[0] is list_input
    assert getattr(captured["spec"], "mode") == "compiled"


def test_native_runtime_prepare_split_uses_forward_guard(monkeypatch) -> None:
    entered_prepare = threading.Event()

    def fake_prepare_split(model, example_inputs, spec):
        del model, example_inputs, spec
        entered_prepare.set()
        return object()

    monkeypatch.setattr(native_runtime.tl, "prepare_split", fake_prepare_split)

    model = TinySplitModel()
    example = torch.randn(2, 4)
    spec = make_split_spec("50%")

    with torchlens_forward_guard():
        thread = threading.Thread(
            target=lambda: native_runtime.prepare_split_runtime(model, example, spec),
        )
        thread.start()
        assert not entered_prepare.wait(timeout=0.05)

    assert entered_prepare.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_torchlens_forward_guard_suppresses_tuple_iterator_warning() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with torchlens_forward_guard():
            warnings.warn(
                "TorchLens intervention-ready output traversal does not support tuple_iterator; "
                "falling back to BFS without stable output paths.",
                UserWarning,
            )

    assert captured == []


def test_tradeoff_validation_accepts_success_reports() -> None:
    from tools.run_split_tradeoff_motivation_experiment import _validate_candidate

    class Candidate:
        candidate_id = "after:tiny"

    class Splitter:
        def validate_candidate(self, candidate):
            return {"success": True, "candidate_id": candidate.candidate_id}

    assert _validate_candidate(Candidate(), Splitter(), None)
