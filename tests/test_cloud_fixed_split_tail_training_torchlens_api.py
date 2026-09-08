from __future__ import annotations

import torch
from torch import nn

import model_management.universal_model_split as split_module
from cloud.feature_cache import FeatureShardStore
from cloud.training.adapters import train_split_suffix_batch
from model_management.payload import boundary_payload_from_tensors
from model_management.split_runtime import (
    BoundaryPayloadCacheCodec,
    make_split_spec,
    prepare_boundary_for_runtime,
    prepare_split_runtime,
)
from model_management.universal_model_split import load_cached_split_batches


class TinySuffixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.stem(x)))


class CountingOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs) -> None:
        del args, kwargs
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class CountingTorchOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls += 1
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return self.optimizer.step(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.optimizer, name)


class CountingRuntime:
    def __init__(self, spec) -> None:
        self.boundary_spec = spec
        self.train_suffix_calls = 0
        self.validated = 0
        self.boundary_tensors = []

    def validate_boundary(self, boundary) -> None:
        self.validated += 1
        boundary.validate(self.boundary_spec, split_id=boundary.split_id)

    def train_suffix(self, boundary, targets, *, loss_fn, optimizer):
        self.train_suffix_calls += 1
        self.boundary_tensors.append(boundary.tensors["x"])
        self.validate_boundary(boundary)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(boundary.tensors["x"], targets)
        optimizer.step()
        return loss, {}


def test_train_split_suffix_batch_delegates_single_optimizer_step_to_runtime() -> None:
    boundary = boundary_payload_from_tensors(
        {"x": torch.ones(2, 3)},
        split_id="after:x",
        graph_signature="graph",
        batch_size=2,
    )
    runtime = CountingRuntime(boundary.spec)
    optimizer = CountingOptimizer()

    loss = train_split_suffix_batch(
        runtime,
        boundary,
        torch.zeros(2, 3),
        lambda outputs, targets: (outputs - targets).mean(),
        optimizer,
    )

    assert float(loss.item()) == 1.0
    assert runtime.train_suffix_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1


def test_train_split_suffix_batch_calls_train_suffix_once_per_batch() -> None:
    boundary = boundary_payload_from_tensors(
        {"x": torch.ones(1, 3)},
        split_id="after:x",
        graph_signature="graph",
        batch_size=1,
    )
    runtime = CountingRuntime(boundary.spec)
    optimizer = CountingOptimizer()

    for _ in range(3):
        train_split_suffix_batch(
            runtime,
            boundary,
            torch.zeros(1, 3),
            lambda outputs, targets: (outputs - targets).sum(),
            optimizer,
        )

    assert runtime.train_suffix_calls == 3
    assert optimizer.step_calls == 3
    source_ptr = boundary.tensors["x"].data_ptr()
    forwarded_ptrs = [tensor.data_ptr() for tensor in runtime.boundary_tensors]
    assert all(ptr != source_ptr for ptr in forwarded_ptrs)
    assert len(set(forwarded_ptrs)) == len(forwarded_ptrs)


def test_trusted_train_split_suffix_batch_uses_prepared_suffix_directly(monkeypatch) -> None:
    torch.manual_seed(13)
    model = TinySuffixModel().train()
    example = torch.randn(2, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )
    prepared = prepare_boundary_for_runtime(runtime, runtime.run_prefix(example), validate=True)
    captured = {}
    original_forward = runtime.segments.suffix.forward

    def forward_spy(boundary):
        captured["boundary"] = boundary
        return original_forward(boundary)

    def fail_validate(_boundary):
        raise AssertionError("trusted suffix training should not revalidate")

    def fail_train_suffix(*_args, **_kwargs):
        raise AssertionError("trusted suffix training should not call runtime.train_suffix")

    monkeypatch.setattr(runtime.segments.suffix, "forward", forward_spy)
    monkeypatch.setattr(runtime, "validate_boundary", fail_validate)
    monkeypatch.setattr(runtime, "train_suffix", fail_train_suffix)

    optimizer = CountingTorchOptimizer(torch.optim.SGD(model.head.parameters(), lr=0.1))
    before = [parameter.detach().clone() for parameter in model.head.parameters()]
    loss = train_split_suffix_batch(
        runtime,
        prepared,
        torch.zeros(2, 2),
        lambda outputs, targets: torch.nn.functional.mse_loss(outputs, targets),
        optimizer,
        trusted_runtime_boundary=True,
    )

    assert not loss.requires_grad
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1
    assert any(
        not torch.equal(snapshot, parameter.detach())
        for snapshot, parameter in zip(before, model.head.parameters(), strict=True)
    )
    assert any(parameter.grad is not None for parameter in model.head.parameters())
    trusted_boundary = captured["boundary"]
    label = next(iter(prepared.tensors))
    assert trusted_boundary.tensors[label].data_ptr() != prepared.tensors[label].data_ptr()
    assert trusted_boundary.tensors[label].requires_grad
    assert (
        trusted_boundary.metadata["use_live_param_sources"]
        == runtime.split_spec.effective_use_live_param_sources
    )


def test_universal_split_retrain_accumulates_low_precision_loss_in_float32(
    monkeypatch,
    tmp_path,
) -> None:
    load_runtime_args = []

    def fake_load_cached_split_batches(**_kwargs):
        load_runtime_args.append(_kwargs.get("runtime"))
        return [
            (["sample-0"], object(), []),
            (["sample-1"], object(), []),
        ]

    calls = []
    prepared_boundaries = []

    def fake_prepare_boundary_for_runtime(_runtime, boundary, *, validate):
        assert validate
        prepared_boundaries.append(boundary)
        return boundary

    def fake_train_split_suffix_batch(*_args, **kwargs):
        calls.append(bool(kwargs.get("trusted_runtime_boundary")))
        return torch.tensor(65504.0, dtype=torch.float16)

    monkeypatch.setattr(
        split_module,
        "load_cached_split_batches",
        fake_load_cached_split_batches,
    )
    monkeypatch.setattr(
        split_module,
        "train_split_suffix_batch",
        fake_train_split_suffix_batch,
    )
    monkeypatch.setattr(
        split_module,
        "prepare_boundary_for_runtime",
        fake_prepare_boundary_for_runtime,
    )

    losses = split_module.universal_split_retrain(
        model=nn.Linear(1, 1),
        sample_input=torch.zeros(1, 1),
        cache_path=str(tmp_path),
        all_indices=["sample-0", "sample-1"],
        gt_annotations={},
        batch_size=1,
        num_epoch=1,
        loss_fn=lambda *_args: torch.tensor(0.0),
        splitter=object(),
        optimizer=object(),
    )

    assert losses == [65504.0]
    assert load_runtime_args == [None]
    assert len(prepared_boundaries) == 2
    assert calls == [True, True]


def test_loaded_cached_split_batch_can_feed_trusted_suffix_training(tmp_path) -> None:
    torch.manual_seed(17)
    model = TinySuffixModel().train()
    example = torch.randn(2, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )
    prepared = prepare_boundary_for_runtime(runtime, runtime.run_prefix(example), validate=True)
    samples = BoundaryPayloadCacheCodec(runtime).split_batch(prepared)
    store = FeatureShardStore(str(tmp_path / "shards"), storage_format="npy_memmap_shard")
    written = store.write_entries(
        [
            {
                "sample": {"sample_id": f"sample-{index}"},
                "record": {"intermediate": sample},
            }
            for index, sample in enumerate(samples)
        ],
        runtime_context={
            "model_id": "unit",
            "model_family": "unit",
            "split_config_id": "split",
            "feature_layout_id": "layout",
            "boundary_id": str(prepared.split_id),
        },
        generation="unit",
        source="test",
    )
    records = {
        f"sample-{index}": {
            "feature_ref": entry["feature_ref"].to_dict(),
            "input_image_size": [32, 32],
            "input_tensor_shape": [1, 4],
            "input_resize_mode": "direct_resize",
        }
        for index, entry in enumerate(written)
    }
    batches = load_cached_split_batches(
        cache_path=str(tmp_path),
        all_indices=list(records),
        annotations={key: {"boxes": [], "labels": []} for key in records},
        batch_size=2,
        runtime=runtime,
        preloaded_records=records,
    )

    assert len(batches) == 1
    _indices, boundary, targets = batches[0]
    optimizer = CountingTorchOptimizer(torch.optim.SGD(model.head.parameters(), lr=0.1))
    loss = train_split_suffix_batch(
        runtime,
        boundary,
        targets,
        lambda outputs, _targets: outputs.square().mean(),
        optimizer,
        trusted_runtime_boundary=True,
    )

    assert not loss.requires_grad
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1
