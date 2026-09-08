from __future__ import annotations

import torch
import torchlens as tl
from torch import nn

from model_management.split_runtime import (
    get_split_runtime_metadata,
    make_runtime_cache_key,
    make_split_spec,
    prepare_split_runtime,
)
from model_management.split_runtime.torchlens_native_runtime import (
    require_torchlens_native_split_api,
)
from model_management.torchlens_optimized_replay import (
    build_torchscript_split_replay,
)
from model_management.universal_model_split import UniversalModelSplitter


class TinyRuntimeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.stem(x)))


def test_torchlens_public_split_api_exists() -> None:
    require_torchlens_native_split_api()
    for name in (
        "SplitSpec",
        "ReplayBoundary",
        "SplitRuntime",
        "prepare_split",
        "prepare_split_replay",
    ):
        assert hasattr(tl, name)


def test_prepare_split_runtime_replays_batch_one_and_two() -> None:
    torch.manual_seed(11)
    model = TinyRuntimeModel().eval()
    example = torch.randn(2, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )

    for batch_size in (1, 2):
        inputs = torch.randn(batch_size, 4)
        with torch.no_grad():
            expected = model(inputs)
            replayed = runtime.replay(inputs)
        assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)

    metadata = get_split_runtime_metadata(runtime)
    assert metadata["runtime_backend"] == "torchlens_native"
    assert metadata["actual_split_id"]


def test_batch_one_trace_collates_batch_boundaries_for_suffix_training() -> None:
    torch.manual_seed(19)
    model = TinyRuntimeModel().train()
    runtime = prepare_split_runtime(
        model,
        torch.randn(1, 4),
        make_split_spec(
            "percent:50",
            dynamic_batch=(1, 4),
            trainable=True,
            trace_batch_mode="batch_1",
        ),
    )
    parts = [runtime.run_prefix(torch.randn(1, 4)) for _index in range(3)]
    collated = tl.ReplayBoundary.collate(parts)

    runtime.validate_boundary(collated)
    replayed = runtime.run_suffix(collated)
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01)
    loss, grads = runtime.train_suffix(
        collated,
        torch.zeros(3, 2),
        loss_fn=lambda outputs, targets: torch.nn.functional.mse_loss(outputs, targets),
        optimizer=optimizer,
    )

    assert int(collated.batch_size) == 3
    assert tuple(replayed.shape) == (3, 2)
    assert not loss.requires_grad
    assert set(grads)


def test_universal_splitter_suffix_fast_path_preserves_validation(monkeypatch) -> None:
    torch.manual_seed(13)
    model = TinyRuntimeModel().eval()
    example = torch.randn(2, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )
    splitter = UniversalModelSplitter(device="cpu").bind_runtime(runtime, model=model)
    boundary = splitter.edge_forward(example)
    validate_calls = {"count": 0}
    original_validate = runtime.validate_boundary

    def validate_spy(payload):
        validate_calls["count"] += 1
        return original_validate(payload)

    def fail_run_suffix(_payload):
        raise AssertionError("cloud_forward should call the validated suffix segment fast path")

    monkeypatch.setattr(runtime, "validate_boundary", validate_spy)
    monkeypatch.setattr(runtime, "run_suffix", fail_run_suffix)

    with torch.no_grad():
        replayed = splitter.cloud_forward(boundary)
        expected = model(example)

    assert validate_calls["count"] == 1
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def test_torchscript_replay_skips_eager_suffix_and_boundary_validation(monkeypatch) -> None:
    torch.manual_seed(17)
    model = TinyRuntimeModel().eval()
    example = torch.randn(2, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )
    splitter = UniversalModelSplitter(device="cpu").bind_runtime(runtime, model=model)
    splitter.prepare_inference_replay(example)
    validate_calls = {"count": 0}
    run_suffix_calls = {"count": 0}

    def validate_spy(_payload):
        validate_calls["count"] += 1
        raise AssertionError("replay_inference should not revalidate trusted prefix payloads")

    def fail_run_suffix(_payload):
        run_suffix_calls["count"] += 1
        raise AssertionError("replay_inference should call the trusted suffix segment directly")

    monkeypatch.setattr(runtime, "validate_boundary", validate_spy)
    monkeypatch.setattr(runtime, "run_suffix", fail_run_suffix)

    with torch.no_grad():
        replayed, boundary = splitter.replay_inference(example, return_split_output=True)
        expected = model(example)

    assert validate_calls["count"] == 0
    assert run_suffix_calls["count"] == 0
    assert set(boundary.tensors)
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def test_torchscript_split_replay_preserves_boundary_output_and_live_parameters() -> None:
    torch.manual_seed(23)
    model = TinyRuntimeModel().eval()
    example = torch.randn(1, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )

    runner = build_torchscript_split_replay(runtime, (example,))
    with torch.inference_mode():
        expected_boundary = runtime.run_prefix(example)
        expected_output = runtime.run_suffix(expected_boundary)
        actual_boundary = runner.run_prefix(example)
        actual_output = runner.run_suffix(actual_boundary)

    assert expected_boundary.tensors.keys() == actual_boundary.tensors.keys()
    assert all(
        torch.equal(expected_boundary.tensors[label], actual_boundary.tensors[label])
        for label in expected_boundary.tensors
    )
    assert torch.equal(expected_output, actual_output)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.125)
    with torch.inference_mode():
        expected_after_update = model(example)

    def fail_full_forward(_inputs):
        raise AssertionError("optimized split replay must not invoke full model forward")

    model.forward = fail_full_forward
    with torch.inference_mode():
        actual_after_update = runner.run_suffix(runner.run_prefix(example))

    assert torch.allclose(
        actual_after_update,
        expected_after_update,
        atol=1e-5,
        rtol=1e-5,
    )


def test_torchscript_split_replay_accepts_dynamic_batch() -> None:
    torch.manual_seed(29)
    model = TinyRuntimeModel().eval()
    example = torch.randn(1, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )
    runner = build_torchscript_split_replay(runtime, (example,))
    dynamic_input = torch.randn(3, 4)

    with torch.inference_mode():
        boundary = runner.run_prefix(dynamic_input)
        actual = runner.run_suffix(boundary)
        expected = model(dynamic_input)

    assert boundary.batch_size == 3
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_runtime_cache_key_ignores_device_for_split_abi() -> None:
    spec = make_split_spec("after:head", dynamic_batch=(1, 8), trainable=True)
    cpu_inputs = torch.randn(2, 4)
    key = make_runtime_cache_key(
        model_name="tiny",
        model_family="tiny",
        split_spec=spec,
        example_inputs=cpu_inputs,
        graph_signature="graph",
        mode="generated_eager",
    )
    payload = key.as_dict()

    assert "adapter_version" not in payload
    assert "runtime_version" not in payload
    assert "device" not in payload
    assert "runtime_identity_id" not in payload
    assert "runtime_batch_validation_signature" not in payload
