from __future__ import annotations

import types

import torch

from cloud.orchestration import runtime_stage
from model_management.payload import boundary_payload_from_tensors
from model_management.universal_model_split import load_cached_split_batches


class _FakeSplitter:
    def __init__(self) -> None:
        self.split_spec = types.SimpleNamespace(dynamic_batch=(2, 8))
        self.seen_batches: list[int] = []

    def cloud_forward(self, boundary, *, candidate=None):
        del candidate
        self.seen_batches.append(int(boundary.batch_size))
        if int(boundary.batch_size) == 4:
            raise RuntimeError("batch 4 incompatible")
        return object()


def _record(payload):
    return {
        "feature_ref": {},
        "input_image_size": [32, 32],
        "input_tensor_shape": [1, 3, 32, 32],
        "input_resize_mode": "direct_resize",
        "feature_payload": payload,
    }


def test_cached_split_loader_is_public() -> None:
    assert callable(load_cached_split_batches)


def test_cached_split_runtime_batch_size_negotiates_first_valid(monkeypatch) -> None:
    payloads = {
        4: boundary_payload_from_tensors(
            {"feat": torch.ones(4, 2)},
            split_id="after:feat",
            graph_signature="unit",
            batch_size=4,
        ),
        3: boundary_payload_from_tensors(
            {"feat": torch.ones(3, 2)},
            split_id="after:feat",
            graph_signature="unit",
            batch_size=3,
        ),
        2: boundary_payload_from_tensors(
            {"feat": torch.ones(2, 2)},
            split_id="after:feat",
            graph_signature="unit",
            batch_size=2,
        ),
    }
    calls: list[int] = []

    def fake_load_batches(**kwargs):
        batch_size = int(kwargs["batch_size"])
        calls.append(batch_size)
        return [(["sample"], payloads[batch_size], [{"boxes": [], "labels": []}])]

    monkeypatch.setattr(
        runtime_stage,
        "load_cached_split_batches",
        fake_load_batches,
    )
    splitter = _FakeSplitter()

    selected = runtime_stage.negotiate_cached_split_runtime_batch_size(
        model_name="unit",
        training_cache_path="/tmp/cache",
        all_sample_ids=["a", "b", "c", "d"],
        gt_annotations={},
        splitter=splitter,
        candidate=object(),
        configured_batch_size=4,
        trace_batch_size=3,
    )

    assert selected == 3
    assert calls == [4, 3]
    assert splitter.seen_batches == [4, 3]


def test_runtime_identity_uses_public_splitter_dynamic_batch_range() -> None:
    class Harness(runtime_stage.FixedSplitRuntimeContractMixin):
        edge_model_name = "rfdetr_nano"

        def _training_window_manifest_context(self, manifest):
            return dict(manifest["training_window_context"])

    split_spec = types.SimpleNamespace(dynamic_batch=(1, 64), mode="generated_eager")
    runtime = types.SimpleNamespace(
        split_spec=split_spec,
        trace_graph=types.SimpleNamespace(graph_shape_hash="graph"),
        symbolic_input_schema={"input": [1, 3, 384, 384]},
        trace_batch_size=2,
    )
    splitter = types.SimpleNamespace(runtime=runtime)

    identity = Harness()._runtime_identity_for_contract(
        manifest={
            "model": {"model_version": "0"},
            "split_plan": {"logical_split_id": "linear_4_32"},
            "training_window_context": {
                "model_id": "rfdetr_nano",
                "front_version": "0",
                "split_config_id": "split-config",
                "canonical_split_key": "after:linear_4_32",
                "input_tensor_shape": [1, 3, 384, 384],
                "input_resize_mode": "direct_resize",
            },
        },
        splitter=splitter,
        cloud_batch_split_id="after:linear_4_32",
        feature_layout_id="layout",
    )

    assert identity["dynamic_batch"] == [1, 64]
