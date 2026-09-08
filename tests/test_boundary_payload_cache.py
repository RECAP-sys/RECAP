from __future__ import annotations

import gzip

import pytest
import torch
from torch import nn

from model_management.payload import BoundaryPayload, boundary_payload_from_tensors
from model_management.split_runtime import (
    BoundaryPayloadCacheCodec,
    make_split_spec,
    prepare_split_runtime,
)


def _single_sample_feature_tensors(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, BoundaryPayload) or int(payload.batch_size) != 1:
        raise ValueError("Expected BoundaryPayload with single-sample tensors")
    return dict(payload.tensors)


class TinyBoundaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.stem(x)))


def _runtime_and_boundary():
    torch.manual_seed(13)
    model = TinyBoundaryModel().eval()
    example = torch.randn(2, 4)
    runtime = prepare_split_runtime(
        model,
        example,
        make_split_spec("percent:50", dynamic_batch=(1, 4), trainable=True),
    )
    return runtime, runtime.run_prefix(example)


def test_replay_boundary_cache_round_trips_and_validates(tmp_path) -> None:
    runtime, boundary = _runtime_and_boundary()
    codec = BoundaryPayloadCacheCodec(runtime)
    path = tmp_path / "boundary.pt.gz"

    codec.save(path, boundary)
    with gzip.open(path, "rb") as handle:
        record = torch.load(handle, map_location="cpu", weights_only=False)
    saved_boundary = record["intermediate"]
    assert all(tensor.device.type == "cpu" for tensor in saved_boundary.tensors.values())

    loaded = codec.load(path)
    runtime.validate_boundary(loaded)
    parts = codec.split_batch(loaded)
    assert len(parts) == int(boundary.batch_size)
    collated = codec.collate(parts)
    assert int(collated.batch_size) == int(boundary.batch_size)
    runtime.validate_boundary(collated)


def test_rfdetr_folded_single_sample_boundary_is_accepted() -> None:
    payload = boundary_payload_from_tensors(
        {"rfdetr": torch.zeros(4, 145, 384)},
        split_id="after:rfdetr",
        graph_signature="graph",
        batch_size=1,
        schema={
            "rfdetr": {
                "canonical_id": "rfdetr",
                "torchlens_label": "rfdetr",
                "module_path": "rfdetr",
                "op_type": "reshape",
                "shape": ("B*4", 145, 384),
                "dtype": torch.float32,
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            }
        },
    )

    tensors = _single_sample_feature_tensors(payload)

    assert tuple(tensors["rfdetr"].shape) == (4, 145, 384)


def test_unstructured_multi_sample_tensor_is_not_misclassified_as_single_sample() -> None:
    with pytest.raises(ValueError, match="single-sample tensors"):
        _single_sample_feature_tensors({"plain": torch.zeros(4, 145, 384)})
