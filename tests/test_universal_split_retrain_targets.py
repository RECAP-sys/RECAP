from __future__ import annotations

import torch

from cloud.feature_cache import FeatureShardStore
from model_management.payload import boundary_payload_from_tensors
from model_management.universal_model_split import load_cached_split_batches


def _record_with_ref(tmp_path, payload):
    store = FeatureShardStore(str(tmp_path / "shards"), storage_format="npy_memmap_shard")
    written = store.write_entries(
        [
            {
                "sample": {"sample_id": "sample-1"},
                "record": {"intermediate": payload},
            }
        ],
        runtime_context={
            "model_id": "unit",
            "model_family": "unit",
            "split_config_id": "split",
            "feature_layout_id": "layout",
            "boundary_id": "after:feat",
        },
        generation="unit",
        source="test",
    )
    return {
        "feature_ref": written[0]["feature_ref"].to_dict(),
        "input_image_size": [1080, 1920],
        "input_tensor_shape": [1, 3, 736, 1280],
        "input_resize_mode": "letterbox",
    }


def test_cached_split_batches_attach_coordinate_metadata_to_targets(tmp_path) -> None:
    payload = boundary_payload_from_tensors(
        {"feat": torch.ones(1, 2, 3)},
        split_id="after:feat",
        graph_signature="unit-test",
        batch_size=1,
    )
    record = _record_with_ref(tmp_path, payload)
    annotation = {
        "boxes": [[10.0, 20.0, 30.0, 40.0]],
        "labels": [1],
        "label_coordinate_space": "original_xyxy",
    }

    batches = load_cached_split_batches(
        cache_path=str(tmp_path),
        all_indices=["sample-1"],
        annotations={"sample-1": annotation},
        batch_size=1,
        runtime=None,
        preloaded_records={"sample-1": record},
    )

    assert len(batches) == 1
    _batch_indices, _boundary, targets = batches[0]
    assert targets[0]["_split_meta"] == {
        "input_image_size": [1080, 1920],
        "input_tensor_shape": [1, 3, 736, 1280],
        "input_resize_mode": "letterbox",
    }
    assert "_split_meta" not in annotation


def test_cached_split_batches_preserve_existing_target_split_meta(tmp_path) -> None:
    payload = boundary_payload_from_tensors(
        {"feat": torch.ones(1, 2, 3)},
        split_id="after:feat",
        graph_signature="unit-test",
        batch_size=1,
    )
    record = _record_with_ref(tmp_path, payload)
    annotation = {
        "boxes": [],
        "labels": [],
        "_split_meta": {
            "input_image_size": [720, 1280],
        },
    }

    batches = load_cached_split_batches(
        cache_path=str(tmp_path),
        all_indices=["sample-1"],
        annotations={"sample-1": annotation},
        batch_size=1,
        runtime=None,
        preloaded_records={"sample-1": record},
    )

    _batch_indices, _boundary, targets = batches[0]
    assert targets[0]["_split_meta"] == {
        "input_image_size": [720, 1280],
        "input_tensor_shape": [1, 3, 736, 1280],
        "input_resize_mode": "letterbox",
    }
