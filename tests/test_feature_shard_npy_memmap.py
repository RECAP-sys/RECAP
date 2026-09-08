from __future__ import annotations

import json
import os

import numpy as np
import torch

import cloud.feature_cache.shard_writer as shard_writer
from cloud.feature_cache import FeatureShardStore
from tests.test_feature_shard_common import make_entries, make_folded_entries, runtime_context


def test_npy_memmap_shard_writes_npy_arrays_and_reads_rows(tmp_path, monkeypatch) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_dtype="float16",
        shard_max_samples=8,
    )
    written = store.write_entries(
        make_entries(4),
        runtime_context=runtime_context(),
        generation="gen_000001",
        source="test",
    )
    refs = [entry["feature_ref"] for entry in written]
    shard_dir = refs[0].shard_dir
    assert shard_dir and os.path.isdir(shard_dir)
    assert np.load(os.path.join(shard_dir, "leaf_0.npy"), mmap_mode="r").shape == (4, 1, 2, 3)
    assert np.load(os.path.join(shard_dir, "leaf_1.npy"), mmap_mode="r").shape == (4, 1, 1, 2)
    with open(refs[0].index_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert "format_version" not in metadata
    assert metadata["leaf_specs"]["leaf_0"]["storage_shape"] == [4, 1, 2, 3]
    assert metadata["leaf_specs"]["leaf_0"]["feature_shape_without_batch"] == [2, 3]

    calls = []
    original_load = np.load

    def counting_load(*args, **kwargs):
        calls.append(kwargs.get("mmap_mode"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)
    batch = store.read_batch([refs[3], refs[1]])
    second = store.read_batch([refs[2], refs[0]])
    assert tuple(batch.tensors["boundary"].shape) == (2, 2, 3)
    assert torch.equal(
        second.tensors["skip"][:, 0, 0], torch.tensor([12.0, 10.0], dtype=torch.float16)
    )
    assert calls == ["r", "r"]


def test_npy_memmap_folded_batch_reads_with_symbolic_multiplier(tmp_path) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_dtype="float16",
        shard_max_samples=8,
    )
    written = store.write_entries(
        make_folded_entries(2),
        runtime_context=runtime_context("folded-layout"),
        generation="gen_folded",
        source="test",
    )
    refs = [entry["feature_ref"] for entry in written]
    shard_dir = refs[0].shard_dir
    assert shard_dir and os.path.isdir(shard_dir)
    assert np.load(os.path.join(shard_dir, "leaf_0.npy"), mmap_mode="r").shape == (2, 4, 2)

    batch = store.read_batch([refs[1], refs[0]])

    assert batch.batch_size == 2
    assert tuple(batch.tensors["folded"].shape) == (8, 2)
    assert torch.equal(
        batch.tensors["folded"][:, 0],
        torch.tensor([100, 102, 104, 106, 0, 2, 4, 6], dtype=torch.float16),
    )


def test_npy_memmap_reader_does_not_call_torch_load(tmp_path, monkeypatch) -> None:
    store = FeatureShardStore(
        str(tmp_path), storage_format="npy_memmap_shard", shard_dtype="float16"
    )
    refs = [
        entry["feature_ref"]
        for entry in store.write_entries(
            make_entries(2),
            runtime_context=runtime_context(),
            generation="gen",
            source="test",
        )
    ]

    def fail_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run on shard read")

    monkeypatch.setattr(torch, "load", fail_load)
    assert store.read_batch(refs).batch_size == 2


def test_npy_memmap_writer_keeps_temporary_paths_short(tmp_path, monkeypatch) -> None:
    context = runtime_context("d" * 40)
    context["model_id"] = "rfdetr_nano"
    generation = "edge_sample_store"
    base_len = len(
        shard_writer.FeatureShardWriter(
            root_dir=str(tmp_path),
            storage_format="npy_memmap_shard",
        )._base_dir(context, generation)
    )
    original_dump = shard_writer._atomic_json_dump
    original_mkstemp = shard_writer.tempfile.mkstemp
    original_mkdtemp = shard_writer.tempfile.mkdtemp
    atomic_targets: list[str] = []
    json_tmp_names: list[str] = []
    tmp_dirs: list[str] = []

    def guarded_dump(path, payload):
        atomic_targets.append(path)
        assert len(path) <= base_len + 70
        return original_dump(path, payload)

    def tracking_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        json_tmp_names.append(os.path.basename(path))
        return fd, path

    def tracking_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        tmp_dirs.append(path)
        return path

    monkeypatch.setattr(shard_writer, "_atomic_json_dump", guarded_dump)
    monkeypatch.setattr(shard_writer.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(shard_writer.tempfile, "mkdtemp", tracking_mkdtemp)

    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_dtype="float16",
    )
    refs = [
        entry["feature_ref"]
        for entry in store.write_entries(
            make_entries(1),
            runtime_context=context,
            generation=generation,
            source="test",
        )
    ]

    assert refs[0].shard_dir and os.path.isdir(refs[0].shard_dir)
    assert atomic_targets
    assert tmp_dirs
    assert all(os.path.basename(path).startswith(".npy-") for path in tmp_dirs)
    assert all(not os.path.exists(path) for path in tmp_dirs)
    assert json_tmp_names
    assert all(name.startswith(".json-") and len(name) <= 32 for name in json_tmp_names)


def test_single_sample_shard_does_not_call_torch_stack(tmp_path, monkeypatch) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_max_samples=1,
    )

    def fail_stack(*_args, **_kwargs):
        raise AssertionError("single-sample shard must not copy tensors through torch.stack")

    monkeypatch.setattr(torch, "stack", fail_stack)

    written = store.write_entries(
        make_entries(1),
        runtime_context=runtime_context(),
        generation="single",
        source="test",
    )

    assert len(written) == 1
