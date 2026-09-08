from __future__ import annotations

import io
import json
import sys
import threading
import time
from queue import Full, Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edge.edge_worker import (
    AsyncSampleCollector,
    AsyncSampleWriter,
    EdgeWorker,
    PendingModelUpdate,
    SampleCollectionJob,
    SampleStatsDelta,
    SampleWriteJob,
    _accepted_uploaded_sample_ids,
    _FixedSplitRuntimeError,
    _lower_current_thread_priority,
)
from edge.info import TASK_STATE
from edge.sample_quality import HIGH_QUALITY, LOW_QUALITY
from edge.task import Task
from edge_client import _write_buffered_task_result, _write_task_result
from model_management import object_detection as object_detection_module
from model_management.detectors import legacy_split_model_adapters as split_adapters
from model_management.inference.artifacts import InferenceArtifacts
from model_management.object_detection import Object_Detection
from model_management.payload import boundary_payload_from_tensors
from model_management.universal_model_split import UniversalModelSplitter


def _collection_job(sample_id: str = "sample-1") -> SampleCollectionJob:
    inference = InferenceArtifacts(
        intermediate=None,
        final_detection_boxes=[],
        final_detection_labels=[],
        final_detection_scores=[],
        low_threshold_boxes=[],
        low_threshold_labels=[],
        low_threshold_scores=[],
        confidence=0.0,
    )
    return SampleCollectionJob(
        sample_id=sample_id,
        frame_index=1,
        frame=None,
        inference=inference,
        split_config_id="split-a",
        model_id="model-a",
        model_version="0",
        front_version="0",
        split_key="after:test",
        feature_abi_id="abi-a",
        runtime_contract={"feature_layout_id": "abi-a"},
    )


def test_write_task_result_includes_latency_and_timing() -> None:
    task = Task(1, 7, None, 10.0, (1, 1, 3))
    task.end_time = 10.125
    task.state = TASK_STATE.FINISHED
    task.result_source = "inference"
    task.timing_ms = {"split_prefix_ms": 12.5, "task_complete_ms": 125.0}
    task.add_result([[1, 2, 3, 4]], [5], [0.9])

    handle = io.StringIO()
    _write_task_result(
        handle,
        task,
        model_name="rfdetr_nano",
        model_version="2",
        metadata={
            "video_source": "video_data/road.mp4",
            "video_slug": "road",
            "scenario_name": "road",
            "edge_id": 1,
            "run_id": "recap_road_001",
            "method": "recap",
            "frame_replayable": True,
            "label_schema": "zero_based",
            "class_names": ["car"],
        },
    )

    payload = json.loads(handle.getvalue())
    assert payload["frame_index"] == 7
    assert payload["latency_ms"] == pytest.approx(125.0)
    assert payload["timing_ms"] == {
        "split_prefix_ms": 12.5,
        "task_complete_ms": 125.0,
    }
    assert payload["model_name"] == "rfdetr_nano"
    assert payload["model_version"] == "2"
    assert payload["timestamp_ms"] == 10000
    assert payload["video_slug"] == "road"
    assert payload["run_id"] == "recap_road_001"
    assert payload["frame_replayable"] is True
    assert payload["result"] == {
        "labels": [5],
        "boxes": [[1, 2, 3, 4]],
        "scores": [0.9],
    }


def test_write_task_result_does_not_flush_per_frame() -> None:
    class FlushTrackingBuffer(io.StringIO):
        flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    task = Task(1, 1, None, 1.0, (1, 1, 3))
    task.end_time = 1.01
    task.state = TASK_STATE.FINISHED
    handle = FlushTrackingBuffer()

    _write_task_result(handle, task)

    assert handle.flush_count == 0


def test_buffered_task_result_flushes_every_configured_frame() -> None:
    class FlushTrackingBuffer(io.StringIO):
        flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    handle = FlushTrackingBuffer()
    pending = 0
    for frame_index in range(1, 31):
        task = Task(1, frame_index, None, 1.0, (1, 1, 3))
        task.end_time = 1.01
        task.state = TASK_STATE.FINISHED
        pending = _write_buffered_task_result(
            handle,
            task,
            unflushed_count=pending,
            flush_every_n_frames=30,
        )

    assert handle.flush_count == 1
    assert pending == 0


def test_fixed_inference_threads_use_replay_device_not_global_cuda(monkeypatch) -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.config = SimpleNamespace(
        split_learning=SimpleNamespace(
            fixed_split=SimpleNamespace(inference_num_threads=6),
        ),
    )
    worker.universal_splitter = SimpleNamespace(device=torch.device("cpu"))
    configured: list[int] = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 16)
    monkeypatch.setattr(torch, "set_num_threads", configured.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    worker._configure_inference_replay_threads(torch.ones(1))

    assert configured == [6]


def test_inference_threads_reject_non_positive_value() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.config = SimpleNamespace(
        split_learning=SimpleNamespace(
            fixed_split=SimpleNamespace(inference_num_threads=0),
        ),
    )
    worker.universal_splitter = SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(ValueError, match="must be positive"):
        worker._configure_inference_replay_threads(torch.ones(1))


def test_uploaded_low_quality_delete_helper_uses_low_quality_filter() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    calls: list[tuple[list[str], str | None]] = []

    class Store:
        def delete_samples(self, sample_ids, *, quality_bucket=None):
            calls.append((list(sample_ids), quality_bucket))
            return len(sample_ids)

    worker.sample_store = Store()

    assert worker._delete_uploaded_low_quality_samples(["low-1", "low-2"]) == 2
    assert calls == [(["low-1", "low-2"], LOW_QUALITY)]


def test_cloud_accepted_low_quality_delete_helper_handles_failed_status_message() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    calls: list[tuple[list[str], str | None]] = []
    metrics = []

    class Store:
        def delete_samples(self, sample_ids, *, quality_bucket=None):
            calls.append((list(sample_ids), quality_bucket))
            return len(sample_ids)

    worker.sample_store = Store()
    worker._record_experiment_metric = (
        lambda name, **payload: metrics.append((name, payload))
    )

    message = (
        "fixed-split training failed: validation split unavailable; "
        'accepted_low_quality_sample_ids_json=["low-2","low-1"]'
    )

    assert (
        worker._delete_cloud_accepted_low_quality_samples(
            message,
            ["low-1", "low-2", "low-3"],
            job_id="job-a",
        )
        == 2
    )
    assert calls == [(["low-2", "low-1"], LOW_QUALITY)]
    assert metrics[0][0] == "uploaded_low_quality_samples_deleted"
    assert metrics[0][1]["job_id"] == "job-a"
    assert metrics[0][1]["deleted_sample_count"] == 2


def test_accepted_uploaded_sample_ids_uses_cloud_confirmed_ids_only() -> None:
    message = (
        "Waiting for enough recent training samples: available=64, required=128, "
        "accepted_low_quality_samples=2, uploaded_low_quality_samples=3; "
        'accepted_low_quality_sample_ids_json=["low-2","other","low-2","low-1"]'
    )

    assert _accepted_uploaded_sample_ids(message, ["low-1", "low-2", "low-3"]) == [
        "low-2",
        "low-1",
    ]


def test_accepted_uploaded_sample_ids_keeps_samples_when_cloud_ids_missing() -> None:
    assert (
        _accepted_uploaded_sample_ids(
            "Waiting for enough recent training samples: available=64, required=128.",
            ["low-1"],
        )
        == []
    )


def test_background_thread_priority_is_best_effort(monkeypatch) -> None:
    monkeypatch.setattr("edge.edge_worker.os.name", "posix")
    monkeypatch.setattr("edge.edge_worker.os.PRIO_PROCESS", 0, raising=False)
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        "edge.edge_worker.os.setpriority",
        lambda which, who, priority: calls.append((which, who, priority)),
        raising=False,
    )
    monkeypatch.setattr("edge.edge_worker.threading.get_native_id", lambda: 17)

    assert _lower_current_thread_priority()
    assert calls == [(0, 17, 5)]


def test_windows_background_thread_priority_uses_pointer_sized_handle(monkeypatch) -> None:
    calls: list[tuple[object, int]] = []

    class Function:
        def __init__(self, result):
            self.result = result
            self.restype = None
            self.argtypes = None

        def __call__(self, *args):
            if args:
                calls.append((args[0], args[1]))
            return self.result

    get_current_thread = Function(-2)
    set_thread_priority = Function(1)
    fake_ctypes = SimpleNamespace(
        c_void_p=object(),
        c_int=object(),
        windll=SimpleNamespace(
            kernel32=SimpleNamespace(
                GetCurrentThread=get_current_thread,
                SetThreadPriority=set_thread_priority,
            )
        ),
    )
    monkeypatch.setattr("edge.edge_worker.os.name", "nt")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    assert _lower_current_thread_priority()
    assert get_current_thread.restype is fake_ctypes.c_void_p
    assert set_thread_priority.argtypes == (
        fake_ctypes.c_void_p,
        fake_ctypes.c_int,
    )
    assert calls == [(-2, -1)]


def test_split_observables_can_skip_feature_spectral_entropy(monkeypatch) -> None:
    calls = {"feature_entropy": 0}

    def fake_feature_entropy(_payload):
        calls["feature_entropy"] += 1
        return 0.25

    def fake_runtime_logits(_model, _outputs):
        return torch.tensor([[4.0, 1.0, 0.0], [0.5, 2.0, 0.0]]), "softmax"

    monkeypatch.setattr(
        split_adapters,
        "_summarize_payload_spectral_entropy",
        fake_feature_entropy,
    )
    monkeypatch.setattr(split_adapters, "_extract_runtime_logits", fake_runtime_logits)

    skipped = split_adapters.summarize_split_runtime_observables(
        torch.nn.Identity(),
        outputs=None,
        split_payload={"payload": torch.ones(2, 4)},
        include_feature_spectral_entropy=False,
    )

    assert calls["feature_entropy"] == 0
    assert skipped["feature_spectral_entropy"] is None
    assert skipped["logit_entropy"] is not None
    assert skipped["logit_margin"] is not None

    default = split_adapters.summarize_split_runtime_observables(
        torch.nn.Identity(),
        outputs=None,
        split_payload={"payload": torch.ones(2, 4)},
    )

    assert calls["feature_entropy"] == 1
    assert default["feature_spectral_entropy"] == 0.25
    assert default["logit_entropy"] == skipped["logit_entropy"]


def test_async_sample_collector_fifo_flush_and_close() -> None:
    seen: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def handler(job: SampleCollectionJob) -> None:
        if job.sample_id == "first":
            entered.set()
            assert release.wait(timeout=2.0)
        seen.append(job.sample_id)

    collector = AsyncSampleCollector(handler, maxsize=1)
    try:
        collector.submit_nowait(_collection_job("first"))
        assert entered.wait(timeout=2.0)
        collector.submit_nowait(_collection_job("second"))
        with pytest.raises(Full):
            collector.submit_nowait(_collection_job("third"))

        release.set()
        assert collector.flush(timeout=2.0)
    finally:
        release.set()
        assert collector.close(timeout=2.0)

    assert seen == ["first", "second"]


def test_async_sample_collector_close_timeout_does_not_block_when_queue_full() -> None:
    seen: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def handler(job: SampleCollectionJob) -> None:
        if job.sample_id == "first":
            entered.set()
            assert release.wait(timeout=2.0)
        seen.append(job.sample_id)

    collector = AsyncSampleCollector(handler, maxsize=1)
    try:
        collector.submit_nowait(_collection_job("first"))
        assert entered.wait(timeout=2.0)
        collector.submit_nowait(_collection_job("second"))

        started = time.perf_counter()
        assert collector.close(timeout=0.05) is False
        assert time.perf_counter() - started < 0.5

        release.set()
        assert collector.close(timeout=2.0)
    finally:
        release.set()
        collector.close(timeout=2.0)

    assert seen == ["first", "second"]


def test_sample_collection_drops_without_sync_fallback_when_queue_is_full() -> None:
    class FullCollector:
        def submit_nowait(self, _job):
            raise Full

    handled: list[str] = []
    worker = SimpleNamespace(
        sample_collector=FullCollector(),
        _collect_data_from_job=lambda job: handled.append(job.sample_id),
    )

    queued = EdgeWorker._submit_sample_collection(worker, _collection_job("fallback"))

    assert queued is False
    assert handled == []


def test_strict_sample_collection_uses_explicit_blocking_submit() -> None:
    submitted: list[str] = []

    class StrictCollector:
        def submit_blocking(self, job):
            submitted.append(job.sample_id)

    worker = SimpleNamespace(
        sample_collector=StrictCollector(),
        strict_sample_collection=True,
    )

    assert EdgeWorker._submit_sample_collection(worker, _collection_job("strict"))
    assert submitted == ["strict"]


def _write_job(sample_id: str, quality_bucket: str) -> SampleWriteJob:
    return SampleWriteJob(
        store_kwargs={
            "sample_id": sample_id,
            "frame_index": 1,
            "quality_bucket": quality_bucket,
        },
        stats_delta=SampleStatsDelta.from_values(quality_bucket=quality_bucket),
    )


def test_async_writer_replaces_queued_low_quality_for_high_quality() -> None:
    entered = threading.Event()
    release = threading.Event()
    stored: list[str] = []

    class BlockingStore:
        def store_sample(self, **kwargs):
            if kwargs["sample_id"] == "active-low":
                entered.set()
                assert release.wait(timeout=2.0)
            stored.append(kwargs["sample_id"])
            return SimpleNamespace(**kwargs)

    writer = AsyncSampleWriter(BlockingStore(), maxsize=1)
    try:
        accepted, dropped = writer.submit_nowait(_write_job("active-low", "low_quality"))
        assert accepted and dropped is None
        assert entered.wait(timeout=2.0)
        accepted, dropped = writer.submit_nowait(_write_job("queued-low", "low_quality"))
        assert accepted and dropped is None

        started = time.perf_counter()
        accepted, dropped = writer.submit_nowait(_write_job("high", "high_quality"))

        assert time.perf_counter() - started < 0.1
        assert accepted
        assert dropped is not None
        assert dropped.store_kwargs["sample_id"] == "queued-low"
        release.set()
        assert writer.flush(timeout=2.0)
    finally:
        release.set()
        writer.close(timeout=2.0)

    assert stored == ["active-low", "high"]


def test_split_inference_summarizes_observables_once_without_second_replay(monkeypatch) -> None:
    detector = Object_Detection.__new__(Object_Detection)
    detector.model = torch.nn.Identity()
    detector.model_lock = threading.Lock()
    detector._split_input_resize_mode = "direct_resize"
    detector.threshold_low = 0.1
    detector.config = SimpleNamespace(final_detection_threshold=0.5)
    detector._parse_prediction_output = lambda _output, _threshold: ([], [], [])

    monkeypatch.setattr(
        object_detection_module,
        "prepare_split_runtime_input",
        lambda *_args, **_kwargs: torch.ones(1, 3, 4, 4),
    )
    monkeypatch.setattr(
        object_detection_module,
        "postprocess_split_runtime_output",
        lambda *_args, **_kwargs: [],
    )
    observables_calls = 0

    def summarize(*_args, **_kwargs):
        nonlocal observables_calls
        observables_calls += 1
        return {
            "feature_spectral_entropy": None,
            "logit_entropy": 0.25,
            "logit_margin": 0.5,
            "logit_energy": 1.0,
        }

    monkeypatch.setattr(
        object_detection_module,
        "summarize_split_runtime_observables",
        summarize,
    )

    class Splitter:
        calls = 0
        outputs = {"pred_logits": torch.ones(1, 1, 2)}

        def replay_inference(self, _input, *, return_split_output, profile):
            self.calls += 1
            assert return_split_output
            profile.update({"split_prefix": 1.0, "split_suffix": 2.0})
            return self.outputs, {"payload": torch.ones(1, 2)}

    splitter = Splitter()
    artifacts = detector.infer_sample(np.zeros((4, 4, 3), dtype=np.uint8), splitter=splitter)

    assert splitter.calls == 1
    assert observables_calls == 1
    assert artifacts.logit_entropy == pytest.approx(0.25)
    assert artifacts.timing_ms["split_prefix_ms"] == pytest.approx(1.0)
    assert artifacts.timing_ms["split_suffix_ms"] == pytest.approx(2.0)


def test_prepare_inference_replay_installs_validated_torchscript_runner(
    monkeypatch,
) -> None:
    payload = boundary_payload_from_tensors(
        {"boundary": torch.tensor([[1.0]])},
        split_id="after:test",
        graph_signature="test",
        batch_size=1,
    )

    class Runtime:
        segments = SimpleNamespace(
            suffix=lambda boundary: {"output": boundary.tensors["boundary"] + 1.0}
        )

    class OptimizedRunner:
        def run_prefix(self, _sample_input):
            return payload

        def run_suffix(self, boundary):
            return {"output": boundary.tensors["boundary"] + 1.0}

    splitter = UniversalModelSplitter()
    splitter.runtime = Runtime()
    splitter.edge_forward = lambda _sample_input: payload
    monkeypatch.setattr(
        "model_management.universal_model_split.build_torchscript_split_replay",
        lambda _runtime, _inputs: OptimizedRunner(),
    )

    splitter.prepare_inference_replay(torch.tensor([[1.0]]))
    outputs, returned_payload = splitter.replay_inference(
        torch.tensor([[1.0]]),
        return_split_output=True,
    )

    assert outputs["output"].item() == pytest.approx(2.0)
    assert returned_payload is payload


def test_prepare_inference_replay_rejects_output_mismatch(monkeypatch) -> None:
    payload = boundary_payload_from_tensors(
        {"boundary": torch.tensor([[1.0]])},
        split_id="after:test",
        graph_signature="test",
        batch_size=1,
    )

    class Runtime:
        segments = SimpleNamespace(
            suffix=lambda boundary: {"output": boundary.tensors["boundary"] + 1.0}
        )

    class MismatchedRunner:
        def run_prefix(self, _sample_input):
            return payload

        def run_suffix(self, boundary):
            return {"output": boundary.tensors["boundary"] + 2.0}

    splitter = UniversalModelSplitter()
    splitter.runtime = Runtime()
    splitter.edge_forward = lambda _sample_input: payload
    monkeypatch.setattr(
        "model_management.universal_model_split.build_torchscript_split_replay",
        lambda _runtime, _inputs: MismatchedRunner(),
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        splitter.prepare_inference_replay(torch.tensor([[1.0]]))

    with pytest.raises(RuntimeError, match="has not been prepared"):
        splitter.replay_inference(torch.tensor([[1.0]]))


def test_inference_replay_propagates_runner_failure() -> None:
    class FailingRunner:
        def run_prefix(self, _sample_input):
            raise RuntimeError("runner failed")

    splitter = UniversalModelSplitter()
    splitter.runtime = object()
    splitter._inference_replay_runner = FailingRunner()

    with pytest.raises(RuntimeError, match="runner failed"):
        splitter.replay_inference(torch.tensor([[1.0]]))


def test_enabled_split_runtime_never_falls_back_when_unavailable() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.split_learning_enabled = True
    worker._fixed_split_init_attempted = True
    worker.universal_split_enabled = False
    worker.universal_splitter = None

    with pytest.raises(RuntimeError, match="runtime is unavailable"):
        worker._resolve_active_splitter(None, (4, 4))


def test_initial_model_can_bootstrap_without_confirmed_drift() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.config = SimpleNamespace(
        resource_aware_trigger=SimpleNamespace(
            bootstrap_without_drift=True,
            min_training_samples=64,
        )
    )
    worker.resource_trigger_enabled = True
    worker.resource_trigger = SimpleNamespace(min_training_samples=64)
    worker.model_version = "0"

    assert worker._bootstrap_training_ready(SimpleNamespace(low_quality_count=63)) is False
    assert worker._bootstrap_training_ready(SimpleNamespace(low_quality_count=64)) is True


def test_updated_model_still_requires_confirmed_drift() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.config = SimpleNamespace(
        resource_aware_trigger=SimpleNamespace(
            bootstrap_without_drift=True,
            min_training_samples=64,
        )
    )
    worker.resource_trigger_enabled = True
    worker.resource_trigger = SimpleNamespace(min_training_samples=64)
    worker.model_version = "1"

    assert worker._bootstrap_training_ready(SimpleNamespace(low_quality_count=128)) is False


def test_high_quality_storage_follows_sync_configuration() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.high_quality_sync_enabled = False

    assert worker._should_store_quality_sample(
        quality_bucket=HIGH_QUALITY,
        raw_sample_selected=False,
    ) is False
    assert worker._should_store_quality_sample(
        quality_bucket=LOW_QUALITY,
        raw_sample_selected=True,
    ) is True
    assert worker._should_store_quality_sample(
        quality_bucket=LOW_QUALITY,
        raw_sample_selected=False,
    ) is False

    worker.high_quality_sync_enabled = True
    assert worker._should_store_quality_sample(
        quality_bucket=HIGH_QUALITY,
        raw_sample_selected=False,
    ) is True


def test_disabled_high_quality_sync_excludes_unsent_feature_bytes_from_trigger() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.high_quality_sync_enabled = False
    worker.sample_store = SimpleNamespace(
        stats=lambda: {
            "total_samples": 2,
            "high_quality_count": 1,
            "low_quality_count": 1,
            "low_quality_rate": 0.5,
            "high_quality_feature_bytes": 10_000,
            "low_quality_feature_bytes": 0,
            "low_quality_raw_bytes": 100,
        }
    )

    stats = worker._stats_for_training_trigger()

    assert stats["high_quality_feature_bytes"] == 0


def test_resource_trigger_uses_window_low_quality_rate() -> None:
    stats = SimpleNamespace(low_quality_rate=0.5)
    drift_state = SimpleNamespace(low_quality_rate=0.8)

    retained_rate = EdgeWorker._apply_window_quality_rate(stats, drift_state)

    assert retained_rate == pytest.approx(0.5)
    assert stats.low_quality_rate == pytest.approx(0.8)


def test_split_runtime_rejects_frame_size_change_without_disabling_split() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.split_learning_enabled = True
    worker._fixed_split_init_attempted = True
    worker.universal_split_enabled = True
    worker.universal_splitter = object()
    worker.split_trace_image_size = (4, 4)

    with pytest.raises(RuntimeError, match="input size changed"):
        worker._resolve_active_splitter(None, (8, 8))

    assert worker.split_learning_enabled is True
    assert worker.universal_split_enabled is True


def test_local_worker_stops_after_fatal_split_failure() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker._stop_event = threading.Event()
    worker.local_queue = Queue()
    worker.config = SimpleNamespace(wait_thresh=100)
    worker.model_version = "0"
    worker._resolve_active_splitter = lambda *_args: (_ for _ in ()).throw(
        _FixedSplitRuntimeError("split failed")
    )
    worker.small_object_detection = SimpleNamespace(
        infer_sample=lambda *_args, **_kwargs: pytest.fail(
            "full inference fallback must not run"
        ),
    )
    task = Task(
        1,
        1,
        np.zeros((4, 4, 3), dtype=np.uint8),
        time.time(),
        (4, 4, 3),
    )
    task.local_queue_enqueued_perf = time.perf_counter()
    pending_task = Task(
        2,
        2,
        np.zeros((4, 4, 3), dtype=np.uint8),
        time.time(),
        (4, 4, 3),
    )
    worker.local_queue.put(task)
    worker.local_queue.put(pending_task)

    worker.local_worker()

    assert task.state == TASK_STATE.TIMEOUT
    assert task.result_source == "inference_error"
    assert task.done_event.is_set()
    assert pending_task.state == TASK_STATE.TIMEOUT
    assert pending_task.result_source == "inference_error"
    assert pending_task.done_event.is_set()
    assert worker._stop_event.is_set()


def test_local_worker_marks_task_done_before_sample_collection() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker._stop_event = threading.Event()
    worker.local_queue = Queue()
    worker.config = SimpleNamespace(wait_thresh=100)
    worker.model_version = "0"
    worker.collect_flag = True
    worker.split_learning_enabled = True
    worker.fixed_split_plan = SimpleNamespace(split_config_id="split")
    worker.latest_result_lock = threading.Lock()
    worker.latest_result = {}
    worker._resolve_active_splitter = lambda *_args: object()
    worker._try_apply_pending_model_update = lambda: False
    inference = InferenceArtifacts(
        intermediate={"payload": torch.ones(1)},
        final_detection_boxes=[[1, 2, 3, 4]],
        final_detection_labels=[1],
        final_detection_scores=[0.9],
        low_threshold_boxes=[[1, 2, 3, 4]],
        low_threshold_labels=[1],
        low_threshold_scores=[0.9],
        confidence=0.9,
        timing_ms={
            "split_preprocess_ms": 1.0,
            "split_prefix_ms": 2.0,
            "split_suffix_ms": 3.0,
            "postprocess_ms": 4.0,
            "parse_filter_ms": 5.0,
        },
    )
    worker.small_object_detection = SimpleNamespace(
        infer_sample=lambda *_args, **_kwargs: inference,
    )
    collection_seen: list[bool] = []

    def collect_data(task, _frame, _inference):
        collection_seen.append(task.done_event.is_set())
        worker._stop_event.set()
        return True

    worker.collect_data = collect_data
    task = Task(
        1,
        1,
        np.zeros((4, 4, 3), dtype=np.uint8),
        time.time(),
        (4, 4, 3),
    )
    task.local_queue_enqueued_perf = time.perf_counter()
    worker.local_queue.put(task)

    worker.local_worker()

    assert task.done_event.is_set()
    assert task.state == TASK_STATE.FINISHED
    assert collection_seen == [True]


def test_task_completion_timestamp_includes_latest_result_snapshot() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.model_version = "0"
    snapshot_finished = threading.Event()

    def remember_latest_result(_task):
        time.sleep(0.02)
        snapshot_finished.set()

    worker._remember_latest_result = remember_latest_result
    task = Task(1, 1, None, time.time(), (1, 1, 3))
    task.result_source = "inference"
    task.set_inference_artifacts({"result_source": "inference"})

    EdgeWorker._set_task_terminal_state(
        worker,
        task,
        TASK_STATE.FINISHED,
        result_source="inference",
    )

    assert snapshot_finished.is_set()
    assert task.done_event.is_set()
    assert task.end_time is not None
    assert task.timing_ms["task_complete_ms"] >= 20.0


def test_latest_result_reuses_owned_frame_buffer() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.model_version = "0"
    worker.latest_result_lock = threading.Lock()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    task = Task(1, 1, frame, time.time(), frame.shape)
    task.result_source = "inference"
    task.set_inference_artifacts({"result_source": "inference"})

    EdgeWorker._remember_latest_result(worker, task)

    assert worker.latest_result["frame"] is frame


def test_task_artifact_snapshot_preserves_entropy_provenance() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker.model_version = "0"
    task = Task(1, 1, None, time.time(), (1, 1, 3))
    task.result_source = "inference"
    inference = _collection_job().inference
    inference.feature_spectral_entropy = 0.25

    feature_snapshot = EdgeWorker._task_artifact_snapshot(
        worker,
        task,
        inference,
        result_source="inference",
    )

    assert feature_snapshot["entropy"] == pytest.approx(0.25)
    assert feature_snapshot["entropy_source"] == "feature_spectral_entropy"
    assert feature_snapshot["logit_entropy"] is None
    assert feature_snapshot["feature_spectral_entropy"] == pytest.approx(0.25)

    inference.logit_entropy = 0.1
    logit_snapshot = EdgeWorker._task_artifact_snapshot(
        worker,
        task,
        inference,
        result_source="inference",
    )

    assert logit_snapshot["entropy"] == pytest.approx(0.1)
    assert logit_snapshot["entropy_source"] == "logit_entropy"
    assert logit_snapshot["logit_entropy"] == pytest.approx(0.1)


def test_cached_frame_boundary_attempts_pending_model_update() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker._stop_event = threading.Event()
    worker.frame_cache = Queue()
    worker.config = SimpleNamespace(diff_flag=True, diff_thresh=1.0)
    worker.diff = 0.0
    worker.edge_processor = SimpleNamespace(
        get_frame_feature=lambda _frame: 0.0,
        cal_frame_diff=lambda _current, _previous: 0.0,
    )
    update_attempted = threading.Event()

    def try_apply():
        update_attempted.set()
        worker._stop_event.set()
        return True

    worker._try_apply_pending_model_update = try_apply
    worker.decision_worker = lambda _task: None
    worker._reuse_latest_result = lambda task: setattr(task, "result_source", "cached")
    worker._finalize_task = lambda task: task.mark_done()
    first = Task(1, 1, np.zeros((2, 2, 3), dtype=np.uint8), time.time(), (2, 2, 3))
    cached = Task(1, 2, np.zeros((2, 2, 3), dtype=np.uint8), time.time(), (2, 2, 3))
    worker.frame_cache.put(first)
    worker.frame_cache.put(cached)

    worker.diff_worker()

    assert cached.done_event.is_set()
    assert update_attempted.is_set()


def test_pending_model_update_defers_when_model_lock_is_busy() -> None:
    worker = EdgeWorker.__new__(EdgeWorker)
    worker._pending_model_update_lock = threading.Lock()
    update = PendingModelUpdate(
        update_payload={},
        state_dict={},
        submitted_model_version="0",
        next_model_version="1",
    )
    worker.pending_model_update = update
    model_lock = threading.Lock()
    worker.small_object_detection = SimpleNamespace(model_lock=model_lock)
    worker._apply_prepared_model_update_locked = lambda item: setattr(
        item, "applied_version", "1"
    )

    model_lock.acquire()
    try:
        assert EdgeWorker._try_apply_pending_model_update(worker) is False
        assert worker.pending_model_update is update
        assert not update.applied_event.is_set()
    finally:
        model_lock.release()

    assert EdgeWorker._try_apply_pending_model_update(worker) is True
    assert worker.pending_model_update is None
    assert update.applied_event.is_set()
