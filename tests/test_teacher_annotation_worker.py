from __future__ import annotations

import cv2
import numpy as np

from cloud.annotation import TeacherAnnotationRequest, TeacherAnnotationWorker, TeacherLabelCache
from cloud.annotation import teacher_worker as teacher_worker_module


def _image(tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.jpg"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return str(path)


def _request(
    tmp_path, sample_id: str, *, image_sha1: str | None = None
) -> TeacherAnnotationRequest:
    return TeacherAnnotationRequest(
        sample_id=sample_id,
        edge_id=1,
        model_id="yolo26n",
        image_path=_image(tmp_path, sample_id),
        image_sha1=image_sha1 or f"{int(sample_id.split('-')[-1]):040x}",
        teacher_model_name="rtdetr_x",
        teacher_weights_fingerprint="weights-a",
        teacher_label_schema="coco_91",
        teacher_num_classes=91,
        teacher_annotation_threshold=0.5,
        label_coordinate_space="original_xyxy",
        metadata={"include_empty": True},
    )


def _prediction(_frame):
    return ([[1, 2, 3, 4]], [1], [0.9])


def test_large_inference_batch_is_called(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    calls: list[int] = []

    def batch(images, threshold):
        assert threshold == 0.5
        calls.append(len(images))
        return [_prediction(image) for image in images]

    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=batch,
        worker_batch_size=4,
        auto_start=False,
    )
    req = _request(tmp_path, "sample-1")

    worker.submit_many([req])
    worker.process_pending_once()

    assert calls == [1]
    assert cache.read(req) == {"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]}


def test_worker_chunks_by_worker_batch_size(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    calls: list[int] = []

    def batch(images, threshold):
        del threshold
        calls.append(len(images))
        return [_prediction(image) for image in images]

    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=batch,
        worker_batch_size=4,
        auto_start=False,
    )
    requests = [_request(tmp_path, f"sample-{index}") for index in range(10)]

    worker.submit_many(requests)
    worker.process_pending_once()

    assert calls == [4, 4, 2]


def test_cuda_oom_halves_batch_size_and_retries(monkeypatch, tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    calls: list[int] = []
    cleanup_calls: list[None] = []

    monkeypatch.setattr(
        teacher_worker_module,
        "_release_cuda_cache_after_oom",
        lambda: cleanup_calls.append(None),
    )

    def batch(images, threshold):
        del threshold
        calls.append(len(images))
        if len(images) == 4:
            raise RuntimeError("CUDA out of memory")
        return [_prediction(image) for image in images]

    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=batch,
        worker_batch_size=4,
        min_worker_batch_size=1,
        auto_start=False,
    )
    requests = [_request(tmp_path, f"sample-{index}") for index in range(4)]

    worker.submit_many(requests)
    worker.process_pending_once()

    assert calls == [4, 2, 2]
    assert len(cleanup_calls) == 1
    assert worker.snapshot_stats()["oom_retry_count"] == 1
    assert all(cache.read(request) is not None for request in requests)


def test_batch_size_one_failure_marks_samples_failed(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))

    def batch(images, threshold):
        del images, threshold
        raise RuntimeError("CUDA out of memory")

    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=batch,
        worker_batch_size=2,
        min_worker_batch_size=1,
        auto_start=False,
    )
    requests = [_request(tmp_path, f"sample-{index}") for index in range(2)]

    worker.submit_many(requests)
    worker.process_pending_once()

    assert worker.snapshot_stats()["failed_count"] == 2
    assert all(cache.read(request) is None for request in requests)


def test_batch_unavailable_falls_back_to_single_inference(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    single_calls = 0

    def single(image, threshold):
        nonlocal single_calls
        assert threshold == 0.5
        single_calls += 1
        return _prediction(image)

    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=None,
        single_inference=single,
        worker_batch_size=4,
        auto_start=False,
    )
    requests = [_request(tmp_path, f"sample-{index}") for index in range(3)]

    worker.submit_many(requests)
    worker.process_pending_once()

    assert single_calls == 3
    assert worker.snapshot_stats()["batch_fallback_count"] == 1


def test_same_cache_key_is_not_queued_twice(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path / "cache"))
    calls: list[int] = []

    def batch(images, threshold):
        del threshold
        calls.append(len(images))
        return [_prediction(image) for image in images]

    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=batch,
        worker_batch_size=4,
        auto_start=False,
    )
    req_a = _request(tmp_path, "sample-1", image_sha1="same")
    req_b = _request(tmp_path, "sample-2", image_sha1="same")

    submit = worker.submit_many([req_a, req_b])
    worker.process_pending_once()

    assert submit.submitted == 1
    assert submit.duplicate == 1
    assert calls == [1]
    assert cache.read(req_b) is not None
