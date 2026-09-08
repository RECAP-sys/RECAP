from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cloud.annotation.label_cache import TeacherLabelCache
from cloud.annotation.remote_service import (
    RemoteTeacherAnnotationService,
    SharedTeacherAnnotationRpcServer,
)
from cloud.annotation.service import TeacherAnnotationService
from cloud.annotation.teacher_worker import TeacherAnnotationWorker
from cloud.annotation.types import TeacherAnnotationRequest


def _request(
    image_path: Path,
    *,
    sample_id: str,
    edge_id: int,
) -> TeacherAnnotationRequest:
    return TeacherAnnotationRequest(
        sample_id=sample_id,
        edge_id=edge_id,
        model_id="yolo26n",
        image_path=str(image_path),
        image_sha1="same-frame",
        teacher_model_name="rtdetr_x",
        teacher_weights_fingerprint="shared-teacher",
        teacher_label_schema="coco_91",
        teacher_num_classes=91,
        teacher_annotation_threshold=0.4,
        label_coordinate_space="original_xyxy",
        metadata={"include_empty": True, "target_model_metadata": {"model_id": "yolo26n"}},
    )


def test_remote_clients_share_teacher_but_not_cross_edge_predictions(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    inference_calls: list[int] = []

    def batch_inference(frames, threshold):
        assert threshold == 0.4
        inference_calls.append(len(frames))
        return [([[1, 1, 6, 6]], [1], [0.9]) for _frame in frames]

    cache = TeacherLabelCache(str(tmp_path / "cache"), enabled=True)
    worker = TeacherAnnotationWorker(
        label_cache=cache,
        batch_inference=batch_inference,
        worker_batch_size=8,
        auto_start=False,
    )
    service = TeacherAnnotationService(label_cache=cache, worker=worker)
    server = SharedTeacherAnnotationRpcServer(
        service=service,
        metadata_provider=lambda: {
            "teacher_model_name": "rtdetr_x",
            "teacher_weights_fingerprint": "shared-teacher",
            "teacher_label_schema": "coco_91",
            "teacher_num_classes": 91,
            "teacher_class_names": [],
        },
    )
    server.start()
    try:
        client_a = RemoteTeacherAnnotationService(server.listen_address, timeout_sec=5)
        client_b = RemoteTeacherAnnotationService(server.listen_address, timeout_sec=5)
        request_a = _request(image_path, sample_id="edge-a", edge_id=1)
        request_b = _request(image_path, sample_id="edge-b", edge_id=2)

        assert client_a.metadata()["teacher_model_name"] == "rtdetr_x"
        assert client_a.submit_many([request_a]).submitted == 1
        assert client_b.submit_many([request_b]).submitted == 1
        worker.start()
        ensured_a = client_a.ensure_many([request_a], wait=True, timeout_sec=5)
        ensured_b = client_b.ensure_many([request_b], wait=True, timeout_sec=5)
        cached = client_a.ensure_many([request_a], wait=True, timeout_sec=5)

        assert ensured_a.unresolved_count == 0
        assert ensured_b.unresolved_count == 0
        assert ensured_a.labels_by_sample_id["edge-a"]["labels"] == [1]
        assert ensured_b.labels_by_sample_id["edge-b"]["labels"] == [1]
        assert cached.cache_hits == 1
        assert inference_calls == [2]
    finally:
        server.shutdown()
        worker.stop()
