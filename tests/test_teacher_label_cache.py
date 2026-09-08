from __future__ import annotations

from dataclasses import replace

from cloud.annotation import TeacherAnnotationRequest, TeacherLabelCache


def _request(**overrides) -> TeacherAnnotationRequest:
    payload = {
        "sample_id": "sample-1",
        "edge_id": 1,
        "model_id": "yolo26n",
        "image_path": "/tmp/sample.jpg",
        "image_sha1": "a" * 40,
        "teacher_model_name": "rtdetr_x",
        "teacher_weights_fingerprint": "weights-a",
        "teacher_label_schema": "coco_91",
        "teacher_num_classes": 91,
        "teacher_annotation_threshold": 0.5,
        "label_coordinate_space": "original_xyxy",
        "metadata": {},
    }
    payload.update(overrides)
    return TeacherAnnotationRequest(**payload)


def test_cache_key_is_stable_and_ignores_batch_size() -> None:
    req_a = _request(metadata={"worker_batch_size": 4})
    req_b = _request(metadata={"worker_batch_size": 16})

    assert req_a.cache_key().digest == req_b.cache_key().digest


def test_cache_key_is_isolated_by_edge_client(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path))
    edge_a = _request(edge_id=1)
    edge_b = _request(edge_id=2)
    cache.write(edge_a, {"boxes": [], "labels": []}, source="test")

    assert edge_a.cache_key().digest != edge_b.cache_key().digest
    assert cache.read(edge_b) is None


def test_threshold_change_causes_cache_miss(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path))
    req = _request()
    cache.write(req, {"boxes": [], "labels": []}, source="test")

    assert cache.read(replace(req, teacher_annotation_threshold=0.7)) is None


def test_weights_fingerprint_change_causes_cache_miss(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path))
    req = _request()
    cache.write(req, {"boxes": [], "labels": []}, source="test")

    assert cache.read(replace(req, teacher_weights_fingerprint="weights-b")) is None


def test_label_schema_change_causes_cache_miss(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path))
    req = _request()
    cache.write(req, {"boxes": [], "labels": []}, source="test")

    assert cache.read(replace(req, teacher_label_schema="zero_based")) is None


def test_target_label_mapping_change_causes_cache_miss(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path))
    req = _request(
        metadata={
            "target_model_metadata": {
                "label_schema": "zero_based",
                "class_names": ["person", "car"],
            }
        }
    )
    cache.write(req, {"boxes": [[1, 2, 3, 4]], "labels": [0]}, source="test")

    changed_schema = replace(
        req,
        metadata={
            "target_model_metadata": {
                "label_schema": "coco_91",
                "class_names": ["person", "car"],
            }
        },
    )
    changed_names = replace(
        req,
        metadata={
            "target_model_metadata": {
                "label_schema": "zero_based",
                "class_names": ["car", "person"],
            }
        },
    )

    assert cache.read(changed_schema) is None
    assert cache.read(changed_names) is None


def test_include_empty_change_causes_cache_miss(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path))
    req = _request(metadata={"include_empty": True})
    cache.write(req, {"boxes": [], "labels": []}, source="test")

    assert cache.read(replace(req, metadata={"include_empty": False})) is None


def test_atomic_write_and_batch_lookup_round_trip(tmp_path) -> None:
    cache = TeacherLabelCache(str(tmp_path))
    req = _request()
    labels = {
        "boxes": [[1.0, 2.0, 3.0, 4.0]],
        "labels": [1],
        "scores": [0.9],
        "label_coordinate_space": "original_xyxy",
    }

    cache.write(req, labels, source="test")
    results, cache_read_time = cache.lookup_many([req])

    assert cache_read_time >= 0.0
    assert len(results) == 1
    assert results[0].labels == labels
