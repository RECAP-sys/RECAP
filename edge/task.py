import threading
import time

import numpy as np


def _to_python_value(value):
    if isinstance(value, np.ndarray):
        return [_to_python_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_python_value(item) for item in value]
    return value


class Task:
    def __init__(self, edge_id, frame_index, frame, start_time, raw_shape):
        self.edge_id = edge_id
        self.frame_index = frame_index
        self.frame_edge = frame
        self.start_time = start_time
        self.created_perf = time.perf_counter()
        self.raw_shape = raw_shape
        self.state = None
        self.ref = None
        self.end_time = None
        self.frame_cloud = None
        self.other = False
        self.directly_cloud = False
        self.edge_process = False

        self.done_event = threading.Event()
        self.result_source = "pending"
        self.timing_ms = {}
        self.inference_artifacts = {
            "boxes": [],
            "labels": [],
            "scores": [],
            "confidence": 0.0,
            "entropy": 0.0,
            "model_version": "",
            "result_source": "pending",
        }

        self.detection_boxes = []
        self.detection_class = []
        self.detection_score = []

    def record_timing(self, name, elapsed_ms):
        value = max(0.0, float(elapsed_ms))
        self.timing_ms[str(name)] = float(self.timing_ms.get(str(name), 0.0)) + value

    def set_timing(self, name, elapsed_ms):
        self.timing_ms[str(name)] = max(0.0, float(elapsed_ms))

    def add_result(self, detection_boxes, detection_class, detection_score):
        if detection_boxes is not None:
            assert len(detection_boxes) == len(detection_class) == len(detection_score)
            for i in range(len(detection_boxes)):
                self.detection_boxes.append(_to_python_value(detection_boxes[i]))
                self.detection_class.append(_to_python_value(detection_class[i]))
                self.detection_score.append(_to_python_value(detection_score[i]))

    def replace_result(self, detection_boxes=None, detection_class=None, detection_score=None):
        self.detection_boxes = [_to_python_value(box) for box in (detection_boxes or [])]
        self.detection_class = [_to_python_value(label) for label in (detection_class or [])]
        self.detection_score = [_to_python_value(score) for score in (detection_score or [])]

    def get_result(self):
        return self.detection_boxes, self.detection_class, self.detection_score

    def set_inference_artifacts(self, artifacts):
        data = dict(artifacts or {})
        data.setdefault("boxes", [list(box) for box in self.detection_boxes])
        data.setdefault("labels", list(self.detection_class))
        data.setdefault("scores", list(self.detection_score))
        data.setdefault("confidence", 0.0)
        data.setdefault("entropy", 0.0)
        data.setdefault("model_version", "")
        data.setdefault("result_source", self.result_source)
        self.inference_artifacts = data

    def mark_done(self):
        self.done_event.set()

    def wait_until_done(self, timeout=None):
        return self.done_event.wait(timeout)
