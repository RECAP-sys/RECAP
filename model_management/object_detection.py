import threading
import time

import cv2
import numpy as np
import torch

from model_management.inference.artifacts import InferenceArtifacts
from model_management.inference.confidence import summarize_detection_confidence
from model_management.inference.prediction_filter import (
    compute_intersection_over_min_area,
    deduplicate_final_predictions,
    resolve_final_dedup_thresholds,
)
from model_management.model_zoo import (
    build_detection_model,
    get_model_detection_thresholds,
    get_model_family,
)
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
    summarize_split_runtime_observables,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bgr_image_to_tensor(img, *, target_device=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb))
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    return tensor.to(target_device or device)


class Object_Detection:
    def __init__(self, config, type):
        self.type = type
        self.config = config
        self.model_lock = threading.Lock()

        if type == "small inference":
            self.model_name = config.lightweight
        else:
            self.model_name = config.golden
        self.model = None
        self._split_input_resize_mode = None
        self.load_model()

    def load_model(self):
        explicit_weights_path = (
            getattr(self.config, "weights_path", None) if self.type == "small inference" else None
        )
        build_kwargs = {}
        if get_model_family(self.model_name) == "tinynext":
            configured_input_size = getattr(self.config, "tinynext_input_size", None)
            if configured_input_size is not None:
                build_kwargs["tinynext_input_size"] = int(configured_input_size)
        self.model = build_detection_model(
            self.model_name,
            pretrained=True,
            device=device,
            weights_path=explicit_weights_path,
            **build_kwargs,
        )

        self.model.to(device)
        self.model.eval()
        get_split_runtime_model(self.model).eval()
        self._split_input_resize_mode = get_split_runtime_input_resize_mode(self.model)
        self.refresh_thresholds_from_model()

    def refresh_thresholds_from_model(self):
        self.threshold_low, self.threshold_high = get_model_detection_thresholds(
            self.model,
            self.model_name,
        )

    def prepare_splitter_input(self, img):
        return prepare_split_runtime_input(self.model, img, device=device)

    def build_split_sample_input(self, image_size=None):
        if image_size is None:
            image_size = (224, 224)
        return build_split_runtime_sample_input(self.model, image_size=image_size, device=device)

    def get_split_runtime_model(self):
        return get_split_runtime_model(self.model)

    def infer_sample(self, img, splitter=None) -> InferenceArtifacts:
        timing_ms: dict[str, float] = {}

        def add_timing(name: str, started: float) -> None:
            elapsed = (time.perf_counter() - started) * 1000.0
            timing_ms[name] = timing_ms.get(name, 0.0) + max(0.0, float(elapsed))

        split_payload = None
        input_tensor_shape = None
        input_resize_mode = None
        observables: dict[str, float | None] = {}
        with self.model_lock:
            with torch.inference_mode():
                if splitter is not None:
                    started = time.perf_counter()
                    splitter_input = prepare_split_runtime_input(self.model, img, device=device)
                    input_resize_mode = self._split_input_resize_mode
                    if isinstance(splitter_input, torch.Tensor):
                        input_tensor_shape = [int(dim) for dim in splitter_input.shape]
                    elif (
                        isinstance(splitter_input, (list, tuple))
                        and splitter_input
                        and isinstance(splitter_input[0], torch.Tensor)
                    ):
                        input_tensor_shape = [int(dim) for dim in splitter_input[0].shape]
                    add_timing("split_preprocess_ms", started)

                    replay_profile: dict[str, float] = {}
                    replayed, split_payload = splitter.replay_inference(
                        splitter_input,
                        return_split_output=True,
                        profile=replay_profile,
                    )
                    for source_name, timing_name in (
                        ("split_prefix", "split_prefix_ms"),
                        ("split_suffix", "split_suffix_ms"),
                    ):
                        if source_name in replay_profile:
                            timing_ms[timing_name] = timing_ms.get(timing_name, 0.0) + float(
                                replay_profile[source_name]
                            )

                    started = time.perf_counter()
                    observables = summarize_split_runtime_observables(
                        self.model,
                        replayed,
                        split_payload,
                        include_feature_spectral_entropy=False,
                    )
                    add_timing("observables_ms", started)

                    started = time.perf_counter()
                    replayed = postprocess_split_runtime_output(
                        self.model,
                        replayed,
                        threshold=self.threshold_low,
                        model_input=splitter_input,
                        orig_image=img,
                    )
                    add_timing("postprocess_ms", started)

                    started = time.perf_counter()
                    pred_boxes, pred_class, pred_score = self._parse_prediction_output(
                        replayed,
                        self.threshold_low,
                    )
                    add_timing("parse_filter_ms", started)
                else:
                    pred_boxes, pred_class, pred_score = self.get_model_prediction(
                        img,
                        self.threshold_low,
                    )

        if pred_boxes is None or pred_score is None:
            return InferenceArtifacts(
                intermediate=split_payload,
                final_detection_boxes=[],
                final_detection_labels=[],
                final_detection_scores=[],
                low_threshold_boxes=[],
                low_threshold_labels=[],
                low_threshold_scores=[],
                confidence=0.0,
                input_tensor_shape=input_tensor_shape,
                input_resize_mode=input_resize_mode,
                proposal_count=0,
                retained_count=0,
                feature_spectral_entropy=observables.get("feature_spectral_entropy"),
                logit_entropy=observables.get("logit_entropy"),
                logit_margin=observables.get("logit_margin"),
                logit_energy=observables.get("logit_energy"),
                timing_ms=timing_ms,
            )

        started = time.perf_counter()
        confidence = self._summarize_detection_confidence(pred_score)
        low_threshold_boxes = list(pred_boxes)
        low_threshold_labels = list(pred_class)
        low_threshold_scores = list(pred_score)
        final_detection_threshold = self._resolve_final_detection_threshold()
        high_keep_indices = [
            index for index, score in enumerate(pred_score) if score > final_detection_threshold
        ]
        if not high_keep_indices:
            detection_boxes = []
            detection_class = []
            detection_score = []
        else:
            detection_boxes = [pred_boxes[index] for index in high_keep_indices]
            detection_class = [pred_class[index] for index in high_keep_indices]
            detection_score = [pred_score[index] for index in high_keep_indices]
            detection_boxes, detection_class, detection_score = self._deduplicate_final_predictions(
                detection_boxes,
                detection_class,
                detection_score,
                threshold=float(final_detection_threshold),
            )
        add_timing("parse_filter_ms", started)

        return InferenceArtifacts(
            intermediate=split_payload,
            final_detection_boxes=detection_boxes,
            final_detection_labels=detection_class,
            final_detection_scores=detection_score,
            low_threshold_boxes=low_threshold_boxes,
            low_threshold_labels=low_threshold_labels,
            low_threshold_scores=low_threshold_scores,
            confidence=confidence,
            input_tensor_shape=input_tensor_shape,
            input_resize_mode=input_resize_mode,
            proposal_count=len(pred_score),
            retained_count=len(detection_score),
            feature_spectral_entropy=observables.get("feature_spectral_entropy"),
            logit_entropy=observables.get("logit_entropy"),
            logit_margin=observables.get("logit_margin"),
            logit_energy=observables.get("logit_energy"),
            timing_ms=timing_ms,
        )

    def small_inference(self, img, splitter=None, return_split_payload=False):
        artifacts = self.infer_sample(img, splitter=splitter)
        if return_split_payload:
            return (
                None,
                artifacts.final_detection_boxes or None,
                artifacts.final_detection_labels or None,
                artifacts.final_detection_scores or None,
                artifacts.intermediate,
            )
        return (
            None,
            artifacts.final_detection_boxes or None,
            artifacts.final_detection_labels or None,
            artifacts.final_detection_scores or None,
        )

    def large_inference(self, img, threshold=None):
        if threshold is None:
            threshold = self.threshold_high
        pred_boxes, pred_class, pred_score = self.get_model_prediction(
            img,
            float(threshold),
        )
        return pred_boxes, pred_class, pred_score

    def large_inference_batch(self, images, threshold=None):
        if threshold is None:
            threshold = self.threshold_high
        frames = list(images or [])
        if not frames:
            return []

        prepared_images = [self._prepare_image_tensor(frame) for frame in frames]
        with torch.inference_mode():
            outputs = self.model(prepared_images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, dict):
            outputs = [outputs]
        if not isinstance(outputs, (list, tuple)):
            return [(None, None, None) for _ in frames]

        predictions = []
        for index in range(len(frames)):
            output = outputs[index] if index < len(outputs) else None
            predictions.append(
                self._parse_prediction_output(
                    [] if output is None else [output],
                    float(threshold),
                )
            )
        return predictions

    def get_model_prediction(self, img, threshold, model=None):
        img = self._prepare_image_tensor(img)
        # get the inference result
        with torch.inference_mode():
            if model is None:
                res = self.model([img])
            else:
                res = model([img])
        return self._parse_prediction_output(res, threshold)

    def _prepare_image_tensor(self, img):
        return bgr_image_to_tensor(img, target_device=device)

    def _prepare_runtime_frame(
        self,
        img,
    ) -> tuple[np.ndarray, tuple[int, int], bool]:
        original_image_size = tuple(int(value) for value in img.shape[:2])
        return img, original_image_size, False

    def _summarize_detection_confidence(self, scores: list[float] | None) -> float:
        return summarize_detection_confidence(scores)

    def _resolve_final_detection_threshold(self) -> float:
        configured_floor = 0.5
        config_obj = getattr(self, "config", None)
        if config_obj is not None:
            configured_floor = float(
                getattr(config_obj, "final_detection_threshold", configured_floor)
            )
        threshold_high = float(getattr(self, "threshold_high", configured_floor))
        return max(threshold_high, configured_floor)

    def _resolve_final_dedup_thresholds(
        self,
        threshold: float,
    ) -> dict[str, tuple[float, float]] | None:
        family = get_model_family(self.model_name)
        threshold_high = float(getattr(self, "threshold_high", threshold))
        return resolve_final_dedup_thresholds(
            family,
            float(threshold),
            threshold_high=threshold_high,
        )

    @staticmethod
    def _compute_intersection_over_min_area(
        candidate_box: torch.Tensor,
        reference_boxes: torch.Tensor,
    ) -> torch.Tensor:
        return compute_intersection_over_min_area(candidate_box, reference_boxes)

    def _deduplicate_final_predictions(
        self,
        boxes: list[list[float]],
        labels: list[int],
        scores: list[float],
        *,
        threshold: float,
    ) -> tuple[list[list[float]], list[int], list[float]]:
        resolved_thresholds = self._resolve_final_dedup_thresholds(float(threshold))
        return deduplicate_final_predictions(
            boxes,
            labels,
            scores,
            thresholds=resolved_thresholds,
        )

    def _parse_prediction_output(self, res, threshold):
        if isinstance(res, tuple):
            res = res[0]
        if isinstance(res, dict):
            res = [res]
        if not isinstance(res, (list, tuple)) or len(res) == 0:
            return None, None, None

        first = res[0]
        if not isinstance(first, dict):
            return None, None, None

        labels_t = first.get("labels")
        boxes_t = first.get("boxes")
        scores_t = first.get("scores")
        if labels_t is None or boxes_t is None or scores_t is None:
            return None, None, None

        prediction_class = labels_t.detach().cpu().tolist()
        prediction_boxes = boxes_t.detach().cpu().tolist()
        prediction_score = scores_t.detach().cpu().tolist()

        keep_indices = [index for index, score in enumerate(prediction_score) if score > threshold]
        if not keep_indices:
            return None, None, None
        pred_boxes = [prediction_boxes[index] for index in keep_indices]
        pred_class = [prediction_class[index] for index in keep_indices]
        pred_score = [prediction_score[index] for index in keep_indices]
        pred_boxes, pred_class, pred_score = self._deduplicate_final_predictions(
            pred_boxes,
            pred_class,
            pred_score,
            threshold=float(threshold),
        )
        if not pred_score:
            return None, None, None
        return pred_boxes, pred_class, pred_score
