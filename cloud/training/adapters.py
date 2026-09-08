from __future__ import annotations

import time
from typing import Any, Mapping, Protocol

from cloud.training.types import (
    EpochTrainResult,
    FixedSplitTrainingContext,
    ProxyEvalResult,
    ProxyMetrics,
)
from model_management.universal_model_split import (
    build_split_retrain_optimizer,
    universal_split_retrain,
)
from model_management.universal_model_split import (
    train_split_suffix_batch as _train_split_suffix_batch,
)


def train_split_suffix_batch(
    runtime: Any,
    boundary: Any,
    targets: Any,
    loss_fn: Any,
    optimizer: Any,
    *,
    trusted_runtime_boundary: bool = False,
) -> Any:
    return _train_split_suffix_batch(
        runtime,
        boundary,
        targets,
        loss_fn,
        optimizer,
        trusted_runtime_boundary=trusted_runtime_boundary,
    )


class DetectionTrainingAdapter(Protocol):
    def build_optimizer(self, context: FixedSplitTrainingContext) -> Any: ...

    def train_one_epoch(
        self,
        context: FixedSplitTrainingContext,
        *,
        epoch: int,
        total_epochs: int,
        optimizer: Any,
    ) -> EpochTrainResult: ...

    def evaluate_proxy(
        self,
        context: FixedSplitTrainingContext,
        *,
        epoch: int | None,
        stage_label: str,
        max_samples: int | None,
    ) -> ProxyEvalResult: ...

    def metric_value(self, metrics: Mapping[str, object] | None) -> float | None: ...

    def metrics_are_better(
        self,
        candidate_metrics: Mapping[str, object] | None,
        incumbent_metrics: Mapping[str, object] | None,
        *,
        min_delta: float,
    ) -> bool: ...


class UniversalSplitTrainingAdapter:
    def build_optimizer(self, context: FixedSplitTrainingContext) -> Any:
        kwargs = context.training_kwargs
        runtime = kwargs.get("splitter")
        if runtime is None:
            return None
        return build_split_retrain_optimizer(
            kwargs["model"],
            runtime=runtime,
            learning_rate=float(context.plan.learning_rate),
            optimizer_name=str(context.plan.optimizer_name or "adam"),
            weight_decay=float(context.plan.weight_decay or 0.0),
            grad_clip_norm=context.plan.grad_clip_norm,
        )

    def train_one_epoch(
        self,
        context: FixedSplitTrainingContext,
        *,
        epoch: int,
        total_epochs: int,
        optimizer: Any,
    ) -> EpochTrainResult:
        profile = context.retrain_profile
        suffix_before = (
            float(getattr(profile, "suffix_forward_backward_time", 0.0))
            if profile is not None
            else 0.0
        )
        started = time.perf_counter()
        losses = universal_split_retrain(
            **context.training_kwargs,
            optimizer=optimizer,
            num_epoch=1,
            epoch_log_context=None,
            log_batches=False,
            log_every_n_epochs=1,
            log_first_epoch=False,
            epoch_log_start=epoch - 1,
            epoch_log_total=total_epochs,
        )
        elapsed = time.perf_counter() - started
        suffix_after = (
            float(getattr(profile, "suffix_forward_backward_time", 0.0))
            if profile is not None
            else suffix_before
        )
        loss = float(losses[-1]) if losses else None
        return EpochTrainResult(
            epoch=int(epoch),
            loss=loss,
            train_time=elapsed,
            suffix_forward_backward_time=max(0.0, suffix_after - suffix_before),
        )

    def evaluate_proxy(
        self,
        context: FixedSplitTrainingContext,
        *,
        epoch: int | None,
        stage_label: str,
        max_samples: int | None,
    ) -> ProxyEvalResult:
        started = time.perf_counter()
        if context.plan.model_family == "tinynext" and context.tinynext_proxy_evaluator:
            metrics = context.tinynext_proxy_evaluator(
                stage_label=stage_label,
                max_samples=max_samples,
            )
        elif context.fixed_proxy_evaluator:
            metrics = context.fixed_proxy_evaluator(
                stage_label=stage_label,
                max_samples=max_samples,
            )
        else:
            metrics = {}
        normalized = _normalize_proxy_metrics(metrics)
        return ProxyEvalResult(
            metrics=normalized,
            metric=self.metric_value(normalized),
            elapsed=time.perf_counter() - started,
            epoch=epoch,
            stage_label=stage_label,
        )

    def metric_value(self, metrics: Mapping[str, object] | None) -> float | None:
        if not metrics:
            return None
        value = metrics.get("primary_metric", metrics.get("map_50_95"))
        if value is None:
            return None
        return float(value)

    def metrics_are_better(
        self,
        candidate_metrics: Mapping[str, object] | None,
        incumbent_metrics: Mapping[str, object] | None,
        *,
        min_delta: float,
    ) -> bool:
        candidate_value = self.metric_value(candidate_metrics)
        if candidate_value is None:
            return False
        incumbent_value = self.metric_value(incumbent_metrics)
        if incumbent_value is None:
            return True
        if candidate_value < incumbent_value + float(min_delta):
            return False
        if candidate_value > incumbent_value + 1e-6:
            return True
        candidate_boxes = int(candidate_metrics.get("total_prediction_boxes", 1 << 30))
        incumbent_boxes = int(incumbent_metrics.get("total_prediction_boxes", 1 << 30))
        return candidate_boxes < incumbent_boxes


def get_training_adapter(
    model_name: str,
    model_family: str,
) -> DetectionTrainingAdapter:
    del model_name, model_family
    return UniversalSplitTrainingAdapter()


def _normalize_proxy_metrics(metrics: Mapping[str, object] | None) -> ProxyMetrics:
    if not metrics:
        return {}
    normalized: ProxyMetrics = {}
    for key, value in metrics.items():
        if value is None:
            normalized[str(key)] = None
        elif isinstance(value, (float, int, str)):
            normalized[str(key)] = value
    return normalized
