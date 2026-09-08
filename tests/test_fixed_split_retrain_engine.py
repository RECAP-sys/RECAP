from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cloud.training.fixed_split_engine import FixedSplitRetrainEngine
from cloud.training.proxy_eval import ProxyEvalConfig
from cloud.training.types import (
    EpochTrainResult,
    FixedSplitTrainingContext,
    FixedSplitTrainingPlan,
    ProxyEvalResult,
)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))


@dataclass
class FakeAdapter:
    metrics: list[float | None]
    train_suffix_time: float = 0.25

    def __post_init__(self) -> None:
        self.trained_epochs: list[int] = []
        self.eval_calls: list[tuple[int | None, str]] = []

    def build_optimizer(self, context):
        del context
        return object()

    def train_one_epoch(self, context, *, epoch: int, total_epochs: int, optimizer):
        del total_epochs, optimizer
        with torch.no_grad():
            context.model.weight.fill_(float(epoch))
        self.trained_epochs.append(epoch)
        return EpochTrainResult(
            epoch=epoch,
            loss=1.0 / epoch,
            train_time=0.5,
            suffix_forward_backward_time=self.train_suffix_time,
        )

    def evaluate_proxy(
        self,
        context,
        *,
        epoch,
        stage_label,
        max_samples,
    ):
        del context, max_samples
        self.eval_calls.append((epoch, stage_label))
        value = self.metrics.pop(0)
        return ProxyEvalResult(
            metrics={
                "primary_metric": value,
                "primary_metric_name": "proxy_mAP_50_95",
                "map_50_95": value,
                "evaluated_samples": 4,
            },
            metric=value,
            elapsed=0.125,
            epoch=epoch,
            stage_label=stage_label,
        )

    def metric_value(self, metrics):
        if not metrics or metrics.get("primary_metric") is None:
            return None
        return float(metrics["primary_metric"])

    def metrics_are_better(self, candidate_metrics, incumbent_metrics, *, min_delta: float):
        candidate = self.metric_value(candidate_metrics)
        incumbent = self.metric_value(incumbent_metrics)
        if candidate is None:
            return False
        if incumbent is None:
            return True
        return candidate >= incumbent + min_delta


def _context(adapter: FakeAdapter, *, model_family: str = "surprise") -> FixedSplitTrainingContext:
    return FixedSplitTrainingContext(
        model=FakeModel(),
        plan=FixedSplitTrainingPlan(
            model_name="fake_detector",
            model_family=model_family,
            total_samples=4,
            epochs=5,
            effective_batch_size=2,
            learning_rate=0.1,
            proxy_eval_config=ProxyEvalConfig(
                eval_final=True,
                interval_epochs=2,
                min_delta=0.05,
                patience=10,
                max_eval_samples=4,
            ),
        ),
        adapter=adapter,
        gt_annotations={"a": {"boxes": [[0, 0, 1, 1]], "labels": [1]}},
        validation_gt_annotations={"v": {"boxes": [[0, 0, 1, 1]], "labels": [1]}},
        validation_sample_ids=["v"],
    )


def test_engine_flow_is_decoupled_from_model_family() -> None:
    adapter = FakeAdapter(metrics=[0.1, 0.2, 0.3, 0.28])
    context = _context(adapter, model_family="not-a-special-family")

    result = FixedSplitRetrainEngine().run(context)

    assert adapter.trained_epochs == [1, 2, 3, 4, 5]
    assert result.trained_epochs == 5
    assert result.best_epoch == 4


def test_engine_saves_and_restores_best_candidate_on_metric_improvement() -> None:
    adapter = FakeAdapter(metrics=[0.4, 0.6, 0.55, 0.5])
    context = _context(adapter)

    result = FixedSplitRetrainEngine().run(context)

    assert result.best_epoch == 2
    assert result.best_proxy_metric == 0.6
    assert context.model.weight.item() == 2.0


def test_engine_early_stop_reduces_training_epochs() -> None:
    adapter = FakeAdapter(metrics=[0.85, 0.9, 0.91, 0.91])
    context = _context(adapter)
    context.plan.proxy_eval_config.interval_epochs = 1
    context.plan.proxy_eval_config.patience = 2
    context.plan.proxy_eval_config.min_delta = 0.05

    result = FixedSplitRetrainEngine().run(context)

    assert result.trained_epochs == 3
    assert adapter.trained_epochs == [1, 2, 3]
    assert result.early_stop_reason is not None


def test_engine_tracks_proxy_and_suffix_times_separately() -> None:
    adapter = FakeAdapter(metrics=[0.1, 0.2, 0.3, 0.35], train_suffix_time=0.75)
    context = _context(adapter)

    result = FixedSplitRetrainEngine().run(context)

    assert result.suffix_forward_backward_time == 5 * 0.75
    assert result.proxy_eval_time == 4 * 0.125


def test_engine_restores_baseline_without_proxy_candidate_eval() -> None:
    adapter = FakeAdapter(metrics=[0.4])
    context = _context(adapter)
    context.plan.proxy_eval_config.eval_final = False
    context.plan.proxy_eval_config.interval_epochs = 100

    result = FixedSplitRetrainEngine().run(context)

    assert adapter.eval_calls == [(0, "proxy evaluation before training")]
    assert result.best_candidate is None
    assert result.result_available is False
    assert result.proxy_metrics_before["primary_metric"] == 0.4
    assert result.proxy_metrics_after["primary_metric"] == 0.4
    assert context.model.weight.item() == 0.0


def test_engine_restores_baseline_when_validation_metric_is_unavailable() -> None:
    adapter = FakeAdapter(metrics=[None, None, None, None])
    context = _context(adapter)

    result = FixedSplitRetrainEngine().run(context)

    assert result.best_candidate is None
    assert result.best_proxy_metric is None
    assert result.result_available is False
    assert result.proxy_metrics_after["primary_metric"] is None
    assert context.model.weight.item() == 0.0


def test_engine_evaluates_actual_baseline_without_candidate_checkpoints() -> None:
    adapter = FakeAdapter(metrics=[0.4])
    context = _context(adapter)
    context.plan.proxy_eval_config.eval_final = False
    context.plan.proxy_eval_config.interval_epochs = 100

    result = FixedSplitRetrainEngine().run(context)

    assert result.proxy_eval_time == 0.125
    assert result.proxy_metrics_before["primary_metric"] == 0.4
    assert result.result_available is False
    assert context.model.weight.item() == 0.0


def test_engine_rejects_candidates_that_regress_from_baseline() -> None:
    adapter = FakeAdapter(metrics=[0.7, 0.6, 0.65, 0.69])
    context = _context(adapter)

    result = FixedSplitRetrainEngine().run(context)

    assert result.best_candidate is None
    assert result.result_available is False
    assert result.proxy_metrics_before["primary_metric"] == 0.7
    assert result.proxy_metrics_after["primary_metric"] == 0.7
    assert context.model.weight.item() == 0.0
