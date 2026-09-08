from __future__ import annotations

import copy
import time
from typing import Any

import torch
from loguru import logger as default_logger

from cloud.training.proxy_eval import ProxyEarlyStopper, ProxyEvalScheduler
from cloud.training.types import (
    CandidateState,
    FixedSplitTrainingContext,
    FixedSplitTrainingResult,
    ProxyEvalResult,
    ProxyMetrics,
)


class FixedSplitRetrainEngine:
    def run(self, context: FixedSplitTrainingContext) -> FixedSplitTrainingResult:
        plan = context.plan
        log = context.logger or default_logger
        adapter = context.adapter
        config = plan.proxy_eval_config
        scheduler = ProxyEvalScheduler(config)
        total_started = time.perf_counter()
        baseline_state = _snapshot_model_state(context.model)
        proxy_results: list[ProxyEvalResult] = []
        proxy_eval_time = 0.0
        epoch_results = []
        trained_epochs = 0
        early_stop_reason: str | None = None
        can_select_by_proxy = bool(config.enabled and context.validation_gt_annotations)

        log.info(
            "[FixedSplitCL][TrainPlan] model_name={} model_family={} total_samples={} "
            "epochs={} effective_batch_size={} eval_interval={} max_eval_samples={}",
            plan.model_name,
            plan.model_family,
            int(plan.total_samples),
            int(plan.epochs),
            int(plan.effective_batch_size),
            int(config.interval_epochs or 0),
            config.max_eval_samples,
        )

        proxy_metrics_before: ProxyMetrics = {}
        baseline_proxy_metric: float | None = None
        if can_select_by_proxy:
            baseline_proxy_eval = adapter.evaluate_proxy(
                context,
                epoch=0,
                stage_label="proxy evaluation before training",
                max_samples=config.max_eval_samples,
            )
            proxy_results.append(baseline_proxy_eval)
            proxy_eval_time += baseline_proxy_eval.elapsed
            proxy_metrics_before = dict(baseline_proxy_eval.metrics)
            baseline_proxy_metric = baseline_proxy_eval.metric
            self._log_proxy_eval(log, plan, baseline_proxy_eval)

        best_candidate: CandidateState | None = None
        early_stopper = ProxyEarlyStopper(config)
        optimizer = adapter.build_optimizer(context)

        for epoch in range(1, int(plan.epochs) + 1):
            epoch_result = adapter.train_one_epoch(
                context,
                epoch=epoch,
                total_epochs=int(plan.epochs),
                optimizer=optimizer,
            )

            trained_epochs = epoch
            epoch_results.append(epoch_result)
            log.info(
                "[FixedSplitCL][Epoch] model_name={} model_family={} epoch {}/{} "
                "avg_loss={} suffix_forward_backward_time={:.3f}s train_time={:.3f}s",
                plan.model_name,
                plan.model_family,
                int(epoch),
                int(plan.epochs),
                "nan" if epoch_result.loss is None else f"{float(epoch_result.loss):.6f}",
                float(epoch_result.suffix_forward_backward_time),
                float(epoch_result.train_time),
            )

            if not (can_select_by_proxy and scheduler.should_eval(epoch, int(plan.epochs))):
                continue

            proxy_eval = adapter.evaluate_proxy(
                context,
                epoch=epoch,
                stage_label=f"proxy evaluation after epoch {epoch}",
                max_samples=config.max_eval_samples,
            )
            proxy_results.append(proxy_eval)
            proxy_eval_time += proxy_eval.elapsed
            self._log_proxy_eval(log, plan, proxy_eval)

            incumbent_metrics = (
                best_candidate.proxy_metrics
                if best_candidate is not None
                else proxy_metrics_before
            )
            improved = adapter.metrics_are_better(
                proxy_eval.metrics,
                incumbent_metrics,
                min_delta=float(config.min_delta),
            )
            if improved:
                best_candidate = CandidateState(
                    epoch=epoch,
                    state_dict=_snapshot_model_state(context.model),
                    proxy_metrics=dict(proxy_eval.metrics),
                    proxy_metric=proxy_eval.metric,
                )
                log.info(
                    "[FixedSplitCL][Candidate] model_name={} model_family={} "
                    "best_epoch={} best_proxy_metric={}",
                    plan.model_name,
                    plan.model_family,
                    int(epoch),
                    _format_metric(proxy_eval.metric),
                )

            decision = early_stopper.record(
                proxy_eval,
                improved=improved,
                best_metric=(
                    best_candidate.proxy_metric
                    if best_candidate is not None
                    else baseline_proxy_metric
                ),
            )
            if decision.should_stop:
                early_stop_reason = decision.reason
                log.info(
                    "[FixedSplitCL][EarlyStop] model_name={} model_family={} epoch={} "
                    "early_stop_reason={}",
                    plan.model_name,
                    plan.model_family,
                    int(epoch),
                    early_stop_reason,
                )
                break

        proxy_metrics_after: ProxyMetrics = {}
        result_available = True
        if can_select_by_proxy and best_candidate is not None:
            if best_candidate.state_dict is not None:
                context.model.load_state_dict(best_candidate.state_dict, strict=False)
                _set_eval(context.model)
            proxy_metrics_after = dict(best_candidate.proxy_metrics or {})
        elif can_select_by_proxy:
            result_available = False
            proxy_metrics_after = dict(proxy_metrics_before)
            context.model.load_state_dict(baseline_state, strict=False)
            _set_eval(context.model)
            log.warning(
                "[FixedSplitCL][Candidate] model_name={} model_family={} "
                "no publishable checkpoint because validation {} did not improve "
                "the pre-training baseline by min_delta={}.",
                plan.model_name,
                plan.model_family,
                "proxy_mAP_50_95",
                float(config.min_delta),
            )
        elif context.validation_gt_annotations and config.enabled and proxy_results:
            proxy_metrics_after = dict(proxy_results[-1].metrics)

        total_retraining_time = time.perf_counter() - total_started
        suffix_forward_backward_time = sum(
            float(result.suffix_forward_backward_time) for result in epoch_results
        )
        best_epoch = best_candidate.epoch if best_candidate is not None else None
        best_proxy_metric = best_candidate.proxy_metric if best_candidate is not None else None
        log.info(
            "[FixedSplitCL][RetrainProfile] model_name={} model_family={} "
            "total_samples={} epochs={} effective_batch_size={} eval_interval={} "
            "max_eval_samples={} best_epoch={} best_proxy_metric={} trained_epochs={} "
            "early_stop_reason={} suffix_forward_backward_time={:.3f}s "
            "proxy_eval_time={:.3f}s total_retraining_time={:.3f}s",
            plan.model_name,
            plan.model_family,
            int(plan.total_samples),
            int(plan.epochs),
            int(plan.effective_batch_size),
            int(config.interval_epochs or 0),
            config.max_eval_samples,
            best_epoch,
            _format_metric(best_proxy_metric),
            int(trained_epochs),
            early_stop_reason,
            suffix_forward_backward_time,
            proxy_eval_time,
            total_retraining_time,
        )

        return FixedSplitTrainingResult(
            proxy_metrics_before=proxy_metrics_before,
            proxy_metrics_after=proxy_metrics_after,
            baseline_state=baseline_state,
            best_candidate=best_candidate,
            epoch_results=epoch_results,
            proxy_results=proxy_results,
            suffix_forward_backward_time=suffix_forward_backward_time,
            proxy_eval_time=proxy_eval_time,
            total_retraining_time=total_retraining_time,
            best_epoch=best_epoch,
            best_proxy_metric=best_proxy_metric,
            trained_epochs=trained_epochs,
            early_stop_reason=early_stop_reason,
            result_available=result_available,
        )

    @staticmethod
    def _log_proxy_eval(
        log: Any,
        plan: Any,
        result: ProxyEvalResult,
    ) -> None:
        log.info(
            "[FixedSplitCL][ProxyEval] model_name={} model_family={} stage={} "
            "epoch={} metric_name={} proxy_metric={} evaluated_samples={} "
            "nonempty_predictions={} total_prediction_boxes={} elapsed={:.3f}s",
            plan.model_name,
            plan.model_family,
            result.stage_label,
            result.epoch,
            result.metrics.get("primary_metric_name", "proxy_mAP_50_95"),
            _format_metric(result.metric),
            int(result.metrics.get("evaluated_samples", 0) or 0),
            int(result.metrics.get("nonempty_predictions", 0) or 0),
            int(result.metrics.get("total_prediction_boxes", 0) or 0),
            float(result.elapsed),
        )

def _snapshot_model_state(model: torch.nn.Module) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            snapshot[key] = value.detach().cpu().clone()
        else:
            snapshot[key] = copy.deepcopy(value)
    return snapshot


def _set_eval(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        module.training = False


def _format_metric(metric: float | None) -> str:
    if metric is None:
        return "None"
    return f"{float(metric):.4f}"
