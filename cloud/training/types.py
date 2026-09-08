from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

ProxyMetricValue = float | int | str | None
ProxyMetrics = dict[str, ProxyMetricValue]


@dataclass
class FixedSplitTrainingPlan:
    model_name: str
    model_family: str
    total_samples: int
    epochs: int
    effective_batch_size: int
    learning_rate: float
    proxy_eval_config: Any
    training_label: str | None = None
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    shuffle_samples: bool = False


@dataclass
class EpochTrainResult:
    epoch: int
    loss: float | None
    train_time: float
    suffix_forward_backward_time: float = 0.0


@dataclass
class ProxyEvalResult:
    metrics: ProxyMetrics
    metric: float | None
    elapsed: float
    epoch: int | None = None
    stage_label: str = "proxy evaluation"


@dataclass
class CandidateState:
    epoch: int
    state_dict: dict[str, object] | None
    proxy_metrics: ProxyMetrics | None = None
    proxy_metric: float | None = None


@dataclass
class EarlyStopDecision:
    should_stop: bool = False
    reason: str | None = None
    stale_evaluations: int = 0


@dataclass
class FixedSplitTrainingResult:
    proxy_metrics_before: ProxyMetrics
    proxy_metrics_after: ProxyMetrics
    baseline_state: dict[str, object]
    best_candidate: CandidateState | None
    epoch_results: list[EpochTrainResult] = field(default_factory=list)
    proxy_results: list[ProxyEvalResult] = field(default_factory=list)
    suffix_forward_backward_time: float = 0.0
    proxy_eval_time: float = 0.0
    total_retraining_time: float = 0.0
    best_epoch: int | None = None
    best_proxy_metric: float | None = None
    trained_epochs: int = 0
    early_stop_reason: str | None = None
    result_available: bool = True


@dataclass
class FixedSplitTrainingContext:
    model: Any
    plan: FixedSplitTrainingPlan
    adapter: Any
    training_kwargs: dict[str, Any] = field(default_factory=dict)
    gt_annotations: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    validation_gt_annotations: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    validation_sample_ids: list[str] = field(default_factory=list)
    fixed_proxy_evaluator: Callable[..., Mapping[str, object]] | None = None
    tinynext_proxy_evaluator: Callable[..., Mapping[str, object]] | None = None
    retrain_profile: Any = None
    logger: Any = None
