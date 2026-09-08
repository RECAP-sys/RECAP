from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import grpc

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.grpc_options import grpc_message_options


@dataclass
class CloudResourceState:
    cpu_utilization: float
    gpu_utilization: float
    memory_utilization: float
    train_queue_size: int
    max_queue_size: int
    timestamp: float = field(default_factory=time.time)

    @property
    def queue_pressure(self) -> float:
        if self.max_queue_size <= 0:
            return 0.0
        return max(0.0, min(1.0, self.train_queue_size / float(self.max_queue_size)))

    @property
    def compute_pressure(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                max(
                    float(self.cpu_utilization),
                    float(self.gpu_utilization),
                    float(self.memory_utilization),
                    self.queue_pressure,
                ),
            ),
        )

    def is_stale(self, max_age_sec: float) -> bool:
        return (time.time() - float(self.timestamp)) > float(max_age_sec)


@dataclass
class PendingTrainingStats:
    total_samples: int
    high_quality_count: int
    low_quality_count: int
    low_quality_rate: float
    drift_detected: bool
    high_quality_feature_bytes: int
    low_quality_feature_bytes: int
    low_quality_raw_bytes: int

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PendingTrainingStats":
        return cls(
            total_samples=int(payload.get("total_samples", 0)),
            high_quality_count=int(payload.get("high_quality_count", 0)),
            low_quality_count=int(payload.get("low_quality_count", 0)),
            low_quality_rate=float(payload.get("low_quality_rate", 0.0)),
            drift_detected=bool(payload.get("drift_detected", False)),
            high_quality_feature_bytes=int(payload.get("high_quality_feature_bytes", 0)),
            low_quality_feature_bytes=int(payload.get("low_quality_feature_bytes", 0)),
            low_quality_raw_bytes=int(payload.get("low_quality_raw_bytes", 0)),
        )

    @property
    def always_sent_bytes(self) -> int:
        return self.high_quality_feature_bytes + self.low_quality_raw_bytes


@dataclass
class TrainingDecision:
    train_now: bool
    send_low_conf_features: bool
    urgency: float
    compute_pressure: float
    bandwidth_pressure: float
    bandwidth_mbps: float = 0.0
    bundle_cap_bytes: int | None = None
    action_scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    cooldown_remaining_decisions: int = 0
    cooldown_remaining_sec: float = 0.0
    trigger_token: int | None = None


class ResourceAwareCLTrigger:
    """Lyapunov-style trigger that chooses training and teacher-needed feature upload.

    The fixed split plan is computed separately at startup. Runtime decisions are
    limited to:
      1. whether continual learning should start now
      2. whether teacher-needed samples should also upload intermediate features

    Only Q_cloud and Q_bw are maintained as virtual resource queues.
    """

    def __init__(
        self,
        *,
        V: float = 10.0,
        K_p: float = 1.0,
        K_d: float = 0.5,
        lambda_cloud: float = 0.5,
        lambda_bw: float = 0.5,
        w_cloud: float = 1.0,
        w_bw: float = 1.0,
        feature_cloud_cost_factor: float = 0.5,
        min_training_samples: int = 1,
        max_training_samples: int = 0,
        drift_bonus: float = 0.35,
        upload_time_budget_sec: float = 5.0,
        bundle_max_bytes: int = 33554432,
        bundle_min_bytes: int = 8388608,
        bundle_target_upload_sec: float = 45.0,
        cooldown_decisions: int = 0,
        min_training_interval_sec: float = 0.0,
    ) -> None:
        self.V = float(V)
        self.K_p = float(K_p)
        self.K_d = float(K_d)
        self.lambda_cloud = float(lambda_cloud)
        self.lambda_bw = float(lambda_bw)
        self.w_cloud = float(w_cloud)
        self.w_bw = float(w_bw)
        self.feature_cloud_cost_factor = max(
            0.0,
            min(1.0, float(feature_cloud_cost_factor)),
        )
        self.min_training_samples = int(min_training_samples)
        self.max_training_samples = max(0, int(max_training_samples))
        self.drift_bonus = float(drift_bonus)
        self.upload_time_budget_sec = float(upload_time_budget_sec)
        self.bundle_max_bytes = max(1, int(bundle_max_bytes))
        self.bundle_min_bytes = max(1, min(int(bundle_min_bytes), self.bundle_max_bytes))
        self.bundle_target_upload_sec = max(1e-6, float(bundle_target_upload_sec))
        self.cooldown_decisions = max(0, int(cooldown_decisions))
        self.min_training_interval_sec = max(0.0, float(min_training_interval_sec))

        self.Q_cloud = 0.0
        self.Q_bw = 0.0
        self.loss_best = float("inf")
        self.loss_prev = 0.0
        self.step = 0
        self.trigger_count = 0
        self._last_trigger_step: int | None = None
        self._last_trigger_monotonic: float | None = None
        self._pending_trigger_costs: dict[int, tuple[float, float]] = {}
        self.history: list[dict[str, Any]] = []

    @property
    def effective_trigger_rate(self) -> float:
        if self.step == 0:
            return 0.0
        return self.trigger_count / float(self.step)

    @property
    def queue_snapshot(self) -> dict[str, float]:
        return {
            "Q_cloud": self.Q_cloud,
            "Q_bw": self.Q_bw,
        }

    def reset(self) -> None:
        self.Q_cloud = 0.0
        self.Q_bw = 0.0
        self.loss_best = float("inf")
        self.loss_prev = 0.0
        self.step = 0
        self.trigger_count = 0
        self._last_trigger_step = None
        self._last_trigger_monotonic = None
        self._pending_trigger_costs.clear()
        self.history.clear()

    def confirm_trigger(self, decision: TrainingDecision) -> None:
        """Commit resource costs and cooldown after the cloud accepts a job."""

        token = decision.trigger_token
        if not decision.train_now or token is None:
            raise ValueError("Only a pending training decision can be confirmed.")
        costs = self._pending_trigger_costs.pop(int(token), None)
        if costs is None:
            raise RuntimeError("Training decision is not pending or was already resolved.")
        selected_cloud_cost, selected_bw_cost = costs
        self.trigger_count += 1
        self._last_trigger_step = int(token)
        self._last_trigger_monotonic = time.monotonic()
        self.Q_cloud = max(0.0, self.Q_cloud + selected_cloud_cost - self.lambda_cloud)
        self.Q_bw = max(0.0, self.Q_bw + selected_bw_cost - self.lambda_bw)

    def cancel_trigger(self, decision: TrainingDecision) -> None:
        """Release an unaccepted decision without arming post-trigger cooldown."""

        token = decision.trigger_token
        if token is None:
            return
        costs = self._pending_trigger_costs.pop(int(token), None)
        if costs is None:
            return
        self.Q_cloud = max(0.0, self.Q_cloud - self.lambda_cloud)
        self.Q_bw = max(0.0, self.Q_bw - self.lambda_bw)

    def _urgency(
        self,
        drift_detected: bool,
        stats: PendingTrainingStats,
    ) -> float:
        threshold_progress = 0.0
        if self.min_training_samples > 0:
            threshold_progress = min(
                1.0,
                stats.low_quality_count / float(self.min_training_samples),
            )
        loss = max(
            0.0,
            (0.70 * float(stats.low_quality_rate)) + (0.30 * threshold_progress),
        )
        if drift_detected or stats.drift_detected:
            loss += self.drift_bonus
        self.loss_best = min(self.loss_best, loss)
        derivative = max(0.0, loss - self.loss_prev)
        urgency = max(
            0.0,
            (self.K_p * (loss - self.loss_best)) + (self.K_d * derivative) + loss,
        )
        self.loss_prev = loss
        return urgency

    def _bandwidth_pressure(self, bandwidth_mbps: float, payload_bytes: int) -> float:
        if payload_bytes <= 0:
            return 0.0
        if bandwidth_mbps <= 0:
            return 1.0
        transfer_sec = (payload_bytes * 8.0) / (bandwidth_mbps * 1_000_000.0)
        return max(0.0, min(1.0, transfer_sec / max(self.upload_time_budget_sec, 1e-6)))

    def effective_bundle_cap_bytes(self, bandwidth_mbps: float) -> int:
        if bandwidth_mbps <= 0:
            return int(self.bundle_max_bytes)
        bandwidth_cap = float(bandwidth_mbps) * self.bundle_target_upload_sec / 8.0 * 1_000_000.0
        return int(
            min(
                self.bundle_max_bytes,
                max(self.bundle_min_bytes, bandwidth_cap),
            )
        )

    def decide(
        self,
        *,
        drift_detected: bool,
        cloud_state: CloudResourceState,
        bandwidth_mbps: float,
        sample_stats: PendingTrainingStats | dict[str, Any],
    ) -> TrainingDecision:
        stats = (
            sample_stats
            if isinstance(sample_stats, PendingTrainingStats)
            else PendingTrainingStats.from_mapping(sample_stats)
        )
        urgency = self._urgency(drift_detected, stats)
        compute_pressure = cloud_state.compute_pressure

        raw_only_payload_bytes = stats.always_sent_bytes
        raw_plus_feature_payload_bytes = stats.always_sent_bytes + stats.low_quality_feature_bytes
        raw_only_bw_pressure = self._bandwidth_pressure(bandwidth_mbps, raw_only_payload_bytes)
        raw_plus_feature_bw_pressure = self._bandwidth_pressure(
            bandwidth_mbps, raw_plus_feature_payload_bytes
        )
        training_disabled = stats.low_quality_count < max(1, self.min_training_samples)
        cooldown_remaining_decisions = 0
        if self._last_trigger_step is not None:
            decisions_since_trigger = max(0, self.step - self._last_trigger_step)
            cooldown_remaining_decisions = max(
                0,
                self.cooldown_decisions - decisions_since_trigger,
            )
        cooldown_remaining_sec = 0.0
        if self._last_trigger_monotonic is not None:
            cooldown_remaining_sec = max(
                0.0,
                self.min_training_interval_sec - (time.monotonic() - self._last_trigger_monotonic),
            )
        cooldown_active = cooldown_remaining_decisions > 0 or cooldown_remaining_sec > 0.0
        low_conf_feature_ratio = stats.low_quality_feature_bytes / float(
            max(raw_plus_feature_payload_bytes, 1)
        )
        raw_only_cloud_cost = compute_pressure
        raw_plus_feature_cloud_cost = self.feature_cloud_cost_factor * compute_pressure

        no_train_score = self.V * urgency
        raw_only_regularizer = (
            self.w_cloud * compute_pressure * (1.0 + compute_pressure)
            + self.w_bw * raw_only_bw_pressure * (1.0 + raw_only_bw_pressure)
            + compute_pressure * low_conf_feature_ratio
        )
        raw_only_score = (
            self.w_cloud * self.Q_cloud * raw_only_cloud_cost
            + self.w_bw * self.Q_bw * raw_only_bw_pressure
            + raw_only_regularizer
        )
        raw_plus_feature_regularizer = (
            self.w_cloud
            * compute_pressure
            * (1.0 + self.feature_cloud_cost_factor * compute_pressure)
            + self.w_bw * raw_plus_feature_bw_pressure * (1.0 + raw_plus_feature_bw_pressure)
            + (1.0 + raw_plus_feature_bw_pressure) * low_conf_feature_ratio
        )
        raw_plus_feature_score = (
            self.w_cloud * self.Q_cloud * raw_plus_feature_cloud_cost
            + self.w_bw * self.Q_bw * raw_plus_feature_bw_pressure
            + raw_plus_feature_regularizer
        )
        if training_disabled or cooldown_active:
            raw_only_score = float("inf")
            raw_plus_feature_score = float("inf")

        action_scores = {
            "skip_training": float(no_train_score),
            "train_raw_only": float(raw_only_score),
            "train_raw_plus_feature": float(raw_plus_feature_score),
        }
        selected_action = min(action_scores, key=action_scores.get)
        train_now = selected_action != "skip_training"
        send_low_conf_features = selected_action == "train_raw_plus_feature"

        selected_cloud_cost = 0.0
        selected_bw_cost = 0.0
        trigger_token: int | None = None
        if train_now:
            selected_cloud_cost = (
                raw_plus_feature_cloud_cost if send_low_conf_features else raw_only_cloud_cost
            )
            selected_bw_cost = (
                raw_plus_feature_bw_pressure if send_low_conf_features else raw_only_bw_pressure
            )
            trigger_token = int(self.step)
            self._pending_trigger_costs[trigger_token] = (
                float(selected_cloud_cost),
                float(selected_bw_cost),
            )

        self.step += 1
        if not train_now:
            self.Q_cloud = max(0.0, self.Q_cloud - self.lambda_cloud)
            self.Q_bw = max(0.0, self.Q_bw - self.lambda_bw)

        if training_disabled:
            reason = (
                "Skipped training until enough teacher-needed samples accumulate "
                f"({stats.low_quality_count}/{max(1, self.min_training_samples)})."
            )
        elif cooldown_active:
            reason = (
                "Skipped training during the post-trigger cooldown "
                f"({cooldown_remaining_decisions} decisions, "
                f"{cooldown_remaining_sec:.1f}s remaining)."
            )
        elif not train_now:
            reason = "Skipped training because Lyapunov penalty outweighed adaptation gain."
        elif send_low_conf_features:
            reason = (
                "Triggered training and included teacher-needed features to reduce "
                "server recomputation under cloud compute pressure."
            )
        else:
            reason = (
                "Triggered training in raw-only teacher-needed mode to reduce bandwidth pressure."
            )

        decision = TrainingDecision(
            train_now=train_now,
            send_low_conf_features=send_low_conf_features,
            urgency=float(urgency),
            compute_pressure=float(compute_pressure),
            bandwidth_pressure=float(
                raw_plus_feature_bw_pressure if send_low_conf_features else raw_only_bw_pressure
            ),
            bandwidth_mbps=float(bandwidth_mbps),
            bundle_cap_bytes=self.effective_bundle_cap_bytes(float(bandwidth_mbps)),
            action_scores=action_scores,
            reason=reason,
            cooldown_remaining_decisions=int(cooldown_remaining_decisions),
            cooldown_remaining_sec=float(cooldown_remaining_sec),
            trigger_token=trigger_token,
        )
        self.history.append(
            {
                "timestamp": time.time(),
                "train_now": train_now,
                "send_low_conf_features": send_low_conf_features,
                "drift_detected": bool(drift_detected),
                "low_quality_rate": float(stats.low_quality_rate),
                "urgency": float(urgency),
                "compute_pressure": float(compute_pressure),
                "bandwidth_mbps": float(bandwidth_mbps),
                "bundle_cap_bytes": decision.bundle_cap_bytes,
                "action_scores": action_scores,
                "cooldown_remaining_decisions": int(cooldown_remaining_decisions),
                "cooldown_remaining_sec": float(cooldown_remaining_sec),
            }
        )
        return decision


def query_cloud_resource(
    server_ip: str, *, edge_id: int = 0, timeout_sec: float = 3.0
) -> CloudResourceState:
    with grpc.insecure_channel(server_ip, options=grpc_message_options()) as channel:
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        reply = stub.query_resource(
            message_transmission_pb2.ResourceRequest(edge_id=int(edge_id)),
            timeout=timeout_sec,
        )
    return CloudResourceState(
        cpu_utilization=float(reply.cpu_utilization),
        gpu_utilization=float(reply.gpu_utilization),
        memory_utilization=float(reply.memory_utilization),
        train_queue_size=int(reply.train_queue_size),
        max_queue_size=int(reply.max_queue_size),
    )


def estimate_bandwidth(
    server_ip: str,
    *,
    probe_size_bytes: int = 64 * 1024,
    timeout_sec: float = 3.0,
) -> float:
    payload = "x" * max(1, int(probe_size_bytes))
    started = time.perf_counter()
    try:
        with grpc.insecure_channel(server_ip, options=grpc_message_options()) as channel:
            stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
            reply = stub.bandwidth_probe(
                message_transmission_pb2.BandwidthProbeRequest(payload=payload),
                timeout=timeout_sec,
            )
        round_trip_sec = max(time.perf_counter() - started, 1e-6)
        echoed_bytes = len(reply.payload.encode("utf-8"))
        return (echoed_bytes * 8.0) / round_trip_sec / 1_000_000.0
    except Exception:
        return 0.0


def create_resource_aware_trigger(config: Any) -> ResourceAwareCLTrigger:
    ra = getattr(config, "resource_aware_trigger", None)
    retrain = getattr(config, "retrain", None)
    model_name = str(getattr(config, "lightweight", "") or "").strip()

    def _get(key: str, default: Any, *sources: Any) -> Any:
        for source in sources:
            if source is None:
                continue
            value = getattr(source, key, None)
            if value is not None:
                return value
        return default

    def _model_value(key: str, default: Any) -> Any:
        values = dict(getattr(ra, f"{key}_by_model", {}) or {})
        return values.get(model_name, _get(key, default, ra))

    return ResourceAwareCLTrigger(
        V=float(_get("V", 10.0, ra)),
        K_p=float(_get("K_p", 1.0, ra)),
        K_d=float(_get("K_d", 0.5, ra)),
        lambda_cloud=float(_get("lambda_cloud", 0.5, ra)),
        lambda_bw=float(_get("lambda_bw", 0.5, ra)),
        w_cloud=float(_get("w_cloud", 1.0, ra)),
        w_bw=float(_get("w_bw", 1.0, ra)),
        feature_cloud_cost_factor=float(_get("feature_cloud_cost_factor", 0.5, ra)),
        min_training_samples=int(
            _model_value("min_training_samples", getattr(retrain, "collect_num", 1))
        ),
        max_training_samples=int(_model_value("max_training_samples", 0)),
        drift_bonus=float(_get("drift_bonus", 0.35, ra)),
        upload_time_budget_sec=float(_get("upload_time_budget_sec", 5.0, ra)),
        bundle_max_bytes=int(_get("bundle_max_bytes", 33554432, ra)),
        bundle_min_bytes=int(_get("bundle_min_bytes", 8388608, ra)),
        bundle_target_upload_sec=float(_get("bundle_target_upload_sec", 45.0, ra)),
        cooldown_decisions=int(_get("cooldown_decisions", 0, ra)),
        min_training_interval_sec=float(_get("min_training_interval_sec", 0.0, ra)),
    )
