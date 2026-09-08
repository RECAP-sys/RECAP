import subprocess

import psutil
from loguru import logger

from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from grpc_server.continual_backends import DisabledContinualLearningBackend
from grpc_server.workspace import (
    normalize_client_cache_path,
)

# ── Resource monitoring helpers ──────────────────
_HAS_PSUTIL = True


def _get_cpu_utilization() -> float:
    """Return CPU utilisation in [0, 1]."""
    if _HAS_PSUTIL:
        return psutil.cpu_percent(interval=0.1) / 100.0
    return 0.0


def _get_memory_utilization() -> float:
    """Return memory utilisation in [0, 1]."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().percent / 100.0
    return 0.0


def _get_gpu_utilization() -> float:
    """Return GPU utilisation in [0, 1] (NVIDIA only)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            vals = [float(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
            if vals:
                return max(vals) / 100.0
    except Exception:
        pass
    return 0.0


class MessageTransmissionServicer(message_transmission_pb2_grpc.MessageTransmissionServicer):
    def __init__(
        self,
        id,
        workspace_root=None,
        edge_registry=None,
        continual_backend=None,
        log_internal_ids: bool = False,
        experiment_result_repository=None,
        experiment_id: str = "",
        experiment_scenario_slug: str = "",
        experiment_edge_count: int = 1,
        experiment_repeat: int = 1,
        experiment_method: str = "",
        experiment_run_id: str = "",
    ):
        self.id = id
        self.workspace_root = workspace_root or "./cache/server_workspace"
        self.edge_registry = edge_registry
        self.continual_backend = continual_backend or DisabledContinualLearningBackend()
        self.log_internal_ids = bool(log_internal_ids)
        self.experiment_result_repository = experiment_result_repository
        self.experiment_id = str(experiment_id or "")
        self.experiment_scenario_slug = str(experiment_scenario_slug or "")
        self.experiment_edge_count = int(experiment_edge_count or 1)
        self.experiment_repeat = int(experiment_repeat or 1)
        self.experiment_method = str(experiment_method or "")
        self.experiment_run_id = str(experiment_run_id or "")

    def _record_experiment_event(self, event: str, **payload) -> None:
        repository = self.experiment_result_repository
        if repository is None:
            return
        try:
            repository.record_cloud_event(
                experiment_id=self.experiment_id,
                scenario_slug=self.experiment_scenario_slug,
                edge_count=self.experiment_edge_count,
                repeat=self.experiment_repeat,
                method=self.experiment_method,
                run_id=self.experiment_run_id,
                event=event,
                **payload,
            )
        except Exception as exc:
            logger.warning("Experiment event recording failed: {}", safe_error_summary(exc))

    def _log_failure(self, label: str, exc: BaseException) -> None:
        logger.error("{} failed: {}", label, safe_error_summary(exc))
        log_diagnostic_debug(
            self,
            f"{label} failure",
            lambda error=exc: {"error": repr(error)},
        )

    def train_model_request(self, request, context):
        """Compatibility endpoint for retired full-frame retraining requests."""
        logger.warning("Rejected full-frame training request: edge={}.", request.edge_id)
        return self.continual_backend.train_model_request(request)

    def continual_learning_request(self, request, context):
        cache_path = normalize_client_cache_path(request.cache_path)
        logger.info(
            "Received continual-learning request: edge={} send_low_conf_features={}.",
            request.edge_id,
            request.send_low_conf_features,
        )
        if cache_path and cache_path != request.cache_path:
            log_diagnostic_debug(
                self,
                "normalized continual-learning cache path",
                lambda: {
                    "original_cache_path": request.cache_path,
                    "normalized_cache_path": cache_path,
                },
            )
        else:
            log_diagnostic_debug(
                self,
                "continual-learning workspace hint",
                lambda: {"cache_path": cache_path or "<uploaded-bundle>"},
            )
        return self.continual_backend.continual_learning_request(request)

    def sync_samples(self, request, context):
        logger.info(
            "Received sample shard: edge={} model={} version={} quality={}.",
            request.edge_id,
            request.model_id,
            request.model_version,
            request.sync_type,
        )
        log_diagnostic_debug(
            self,
            "sample sync request details",
            lambda: {
                "split_config_id": request.split_config_id,
                "payload_zip_bytes": len(getattr(request, "payload_zip", b"") or b""),
            },
        )
        return self.continual_backend.sync_samples(request)

    def submit_training_job(self, request, context):
        logger.info(
            "Received training request: edge={} type={}.",
            request.edge_id,
            request.job_type,
        )
        log_diagnostic_debug(
            self,
            "submit_training_job request details",
            lambda: {
                "request_id": request.request_id,
                "cache_path": request.cache_path,
                "payload_zip_bytes": len(getattr(request, "payload_zip", b"") or b""),
            },
        )

        reply = self.continual_backend.submit_training_job(request)
        if bool(getattr(reply, "accepted", False)):
            self._record_experiment_event(
                "training_job_submitted",
                edge_id=int(request.edge_id),
                job_id=str(getattr(reply, "job_id", "") or ""),
                job_type=int(request.job_type),
                status=str(getattr(reply, "status", "") or ""),
            )
        return reply

    def get_training_job_status(self, request, context):
        reply = self.continual_backend.get_training_job_status(request)
        status = str(getattr(reply, "status", "") or "").upper()
        if bool(getattr(reply, "found", False)) and status in {"RUNNING", "SUCCEEDED"}:
            self._record_experiment_event(
                "training_job_started" if status == "RUNNING" else "training_job_succeeded",
                edge_id=int(request.edge_id),
                job_id=str(request.job_id),
                status=status,
            )
        return reply

    def download_trained_model(self, request, context):
        return self.continual_backend.download_trained_model(request)

    def cancel_training_job(self, request, context):
        """Cancel a queued training job by edge_id and job_id."""
        reply = self.continual_backend.cancel_training_job(request)
        logger.info(
            "Training job cancellation requested: edge={} cancelled={} reason={}.",
            request.edge_id,
            reply.cancelled,
            reply.message,
        )
        log_diagnostic_debug(
            self,
            "cancel_training_job details",
            lambda: {"job_id": request.job_id},
        )
        return reply

    def report_edge_model_version(self, request, context):
        del context
        reply = self.continual_backend.report_edge_model_version(request)
        if bool(getattr(reply, "success", False)):
            self._record_experiment_event(
                "model_update_applied_ack",
                edge_id=int(request.edge_id),
                model_id=str(request.model_id),
                model_version=str(request.model_version),
            )
        return reply

    # ---- Resource-aware CL trigger: cloud resource query ----

    def query_resource(self, request, context):
        """Return current cloud resource utilisation for the edge's
        Lyapunov-based CL trigger decision.
        """
        # Track edge heartbeat in registry
        if self.edge_registry is not None:
            self.edge_registry.touch(int(request.edge_id))

        cpu = _get_cpu_utilization()
        gpu = _get_gpu_utilization()
        mem = _get_memory_utilization()

        train_q, max_q = self.continual_backend.training_queue_state()
        if max_q <= 0:
            max_q = 10

        return message_transmission_pb2.ResourceReply(
            cpu_utilization=cpu,
            gpu_utilization=gpu,
            memory_utilization=mem,
            train_queue_size=train_q,
            max_queue_size=max_q,
        )

    def bandwidth_probe(self, request, context):
        """Echo the payload back for edge-side RTT / bandwidth estimation."""
        return message_transmission_pb2.BandwidthProbeReply(
            payload=request.payload,
        )

    def UploadExperimentResult(self, request, context):
        del context
        if self.experiment_result_repository is None:
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=False,
                message="experiment result repository is not configured",
            )
        try:
            stored = self.experiment_result_repository.store_artifacts(request)
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=True,
                message="experiment artifacts stored",
                stored_paths=[str(path) for path in stored],
            )
        except Exception as exc:
            self._log_failure("UploadExperimentResult", exc)
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=False,
                message=str(exc),
            )
