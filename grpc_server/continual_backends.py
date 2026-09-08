from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from uuid import uuid4

from loguru import logger

from common.logging_sanitizer import safe_error_summary
from grpc_server import message_transmission_pb2
from grpc_server.training_jobs import JOB_STATUS_SUCCEEDED


class ContinualLearningBackend(Protocol):
    def train_model_request(self, request) -> message_transmission_pb2.TrainReply: ...

    def continual_learning_request(
        self, request
    ) -> message_transmission_pb2.ContinualLearningReply: ...

    def sync_samples(self, request) -> message_transmission_pb2.SampleSyncReply: ...

    def submit_training_job(self, request) -> message_transmission_pb2.SubmitTrainingJobReply: ...

    def get_training_job_status(
        self, request
    ) -> message_transmission_pb2.TrainingJobStatusReply: ...

    def download_trained_model(
        self, request
    ) -> message_transmission_pb2.DownloadTrainedModelReply: ...

    def cancel_training_job(self, request) -> message_transmission_pb2.CancelTrainingJobReply: ...

    def report_edge_model_version(
        self, request
    ) -> message_transmission_pb2.ReportEdgeModelVersionReply: ...

    def training_queue_state(self) -> tuple[int, int]: ...


class DisabledContinualLearningBackend:
    def train_model_request(self, request) -> message_transmission_pb2.TrainReply:
        del request
        return message_transmission_pb2.TrainReply(
            success=False,
            model_data="",
            message="continual learning backend is not configured",
        )

    def continual_learning_request(
        self, request
    ) -> message_transmission_pb2.ContinualLearningReply:
        return message_transmission_pb2.ContinualLearningReply(
            success=False,
            model_data="",
            message="continual learning backend is not configured",
            protocol_version=str(getattr(request, "protocol_version", "") or ""),
        )

    def sync_samples(self, request) -> message_transmission_pb2.SampleSyncReply:
        del request
        return message_transmission_pb2.SampleSyncReply(
            success=False,
            message="continual learning backend is not configured",
        )

    def submit_training_job(self, request) -> message_transmission_pb2.SubmitTrainingJobReply:
        del request
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=False,
            job_id="",
            status="",
            queue_position=-1,
            message="continual learning backend is not configured",
        )

    def get_training_job_status(
        self, request
    ) -> message_transmission_pb2.TrainingJobStatusReply:
        return message_transmission_pb2.TrainingJobStatusReply(
            found=False,
            job_id=str(getattr(request, "job_id", "") or ""),
            edge_id=int(getattr(request, "edge_id", 0) or 0),
            status="",
            queue_position=-1,
            message="continual learning backend is not configured",
        )

    def download_trained_model(
        self, request
    ) -> message_transmission_pb2.DownloadTrainedModelReply:
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=False,
            job_id=str(getattr(request, "job_id", "") or ""),
            status="",
            model_data="",
            message="continual learning backend is not configured",
            protocol_version="",
        )

    def cancel_training_job(self, request) -> message_transmission_pb2.CancelTrainingJobReply:
        del request
        return message_transmission_pb2.CancelTrainingJobReply(
            cancelled=False,
            message="continual learning backend is not configured",
        )

    def report_edge_model_version(
        self, request
    ) -> message_transmission_pb2.ReportEdgeModelVersionReply:
        del request
        return message_transmission_pb2.ReportEdgeModelVersionReply(
            success=False,
            message="continual learning backend is not configured",
        )

    def training_queue_state(self) -> tuple[int, int]:
        return 0, 0


class LocalContinualLearningBackend:
    """Backend used inside edge worker processes."""

    def __init__(
        self,
        *,
        continual_learner=None,
        workspace_root: str = "./cache/server_workspace",
        training_job_manager=None,
        edge_registry=None,
        log_internal_ids: bool = False,
    ) -> None:
        self.continual_learner = continual_learner
        self.workspace_root = workspace_root or "./cache/server_workspace"
        self.training_job_manager = training_job_manager
        self.edge_registry = edge_registry
        self.log_internal_ids = bool(log_internal_ids)

    def train_model_request(self, request) -> message_transmission_pb2.TrainReply:
        del request
        return message_transmission_pb2.TrainReply(
            success=False,
            model_data="",
            message="full-frame retrain is unavailable; use fixed-split continual learning",
        )

    def continual_learning_request(
        self, request
    ) -> message_transmission_pb2.ContinualLearningReply:
        return message_transmission_pb2.ContinualLearningReply(
            success=False,
            model_data="",
            message="synchronous continual learning is unavailable; use submit_training_job",
            protocol_version=request.protocol_version,
        )

    def sync_samples(self, request) -> message_transmission_pb2.SampleSyncReply:
        if self.edge_registry is not None:
            self.edge_registry.touch(int(request.edge_id))
        if self.continual_learner is None:
            return message_transmission_pb2.SampleSyncReply(
                success=False,
                message="continual_learner not configured",
            )
        sync_method = getattr(self.continual_learner, "sync_samples", None)
        if sync_method is None:
            return message_transmission_pb2.SampleSyncReply(
                success=False,
                message="continual_learner has no sample sync method",
            )
        try:
            result = sync_method(
                edge_id=int(request.edge_id),
                protocol_version=str(request.protocol_version or ""),
                sync_type=str(request.sync_type or ""),
                payload_zip=bytes(request.payload_zip or b""),
                model_id=str(request.model_id or ""),
                model_version=str(request.model_version or ""),
                split_config_id=str(request.split_config_id or ""),
            )
            return _sample_sync_reply(result)
        except Exception as exc:
            logger.error("sync_samples failed: {}", safe_error_summary(exc))
            return message_transmission_pb2.SampleSyncReply(success=False, message=str(exc))

    def submit_training_job(self, request) -> message_transmission_pb2.SubmitTrainingJobReply:
        if self.continual_learner is None or self.training_job_manager is None:
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=False,
                job_id="",
                status="",
                queue_position=-1,
                message="async training is not configured",
            )
        try:
            if int(request.job_type) == message_transmission_pb2.TRAINING_JOB_TYPE_FULL_FRAME:
                return message_transmission_pb2.SubmitTrainingJobReply(
                    accepted=False,
                    job_id="",
                    status="",
                    queue_position=-1,
                    message="full-frame retrain is unavailable; use fixed-split continual learning",
                )
            request_kind = _request_kind_for_job_type(int(request.job_type))
            return self._submit_training_job_from_workspace(
                request,
                workspace=request.cache_path,
                request_kind=request_kind,
                payload_zip=getattr(request, "payload_zip", b""),
            )
        except Exception as exc:
            logger.error("submit_training_job failed: {}", safe_error_summary(exc))
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=False,
                job_id="",
                status="",
                queue_position=-1,
                message=str(exc),
            )

    def get_training_job_status(
        self, request
    ) -> message_transmission_pb2.TrainingJobStatusReply:
        if self.training_job_manager is None:
            return message_transmission_pb2.TrainingJobStatusReply(
                found=False,
                job_id=str(request.job_id or ""),
                edge_id=int(request.edge_id),
                status="",
                queue_position=-1,
                message="async training is not configured",
            )
        job = self.training_job_manager.get_job(
            edge_id=int(request.edge_id),
            job_id=str(request.job_id or ""),
        )
        if job is None:
            return message_transmission_pb2.TrainingJobStatusReply(
                found=False,
                job_id=str(request.job_id or ""),
                edge_id=int(request.edge_id),
                status="",
                queue_position=-1,
                message="Training job not found.",
            )
        queue_position = self.training_job_manager.queue_position(job.job_id)
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=job.job_id,
            edge_id=job.edge_id,
            status=job.status,
            queue_position=queue_position,
            message=job.message,
            request_id=job.request_id,
            job_type=job.job_type,
            result_available=(job.status == JOB_STATUS_SUCCEEDED and bool(job.model_data)),
            submitted_at_ms=job.submitted_at_ms,
            started_at_ms=job.started_at_ms,
            finished_at_ms=job.finished_at_ms,
            protocol_version=job.protocol_version,
            base_model_version=job.base_model_version,
            result_model_version=job.result_model_version,
            worker_id=str(getattr(job, "worker_id", "") or ""),
        )

    def download_trained_model(
        self, request
    ) -> message_transmission_pb2.DownloadTrainedModelReply:
        if self.training_job_manager is None:
            return message_transmission_pb2.DownloadTrainedModelReply(
                success=False,
                job_id=str(request.job_id or ""),
                status="",
                model_data="",
                message="async training is not configured",
                protocol_version="",
            )
        success, job, message = self.training_job_manager.download_result(
            edge_id=int(request.edge_id),
            job_id=str(request.job_id or ""),
        )
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=success,
            job_id=job.job_id if job is not None else str(request.job_id or ""),
            status=job.status if job is not None else "",
            model_data=job.model_data if success and job is not None else "",
            message=message,
            protocol_version=job.protocol_version if job is not None else "",
            result_model_version=job.result_model_version if job is not None else "",
        )

    def cancel_training_job(self, request) -> message_transmission_pb2.CancelTrainingJobReply:
        if self.training_job_manager is None:
            return message_transmission_pb2.CancelTrainingJobReply(
                cancelled=False,
                message="async training is not configured",
            )
        cancelled, message = self.training_job_manager.cancel_job(
            edge_id=int(request.edge_id),
            job_id=str(request.job_id or ""),
        )
        return message_transmission_pb2.CancelTrainingJobReply(
            cancelled=cancelled,
            message=message,
        )

    def report_edge_model_version(
        self, request
    ) -> message_transmission_pb2.ReportEdgeModelVersionReply:
        if self.edge_registry is not None:
            self.edge_registry.touch(
                int(request.edge_id),
                model_id=str(request.model_id or ""),
                model_version=str(request.model_version or ""),
            )
        if self.training_job_manager is not None:
            self.training_job_manager.update_edge_model_version(
                int(request.edge_id),
                str(request.model_version or ""),
            )
        return message_transmission_pb2.ReportEdgeModelVersionReply(
            success=True,
            message="edge model version recorded",
        )

    def training_queue_state(self) -> tuple[int, int]:
        if self.training_job_manager is not None:
            return self.training_job_manager.training_queue_state()
        if self.continual_learner is not None and hasattr(
            self.continual_learner, "training_queue_state"
        ):
            return self.continual_learner.training_queue_state()
        return 0, 0

    def _submit_training_job_from_workspace(
        self,
        request,
        *,
        workspace,
        request_kind: str,
        payload_zip: bytes = b"",
    ) -> message_transmission_pb2.SubmitTrainingJobReply:
        base_model_version = str(getattr(request, "base_model_version", "") or "0")
        try:
            manifest = _read_trigger_manifest(workspace=workspace, payload_zip=payload_zip)
            if manifest is not None:
                model_info = manifest.get("model", {}) or {
                    "model_id": manifest.get("model_id", ""),
                    "model_version": manifest.get("model_version", ""),
                }
                base_model_version = str(model_info.get("model_version", base_model_version))
                if self.edge_registry is not None:
                    self.edge_registry.touch(
                        int(request.edge_id),
                        model_id=str(model_info.get("model_id", "")),
                        model_version=base_model_version,
                    )
        except Exception:
            pass
        job, created = self.training_job_manager.submit(
            edge_id=int(request.edge_id),
            request_id=str(request.request_id or ""),
            job_type=int(request.job_type),
            workspace=str(workspace),
            protocol_version=str(request.protocol_version or ""),
            workspace_root=str(self.workspace_root),
            request_kind=str(request_kind),
            payload_zip=payload_zip,
            send_low_conf_features=bool(request.send_low_conf_features),
            frame_indices=[int(index) for index in request.frame_indices],
            exclusive_gpu_lease=bool(getattr(request, "exclusive_gpu_lease", False)),
            base_model_version=base_model_version,
        )
        if self.edge_registry is not None and created:
            self.edge_registry.record_job_submitted(int(request.edge_id), job.job_id)
        queue_position = self.training_job_manager.queue_position(job.job_id)
        message = "Training job accepted." if created else "Training job already exists."
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=True,
            job_id=job.job_id,
            status=job.status,
            queue_position=queue_position,
            message=message,
        )


class EdgeWorkerRoutedContinualLearningBackend:
    def __init__(self, *, worker_pool, edge_registry=None, gpu_lease_manager=None) -> None:
        self.worker_pool = worker_pool
        self.edge_registry = edge_registry
        self.gpu_lease_manager = gpu_lease_manager
        self._submitted_jobs: dict[tuple[int, str], _SubmitRequestSnapshot] = {}

    def close(self) -> None:
        self._submitted_jobs.clear()

    def _client(self, edge_id: int):
        if self.edge_registry is not None:
            self.edge_registry.touch(int(edge_id))
        return self.worker_pool.client_for_edge(int(edge_id))

    def train_model_request(self, request) -> message_transmission_pb2.TrainReply:
        return message_transmission_pb2.TrainReply(
            success=False,
            model_data="",
            message="full-frame retrain is unavailable; use fixed-split continual learning",
        )

    def continual_learning_request(
        self, request
    ) -> message_transmission_pb2.ContinualLearningReply:
        submit_request = message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version=request.protocol_version,
            edge_id=request.edge_id,
            request_id="",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            cache_path=request.cache_path,
            send_low_conf_features=request.send_low_conf_features,
            payload_zip=request.payload_zip,
        )
        reply = self.submit_training_job(submit_request)
        return message_transmission_pb2.ContinualLearningReply(
            success=bool(reply.accepted),
            model_data="",
            message=reply.message,
            protocol_version=request.protocol_version,
        )

    def sync_samples(self, request) -> message_transmission_pb2.SampleSyncReply:
        return self._client(request.edge_id).sync_samples(request)

    def submit_training_job(self, request) -> message_transmission_pb2.SubmitTrainingJobReply:
        routed_request = self._materialize_payload_bundle(request)
        routed_bundle_path_value = str(
            getattr(routed_request, "payload_bundle_path", "") or ""
        ).strip()
        routed_bundle_path = Path(routed_bundle_path_value) if routed_bundle_path_value else None
        exclusive_gpu_lease = bool(
            getattr(routed_request, "exclusive_gpu_lease", False)
            or getattr(request, "exclusive_gpu_lease", False)
        )
        reply = self._client(routed_request.edge_id).submit_training_job(
            routed_request,
            exclusive_gpu_lease=exclusive_gpu_lease,
        )
        if reply.accepted and reply.job_id:
            self._submitted_jobs[(int(routed_request.edge_id), str(reply.job_id))] = (
                _SubmitRequestSnapshot.from_request(routed_request)
            )
            if routed_bundle_path is not None:
                routed_bundle_path.unlink(missing_ok=True)
        return reply

    def get_training_job_status(
        self, request
    ) -> message_transmission_pb2.TrainingJobStatusReply:
        edge_id = int(request.edge_id)
        job_id = str(request.job_id or "")
        expired_reply = self._expired_lease_status_reply(
            edge_id=edge_id,
            original_job_id=job_id,
            lease_job_id=job_id,
        )
        if expired_reply is not None:
            return expired_reply

        reply = self._client(edge_id).get_training_job_status(request)
        return reply

    def download_trained_model(
        self, request
    ) -> message_transmission_pb2.DownloadTrainedModelReply:
        edge_id = int(request.edge_id)
        return self._client(edge_id).download_trained_model(request)

    def cancel_training_job(self, request) -> message_transmission_pb2.CancelTrainingJobReply:
        edge_id = int(request.edge_id)
        return self._client(edge_id).cancel_training_job(request)

    def report_edge_model_version(
        self, request
    ) -> message_transmission_pb2.ReportEdgeModelVersionReply:
        if self.edge_registry is not None:
            self.edge_registry.touch(
                int(request.edge_id),
                model_id=str(request.model_id or ""),
                model_version=str(request.model_version or ""),
            )
        return self._client(request.edge_id).report_edge_model_version(request)

    def training_queue_state(self) -> tuple[int, int]:
        return 0, 0

    def _expired_lease_status_reply(
        self,
        *,
        edge_id: int,
        original_job_id: str,
        lease_job_id: str,
    ) -> message_transmission_pb2.TrainingJobStatusReply | None:
        reason_method = getattr(self.gpu_lease_manager, "expired_job_reason", None)
        if not callable(reason_method):
            return None
        reason = str(reason_method(str(lease_job_id)) or "")
        if not reason:
            return None
        snapshot = self._submitted_jobs.get((edge_id, original_job_id)) or self._submitted_jobs.get(
            (edge_id, lease_job_id)
        )
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=original_job_id,
            edge_id=edge_id,
            status="FAILED",
            queue_position=-1,
            message=f"retryable failure: {reason}",
            request_id=snapshot.request_id if snapshot is not None else "",
            job_type=(
                snapshot.job_type
                if snapshot is not None
                else message_transmission_pb2.TRAINING_JOB_TYPE_UNSPECIFIED
            ),
            protocol_version=snapshot.protocol_version if snapshot is not None else "",
            base_model_version=snapshot.base_model_version if snapshot is not None else "",
        )

    def _materialize_payload_bundle(self, request):
        payload_zip = bytes(getattr(request, "payload_zip", b"") or b"")
        if not payload_zip:
            return request
        ensure_worker = getattr(self.worker_pool, "ensure_worker", None)
        if not callable(ensure_worker):
            return request
        edge_id = int(request.edge_id)
        assignment = ensure_worker(edge_id)
        bundle_dir = Path(assignment.workspace_root) / "incoming_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / f"{uuid4().hex}.zip"
        bundle_path.write_bytes(payload_zip)
        return SimpleNamespace(
            protocol_version=str(request.protocol_version or ""),
            edge_id=edge_id,
            request_id=str(request.request_id or ""),
            job_type=int(request.job_type),
            cache_path=str(request.cache_path or ""),
            send_low_conf_features=bool(request.send_low_conf_features),
            frame_indices=[int(value) for value in request.frame_indices],
            payload_zip=b"",
            payload_bundle_path=str(bundle_path),
            base_model_version=str(request.base_model_version or ""),
            exclusive_gpu_lease=bool(getattr(request, "exclusive_gpu_lease", False)),
        )

@dataclass(frozen=True)
class _SubmitRequestSnapshot:
    protocol_version: str
    request_id: str
    job_type: int
    base_model_version: str

    @classmethod
    def from_request(cls, request) -> "_SubmitRequestSnapshot":
        return cls(
            protocol_version=str(getattr(request, "protocol_version", "") or ""),
            request_id=str(request.request_id or ""),
            job_type=int(request.job_type),
            base_model_version=str(request.base_model_version or ""),
        )


def _sample_sync_reply(result) -> message_transmission_pb2.SampleSyncReply:
    if isinstance(result, message_transmission_pb2.SampleSyncReply):
        return result
    if isinstance(result, tuple):
        values = list(result)
        return message_transmission_pb2.SampleSyncReply(
            success=bool(values[0]) if values else True,
            message=str(values[1]) if len(values) > 1 else "sample sync completed",
            committed_samples=int(values[2] or 0) if len(values) > 2 else 0,
        )
    if isinstance(result, dict):
        return message_transmission_pb2.SampleSyncReply(
            success=bool(result.get("success", True)),
            message=str(result.get("message", "sample sync completed")),
            committed_samples=int(result.get("committed_samples", 0) or 0),
        )
    return message_transmission_pb2.SampleSyncReply(
        success=bool(result),
        message="sample sync completed",
    )


def _read_trigger_manifest(*, workspace, payload_zip: bytes) -> dict | None:
    if payload_zip:
        with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as archive:
            if "trigger_manifest.json" not in archive.namelist():
                return None
            with archive.open("trigger_manifest.json", "r") as handle:
                return json.loads(handle.read().decode("utf-8"))
    trigger_manifest_path = Path(workspace) / "trigger_manifest.json"
    if trigger_manifest_path.exists():
        return json.loads(trigger_manifest_path.read_text(encoding="utf-8"))
    return None


def _request_kind_for_job_type(job_type: int) -> str:
    if job_type == message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING:
        return "continual_learning"
    raise ValueError(f"Unsupported training job type: {job_type!r}")
