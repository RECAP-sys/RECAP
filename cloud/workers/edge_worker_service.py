from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cloud.annotation.remote_service import RemoteTeacherAnnotationService
from cloud.orchestrator import CloudFixedSplitOrchestrator
from cloud.workers.worker_client import GpuLeaseHttpClient, decode_payload_zip
from config import load_runtime_config
from grpc_server import message_transmission_pb2
from grpc_server.continual_backends import LocalContinualLearningBackend
from grpc_server.training_jobs import TrainingJobManager


class LazyObjectDetection:
    def __init__(self, config: object, detector_type: str) -> None:
        self.config = config
        self.detector_type = detector_type
        self._detector = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    from model_management.object_detection import Object_Detection

                    self._detector = Object_Detection(
                        self.config,
                        type=self.detector_type,
                    )
        return self._detector

    def __getattr__(self, name: str):
        return getattr(self._ensure(), name)

    def large_inference(self, *args, **kwargs):
        return self._ensure().large_inference(*args, **kwargs)

    def large_inference_batch(self, *args, **kwargs):
        return self._ensure().large_inference_batch(*args, **kwargs)


class EdgeWorkerService:
    def __init__(
        self,
        *,
        edge_id: int,
        worker_id: str,
        yaml_path: str,
        workspace_root: str,
        lease_address: str,
        teacher_annotation_address: str = "",
    ) -> None:
        self.edge_id = int(edge_id)
        self.worker_id = str(worker_id)
        runtime_config = load_runtime_config(yaml_path)
        self.config = runtime_config.server
        self._override_worker_paths(workspace_root)
        lease_cfg = self.config.edge_affine_workers.gpu_lease
        self.lease_client = GpuLeaseHttpClient(
            lease_address,
            timeout_sec=float(self.config.edge_affine_workers.worker.request_timeout_sec),
            heartbeat_interval_sec=float(lease_cfg.heartbeat_interval_sec),
        )
        self.teacher_annotation_service = None
        teacher_annotation_metadata = None
        if str(teacher_annotation_address or "").strip():
            self.teacher_annotation_service = RemoteTeacherAnnotationService(
                str(teacher_annotation_address),
                timeout_sec=float(self.config.edge_affine_workers.worker.request_timeout_sec),
            )
            teacher_annotation_metadata = self.teacher_annotation_service.metadata()
            large_od = None
        else:
            large_od = LazyObjectDetection(self.config, "large inference")
        self.learner = CloudFixedSplitOrchestrator(
            self.config,
            large_od,
            gpu_lease_client=self.lease_client,
            worker_id=self.worker_id,
            teacher_annotation_service=self.teacher_annotation_service,
            teacher_annotation_metadata=teacher_annotation_metadata,
        )
        self.training_jobs = TrainingJobManager(
            continual_learner=self.learner,
            max_concurrent_jobs=1,
            edge_registry=None,
            log_internal_ids=bool(getattr(self.learner, "log_internal_ids", False)),
        )
        self.backend = LocalContinualLearningBackend(
            continual_learner=self.learner,
            workspace_root=str(workspace_root),
            training_job_manager=self.training_jobs,
            edge_registry=None,
            log_internal_ids=bool(getattr(self.learner, "log_internal_ids", False)),
        )

    def close(self) -> None:
        self.training_jobs.close()
        self.learner.close()

    def routes(self) -> dict[str, Any]:
        return {
            "/sync_samples": self.sync_samples,
            "/submit_training_job": self.submit_training_job,
            "/get_training_job_status": self.get_training_job_status,
            "/download_trained_model": self.download_trained_model,
            "/cancel_training_job": self.cancel_training_job,
            "/report_edge_model_version": self.report_edge_model_version,
        }

    def sync_samples(self, payload: dict[str, Any]) -> dict[str, Any]:
        reply = self.backend.sync_samples(
            message_transmission_pb2.SampleSyncRequest(
                protocol_version=str(payload.get("protocol_version", "")),
                edge_id=int(payload.get("edge_id", self.edge_id) or self.edge_id),
                model_id=str(payload.get("model_id", "")),
                model_version=str(payload.get("model_version", "")),
                split_config_id=str(payload.get("split_config_id", "")),
                sync_type=str(payload.get("sync_type", "")),
                payload_zip=decode_payload_zip(payload),
            )
        )
        return {
            "success": bool(reply.success),
            "message": reply.message,
            "committed_samples": int(reply.committed_samples),
        }

    def submit_training_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload_zip = decode_payload_zip(payload)
        payload_bundle_path = str(payload.get("payload_bundle_path", "") or "")
        if payload_bundle_path and not payload_zip:
            try:
                payload_zip = Path(payload_bundle_path).read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Unable to read routed payload bundle: {payload_bundle_path}"
                ) from exc
        reply = self.backend.submit_training_job(
            SimpleNamespace(
                protocol_version=str(payload.get("protocol_version", "")),
                edge_id=int(payload.get("edge_id", self.edge_id) or self.edge_id),
                request_id=str(payload.get("request_id", "")),
                job_type=int(payload.get("job_type", 0) or 0),
                cache_path=str(payload.get("cache_path", "")),
                send_low_conf_features=bool(payload.get("send_low_conf_features", False)),
                frame_indices=[
                    int(value) for value in list(payload.get("frame_indices", []) or [])
                ],
                payload_zip=payload_zip,
                base_model_version=str(payload.get("base_model_version", "")),
                exclusive_gpu_lease=bool(payload.get("exclusive_gpu_lease", False)),
            )
        )
        return {
            "accepted": bool(reply.accepted),
            "job_id": reply.job_id,
            "status": reply.status,
            "queue_position": int(reply.queue_position),
            "message": reply.message,
        }

    def get_training_job_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        reply = self.backend.get_training_job_status(
            message_transmission_pb2.TrainingJobStatusRequest(
                edge_id=int(payload.get("edge_id", self.edge_id) or self.edge_id),
                job_id=str(payload.get("job_id", "")),
            )
        )
        return {
            "found": bool(reply.found),
            "job_id": reply.job_id,
            "edge_id": int(reply.edge_id),
            "status": reply.status,
            "queue_position": int(reply.queue_position),
            "message": reply.message,
            "request_id": reply.request_id,
            "job_type": int(reply.job_type),
            "result_available": bool(reply.result_available),
            "submitted_at_ms": int(reply.submitted_at_ms),
            "started_at_ms": int(reply.started_at_ms),
            "finished_at_ms": int(reply.finished_at_ms),
            "protocol_version": reply.protocol_version,
            "base_model_version": reply.base_model_version,
            "result_model_version": reply.result_model_version,
            "worker_id": reply.worker_id,
        }

    def download_trained_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        reply = self.backend.download_trained_model(
            message_transmission_pb2.DownloadTrainedModelRequest(
                edge_id=int(payload.get("edge_id", self.edge_id) or self.edge_id),
                job_id=str(payload.get("job_id", "")),
            )
        )
        return {
            "success": bool(reply.success),
            "job_id": reply.job_id,
            "status": reply.status,
            "model_data": reply.model_data,
            "message": reply.message,
            "protocol_version": reply.protocol_version,
            "result_model_version": reply.result_model_version,
        }

    def cancel_training_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        reply = self.backend.cancel_training_job(
            message_transmission_pb2.CancelTrainingJobRequest(
                edge_id=int(payload.get("edge_id", self.edge_id) or self.edge_id),
                job_id=str(payload.get("job_id", "")),
            )
        )
        return {"cancelled": bool(reply.cancelled), "message": reply.message}

    def report_edge_model_version(self, payload: dict[str, Any]) -> dict[str, Any]:
        reply = self.backend.report_edge_model_version(
            message_transmission_pb2.ReportEdgeModelVersionRequest(
                edge_id=int(payload.get("edge_id", self.edge_id) or self.edge_id),
                model_id=str(payload.get("model_id", "")),
                model_version=str(payload.get("model_version", "")),
            )
        )
        return {"success": bool(reply.success), "message": reply.message}

    def _override_worker_paths(self, workspace_root: str) -> None:
        root = os.path.abspath(str(workspace_root))
        self.config.workspace_root = root
        self.config.continual_learning.max_concurrent_jobs = 1
        self.config.continual_learning.recent_training_window_root = os.path.join(
            root,
            "recent_training_windows",
        )
        self.config.continual_learning.split_contract_root = os.path.join(
            root,
            "split_contracts",
        )
        feature_cache = self.config.continual_learning.feature_cache
        feature_cache.view_root_dir = os.path.join(root, "cloud_training_views")
        feature_cache.store_root_dir = os.path.join(root, "cloud_feature_shards")
        feature_cache.shard_root_dir = os.path.join(root, "cloud_feature_shards")
        self.config.continual_learning.teacher_annotation.cache_root_dir = os.path.join(
            root,
            "teacher_label_cache",
        )
