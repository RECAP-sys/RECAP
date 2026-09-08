from __future__ import annotations

import json
import time
from dataclasses import asdict
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from cloud.workers.gpu_lease_manager import LeaseRequest
from cloud.workers.worker_protocol import (
    WORKER_PORT_CONFLICT,
    WORKER_REQUEST_TIMEOUT,
    WORKER_RPC_UNAVAILABLE,
    JsonRpcError,
    WorkerHealth,
    decode_bytes,
    encode_bytes,
    open_direct,
    post_json,
)
from grpc_server import message_transmission_pb2


class EdgeWorkerClient:
    def __init__(self, endpoint: str, *, timeout_sec: float = 600.0) -> None:
        self.endpoint = str(endpoint)
        self.timeout_sec = float(timeout_sec)

    def get_health(self) -> WorkerHealth:
        try:
            with open_direct(
                Request(f"http://{self.endpoint}/health"),
                timeout=min(5.0, self.timeout_sec),
            ) as response:
                if response.status != 200:
                    raise JsonRpcError(
                        f"worker health returned HTTP {response.status}",
                        error_type=WORKER_RPC_UNAVAILABLE,
                        status=int(response.status),
                    )
                payload = json.loads(response.read().decode("utf-8") or "{}")
                return WorkerHealth.from_payload(dict(payload or {}))
        except HTTPError as exc:
            raise JsonRpcError(
                f"worker health returned HTTP {exc.code}",
                error_type=WORKER_RPC_UNAVAILABLE,
                status=int(exc.code),
            ) from exc
        except (JSONDecodeError, TypeError, ValueError) as exc:
            raise JsonRpcError(
                f"invalid worker health response: {exc}",
                error_type=WORKER_RPC_UNAVAILABLE,
            ) from exc
        except (TimeoutError, URLError) as exc:
            raise JsonRpcError(
                str(exc),
                error_type=_classify_transport_error(exc),
            ) from exc

    def health(
        self,
        *,
        expected_worker_id: str = "",
        expected_run_id: str = "",
        expected_lease_address: str = "",
    ) -> bool:
        try:
            health = self.get_health()
            if expected_worker_id and health.worker_id != str(expected_worker_id):
                return False
            if expected_run_id and health.run_id != str(expected_run_id):
                return False
            if expected_lease_address and health.lease_address != str(expected_lease_address):
                return False
            return bool(health.ok and health.state == "READY")
        except JsonRpcError:
            return False

    def sync_samples(self, request) -> message_transmission_pb2.SampleSyncReply:
        result = post_json(
            self.endpoint,
            "/sync_samples",
            {
                "protocol_version": request.protocol_version,
                "edge_id": int(request.edge_id),
                "model_id": request.model_id,
                "model_version": request.model_version,
                "split_config_id": request.split_config_id,
                "sync_type": request.sync_type,
                "payload_zip": encode_bytes(request.payload_zip),
            },
            timeout=self.timeout_sec,
        )
        return message_transmission_pb2.SampleSyncReply(
            success=bool(result.get("success", False)),
            message=str(result.get("message", "")),
            committed_samples=int(result.get("committed_samples", 0) or 0),
        )

    def submit_training_job(
        self,
        request,
        *,
        exclusive_gpu_lease: bool = False,
    ) -> message_transmission_pb2.SubmitTrainingJobReply:
        payload_bundle_path = str(getattr(request, "payload_bundle_path", "") or "")
        payload_zip = b"" if payload_bundle_path else bytes(request.payload_zip or b"")
        result = post_json(
            self.endpoint,
            "/submit_training_job",
            {
                "protocol_version": request.protocol_version,
                "edge_id": int(request.edge_id),
                "request_id": request.request_id,
                "job_type": int(request.job_type),
                "cache_path": request.cache_path,
                "send_low_conf_features": bool(request.send_low_conf_features),
                "frame_indices": [int(value) for value in request.frame_indices],
                "payload_zip": encode_bytes(payload_zip),
                "payload_bundle_path": payload_bundle_path,
                "base_model_version": request.base_model_version,
                "exclusive_gpu_lease": bool(exclusive_gpu_lease),
            },
            timeout=self.timeout_sec,
        )
        return message_transmission_pb2.SubmitTrainingJobReply(
            accepted=bool(result.get("accepted", False)),
            job_id=str(result.get("job_id", "")),
            status=str(result.get("status", "")),
            queue_position=int(result.get("queue_position", -1) or -1),
            message=str(result.get("message", "")),
        )

    def get_training_job_status(self, request) -> message_transmission_pb2.TrainingJobStatusReply:
        result = post_json(
            self.endpoint,
            "/get_training_job_status",
            {"edge_id": int(request.edge_id), "job_id": request.job_id},
            timeout=self.timeout_sec,
        )
        return message_transmission_pb2.TrainingJobStatusReply(
            found=bool(result.get("found", False)),
            job_id=str(result.get("job_id", request.job_id or "")),
            edge_id=int(result.get("edge_id", request.edge_id) or 0),
            status=str(result.get("status", "")),
            queue_position=int(result.get("queue_position", -1) or -1),
            message=str(result.get("message", "")),
            request_id=str(result.get("request_id", "")),
            job_type=int(result.get("job_type", 0) or 0),
            result_available=bool(result.get("result_available", False)),
            submitted_at_ms=int(result.get("submitted_at_ms", 0) or 0),
            started_at_ms=int(result.get("started_at_ms", 0) or 0),
            finished_at_ms=int(result.get("finished_at_ms", 0) or 0),
            protocol_version=str(result.get("protocol_version", "")),
            base_model_version=str(result.get("base_model_version", "")),
            result_model_version=str(result.get("result_model_version", "")),
            worker_id=str(result.get("worker_id", "")),
        )

    def download_trained_model(self, request) -> message_transmission_pb2.DownloadTrainedModelReply:
        result = post_json(
            self.endpoint,
            "/download_trained_model",
            {"edge_id": int(request.edge_id), "job_id": request.job_id},
            timeout=self.timeout_sec,
        )
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=bool(result.get("success", False)),
            job_id=str(result.get("job_id", request.job_id or "")),
            status=str(result.get("status", "")),
            model_data=str(result.get("model_data", "")),
            message=str(result.get("message", "")),
            protocol_version=str(result.get("protocol_version", "")),
            result_model_version=str(result.get("result_model_version", "")),
        )

    def cancel_training_job(self, request) -> message_transmission_pb2.CancelTrainingJobReply:
        result = post_json(
            self.endpoint,
            "/cancel_training_job",
            {"edge_id": int(request.edge_id), "job_id": request.job_id},
            timeout=self.timeout_sec,
        )
        return message_transmission_pb2.CancelTrainingJobReply(
            cancelled=bool(result.get("cancelled", False)),
            message=str(result.get("message", "")),
        )

    def report_edge_model_version(
        self,
        request,
    ) -> message_transmission_pb2.ReportEdgeModelVersionReply:
        result = post_json(
            self.endpoint,
            "/report_edge_model_version",
            {
                "edge_id": int(request.edge_id),
                "model_id": request.model_id,
                "model_version": request.model_version,
            },
            timeout=self.timeout_sec,
        )
        return message_transmission_pb2.ReportEdgeModelVersionReply(
            success=bool(result.get("success", False)),
            message=str(result.get("message", "")),
        )

    def shutdown(self) -> dict[str, Any]:
        return post_json(
            self.endpoint,
            "/shutdown",
            {},
            timeout=min(5.0, self.timeout_sec),
        )


class GpuLeaseHttpClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_sec: float = 600.0,
        heartbeat_interval_sec: float = 10.0,
    ) -> None:
        self.endpoint = str(endpoint)
        self.timeout_sec = float(timeout_sec)
        self.heartbeat_interval_sec = float(heartbeat_interval_sec)

    def acquire(self, request: LeaseRequest, *, wait_timeout_sec: float | None = None):
        payload = asdict(request)
        if wait_timeout_sec is not None:
            payload["timeout_sec"] = float(wait_timeout_sec)
        result = post_json(
            self.endpoint,
            "/gpu_lease/acquire",
            payload,
            timeout=self.timeout_sec,
        )
        if not bool(result.get("success", False)):
            raise RuntimeError(str(result.get("message", "GPU lease denied")))
        return GpuLeaseHandle(
            client=self,
            lease_id=str(result["lease_id"]),
            observed_peak_memory_gb=0.0,
        )

    def release(self, lease_id: str, *, observed_peak_memory_gb: float = 0.0) -> None:
        post_json(
            self.endpoint,
            "/gpu_lease/release",
            {
                "lease_id": str(lease_id),
                "observed_peak_memory_gb": float(observed_peak_memory_gb or 0.0),
            },
            timeout=min(30.0, self.timeout_sec),
        )

    def heartbeat(self, lease_id: str) -> bool:
        result = post_json(
            self.endpoint,
            "/gpu_lease/heartbeat",
            {"lease_id": str(lease_id)},
            timeout=min(30.0, self.timeout_sec),
        )
        return bool(result.get("success", False))

    def mark_oom(self, *, job_id: str, message: str) -> dict[str, Any]:
        return post_json(
            self.endpoint,
            "/gpu_lease/oom",
            {"job_id": str(job_id), "message": str(message)},
            timeout=min(30.0, self.timeout_sec),
        )


class GpuLeaseHandle:
    def __init__(
        self,
        *,
        client: GpuLeaseHttpClient,
        lease_id: str,
        observed_peak_memory_gb: float,
    ) -> None:
        self.client = client
        self.lease_id = str(lease_id)
        self.observed_peak_memory_gb = float(observed_peak_memory_gb or 0.0)
        self._closed = False
        self._stop = False

    def __enter__(self):
        import threading

        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"gpu-lease-heartbeat-{self.lease_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if self._closed:
            return
        self._stop = True
        self._closed = True
        self.client.release(
            self.lease_id,
            observed_peak_memory_gb=self.observed_peak_memory_gb,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop:
            time.sleep(max(1.0, self.client.heartbeat_interval_sec))
            if self._stop:
                return
            if not self.client.heartbeat(self.lease_id):
                return


def decode_payload_zip(payload: dict[str, Any]) -> bytes:
    return decode_bytes(payload.get("payload_zip"))


def _classify_transport_error(exc: BaseException) -> str:
    text = str(exc).lower()
    reason = getattr(exc, "reason", None)
    errno_value = getattr(reason, "errno", None)
    if errno_value == 98 or "address already in use" in text:
        return WORKER_PORT_CONFLICT
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        return WORKER_REQUEST_TIMEOUT
    return WORKER_RPC_UNAVAILABLE
