from __future__ import annotations

from typing import Any

from cloud.workers.gpu_lease_manager import GpuLeaseManager, LeaseRequest
from cloud.workers.worker_protocol import JsonRpcError, JsonRpcServer


class GpuLeaseService:
    def __init__(self, *, listen_address: str, manager: GpuLeaseManager) -> None:
        self.listen_address = str(listen_address)
        self.manager = manager
        self.server = JsonRpcServer(
            listen_address=self.listen_address,
            routes={
                "/gpu_lease/acquire": self._acquire,
                "/gpu_lease/release": self._release,
                "/gpu_lease/heartbeat": self._heartbeat,
                "/gpu_lease/oom": self._oom,
            },
        )

    def start(self) -> None:
        self.server.start()
        self.listen_address = self.server.listen_address

    def shutdown(self) -> None:
        self.server.shutdown()

    def _acquire(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout_sec = payload.get("timeout_sec", None)
        try:
            lease = self.manager.acquire(
                LeaseRequest(
                    edge_id=int(payload.get("edge_id", 0) or 0),
                    worker_id=str(payload.get("worker_id", "")),
                    job_id=str(payload.get("job_id", "")),
                    model_name=str(payload.get("model_name", "")),
                    split_key=str(payload.get("split_key", "")),
                    batch_size=int(payload.get("batch_size", 0) or 0),
                    train_samples=int(payload.get("train_samples", 0) or 0),
                    estimated_peak_memory_gb=float(
                        payload.get("estimated_peak_memory_gb", 0.0) or 0.0
                    ),
                    exclusive=bool(payload.get("exclusive", False)),
                ),
                timeout_sec=None if timeout_sec is None else float(timeout_sec),
            )
        except TimeoutError as exc:
            raise JsonRpcError(
                str(exc),
                error_type="GPU_LEASE_BUSY",
                status=409,
            ) from exc
        return {
            "success": True,
            "lease_id": lease.lease_id,
            "estimated_peak_memory_gb": lease.estimated_peak_memory_gb,
        }

    def _release(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.manager.release(
            str(payload.get("lease_id", "")),
            observed_peak_memory_gb=float(payload.get("observed_peak_memory_gb", 0.0) or 0.0),
        )
        return {"success": True}

    def _heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"success": self.manager.heartbeat(str(payload.get("lease_id", "")))}

    def _oom(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.manager.mark_oom(
            job_id=str(payload.get("job_id", "")),
            message=str(payload.get("message", "")),
        )
        return {"success": True, **result}
