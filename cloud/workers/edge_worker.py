from __future__ import annotations

import argparse
import os
import threading
from typing import Any, Callable

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

from loguru import logger

from cloud.workers.edge_worker_service import EdgeWorkerService
from cloud.workers.worker_protocol import (
    WORKER_STARTUP_FAILED,
    JsonRpcServer,
    WorkerHealth,
    WorkerState,
)


class EdgeWorkerServiceManager:
    def __init__(
        self,
        *,
        edge_id: int,
        worker_id: str,
        run_id: str,
        yaml_path: str,
        workspace_root: str,
        lease_address: str,
        teacher_annotation_address: str = "",
    ) -> None:
        self.edge_id = int(edge_id)
        self.worker_id = str(worker_id)
        self.run_id = str(run_id or "")
        self.yaml_path = str(yaml_path)
        self.workspace_root = str(workspace_root)
        self.lease_address = str(lease_address)
        self.teacher_annotation_address = str(teacher_annotation_address or "")
        self._lock = threading.RLock()
        self._state: WorkerState = "STARTING"
        self._message = "edge worker is still starting"
        self._error_type = ""
        self._service: EdgeWorkerService | None = None
        self._thread: threading.Thread | None = None
        self._shutdown_callback: Callable[[], None] | None = None
        self._closing = False

    def set_shutdown_callback(self, callback: Callable[[], None]) -> None:
        self._shutdown_callback = callback

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._initialize,
                name=f"edge-worker-init-{self.worker_id}",
                daemon=True,
            )
            self._thread.start()

    def health(self) -> WorkerHealth:
        with self._lock:
            return WorkerHealth(
                ok=self._state == "READY",
                state=self._state,
                edge_id=self.edge_id,
                worker_id=self.worker_id,
                message=self._message,
                error_type=self._error_type,
                run_id=self.run_id,
                lease_address=self.lease_address,
            )

    def routes(self) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
        return {
            "/sync_samples": self._call("sync_samples"),
            "/submit_training_job": self._call("submit_training_job"),
            "/get_training_job_status": self._call("get_training_job_status"),
            "/download_trained_model": self._call("download_trained_model"),
            "/cancel_training_job": self._call("cancel_training_job"),
            "/report_edge_model_version": self._call("report_edge_model_version"),
            "/shutdown": self.shutdown,
        }

    def close(self) -> None:
        with self._lock:
            self._closing = True
            service = self._service
            self._service = None
            if self._state != "FAILED":
                self._state = "STOPPING"
                self._message = "edge worker is stopping"
        if service is not None:
            service.close()

    def shutdown(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        self.close()
        callback = self._shutdown_callback
        if callback is not None:
            threading.Thread(
                target=callback,
                name=f"edge-worker-shutdown-{self.worker_id}",
                daemon=True,
            ).start()
        return {"success": True, "message": "worker shutdown"}

    def _initialize(self) -> None:
        logger.info(
            "[EdgeWorker] service initializing worker={} edge={}",
            self.worker_id,
            self.edge_id,
        )
        try:
            service_kwargs = dict(
                edge_id=self.edge_id,
                worker_id=self.worker_id,
                yaml_path=self.yaml_path,
                workspace_root=self.workspace_root,
                lease_address=self.lease_address,
            )
            if self.teacher_annotation_address:
                service_kwargs["teacher_annotation_address"] = (
                    self.teacher_annotation_address
                )
            service = EdgeWorkerService(**service_kwargs)
        except Exception as exc:
            with self._lock:
                if self._closing:
                    self._state = "STOPPING"
                    self._message = "edge worker is stopping"
                    self._error_type = ""
                    return
                self._state = "FAILED"
                self._message = str(exc)
                self._error_type = WORKER_STARTUP_FAILED
            logger.exception(
                "[EdgeWorker] service startup failed worker={} edge={}",
                self.worker_id,
                self.edge_id,
            )
            return
        should_close = False
        with self._lock:
            if self._closing or self._state == "STOPPING":
                should_close = True
            else:
                self._service = service
                self._state = "READY"
                self._message = ""
                self._error_type = ""
        if should_close:
            service.close()
            logger.info(
                "[EdgeWorker] service initialized during shutdown worker={} edge={}",
                self.worker_id,
                self.edge_id,
            )
            return
        logger.info(
            "[EdgeWorker] service ready worker={} edge={}",
            self.worker_id,
            self.edge_id,
        )

    def _call(self, method_name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def handler(payload: dict[str, Any]) -> dict[str, Any]:
            with self._lock:
                service = self._service
                state = self._state
                message = self._message
            if service is None or state != "READY":
                raise RuntimeError(message or "edge worker is not ready")
            method = getattr(service, method_name, None)
            if not callable(method):
                raise RuntimeError(f"edge worker route is not configured: {method_name}")
            return method(payload)

        return handler


def main() -> None:
    parser = argparse.ArgumentParser(description="RECAP edge-affine cloud worker")
    parser.add_argument("--edge_id", type=int, required=True)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--yaml_path", default="./config/config.yaml")
    parser.add_argument("--listen_address", required=True)
    parser.add_argument("--workspace_root", required=True)
    parser.add_argument("--lease_address", required=True)
    parser.add_argument("--teacher_annotation_address", default="")
    parser.add_argument("--lazy_cuda_init", default="true")
    args = parser.parse_args()

    os.environ.setdefault("RECAP_EDGE_WORKER_ID", str(args.worker_id))
    os.environ.setdefault("RECAP_EDGE_ID", str(args.edge_id))
    manager = EdgeWorkerServiceManager(
        edge_id=args.edge_id,
        worker_id=args.worker_id,
        run_id=args.run_id,
        yaml_path=args.yaml_path,
        workspace_root=args.workspace_root,
        lease_address=args.lease_address,
        teacher_annotation_address=args.teacher_annotation_address,
    )
    server = JsonRpcServer(
        listen_address=args.listen_address,
        routes=manager.routes(),
        health_payload={
            "edge_id": int(args.edge_id),
            "worker_id": str(args.worker_id),
            "run_id": str(args.run_id),
            "lease_address": str(args.lease_address),
        },
        health_provider=manager.health,
        always_available_routes={"/shutdown"},
    )
    manager.set_shutdown_callback(server.shutdown)
    logger.info(
        "[EdgeWorker] rpc server bound worker={} edge={} endpoint={} lazy_cuda={}",
        args.worker_id,
        args.edge_id,
        server.listen_address,
        args.lazy_cuda_init,
    )
    manager.start()
    try:
        server.serve_forever()
    finally:
        try:
            manager.close()
        except Exception:
            logger.exception(
                "[EdgeWorker] service close failed worker={} edge={}",
                args.worker_id,
                args.edge_id,
            )


if __name__ == "__main__":
    main()
