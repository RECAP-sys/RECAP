from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from cloud.workers.assignment_store import EdgeAssignment, EdgeAssignmentStore
from cloud.workers.mps_runtime import MpsEnvironment
from cloud.workers.worker_client import EdgeWorkerClient
from cloud.workers.worker_protocol import (
    WORKER_RPC_UNAVAILABLE,
    WORKER_STARTUP_FAILED,
    JsonRpcError,
    WorkerHealth,
)


class WorkerStartupError(RuntimeError):
    def __init__(self, message: str, *, error_type: str = WORKER_STARTUP_FAILED) -> None:
        super().__init__(str(message))
        self.error_type = str(error_type or WORKER_STARTUP_FAILED)


class WorkerPoolClosingError(WorkerStartupError):
    def __init__(self) -> None:
        super().__init__("edge worker pool is closing", error_type="WORKER_POOL_CLOSING")


@dataclass(slots=True)
class _HealthProbe:
    state: str
    message: str = ""
    error_type: str = ""
    health: WorkerHealth | None = None

    @property
    def ready(self) -> bool:
        return (
            self.state == "READY"
            and self.health is not None
            and self.health.ok
            and self.health.state == "READY"
        )

    @property
    def failed(self) -> bool:
        return self.state == "FAILED"

    @property
    def unreachable(self) -> bool:
        return self.state == "UNREACHABLE"


class EdgeWorkerPool:
    def __init__(
        self,
        *,
        yaml_path: str,
        run_id: str,
        mode: str,
        assignment_store: EdgeAssignmentStore,
        edge_workers_config: object,
        worker_service_config: object,
        mps_env: MpsEnvironment,
        lease_address: str,
        teacher_annotation_address: str = "",
        log_internal_ids: bool = False,
    ) -> None:
        self.yaml_path = str(yaml_path)
        self.run_id = str(run_id)
        self.mode = str(mode)
        self.assignment_store = assignment_store
        self.edge_workers_config = edge_workers_config
        self.worker_service_config = worker_service_config
        self.mps_env = mps_env
        self.lease_address = str(lease_address)
        self.teacher_annotation_address = str(teacher_annotation_address or "")
        self.log_internal_ids = bool(log_internal_ids)
        self.request_timeout_sec = float(
            getattr(worker_service_config, "request_timeout_sec", 600.0)
        )
        self.startup_timeout_sec = float(
            getattr(worker_service_config, "startup_timeout_sec", 30.0)
        )
        self.startup_max_retries = max(
            0,
            int(getattr(worker_service_config, "startup_max_retries", 2)),
        )
        self.ready_cache_ttl_sec = max(
            0.0,
            float(getattr(worker_service_config, "healthcheck_interval_sec", 10.0)),
        )
        self.worker_base_port = int(getattr(edge_workers_config, "worker_base_port", 56000))
        self.worker_workspace_root = str(
            getattr(edge_workers_config, "workspace_root", "./cache/server_workspace/workers")
        )
        self.lazy_cuda_init = bool(getattr(edge_workers_config, "lazy_cuda_init", True))
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen] = {}
        self._states: dict[int, str] = {}
        self._startup_events: dict[int, threading.Event] = {}
        self._startup_errors: dict[int, str] = {}
        self._edge_locks: dict[int, threading.Lock] = {}
        self._ready_cache: dict[int, tuple[float, EdgeAssignment]] = {}
        self._ready_logged: set[int] = set()
        self._reserved_ports: set[int] = {
            port
            for assignment in self.assignment_store.all()
            for port in [_endpoint_port(assignment.endpoint)]
            if port > 0
        }
        self._bad_ports: set[int] = set()
        self._closing = False

    def client_for_edge(self, edge_id: int) -> EdgeWorkerClient:
        assignment = self.ensure_worker(edge_id)
        return EdgeWorkerClient(assignment.endpoint, timeout_sec=self.request_timeout_sec)

    def ensure_worker(self, edge_id: int) -> EdgeAssignment:
        edge = int(edge_id)
        with self._edge_lock(edge):
            return self._ensure_worker_locked_by_edge(edge)

    def _ensure_worker_locked_by_edge(self, edge: int) -> EdgeAssignment:
        cached = self._ready_assignment_if_fresh(edge)
        if cached is not None:
            return cached
        max_attempts = self.startup_max_retries + 1
        last_error = ""
        last_error_type = WORKER_STARTUP_FAILED
        for attempt in range(max_attempts):
            assignment = self._prepare_startup_attempt(edge)
            try:
                ready = self._wait_until_ready_or_failed(edge, assignment)
                with self._lock:
                    self._mark_ready_locked(edge, ready)
                return ready
            except WorkerStartupError as exc:
                last_error = str(exc)
                last_error_type = exc.error_type
                with self._lock:
                    if self._closing:
                        raise WorkerPoolClosingError() from exc
                    process = self._processes.pop(edge, None)
                    self._ready_cache.pop(edge, None)
                    if process is not None:
                        self._states[edge] = "STOPPING"
                    self._startup_errors[edge] = last_error
                    event = self._startup_events.setdefault(edge, threading.Event())
                    event.set()
                if process is not None:
                    self._stop_process(process, timeout=5.0)
                with self._lock:
                    self._mark_bad_endpoint_locked(assignment.endpoint)
                    self._states[edge] = "FAILED"
                    if attempt + 1 < max_attempts:
                        self._replace_endpoint_locked(edge, assignment)
                        logger.warning(
                            "[EdgeWorkerPool] worker startup retry: edge={} attempt={} "
                            "reason={} endpoint={}",
                            edge,
                            attempt + 1,
                            last_error,
                            assignment.endpoint,
                        )
        raise WorkerStartupError(
            last_error or "edge worker did not become ready",
            error_type=last_error_type,
        )

    def restart_worker(self, edge_id: int) -> EdgeAssignment:
        edge = int(edge_id)
        with self._edge_lock(edge):
            return self._restart_worker_locked_by_edge(edge)

    def _restart_worker_locked_by_edge(self, edge: int) -> EdgeAssignment:
        with self._lock:
            if self._closing:
                raise WorkerPoolClosingError()
            process = self._processes.pop(edge, None)
            self._ready_cache.pop(edge, None)
            self._ready_logged.discard(edge)
            assignment = self._get_or_create_assignment_locked(edge)
            self._states[edge] = "STOPPING"
        if process is not None:
            self._stop_process(process, timeout=5.0)
        with self._lock:
            port = _endpoint_port(assignment.endpoint)
            if port <= 0 or not _port_available("127.0.0.1", port):
                assignment = self._replace_endpoint_locked(edge, assignment)
            self._start_worker_process_locked(edge, assignment)
        return self._wait_until_ready_or_failed(edge, assignment)

    def close(self) -> None:
        with self._lock:
            self._closing = True
            items = [
                (edge, process, self.assignment_store.get(edge))
                for edge, process in self._processes.items()
            ]
            self._processes.clear()
            for edge in list(self._states):
                self._states[edge] = "STOPPING"
            self._ready_cache.clear()
        for _edge, process, assignment in items:
            if assignment is not None and process.poll() is None:
                self._request_worker_shutdown(assignment)
        deadline = time.monotonic() + 5.0
        for _edge, process, _assignment in items:
            if process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=min(2.0, remaining))
            except subprocess.TimeoutExpired:
                pass
        for _edge, process, _assignment in items:
            remaining = max(0.0, deadline - time.monotonic())
            self._stop_process(process, timeout=remaining)

    def _prepare_startup_attempt(self, edge: int) -> EdgeAssignment:
        with self._lock:
            if self._closing:
                raise WorkerPoolClosingError()
            assignment = self._get_or_create_assignment_locked(edge)
            process = self._processes.get(edge)
            if process is not None and process.poll() is None:
                probe = self._health_probe(assignment)
                if probe.ready:
                    self._mark_ready_locked(edge, assignment)
                    return assignment
                if probe.failed:
                    self._states[edge] = "FAILED"
                    self._startup_errors[edge] = probe.message
                    self._processes.pop(edge, None)
                    self._ready_cache.pop(edge, None)
                    self._mark_bad_endpoint_locked(assignment.endpoint)
                    process_to_stop = process
                else:
                    self._states[edge] = "STARTING"
                    return assignment
            else:
                process_to_stop = None
                if process is not None:
                    self._processes.pop(edge, None)
                    self._ready_cache.pop(edge, None)
                    self._mark_bad_endpoint_locked(assignment.endpoint)
                    assignment = self._replace_endpoint_locked(edge, assignment)
                else:
                    probe = self._health_probe(assignment)
                    if probe.ready:
                        logger.debug(
                            "[EdgeWorkerPool] using existing healthy worker={} endpoint={} edge={}",
                            assignment.worker_id,
                            assignment.endpoint,
                            assignment.edge_id,
                        )
                        self._mark_ready_locked(edge, assignment)
                        return assignment
                    if probe.state == "STARTING":
                        self._states[edge] = "STARTING"
                        return assignment
                    port = _endpoint_port(assignment.endpoint)
                    if port <= 0 or not _port_available("127.0.0.1", port):
                        old_endpoint = assignment.endpoint
                        assignment = self._replace_endpoint_locked(edge, assignment)
                        logger.warning(
                            "[EdgeWorkerPool] endpoint conflict: edge={} old_endpoint={} "
                            "retry_endpoint={}",
                            edge,
                            old_endpoint,
                            assignment.endpoint,
                        )
                self._start_worker_process_locked(edge, assignment)
                return assignment
        if process_to_stop is not None:
            self._stop_process(process_to_stop, timeout=5.0)
        with self._lock:
            if self._closing:
                raise WorkerPoolClosingError()
            assignment = self._replace_endpoint_locked(edge, assignment)
            self._start_worker_process_locked(edge, assignment)
            return assignment

    def _get_or_create_assignment_locked(self, edge: int) -> EdgeAssignment:
        assignment = self.assignment_store.get(edge)
        if assignment is None or not assignment.endpoint:
            endpoint = self._allocate_endpoint_locked()
            assignment = self.assignment_store.assign(edge_id=edge, endpoint=endpoint)
        return assignment

    def _edge_lock(self, edge: int) -> threading.Lock:
        with self._lock:
            lock = self._edge_locks.get(edge)
            if lock is None:
                lock = threading.Lock()
                self._edge_locks[edge] = lock
            return lock

    def _replace_endpoint_locked(
        self,
        edge: int,
        assignment: EdgeAssignment,
    ) -> EdgeAssignment:
        self._mark_bad_endpoint_locked(assignment.endpoint)
        endpoint = self._allocate_endpoint_locked()
        return self.assignment_store.update_endpoint(edge_id=edge, endpoint=endpoint)

    def _allocate_endpoint_locked(self) -> str:
        port = self.worker_base_port
        while port < 65535:
            if (
                port not in self._reserved_ports
                and port not in self._bad_ports
                and _port_available("127.0.0.1", port)
            ):
                self._reserved_ports.add(port)
                return f"127.0.0.1:{port}"
            port += 1
        raise RuntimeError("No available local worker port")

    def _start_worker_process_locked(
        self,
        edge: int,
        assignment: EdgeAssignment,
    ) -> subprocess.Popen:
        process = self._start_worker_process(assignment)
        self._processes[edge] = process
        self._states[edge] = "STARTING"
        self._ready_cache.pop(edge, None)
        self._ready_logged.discard(edge)
        self._startup_errors.pop(edge, None)
        self._startup_events[edge] = threading.Event()
        return process

    def _start_worker_process(self, assignment: EdgeAssignment) -> subprocess.Popen:
        Path(assignment.workspace_root).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(self.mps_env.as_env())
        cmd = [
            sys.executable,
            "-m",
            "cloud.workers.edge_worker",
            "--edge_id",
            str(assignment.edge_id),
            "--worker_id",
            assignment.worker_id,
            "--run_id",
            self.run_id,
            "--yaml_path",
            self.yaml_path,
            "--listen_address",
            assignment.endpoint,
            "--workspace_root",
            assignment.workspace_root,
            "--lease_address",
            self.lease_address,
            "--lazy_cuda_init",
            "true" if self.lazy_cuda_init else "false",
        ]
        if self.teacher_annotation_address:
            cmd.extend(
                ["--teacher_annotation_address", self.teacher_annotation_address]
            )
        process = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env)
        logger.info(
            "[EdgeWorkerPool] starting worker={} edge={} endpoint={} state=STARTING lazy_cuda={}",
            assignment.worker_id,
            assignment.edge_id,
            assignment.endpoint,
            self.lazy_cuda_init,
        )
        return process

    def _wait_until_ready_or_failed(
        self,
        edge: int,
        assignment: EdgeAssignment,
    ) -> EdgeAssignment:
        deadline = time.monotonic() + self.startup_timeout_sec
        last_message = ""
        last_error_type = WORKER_RPC_UNAVAILABLE
        while time.monotonic() < deadline:
            with self._lock:
                if self._closing:
                    raise WorkerPoolClosingError()
                process = self._processes.get(edge)
            probe = self._health_probe(assignment)
            if probe.ready:
                with self._lock:
                    if edge not in self._ready_logged:
                        logger.info(
                            "[EdgeWorkerPool] worker ready worker={} edge={} endpoint={}",
                            assignment.worker_id,
                            assignment.edge_id,
                            assignment.endpoint,
                        )
                    else:
                        logger.debug(
                            "[EdgeWorkerPool] worker ready worker={} edge={} endpoint={}",
                            assignment.worker_id,
                            assignment.edge_id,
                            assignment.endpoint,
                        )
                    self._mark_ready_locked(edge, assignment)
                return assignment
            last_message = probe.message
            last_error_type = probe.error_type or last_error_type
            if probe.failed:
                raise WorkerStartupError(
                    probe.message or f"edge worker {assignment.worker_id} startup failed",
                    error_type=probe.error_type or WORKER_STARTUP_FAILED,
                )
            if process is not None and process.poll() is not None:
                raise WorkerStartupError(
                    f"edge worker {assignment.worker_id} exited during startup",
                    error_type=last_error_type,
                )
            time.sleep(0.25)
        raise WorkerStartupError(
            last_message or f"edge worker {assignment.worker_id} did not become ready",
            error_type=last_error_type,
        )

    def _health(self, assignment: EdgeAssignment) -> bool:
        return self._health_probe(assignment).ready

    def _ready_assignment_if_fresh(self, edge: int) -> EdgeAssignment | None:
        with self._lock:
            cached = self._ready_cache.get(edge)
            if cached is None:
                return None
            expires_at, assignment = cached
            if time.monotonic() >= expires_at:
                self._ready_cache.pop(edge, None)
                return None
            process = self._processes.get(edge)
            if process is not None and process.poll() is not None:
                self._ready_cache.pop(edge, None)
                self._states[edge] = "FAILED"
                return None
            return assignment

    def _mark_ready_locked(self, edge: int, assignment: EdgeAssignment) -> None:
        self._states[edge] = "READY"
        ttl = float(getattr(self, "ready_cache_ttl_sec", 0.0) or 0.0)
        if ttl > 0.0:
            self._ready_cache[edge] = (time.monotonic() + ttl, assignment)
        self._ready_logged.add(edge)
        event = self._startup_events.setdefault(edge, threading.Event())
        event.set()

    def _health_probe(self, assignment: EdgeAssignment) -> _HealthProbe:
        try:
            health = EdgeWorkerClient(assignment.endpoint, timeout_sec=2.0).get_health()
        except JsonRpcError as exc:
            return _HealthProbe(
                state="UNREACHABLE",
                message=exc.message,
                error_type=exc.error_type,
            )
        if assignment.worker_id and health.worker_id != assignment.worker_id:
            return _HealthProbe(
                state="FAILED",
                message=(
                    f"worker identity mismatch: expected {assignment.worker_id}, "
                    f"got {health.worker_id}"
                ),
                error_type=WORKER_RPC_UNAVAILABLE,
                health=health,
            )
        if self.run_id and health.run_id != self.run_id:
            return _HealthProbe(
                state="FAILED",
                message=f"worker run_id mismatch: expected {self.run_id}, got {health.run_id}",
                error_type=WORKER_RPC_UNAVAILABLE,
                health=health,
            )
        if self.lease_address and health.lease_address != self.lease_address:
            return _HealthProbe(
                state="FAILED",
                message=(
                    "worker lease address mismatch: expected "
                    f"{self.lease_address}, got {health.lease_address}"
                ),
                error_type=WORKER_RPC_UNAVAILABLE,
                health=health,
            )
        return _HealthProbe(
            state=health.state,
            message=health.message,
            error_type=health.error_type,
            health=health,
        )

    def _request_worker_shutdown(self, assignment: EdgeAssignment) -> None:
        try:
            EdgeWorkerClient(assignment.endpoint, timeout_sec=2.0).shutdown()
        except Exception as exc:
            logger.debug(
                "[EdgeWorkerPool] worker shutdown RPC failed: worker={} edge={} reason={}",
                assignment.worker_id,
                assignment.edge_id,
                exc,
            )

    def _mark_bad_endpoint_locked(self, endpoint: str) -> None:
        port = _endpoint_port(endpoint)
        if port > 0:
            self._bad_ports.add(port)

    @staticmethod
    def _stop_process(process: subprocess.Popen, *, timeout: float) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=max(0.0, float(timeout)))
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass


def _endpoint_port(endpoint: str) -> int:
    try:
        return int(str(endpoint).rsplit(":", 1)[1])
    except (IndexError, TypeError, ValueError):
        return 0


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True
