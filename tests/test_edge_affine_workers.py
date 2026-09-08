from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import cloud.workers.edge_worker_pool as edge_worker_pool_module
from cloud.workers.assignment_store import EdgeAssignment, EdgeAssignmentStore
from cloud.workers.edge_worker_pool import (
    EdgeWorkerPool,
    WorkerPoolClosingError,
    WorkerStartupError,
)
from cloud.workers.gpu_lease_manager import GpuLeaseManager, LeaseRequest
from cloud.workers.lease_service import GpuLeaseService
from cloud.workers.mps_runtime import MpsEnvironment, resolve_active_thread_percentage
from cloud.workers.worker_client import EdgeWorkerClient, GpuLeaseHttpClient
from cloud.workers.worker_protocol import (
    WORKER_NOT_READY,
    WORKER_RPC_UNAVAILABLE,
    JsonRpcError,
    JsonRpcServer,
    WorkerHealth,
    post_json,
)
from config import load_runtime_config
from grpc_server import message_transmission_pb2
from grpc_server.continual_backends import EdgeWorkerRoutedContinualLearningBackend


def test_edge_worker_module_imports() -> None:
    import cloud.workers.edge_worker as edge_worker

    assert edge_worker.EdgeWorkerService is not None


def test_edge_worker_health_is_available_before_service_ready(monkeypatch) -> None:
    from cloud.workers.edge_worker import EdgeWorkerServiceManager

    init_can_finish = threading.Event()

    class FakeService:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            init_can_finish.wait(timeout=2.0)

        def close(self) -> None:
            pass

        def sync_samples(self, payload):
            del payload
            return {"success": True, "message": "ok", "committed_samples": 0}

    monkeypatch.setattr("cloud.workers.edge_worker.EdgeWorkerService", FakeService)
    manager = EdgeWorkerServiceManager(
        edge_id=1,
        worker_id="edge_1",
        run_id="run-a",
        yaml_path="./config/config.yaml",
        workspace_root="/tmp/edge_1",
        lease_address="127.0.0.1:55999",
    )
    server = JsonRpcServer(
        listen_address="127.0.0.1:0",
        routes=manager.routes(),
        health_provider=manager.health,
        health_payload={"edge_id": 1, "worker_id": "edge_1"},
    )
    manager.set_shutdown_callback(server.shutdown)
    server.start()
    try:
        manager.start()
        health = EdgeWorkerClient(server.listen_address, timeout_sec=1).get_health()
        assert health.state == "STARTING"
        assert health.ok is False
        with pytest.raises(JsonRpcError) as exc_info:
            post_json(server.listen_address, "/sync_samples", {}, timeout=1)
        assert exc_info.value.error_type == WORKER_NOT_READY

        init_can_finish.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            health = EdgeWorkerClient(server.listen_address, timeout_sec=1).get_health()
            if health.state == "READY":
                break
            time.sleep(0.01)
        assert health.state == "READY"
        assert health.ok is True
    finally:
        init_can_finish.set()
        manager.close()
        server.shutdown()


def test_assignment_store_sticky_and_isolated(tmp_path: Path) -> None:
    store_path = tmp_path / "worker_assignments.json"
    store = EdgeAssignmentStore(
        store_path,
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )

    edge_1 = store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    edge_110 = store.assign(edge_id=110, endpoint="127.0.0.1:56001")

    assert store.assign(edge_id=1, endpoint="127.0.0.1:56099") == edge_1
    assert edge_1.worker_id == "edge_1"
    assert edge_110.worker_id == "edge_110"
    assert edge_1.workspace_root != edge_110.workspace_root

    reloaded = EdgeAssignmentStore(
        store_path,
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    assert reloaded.get(1).worker_id == "edge_1"
    assert reloaded.get(1).endpoint == "127.0.0.1:56000"


def test_worker_pool_allocates_dynamic_ports(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_port_available(host: str, port: int) -> bool:
        del host
        calls.append(port)
        return port >= 56002

    monkeypatch.setattr("cloud.workers.edge_worker_pool._port_available", fake_port_available)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(
            request_timeout_sec=1,
            startup_timeout_sec=1,
        ),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assert pool._allocate_endpoint_locked() == "127.0.0.1:56002"
    assert calls == [56000, 56001, 56002]


def test_worker_pool_reserves_persisted_assignment_ports(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_port_available(host: str, port: int) -> bool:
        del host
        calls.append(port)
        return True

    monkeypatch.setattr("cloud.workers.edge_worker_pool._port_available", fake_port_available)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")

    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assert pool._allocate_endpoint_locked() == "127.0.0.1:56001"
    assert calls == [56001]


def test_worker_pool_health_rejects_wrong_worker_id(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            captured["endpoint"] = endpoint
            captured["timeout_sec"] = str(timeout_sec)

        def get_health(self) -> WorkerHealth:
            return WorkerHealth(
                ok=True,
                state="READY",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=EdgeAssignmentStore(
            tmp_path / "assignments.json",
            run_id="run-a",
            mode="edge_affine_single_gpu_mps",
            worker_workspace_root=tmp_path / "workers",
        ),
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assert pool._health(
        EdgeAssignment(
            edge_id=1,
            worker_id="edge_1",
            endpoint="127.0.0.1:56000",
            workspace_root=str(tmp_path / "workers" / "edge_1"),
        )
    )
    assert not pool._health(
        EdgeAssignment(
            edge_id=2,
            worker_id="edge_2",
            endpoint="127.0.0.1:56000",
            workspace_root=str(tmp_path / "workers" / "edge_2"),
        )
    )
    assert captured["endpoint"] == "127.0.0.1:56000"
    assert captured["timeout_sec"] == "2.0"


def test_worker_pool_reallocates_port_for_stale_worker_health(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"started": []}

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del timeout_sec
            self.endpoint = endpoint

        def get_health(self) -> WorkerHealth:
            if self.endpoint != "127.0.0.1:56001":
                raise JsonRpcError(
                    "connection refused",
                    error_type=WORKER_RPC_UNAVAILABLE,
                )
            return WorkerHealth(
                ok=True,
                state="READY",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    class FakeProcess:
        def poll(self):
            return None

    def fake_port_available(host: str, port: int) -> bool:
        del host
        return port != 56000

    def fake_popen(cmd, cwd, env):
        del cwd, env
        captured["started"].append(cmd)
        return FakeProcess()

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    monkeypatch.setattr("cloud.workers.edge_worker_pool._port_available", fake_port_available)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assignment = pool.ensure_worker(1)

    assert assignment.endpoint == "127.0.0.1:56001"
    assert len(captured["started"]) == 1
    assert "--run_id" in captured["started"][0]


def test_worker_pool_does_not_spawn_over_matching_untracked_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del endpoint, timeout_sec

        def get_health(self) -> WorkerHealth:
            return WorkerHealth(
                ok=True,
                state="READY",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("matching healthy worker should not be respawned")

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assignment = pool.ensure_worker(1)

    assert assignment.endpoint == "127.0.0.1:56000"


def test_worker_pool_logs_ready_once_with_ttl_cache(monkeypatch, tmp_path: Path) -> None:
    messages: list[str] = []
    started = False

    class FakeLogger:
        def info(self, message, *args, **kwargs):
            del args, kwargs
            messages.append(str(message))

        def debug(self, message, *args, **kwargs):
            del message, args, kwargs

        def warning(self, message, *args, **kwargs):
            del message, args, kwargs

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del endpoint, timeout_sec

        def get_health(self) -> WorkerHealth:
            if not started:
                raise JsonRpcError("connection refused", error_type=WORKER_RPC_UNAVAILABLE)
            return WorkerHealth(
                ok=True,
                state="READY",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None):
            del timeout
            return 0

    def fake_popen(*args, **kwargs):
        del args, kwargs
        nonlocal started
        started = True
        return FakeProcess()

    monkeypatch.setattr(edge_worker_pool_module, "logger", FakeLogger())
    monkeypatch.setattr(edge_worker_pool_module, "EdgeWorkerClient", FakeClient)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=EdgeAssignmentStore(
            tmp_path / "assignments.json",
            run_id="run-a",
            mode="edge_affine_single_gpu_mps",
            worker_workspace_root=tmp_path / "workers",
        ),
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(
            request_timeout_sec=1,
            startup_timeout_sec=1,
            healthcheck_interval_sec=60,
        ),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )
    try:
        first = pool.ensure_worker(1)
        second = pool.ensure_worker(1)
    finally:
        pool.close()

    assert first.endpoint == second.endpoint
    ready_logs = [message for message in messages if "[EdgeWorkerPool] worker ready" in message]
    assert len(ready_logs) == 1


def test_worker_pool_serializes_failed_worker_replacement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"started": []}

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del timeout_sec
            self.endpoint = endpoint

        def get_health(self) -> WorkerHealth:
            if self.endpoint == "127.0.0.1:56000":
                return WorkerHealth(
                    ok=False,
                    state="FAILED",
                    edge_id=1,
                    worker_id="edge_1",
                    run_id="run-a",
                    lease_address="127.0.0.1:55999",
                    message="startup failed",
                )
            return WorkerHealth(
                ok=True,
                state="READY",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            del timeout
            self.terminated = True
            return 0

        def kill(self):
            self.terminated = True

    def fake_stop(process, *, timeout: float) -> None:
        del timeout
        time.sleep(0.05)
        process.terminated = True

    def fake_popen(cmd, cwd, env):
        del cwd, env
        captured["started"].append(cmd)
        return FakeProcess()

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    monkeypatch.setattr("cloud.workers.edge_worker_pool._port_available", lambda *_: True)
    monkeypatch.setattr(EdgeWorkerPool, "_stop_process", staticmethod(fake_stop))
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )
    pool._processes[1] = FakeProcess()
    start = threading.Event()
    assignments: list[str] = []

    def ensure() -> None:
        start.wait(timeout=1.0)
        assignments.append(pool.ensure_worker(1).endpoint)

    threads = [threading.Thread(target=ensure) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=2.0)

    assert assignments == ["127.0.0.1:56001", "127.0.0.1:56001"]
    assert len(captured["started"]) == 1


def test_worker_pool_waits_for_alive_starting_worker_without_respawn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = {"health": 0}

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del endpoint, timeout_sec

        def get_health(self) -> WorkerHealth:
            calls["health"] += 1
            if calls["health"] < 3:
                return WorkerHealth(
                    ok=False,
                    state="STARTING",
                    edge_id=1,
                    worker_id="edge_1",
                    run_id="run-a",
                    lease_address="127.0.0.1:55999",
                )
            return WorkerHealth(
                ok=True,
                state="READY",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    class FakeProcess:
        def poll(self):
            return None

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("alive STARTING worker should not be respawned")

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(
            request_timeout_sec=1,
            startup_timeout_sec=1,
        ),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )
    pool._processes[1] = FakeProcess()

    assignment = pool.ensure_worker(1)

    assert assignment.endpoint == "127.0.0.1:56000"
    assert calls["health"] >= 3


def test_worker_pool_startup_timeout_stops_alive_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del endpoint, timeout_sec

        def get_health(self) -> WorkerHealth:
            return WorkerHealth(
                ok=False,
                state="STARTING",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.terminated or self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            del timeout
            return 0

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(
            request_timeout_sec=1,
            startup_timeout_sec=0.01,
            startup_max_retries=0,
        ),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )
    process = FakeProcess()
    pool._processes[1] = process

    with pytest.raises(WorkerStartupError):
        pool.ensure_worker(1)

    assert process.terminated
    assert pool._processes == {}


def test_worker_pool_close_requests_shutdown_and_blocks_new_starts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    shutdown_calls: list[str] = []

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del timeout_sec
            self.endpoint = endpoint

        def shutdown(self):
            shutdown_calls.append(self.endpoint)
            return {"success": True}

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

        def wait(self, timeout=None):
            if self.terminated:
                return 0
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )
    process = FakeProcess()
    pool._processes[1] = process

    pool.close()

    assert shutdown_calls == ["127.0.0.1:56000"]
    assert process.terminated
    with pytest.raises(WorkerPoolClosingError):
        pool.ensure_worker(1)


def test_worker_pool_restart_reallocates_occupied_untracked_endpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"started": []}

    class FakeClient:
        def __init__(self, endpoint: str, *, timeout_sec: float) -> None:
            del timeout_sec
            self.endpoint = endpoint

        def get_health(self) -> WorkerHealth:
            if self.endpoint != "127.0.0.1:56001":
                raise JsonRpcError(
                    "connection refused",
                    error_type=WORKER_RPC_UNAVAILABLE,
                )
            return WorkerHealth(
                ok=True,
                state="READY",
                edge_id=1,
                worker_id="edge_1",
                run_id="run-a",
                lease_address="127.0.0.1:55999",
            )

    class FakeProcess:
        def poll(self):
            return None

    def fake_port_available(host: str, port: int) -> bool:
        del host
        return port != 56000

    def fake_popen(cmd, cwd, env):
        del cwd, env
        captured["started"].append(cmd)
        return FakeProcess()

    monkeypatch.setattr("cloud.workers.edge_worker_pool.EdgeWorkerClient", FakeClient)
    monkeypatch.setattr("cloud.workers.edge_worker_pool._port_available", fake_port_available)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    store.assign(edge_id=1, endpoint="127.0.0.1:56000")
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/mps", "/tmp/mps-log", "50"),
        lease_address="127.0.0.1:55999",
    )

    assignment = pool.restart_worker(1)

    assert assignment.endpoint == "127.0.0.1:56001"
    assert len(captured["started"]) == 1


def test_worker_pool_process_env_includes_mps(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(cmd, cwd, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    store = EdgeAssignmentStore(
        tmp_path / "assignments.json",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        worker_workspace_root=tmp_path / "workers",
    )
    pool = EdgeWorkerPool(
        yaml_path="./config/config.yaml",
        run_id="run-a",
        mode="edge_affine_single_gpu_mps",
        assignment_store=store,
        edge_workers_config=SimpleNamespace(
            worker_base_port=56000,
            workspace_root=str(tmp_path / "workers"),
            lazy_cuda_init=True,
        ),
        worker_service_config=SimpleNamespace(request_timeout_sec=1, startup_timeout_sec=1),
        mps_env=MpsEnvironment("0", "/tmp/nvidia-mps", "/tmp/nvidia-mps-log", "50"),
        lease_address="127.0.0.1:55999",
        teacher_annotation_address="127.0.0.1:55998",
    )
    assignment = EdgeAssignment(
        edge_id=1,
        worker_id="edge_1",
        endpoint="127.0.0.1:56000",
        workspace_root=str(tmp_path / "workers" / "edge_1"),
    )

    pool._start_worker_process(assignment)

    env = captured["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/nvidia-mps-log"
    assert "--edge_id" in captured["cmd"]
    assert "--run_id" in captured["cmd"]
    teacher_address_index = captured["cmd"].index("--teacher_annotation_address")
    assert captured["cmd"][teacher_address_index + 1] == "127.0.0.1:55998"


def test_gpu_lease_grant_wait_release() -> None:
    manager = GpuLeaseManager(
        memory_usage_threshold=0.85,
        reserve_memory_gb=4,
        max_active_gpu_workers="auto",
        default_estimated_job_memory_gb=18,
        lease_ttl_sec=10,
        query_total_memory_gb=lambda: 48,
    )
    try:
        lease_1 = manager.acquire(_lease_request(edge_id=1, job_id="job-1"))
        lease_2 = manager.acquire(_lease_request(edge_id=110, job_id="job-2"))
        with pytest.raises(TimeoutError):
            manager.acquire(_lease_request(edge_id=2, job_id="job-3"), timeout_sec=0.01)
        manager.release(lease_1.lease_id, observed_peak_memory_gb=16.4)
        lease_3 = manager.acquire(_lease_request(edge_id=2, job_id="job-3"), timeout_sec=0.1)
        assert {lease_2.edge_id, lease_3.edge_id} == {110, 2}
    finally:
        manager.close()


def test_gpu_lease_memory_only_admits_all_jobs_that_fit_memory() -> None:
    manager = GpuLeaseManager(
        memory_usage_threshold=0.85,
        reserve_memory_gb=4,
        max_active_gpu_workers="memory_only",
        default_estimated_job_memory_gb=4,
        lease_ttl_sec=10,
        query_total_memory_gb=lambda: 48,
    )
    try:
        leases = [
            manager.acquire(
                _lease_request(
                    edge_id=edge_id,
                    job_id=f"job-{edge_id}",
                    estimated_peak_memory_gb=4,
                )
            )
            for edge_id in range(1, 5)
        ]
        snapshot = manager.snapshot()
        assert len(snapshot["active"]) == 4
        assert snapshot["max_active_gpu_workers"] is None
        assert snapshot["admission_policy"] == "memory_only"
        for lease in leases:
            manager.release(lease.lease_id)
    finally:
        manager.close()


def test_memory_only_mps_auto_does_not_impose_a_fixed_thread_quota() -> None:
    assert (
        resolve_active_thread_percentage(
            "auto",
            max_active_gpu_workers=None,
        )
        == "100"
    )


def test_gpu_lease_memory_only_waits_when_estimated_memory_is_insufficient() -> None:
    manager = GpuLeaseManager(
        memory_usage_threshold=0.75,
        reserve_memory_gb=4,
        max_active_gpu_workers="memory_only",
        default_estimated_job_memory_gb=6,
        lease_ttl_sec=10,
        query_total_memory_gb=lambda: 16,
    )
    try:
        lease = manager.acquire(
            _lease_request(
                edge_id=1,
                job_id="job-1",
                estimated_peak_memory_gb=6,
            )
        )
        with pytest.raises(TimeoutError):
            manager.acquire(
                _lease_request(
                    edge_id=2,
                    job_id="job-2",
                    estimated_peak_memory_gb=6,
                ),
                timeout_sec=0.01,
            )
        manager.release(lease.lease_id)
        replacement = manager.acquire(
            _lease_request(
                edge_id=2,
                job_id="job-2",
                estimated_peak_memory_gb=6,
            ),
            timeout_sec=0.1,
        )
        manager.release(replacement.lease_id)
    finally:
        manager.close()


def test_gpu_lease_memory_only_accounts_for_live_free_memory() -> None:
    manager = GpuLeaseManager(
        memory_usage_threshold=0.85,
        reserve_memory_gb=2,
        max_active_gpu_workers="memory_only",
        default_estimated_job_memory_gb=7,
        lease_ttl_sec=10,
        query_total_memory_gb=lambda: 48,
        query_free_memory_gb=lambda: 6,
    )
    try:
        with pytest.raises(TimeoutError):
            manager.acquire(
                _lease_request(
                    edge_id=1,
                    job_id="job-1",
                    estimated_peak_memory_gb=7,
                ),
                timeout_sec=0.01,
            )
        assert manager.snapshot()["available_memory_gb"] == pytest.approx(4.0)
    finally:
        manager.close()


def test_gpu_lease_memory_only_preserves_exclusive_oom_retry() -> None:
    manager = GpuLeaseManager(
        max_active_gpu_workers="memory_only",
        default_estimated_job_memory_gb=4,
        query_total_memory_gb=lambda: 48,
    )
    try:
        ordinary = manager.acquire(
            _lease_request(
                edge_id=1,
                job_id="ordinary",
                estimated_peak_memory_gb=4,
            )
        )
        exclusive_request = _lease_request(
            edge_id=2,
            job_id="exclusive",
            estimated_peak_memory_gb=4,
        )
        exclusive_request.exclusive = True
        with pytest.raises(TimeoutError):
            manager.acquire(exclusive_request, timeout_sec=0.01)
        manager.release(ordinary.lease_id)

        exclusive = manager.acquire(exclusive_request, timeout_sec=0.1)
        with pytest.raises(TimeoutError):
            manager.acquire(
                _lease_request(edge_id=3, job_id="blocked-by-exclusive"),
                timeout_sec=0.01,
            )
        assert manager.mark_oom(
            job_id="exclusive",
            message="CUDA out of memory",
        )["retry_exclusive"]
        manager.release(exclusive.lease_id)
    finally:
        manager.close()


def test_internal_worker_rpc_ignores_http_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    manager = GpuLeaseManager(
        max_active_gpu_workers=1,
        query_total_memory_gb=lambda: 48,
    )
    service = GpuLeaseService(listen_address="127.0.0.1:0", manager=manager)
    service.start()
    try:
        assert EdgeWorkerClient(service.listen_address, timeout_sec=1).health()
        lease = GpuLeaseHttpClient(service.listen_address, timeout_sec=1).acquire(
            _lease_request(edge_id=1, job_id="direct-rpc")
        )
        lease.release()
    finally:
        service.shutdown()
        manager.close()


def test_worker_health_malformed_response_is_unavailable(monkeypatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

        def read(self):
            return b"not-json"

    monkeypatch.setattr(
        "cloud.workers.worker_client.open_direct",
        lambda request, *, timeout: FakeResponse(),
    )

    with pytest.raises(JsonRpcError) as exc_info:
        EdgeWorkerClient("127.0.0.1:56000", timeout_sec=1).get_health()

    assert exc_info.value.error_type == WORKER_RPC_UNAVAILABLE


def test_gpu_lease_ttl_expires_stale_worker() -> None:
    manager = GpuLeaseManager(
        lease_ttl_sec=0.1,
        query_total_memory_gb=lambda: 48,
    )
    try:
        manager.acquire(_lease_request(edge_id=1, job_id="job-expire"))
        time.sleep(0.35)
        snapshot = manager.snapshot()
        assert snapshot["active"] == []
        assert snapshot["expired_jobs"]["job-expire"] == "GPU lease heartbeat expired"
    finally:
        manager.close()


def test_routed_backend_forwards_main_rpc_calls() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def sync_samples(self, request):
            self.calls.append("sync_samples")
            return message_transmission_pb2.SampleSyncReply(success=True)

        def submit_training_job(self, request, *, exclusive_gpu_lease: bool = False):
            del exclusive_gpu_lease
            self.calls.append("submit_training_job")
            return message_transmission_pb2.SubmitTrainingJobReply(accepted=True, job_id="j")

        def get_training_job_status(self, request):
            self.calls.append("get_training_job_status")
            return message_transmission_pb2.TrainingJobStatusReply(found=True, job_id="j")

        def download_trained_model(self, request):
            self.calls.append("download_trained_model")
            return message_transmission_pb2.DownloadTrainedModelReply(success=True)

        def report_edge_model_version(self, request):
            self.calls.append("report_edge_model_version")
            return message_transmission_pb2.ReportEdgeModelVersionReply(success=True)

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(worker_pool=pool)
    backend.sync_samples(message_transmission_pb2.SampleSyncRequest(edge_id=1))
    backend.submit_training_job(
        message_transmission_pb2.SubmitTrainingJobRequest(
            edge_id=1,
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
        )
    )
    backend.get_training_job_status(
        message_transmission_pb2.TrainingJobStatusRequest(edge_id=1, job_id="j")
    )
    backend.download_trained_model(
        message_transmission_pb2.DownloadTrainedModelRequest(edge_id=1, job_id="j")
    )
    backend.report_edge_model_version(
        message_transmission_pb2.ReportEdgeModelVersionRequest(
            edge_id=1,
            model_id="m",
            model_version="1",
        )
    )

    assert pool.client.calls == [
        "sync_samples",
        "submit_training_job",
        "get_training_job_status",
        "download_trained_model",
        "report_edge_model_version",
    ]


def test_routed_backend_materializes_uploaded_bundle_for_worker(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.request = None

        def submit_training_job(self, request, *, exclusive_gpu_lease: bool = False):
            del exclusive_gpu_lease
            self.request = request
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=True,
                job_id="j",
                status="QUEUED",
            )

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()

        def ensure_worker(self, edge_id: int):
            assert edge_id == 1
            return SimpleNamespace(workspace_root=str(tmp_path / "edge_1"))

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(worker_pool=pool)
    backend.submit_training_job(
        message_transmission_pb2.SubmitTrainingJobRequest(
            edge_id=1,
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            payload_zip=b"bundle-bytes",
        )
    )

    routed_request = pool.client.request
    bundle_path = Path(routed_request.payload_bundle_path)
    assert routed_request.payload_zip == b""
    assert bundle_path.parent.name == "incoming_bundles"
    assert not bundle_path.exists()


def test_routed_backend_forwards_exclusive_gpu_lease_flag() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.exclusive_flags: list[bool] = []

        def submit_training_job(self, request, *, exclusive_gpu_lease: bool = False):
            del request
            self.exclusive_flags.append(bool(exclusive_gpu_lease))
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=True,
                job_id="exclusive-job",
                status="QUEUED",
            )

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(worker_pool=pool)
    backend.submit_training_job(
        SimpleNamespace(
            edge_id=1,
            request_id="req-exclusive",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            cache_path="/tmp/cache",
            send_low_conf_features=False,
            frame_indices=[1],
            payload_zip=b"",
            base_model_version="0",
            exclusive_gpu_lease=True,
        )
    )

    assert pool.client.exclusive_flags == [True]


def test_routed_backend_reports_expired_lease_as_retryable_failure() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.status_calls = 0

        def submit_training_job(self, request, *, exclusive_gpu_lease: bool = False):
            del request, exclusive_gpu_lease
            return message_transmission_pb2.SubmitTrainingJobReply(
                accepted=True,
                job_id="expired-job",
                status="RUNNING",
            )

        def get_training_job_status(self, request):
            del request
            self.status_calls += 1
            return message_transmission_pb2.TrainingJobStatusReply(found=False)

    class FakePool:
        def __init__(self) -> None:
            self.client = FakeClient()

        def client_for_edge(self, edge_id: int):
            assert edge_id == 1
            return self.client

    class FakeLeaseManager:
        def expired_job_reason(self, job_id: str) -> str:
            return "GPU lease heartbeat expired" if job_id == "expired-job" else ""

    pool = FakePool()
    backend = EdgeWorkerRoutedContinualLearningBackend(
        worker_pool=pool,
        gpu_lease_manager=FakeLeaseManager(),
    )
    submit_reply = backend.submit_training_job(
        message_transmission_pb2.SubmitTrainingJobRequest(
            edge_id=1,
            request_id="req-a",
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            base_model_version="0",
        )
    )

    status_reply = backend.get_training_job_status(
        message_transmission_pb2.TrainingJobStatusRequest(
            edge_id=1,
            job_id=submit_reply.job_id,
        )
    )

    assert pool.client.status_calls == 0
    assert status_reply.found
    assert status_reply.job_id == "expired-job"
    assert status_reply.status == "FAILED"
    assert "retryable failure" in status_reply.message
    assert status_reply.request_id == "req-a"


def test_cloud_server_worker_pool_does_not_create_local_training_objects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from cloud_server import CloudServer

    captured_pool_kwargs: dict[str, object] = {}

    class FakeLeaseManager:
        max_active_gpu_workers = 2

        def __init__(self, **kwargs):
            del kwargs

        def close(self):
            pass

    class FakeLeaseService:
        listen_address = "127.0.0.1:55555"

        def __init__(self, **kwargs):
            del kwargs

        def start(self):
            pass

        def shutdown(self):
            pass

    class FakePool:
        def __init__(self, **kwargs):
            captured_pool_kwargs.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr("cloud_server.GpuLeaseManager", FakeLeaseManager)
    monkeypatch.setattr("cloud_server.GpuLeaseService", FakeLeaseService)
    monkeypatch.setattr("cloud_server.EdgeWorkerPool", FakePool)
    monkeypatch.setattr(
        "cloud_server.ensure_mps_runtime",
        lambda *_args, **_kwargs: MpsEnvironment("0", "/tmp/mps", "/tmp/log", "50"),
    )

    config = load_runtime_config("./config/config.yaml").server
    config.workspace_root = str(tmp_path)
    config.edge_affine_workers.enabled = True

    server = CloudServer(config, yaml_path="./config/config.yaml")
    try:
        assert server.large_object_detection is not None
        assert server.large_object_detection._detector is None
        assert server.shared_teacher_annotation_worker is not None
        assert captured_pool_kwargs["teacher_annotation_address"] == (
            server.shared_teacher_rpc_server.listen_address
        )
        assert not hasattr(server, "continual_learner")
        assert not hasattr(server, "training_job_manager")
        assert isinstance(server.continual_backend, EdgeWorkerRoutedContinualLearningBackend)
    finally:
        server.close()


def test_cloud_server_main_requires_edge_affine_workers(tmp_path: Path) -> None:
    from cloud_server import CloudServer

    config = load_runtime_config("./config/config.yaml").server
    config.workspace_root = str(tmp_path)
    config.edge_affine_workers.enabled = False

    with pytest.raises(ValueError, match=r"edge_affine_workers\.enabled=true"):
        CloudServer(config, yaml_path="./config/config.yaml")


def test_torchlens_prepare_split_error_message(monkeypatch) -> None:
    import model_management.split_runtime.torchlens_native_runtime as runtime

    monkeypatch.setattr(runtime, "require_torchlens_native_split_api", lambda: None)

    def raise_runtime(*_args, **_kwargs):
        raise RuntimeError("trace failed")

    monkeypatch.setattr(runtime.tl, "prepare_split", raise_runtime)

    with pytest.raises(RuntimeError) as exc_info:
        runtime.prepare_split_runtime(object(), object(), "after:x")

    message = str(exc_info.value)
    assert "TorchLens native split runtime construction failed during tl.prepare_split" in message
    assert "concurrent TorchLens execution" in message
    assert "CUDA OOM" in message
    assert "worker/MPS resource contention" in message
    assert "cause=RuntimeError: trace failed" in message


def _lease_request(
    *,
    edge_id: int,
    job_id: str,
    estimated_peak_memory_gb: float = 18,
) -> LeaseRequest:
    return LeaseRequest(
        edge_id=edge_id,
        worker_id=f"edge_{edge_id}",
        job_id=job_id,
        estimated_peak_memory_gb=estimated_peak_memory_gb,
    )


def test_lazy_teacher_detector_initializes_once_across_threads(monkeypatch) -> None:
    from cloud.workers.edge_worker_service import LazyObjectDetection

    created: list[object] = []

    class FakeDetector:
        def __init__(self, _config, *, type):
            del _config, type
            time.sleep(0.01)
            created.append(self)

    monkeypatch.setattr(
        "model_management.object_detection.Object_Detection",
        FakeDetector,
    )
    lazy_detector = LazyObjectDetection(SimpleNamespace(), "large inference")
    resolved: list[object] = []
    threads = [
        threading.Thread(target=lambda: resolved.append(lazy_detector._ensure()))
        for _index in range(8)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1
    assert resolved == [created[0]] * 8


def test_shared_teacher_budget_admits_three_initial_training_workers() -> None:
    manager = GpuLeaseManager(
        memory_usage_threshold=0.85,
        reserve_memory_gb=4,
        teacher_reserved_memory_gb=5,
        max_active_gpu_workers="memory_only",
        default_estimated_job_memory_gb=7.25,
        query_total_memory_gb=lambda: 36.8,
    )
    leases = []
    try:
        for edge_id in (1, 2, 3):
            leases.append(
                manager.acquire(
                    _lease_request(
                        edge_id=edge_id,
                        job_id=f"job-{edge_id}",
                        estimated_peak_memory_gb=7.25,
                    ),
                    timeout_sec=0,
                )
            )
        assert len(manager.snapshot()["active"]) == 3
        with pytest.raises(TimeoutError):
            manager.acquire(
                _lease_request(
                    edge_id=4,
                    job_id="job-4",
                    estimated_peak_memory_gb=7.25,
                ),
                timeout_sec=0,
            )
    finally:
        for lease in leases:
            manager.release(lease.lease_id)
        manager.close()
