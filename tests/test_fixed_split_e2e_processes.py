from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
import yaml

from common.logging_sanitizer import find_forbidden_log_content

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_E2E_ENV = "RECAP_RUN_YOLO26N_E2E"
RUN_FULL_RETRAIN_ENV = "RECAP_RUN_YOLO26N_E2E_FULL_RETRAIN"
E2E_EDGE_MODEL_ENV = "RECAP_E2E_EDGE_MODEL"
E2E_MODEL_ARTIFACTS = {
    "yolo26n": PROJECT_ROOT / "model_management" / "models" / "yolo26n.pt",
    "rfdetr_nano": (
        PROJECT_ROOT / "model_management" / "models" / "rf-detr-nano-origin.pth"
    ),
}
DEEP_COPY_ERROR = "Only Tensors created explicitly by the user"
COORDINATE_METADATA_ERROR = "Missing coordinate metadata required for split retraining"
FIXED_SPLIT_FAILURE = "fixed-split training failed"


class _ProcessCapture:
    def __init__(
        self,
        args: list[str],
        *,
        env: dict[str, str],
        name: str,
    ) -> None:
        self.name = name
        self.lines: list[str] = []
        self.process = subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._thread.start()

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        for line in stream:
            self.lines.append(line)
            print(f"[{self.name}] {line}", end="", flush=True)

    @property
    def text(self) -> str:
        return "".join(self.lines)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self._thread.join(timeout=2)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, *, process: _ProcessCapture, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            pytest.fail(f"{process.name} exited early:\n{process.text[-4000:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    pytest.fail(f"{process.name} did not open port {port}:\n{process.text[-4000:]}")


def _wait_for_patterns(
    processes: list[_ProcessCapture],
    patterns: list[str],
    *,
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    compiled = [re.compile(pattern) for pattern in patterns]
    while time.monotonic() < deadline:
        combined = "\n".join(process.text for process in processes)
        if DEEP_COPY_ERROR in combined:
            pytest.fail(f"TorchLens deepcopy regression appeared:\n{combined[-8000:]}")
        if COORDINATE_METADATA_ERROR in combined:
            pytest.fail(f"Coordinate metadata regression appeared:\n{combined[-8000:]}")
        if FIXED_SPLIT_FAILURE in combined:
            pytest.fail(f"Fixed-split training failed:\n{combined[-8000:]}")
        if all(pattern.search(combined) for pattern in compiled):
            return combined
        for process in processes:
            if process.process.poll() is not None:
                pytest.fail(
                    f"{process.name} exited before E2E smoke completed:\n{combined[-12000:]}"
                )
        time.sleep(1.0)
    combined = "\n".join(process.text for process in processes)
    missing = [pattern.pattern for pattern in compiled if not pattern.search(combined)]
    pytest.fail(f"Timed out waiting for E2E log patterns {missing!r}:\n{combined[-12000:]}")


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["LOGURU_COLORIZE"] = "0"
    env["LOGURU_LEVEL"] = "INFO"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


def _snapshot_log_files() -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for relative_root in ("log/client", "log/server", "log/cloud"):
        root = PROJECT_ROOT / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            stat = path.stat()
            snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _read_changed_log_files(snapshot: dict[Path, tuple[int, int]]) -> str:
    chunks: list[str] = []
    for relative_root in ("log/client", "log/server", "log/cloud"):
        root = PROJECT_ROOT / relative_root
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.log")):
            stat = path.stat()
            previous = snapshot.get(path)
            current = (stat.st_mtime_ns, stat.st_size)
            if previous == current:
                continue
            if previous is not None and previous[1] < stat.st_size:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(previous[1])
                    chunks.append(handle.read())
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def test_read_changed_log_files_reads_only_appended_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "PROJECT_ROOT", tmp_path)
    log_path = tmp_path / "log" / "client" / "run.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("old line\n", encoding="utf-8")

    snapshot = _snapshot_log_files()

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("new line\n")

    assert _read_changed_log_files(snapshot) == "new line\n"


def _write_e2e_config(
    tmp_path: Path,
    port: int,
    *,
    edge_model_name: str,
    weights_path: Path,
    full_retrain: bool,
) -> Path:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_root = tmp_path / "cache"
    video_path = PROJECT_ROOT / "video_data" / "road.mp4"

    config["server"]["continual_learning"]["training_frame_count"] = 2
    config["client"].update(
        {
            "server_ip": f"127.0.0.1:{port}",
            "diff_flag": False,
            "final_detection_threshold": 1.0,
            "lightweight": edge_model_name,
            "weights_path": str(weights_path),
        }
    )
    config["client"]["source"].update(
        {
            "video_path": str(video_path),
            "max_count": 160,
        }
    )
    config["client"]["retrain"].update(
        {
            "cache_path": str(cache_root / "edge"),
            "collect_num": 2,
            "min_low_quality_samples": 2,
            "status_not_found_grace_sec": 120,
            "poll_interval_sec": 1,
            "raw_jpeg_quality": 70,
        }
    )
    config["client"]["resource_aware_trigger"].update({"enabled": False})
    config["client"]["split_learning"].update(
        {
            "enabled": True,
            "warmup_iterations": 0,
        }
    )
    config["client"]["sample_quality"].update(
        {
            "output_entropy": {"warmup_samples": 0},
            "boundary_feature_entropy": {"warmup_samples": 0},
        }
    )
    config["client"]["window_drift"].update(
        {
            "window_size": 4,
            "min_window_size": 2,
            "low_quality_rate_threshold": 0.0,
            "persistence_windows": 1,
        }
    )

    config["server"].update(
        {
            "listen_address": f"127.0.0.1:{port}",
            "golden": "yolo26n",
            "edge_model_name": edge_model_name,
            "weights_path": str(weights_path),
            "workspace_root": str(cache_root / "server_workspace"),
            "grpc_max_workers": 8,
        }
    )
    config["server"]["continual_learning"].update(
        {
            "num_epoch": 1,
            "trace_batch_size": 2,
            "batch_size": 2,
            "teacher_batch_size": 2,
            "proxy_eval_max_samples": 4,
            "proxy_eval_interval_rounds": 1,
            "proxy_eval_patience": 0,
            "max_concurrent_jobs": 1,
            "recent_training_window_root": str(cache_root / "recent_training_windows"),
            "connectivity_smoke_only": not full_retrain,
        }
    )

    output_path = tmp_path / "e2e_config.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


@pytest.mark.integration
def test_fixed_split_continual_learning_process_e2e(tmp_path: Path) -> None:
    if os.environ.get(RUN_E2E_ENV) != "1":
        pytest.skip(
            f"set {RUN_E2E_ENV}=1 to run the GPU/CPU process E2E smoke; "
            f"also set {RUN_FULL_RETRAIN_ENV}=1 to run full retraining"
        )
    if not torch.cuda.is_available():
        pytest.skip("GPU cloud process requested, but CUDA is not available")
    edge_model_name = os.environ.get(E2E_EDGE_MODEL_ENV, "yolo26n").strip().lower()
    if edge_model_name not in E2E_MODEL_ARTIFACTS:
        pytest.fail(
            f"unsupported {E2E_EDGE_MODEL_ENV}={edge_model_name!r}; "
            f"choose from {sorted(E2E_MODEL_ARTIFACTS)}"
        )
    weights_path = E2E_MODEL_ARTIFACTS[edge_model_name]
    if not weights_path.exists():
        pytest.skip(f"{edge_model_name} weights are not available")
    if not (PROJECT_ROOT / "video_data" / "road.mp4").exists():
        pytest.skip("road.mp4 is not available")

    port = _free_port()
    full_retrain = os.environ.get(RUN_FULL_RETRAIN_ENV) == "1"
    config_path = _write_e2e_config(
        tmp_path,
        port,
        edge_model_name=edge_model_name,
        weights_path=weights_path,
        full_retrain=full_retrain,
    )
    cloud_env = _base_env()
    cloud_env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "RECAP_E2E_CUDA_VISIBLE_DEVICES",
        "0",
    )
    edge_env = _base_env()
    edge_env["CUDA_VISIBLE_DEVICES"] = ""
    log_snapshot = _snapshot_log_files()

    cloud = _ProcessCapture(
        [sys.executable, "cloud_server.py", "--yaml_path", str(config_path)],
        env=cloud_env,
        name="cloud_server",
    )
    edge: _ProcessCapture | None = None
    combined = ""
    try:
        _wait_for_port(port, process=cloud, timeout=120)
        edge = _ProcessCapture(
            [
                sys.executable,
                "edge_client.py",
                "--yaml_path",
                str(config_path),
                "--headless",
            ],
            env=edge_env,
            name="edge_client",
        )
        patterns = [
            r"Preparing fixed split runtime",
            r"\[EdgeCL\] training accepted",
            r"Received training request: edge=1",
            r"\[ShardCL\]\[CloudUnpack\] materialized low-quality trigger shards",
            r"TrainingCacheView\(source=recent_training_window\)",
        ]
        if full_retrain:
            patterns.extend(
                [
                    r"\[FixedSplitCL\]\[TrainPlan\].*epochs=1",
                    r"epoch 1/1 avg_loss=",
                    r"\[EdgeCL\] model update successful",
                ]
            )
        else:
            patterns.extend(
                [
                    r"\[FixedSplitCL\]\[ConnectivitySmoke\]",
                    r"\[EdgeCL\] model update successful",
                ]
            )
        _wait_for_patterns(
            [cloud, edge],
            patterns,
            timeout=900 if full_retrain else 360,
        )
    finally:
        if edge is not None:
            edge.stop()
        cloud.stop()
        combined = "\n".join(
            process.text for process in (cloud, edge) if process is not None
        )

    complete_logs = "\n".join((combined, _read_changed_log_files(log_snapshot)))
    findings = find_forbidden_log_content(complete_logs)
    assert not findings, f"default E2E logs exposed diagnostics: {findings}"
    assert DEEP_COPY_ERROR not in complete_logs
