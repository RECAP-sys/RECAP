from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True, slots=True)
class MpsEnvironment:
    cuda_visible_devices: str
    pipe_directory: str
    log_directory: str
    active_thread_percentage: str

    def as_env(self) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_directory,
            "CUDA_MPS_LOG_DIRECTORY": self.log_directory,
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": self.active_thread_percentage,
        }


def resolve_active_thread_percentage(
    value: object,
    *,
    max_active_gpu_workers: int | None,
) -> str:
    text = str(value or "auto").strip().lower()
    if text == "auto":
        if max_active_gpu_workers is None:
            return "100"
        return str(max(1, int(100 / max(1, int(max_active_gpu_workers)))))
    return str(int(text))


def build_mps_environment(
    config: object,
    *,
    max_active_gpu_workers: int | None,
) -> MpsEnvironment:
    return MpsEnvironment(
        cuda_visible_devices=str(getattr(config, "cuda_visible_devices", "0")),
        pipe_directory=str(getattr(config, "pipe_directory", "/tmp/nvidia-mps")),
        log_directory=str(getattr(config, "log_directory", "/tmp/nvidia-mps-log")),
        active_thread_percentage=resolve_active_thread_percentage(
            getattr(config, "active_thread_percentage", "auto"),
            max_active_gpu_workers=max_active_gpu_workers,
        ),
    )


def ensure_mps_runtime(
    config: object,
    *,
    max_active_gpu_workers: int | None,
) -> MpsEnvironment:
    env = build_mps_environment(config, max_active_gpu_workers=max_active_gpu_workers)
    Path(env.pipe_directory).mkdir(parents=True, exist_ok=True)
    Path(env.log_directory).mkdir(parents=True, exist_ok=True)
    if bool(getattr(config, "auto_start", False)):
        runtime_env = dict(os.environ)
        runtime_env.update(env.as_env())
        try:
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                env=runtime_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            logger.info("[MPS] auto_start requested; nvidia-cuda-mps-control invoked.")
        except Exception as exc:
            logger.warning("[MPS] auto_start failed: {}", exc)
    return env
