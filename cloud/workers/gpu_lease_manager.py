from __future__ import annotations

import json
import math
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

from cloud.workers.gpu_memory_estimator import GpuJobEstimate, GpuMemoryEstimator

OOM_ERROR_MARKERS = (
    "CUDA out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory",
    "out of memory",
)


def is_oom_message(message: object) -> bool:
    text = str(message or "")
    lower = text.lower()
    return any(marker.lower() in lower for marker in OOM_ERROR_MARKERS)


@dataclass(slots=True)
class LeaseRequest:
    edge_id: int
    worker_id: str
    job_id: str
    model_name: str = ""
    split_key: str = ""
    batch_size: int = 0
    train_samples: int = 0
    estimated_peak_memory_gb: float = 0.0
    exclusive: bool = False


@dataclass(slots=True)
class LeaseRecord:
    lease_id: str
    edge_id: int
    worker_id: str
    job_id: str
    model_name: str
    split_key: str
    batch_size: int
    train_samples: int
    estimated_peak_memory_gb: float
    exclusive: bool = False
    granted_at: float = field(default_factory=time.monotonic)
    last_heartbeat: float = field(default_factory=time.monotonic)
    observed_peak_memory_gb: float = 0.0


class GpuLeaseManager:
    """Process-local GPU lease scheduler.

    This class intentionally does not import torch or call torch.cuda.  GPU
    memory discovery uses NVML when available, then falls back to nvidia-smi.
    """

    def __init__(
        self,
        *,
        memory_usage_threshold: float = 0.85,
        reserve_memory_gb: float = 4.0,
        max_active_gpu_workers: int | str = "auto",
        default_estimated_job_memory_gb: float = 18.0,
        lease_ttl_sec: float = 120.0,
        teacher_reserved_memory_gb: float = 0.0,
        query_total_memory_gb=None,
        query_free_memory_gb=None,
    ) -> None:
        self.memory_usage_threshold = float(memory_usage_threshold)
        self.reserve_memory_gb = float(reserve_memory_gb)
        self.lease_ttl_sec = float(lease_ttl_sec)
        self.teacher_reserved_memory_gb = float(teacher_reserved_memory_gb)
        # Tests and callers that inject total memory generally do not have a
        # live GPU to query.  Production instances use the live free-memory
        # probe unless a caller explicitly supplies a probe (or disables it
        # with ``query_free_memory_gb=None`` alongside an injected total).
        self._query_free_memory_gb = (
            query_free_memory_gb
            if query_free_memory_gb is not None
            else (None if query_total_memory_gb is not None else query_gpu_free_memory_gb)
        )
        self.estimator = GpuMemoryEstimator(
            default_estimated_job_memory_gb=default_estimated_job_memory_gb
        )
        total_memory = (
            query_total_memory_gb()
            if query_total_memory_gb is not None
            else query_gpu_total_memory_gb()
        )
        self.total_memory_gb = float(total_memory)
        self.max_active_gpu_workers = resolve_max_active_gpu_workers(
            total_memory_gb=self.total_memory_gb,
            memory_usage_threshold=self.memory_usage_threshold,
            reserve_memory_gb=self.reserve_memory_gb + self.teacher_reserved_memory_gb,
            default_estimated_job_memory_gb=default_estimated_job_memory_gb,
            configured=max_active_gpu_workers,
        )
        self._cv = threading.Condition()
        self._active: dict[str, LeaseRecord] = {}
        self._expired_jobs: dict[str, str] = {}
        self._exclusive_waiters = 0
        self._ordinary_paused = False
        self._closed = False
        self._reaper = threading.Thread(
            target=self._reap_expired_leases,
            name="gpu-lease-reaper",
            daemon=True,
        )
        self._reaper.start()

    @property
    def allowed_memory_gb(self) -> float:
        allowed = self.total_memory_gb * self.memory_usage_threshold
        allowed -= self.reserve_memory_gb + self.teacher_reserved_memory_gb
        return max(0.0, float(allowed))

    @property
    def available_memory_gb(self) -> float:
        """Return memory available for one more lease.

        The static budget protects against delayed allocations.  When a live
        free-memory probe is available, the observed free memory is also used
        so unrelated GPU processes are accounted for before admitting work.
        """
        reserved = self._reserved_memory_locked()
        configured_available = max(0.0, self.allowed_memory_gb - reserved)
        query = self._query_free_memory_gb
        if query is None:
            return configured_available
        try:
            free_memory = float(query())
        except Exception:
            return configured_available
        if not math.isfinite(free_memory) or free_memory < 0.0:
            return configured_available
        observed_available = max(
            0.0,
            free_memory - self.reserve_memory_gb - self.teacher_reserved_memory_gb,
        )
        return min(configured_available, observed_available)

    @property
    def admission_policy(self) -> str:
        return (
            "memory_only"
            if self.max_active_gpu_workers is None
            else "count_and_memory"
        )

    def acquire(
        self,
        request: LeaseRequest,
        *,
        timeout_sec: float | None = None,
    ) -> LeaseRecord:
        requested_at = time.monotonic()
        estimate = self.estimator.estimate(
            GpuJobEstimate(
                model_name=request.model_name,
                split_key=request.split_key,
                batch_size=request.batch_size,
                train_samples=request.train_samples,
                estimated_peak_memory_gb=request.estimated_peak_memory_gb,
            )
        )
        deadline = None if timeout_sec is None else requested_at + max(0.0, float(timeout_sec))
        with self._cv:
            if request.exclusive:
                self._exclusive_waiters += 1
                self._cv.notify_all()
            try:
                while True:
                    if self._closed:
                        raise RuntimeError("GPU lease manager is closed")
                    if self._can_grant_locked(estimate, exclusive=bool(request.exclusive)):
                        lease = LeaseRecord(
                            lease_id=uuid.uuid4().hex,
                            edge_id=int(request.edge_id),
                            worker_id=str(request.worker_id),
                            job_id=str(request.job_id),
                            model_name=str(request.model_name or ""),
                            split_key=str(request.split_key or ""),
                            batch_size=int(request.batch_size or 0),
                            train_samples=int(request.train_samples or 0),
                            estimated_peak_memory_gb=float(estimate),
                            exclusive=bool(request.exclusive),
                        )
                        self._active[lease.lease_id] = lease
                        logger.info(
                            "[GpuLease] granted edge={} job={} estimated_peak={:.1f}GB "
                            "active={} reserved={:.1f}GB exclusive={}",
                            lease.edge_id,
                            lease.job_id,
                            lease.estimated_peak_memory_gb,
                            len(self._active),
                            self._reserved_memory_locked(),
                            lease.exclusive,
                        )
                        return lease
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for GPU lease")
                    logger.info(
                        "[GpuLease] waiting edge={} job={} reason=memory_threshold "
                        "reserved={:.1f}GB available={:.1f}GB allowed={:.1f}GB exclusive={}",
                        request.edge_id,
                        request.job_id,
                        self._reserved_memory_locked(),
                        self.available_memory_gb,
                        self.allowed_memory_gb,
                        bool(request.exclusive),
                    )
                    wait_timeout = (
                        1.0
                        if deadline is None
                        else max(0.0, min(1.0, deadline - time.monotonic()))
                    )
                    self._cv.wait(timeout=wait_timeout)
            finally:
                if request.exclusive:
                    self._exclusive_waiters = max(0, self._exclusive_waiters - 1)
                    self._cv.notify_all()

    def heartbeat(self, lease_id: str) -> bool:
        with self._cv:
            lease = self._active.get(str(lease_id))
            if lease is None:
                return False
            lease.last_heartbeat = time.monotonic()
            return True

    def release(
        self,
        lease_id: str,
        *,
        observed_peak_memory_gb: float = 0.0,
    ) -> LeaseRecord | None:
        with self._cv:
            lease = self._active.pop(str(lease_id), None)
            if lease is None:
                return None
            lease.observed_peak_memory_gb = float(observed_peak_memory_gb or 0.0)
            if lease.observed_peak_memory_gb > 0:
                self.estimator.observe(
                    model_name=lease.model_name,
                    split_key=lease.split_key,
                    batch_size=lease.batch_size,
                    observed_peak_memory_gb=lease.observed_peak_memory_gb,
                )
            logger.info(
                "[GpuLease] released edge={} job={} observed_peak={:.1f}GB",
                lease.edge_id,
                lease.job_id,
                lease.observed_peak_memory_gb,
            )
            self._cv.notify_all()
            return lease

    def mark_oom(self, *, job_id: str, message: str) -> dict[str, Any]:
        retry_exclusive = is_oom_message(message)
        if retry_exclusive:
            logger.warning(
                "[GpuLease] CUDA OOM detected; retrying job with exclusive GPU lease"
            )
        return {"retry_exclusive": bool(retry_exclusive), "job_id": str(job_id)}

    def pause_ordinary(self) -> None:
        with self._cv:
            self._ordinary_paused = True

    def resume_ordinary(self) -> None:
        with self._cv:
            self._ordinary_paused = False
            self._cv.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._cv:
            return {
                "total_memory_gb": self.total_memory_gb,
                "allowed_memory_gb": self.allowed_memory_gb,
                "available_memory_gb": self.available_memory_gb,
                "max_active_gpu_workers": self.max_active_gpu_workers,
                "admission_policy": self.admission_policy,
                "active": [asdict(lease) for lease in self._active.values()],
                "expired_jobs": dict(self._expired_jobs),
            }

    def expired_job_reason(self, job_id: str) -> str:
        with self._cv:
            return str(self._expired_jobs.get(str(job_id), ""))

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._reaper.join(timeout=2.0)

    def _reserved_memory_locked(self) -> float:
        return sum(float(lease.estimated_peak_memory_gb) for lease in self._active.values())

    def _can_grant_locked(self, estimate_gb: float, *, exclusive: bool) -> bool:
        if exclusive:
            return not self._active and float(estimate_gb) <= self.available_memory_gb
        if self._ordinary_paused:
            return False
        if self._exclusive_waiters > 0:
            return False
        if any(lease.exclusive for lease in self._active.values()):
            return False
        if (
            self.max_active_gpu_workers is not None
            and len(self._active) >= int(self.max_active_gpu_workers)
        ):
            return False
        return float(estimate_gb) <= self.available_memory_gb

    def _reap_expired_leases(self) -> None:
        while True:
            with self._cv:
                if self._closed:
                    return
                now = time.monotonic()
                expired = [
                    lease_id
                    for lease_id, lease in self._active.items()
                    if now - float(lease.last_heartbeat) > self.lease_ttl_sec
                ]
                for lease_id in expired:
                    lease = self._active.pop(lease_id)
                    self._expired_jobs[lease.job_id] = "GPU lease heartbeat expired"
                    logger.warning(
                        "[GpuLease] expired edge={} job={} lease={}",
                        lease.edge_id,
                        lease.job_id,
                        lease.lease_id,
                    )
                if expired:
                    self._cv.notify_all()
                reap_interval = max(0.01, min(10.0, self.lease_ttl_sec / 2.0))
                self._cv.wait(timeout=reap_interval)


def resolve_max_active_gpu_workers(
    *,
    total_memory_gb: float,
    memory_usage_threshold: float,
    reserve_memory_gb: float,
    default_estimated_job_memory_gb: float,
    configured: int | str,
) -> int | None:
    configured_text = str(configured).strip().lower()
    if configured_text == "memory_only":
        return None
    if configured_text != "auto":
        return max(1, int(configured))
    allowed = float(total_memory_gb) * float(memory_usage_threshold) - float(reserve_memory_gb)
    value = math.floor(allowed / float(default_estimated_job_memory_gb))
    return max(1, int(value))


def query_gpu_total_memory_gb() -> float:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.total) / (1024.0**3)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            first = result.stdout.strip().splitlines()[0]
            return float(first.strip()) / 1024.0
    except Exception:
        pass
    return 48.0


def query_gpu_free_memory_gb() -> float | None:
    """Return free memory on GPU 0 in GiB, or ``None`` when unavailable."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.free) / (1024.0**3)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            first = result.stdout.strip().splitlines()[0]
            return float(first.strip()) / 1024.0
    except Exception:
        pass
    return None


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)
