from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuJobEstimate:
    model_name: str = ""
    split_key: str = ""
    batch_size: int = 0
    train_samples: int = 0
    estimated_peak_memory_gb: float = 18.0


class GpuMemoryEstimator:
    def __init__(self, *, default_estimated_job_memory_gb: float = 18.0) -> None:
        self.default_estimated_job_memory_gb = float(default_estimated_job_memory_gb)
        self._observed: dict[tuple[str, str, int], float] = {}

    def estimate(self, job: GpuJobEstimate) -> float:
        key = (str(job.model_name), str(job.split_key), int(job.batch_size or 0))
        observed = self._observed.get(key)
        if observed is not None and observed > 0:
            return float(observed)
        configured = float(job.estimated_peak_memory_gb or 0.0)
        return configured if configured > 0 else self.default_estimated_job_memory_gb

    def observe(
        self,
        *,
        model_name: str,
        split_key: str,
        batch_size: int,
        observed_peak_memory_gb: float,
    ) -> None:
        observed = float(observed_peak_memory_gb or 0.0)
        if observed <= 0:
            return
        self._observed[(str(model_name), str(split_key), int(batch_size or 0))] = observed
