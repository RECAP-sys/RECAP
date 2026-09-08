from __future__ import annotations

import time

from loguru import logger


class StageLoggingMixin:
    @staticmethod
    def _preview_ids(sample_ids: list[str], *, limit: int = 5) -> list[str]:
        return [str(sample_id) for sample_id in sample_ids[: max(0, int(limit))]]

    @staticmethod
    def _log_stage_duration(stage: str, started_at: float) -> float:
        elapsed = time.perf_counter() - started_at
        logger.info("[FixedSplitCL] {} took {:.3f}s.", stage, elapsed)
        return elapsed

    @staticmethod
    def _log_stage_elapsed(stage: str, elapsed: float | None) -> float:
        duration = max(0.0, float(elapsed or 0.0))
        logger.info("[FixedSplitCL] {} took {:.3f}s.", stage, duration)
        return duration
