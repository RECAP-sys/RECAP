"""Cloud-side edge registry for tracking connected edge nodes.

Maintains a thread-safe registry of all known edge nodes, their last
heartbeat timestamp, current model version, and active training state.
This enables the cloud to monitor multi-edge deployments and make
scheduling decisions based on edge health.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(slots=True)
class EdgeRecord:
    """Snapshot of one registered edge node."""

    edge_id: int
    model_id: str = ""
    model_version: str = "0"
    last_seen_ms: int = 0
    registered_at_ms: int = 0
    active_job_id: str = ""
    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_jobs_failed: int = 0


class EdgeRegistry:
    """Thread-safe registry that tracks connected edge nodes.

    Edges are registered implicitly when they submit jobs or query
    resources.  The registry records the latest model version reported
    by each edge so that the cloud can detect *stale* training results
    (a completed job whose ``base_model_version`` is older than the
    edge's current version).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._edges: dict[int, EdgeRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def touch(
        self,
        edge_id: int,
        *,
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> EdgeRecord:
        """Update (or create) the record for *edge_id* and bump its
        ``last_seen_ms`` timestamp.  Returns the updated record.
        """
        now_ms = _now_ms()
        with self._lock:
            record = self._edges.get(int(edge_id))
            if record is None:
                record = EdgeRecord(
                    edge_id=int(edge_id),
                    registered_at_ms=now_ms,
                )
                self._edges[int(edge_id)] = record
            record.last_seen_ms = now_ms
            if model_id is not None:
                record.model_id = str(model_id)
            if model_version is not None:
                record.model_version = str(model_version)
            return record

    def record_job_submitted(self, edge_id: int, job_id: str) -> None:
        with self._lock:
            record = self._edges.get(int(edge_id))
            if record is not None:
                record.active_job_id = str(job_id)
                record.total_jobs_submitted += 1

    def record_job_completed(self, edge_id: int, *, success: bool) -> None:
        with self._lock:
            record = self._edges.get(int(edge_id))
            if record is not None:
                record.active_job_id = ""
                if success:
                    record.total_jobs_completed += 1
                else:
                    record.total_jobs_failed += 1

    def get(self, edge_id: int) -> EdgeRecord | None:
        with self._lock:
            return self._edges.get(int(edge_id))

    def get_model_version(self, edge_id: int) -> str:
        """Return the last known model version for *edge_id*, or ``"0"``."""
        with self._lock:
            record = self._edges.get(int(edge_id))
            if record is None:
                return "0"
            return record.model_version

    def all_edges(self) -> list[EdgeRecord]:
        with self._lock:
            return list(self._edges.values())

    def active_edge_count(self) -> int:
        with self._lock:
            return len(self._edges)

    def summary(self) -> dict[str, object]:
        """Return a JSON-friendly summary of all registered edges."""
        with self._lock:
            edges = []
            for record in self._edges.values():
                edges.append(
                    {
                        "edge_id": record.edge_id,
                        "model_id": record.model_id,
                        "model_version": record.model_version,
                        "last_seen_ms": record.last_seen_ms,
                        "active_job_id": record.active_job_id,
                        "total_jobs_submitted": record.total_jobs_submitted,
                        "total_jobs_completed": record.total_jobs_completed,
                        "total_jobs_failed": record.total_jobs_failed,
                    }
                )
            return {
                "total_edges": len(self._edges),
                "edges": edges,
            }


def _now_ms() -> int:
    return int(time.time() * 1000)
