from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cloud.workers.edge_assignment import worker_id_for_edge, workspace_for_worker


@dataclass(slots=True)
class EdgeAssignment:
    edge_id: int
    worker_id: str
    endpoint: str
    workspace_root: str


class EdgeAssignmentStore:
    """Persistent edge -> worker assignment map for sticky routing."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        mode: str,
        worker_workspace_root: str | Path,
    ) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.mode = str(mode)
        self.worker_workspace_root = Path(worker_workspace_root)
        self._lock = threading.Lock()
        self._assignments: dict[int, EdgeAssignment] = {}
        self._load()

    def get(self, edge_id: int) -> EdgeAssignment | None:
        with self._lock:
            return self._assignments.get(int(edge_id))

    def all(self) -> list[EdgeAssignment]:
        with self._lock:
            return list(self._assignments.values())

    def assign(self, *, edge_id: int, endpoint: str) -> EdgeAssignment:
        edge = int(edge_id)
        with self._lock:
            existing = self._assignments.get(edge)
            if existing is not None:
                return existing
            worker_id = worker_id_for_edge(edge)
            assignment = EdgeAssignment(
                edge_id=edge,
                worker_id=worker_id,
                endpoint=str(endpoint),
                workspace_root=workspace_for_worker(self.worker_workspace_root, worker_id),
            )
            self._assignments[edge] = assignment
            self._save_locked()
            return assignment

    def update_endpoint(self, *, edge_id: int, endpoint: str) -> EdgeAssignment:
        edge = int(edge_id)
        with self._lock:
            assignment = self._assignments.get(edge)
            if assignment is None:
                worker_id = worker_id_for_edge(edge)
                assignment = EdgeAssignment(
                    edge_id=edge,
                    worker_id=worker_id,
                    endpoint=str(endpoint),
                    workspace_root=workspace_for_worker(self.worker_workspace_root, worker_id),
                )
                self._assignments[edge] = assignment
                self._save_locked()
                return assignment
            assignment.endpoint = str(endpoint)
            self._save_locked()
            return assignment

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if str(payload.get("run_id", "")) != self.run_id:
            return
        if str(payload.get("mode", "")) != self.mode:
            return
        raw_assignments = dict(payload.get("assignments", {}) or {})
        for edge_key, value in raw_assignments.items():
            if not isinstance(value, dict):
                continue
            try:
                edge_id = int(edge_key)
            except (TypeError, ValueError):
                continue
            worker_id = str(value.get("worker_id") or worker_id_for_edge(edge_id))
            self._assignments[edge_id] = EdgeAssignment(
                edge_id=edge_id,
                worker_id=worker_id,
                endpoint=str(value.get("endpoint", "")),
                workspace_root=str(
                    value.get("workspace_root")
                    or workspace_for_worker(self.worker_workspace_root, worker_id)
                ),
            )

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "mode": self.mode,
            "assignments": {
                str(edge_id): asdict(assignment)
                for edge_id, assignment in sorted(self._assignments.items())
            },
        }
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)
