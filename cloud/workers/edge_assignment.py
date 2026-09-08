from __future__ import annotations

import re
from pathlib import Path


def safe_edge_id(edge_id: int | str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(edge_id).strip())
    return value or "unknown"


def worker_id_for_edge(edge_id: int | str) -> str:
    return f"edge_{safe_edge_id(edge_id)}"


def workspace_for_worker(workspace_root: str | Path, worker_id: str) -> str:
    return str(Path(workspace_root).expanduser().resolve() / str(worker_id))
