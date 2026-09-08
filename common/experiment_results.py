from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

ArtifactContent = bytes | str | Path

RECAP_METHOD = "recap"
EXPERIMENT_METHODS: tuple[str, ...] = (RECAP_METHOD,)
SUPPORTED_EXPERIMENT_METHODS: tuple[str, ...] = EXPERIMENT_METHODS


def sanitize_component(value: str) -> str:
    component = str(value or "").strip()
    if not component:
        raise ValueError("path component must be non-empty")
    if Path(component).is_absolute():
        raise ValueError(f"absolute path component is not allowed: {component!r}")
    if component in {".", ".."} or ".." in PurePath(component).parts:
        raise ValueError(f"path traversal is not allowed: {component!r}")
    if "/" in component or "\\" in component or "\x00" in component:
        raise ValueError(f"path separators are not allowed in component: {component!r}")
    return component


def sanitize_method(method: str) -> str:
    value = sanitize_component(method)
    if value not in SUPPORTED_EXPERIMENT_METHODS:
        raise ValueError(
            "unknown experiment method "
            f"{value!r}; expected one of {', '.join(SUPPORTED_EXPERIMENT_METHODS)}"
        )
    return value


def normalize_scenario_slug(value: str) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    normalized = "-".join(part for part in raw.replace(" ", "-").split("-") if part)
    return sanitize_component(normalized)


def normalize_edge_count(value: int | str) -> int:
    edge_count = int(value)
    if edge_count <= 0:
        raise ValueError("edge_count must be a positive integer")
    return edge_count


def normalize_edge_id(value: int | str) -> int:
    edge_id = int(value)
    if edge_id <= 0:
        raise ValueError("edge_id must be a positive integer")
    return edge_id


def normalize_edge_id_for_count(edge_id: int | str, edge_count: int | str) -> int:
    resolved_edge_id = normalize_edge_id(edge_id)
    resolved_edge_count = normalize_edge_count(edge_count)
    if resolved_edge_id > resolved_edge_count:
        raise ValueError("edge_id must be <= edge_count")
    return resolved_edge_id


def normalize_repeat(value: int | str) -> int:
    raw = "" if value is None else str(value).strip().lower()
    if raw.startswith("r"):
        raw = raw[1:]
    try:
        repeat = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("repeat must be a positive integer") from exc
    if repeat <= 0:
        raise ValueError("repeat must be a positive integer")
    return repeat


def repeat_label(value: int | str) -> str:
    return f"r{normalize_repeat(value):02d}"


def edge_count_label(value: int | str) -> str:
    return f"n{normalize_edge_count(value)}"


def default_experiment_run_id(
    *,
    scenario_slug: str,
    edge_count: int | str,
    repeat: int | str,
    method: str,
) -> str:
    return sanitize_component(
        f"{normalize_scenario_slug(scenario_slug)}_"
        f"{edge_count_label(edge_count)}_"
        f"{repeat_label(repeat)}_"
        f"{sanitize_method(method)}"
    )


@dataclass(frozen=True)
class ExperimentIdentity:
    experiment_id: str
    scenario_slug: str
    edge_count: int
    repeat: int
    method: str
    run_id: str

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        scenario_slug: str,
        edge_count: int | str,
        repeat: int | str,
        method: str,
        run_id: str | None = None,
    ) -> "ExperimentIdentity":
        resolved_method = sanitize_method(method)
        resolved_scenario = normalize_scenario_slug(scenario_slug)
        resolved_edge_count = normalize_edge_count(edge_count)
        resolved_repeat = normalize_repeat(repeat)
        resolved_run_id = (
            sanitize_component(run_id)
            if str(run_id or "").strip()
            else default_experiment_run_id(
                scenario_slug=resolved_scenario,
                edge_count=resolved_edge_count,
                repeat=resolved_repeat,
                method=resolved_method,
            )
        )
        return cls(
            experiment_id=sanitize_component(experiment_id),
            scenario_slug=resolved_scenario,
            edge_count=resolved_edge_count,
            repeat=resolved_repeat,
            method=resolved_method,
            run_id=resolved_run_id,
        )

    @property
    def repeat_label(self) -> str:
        return repeat_label(self.repeat)

    @property
    def edge_count_label(self) -> str:
        return edge_count_label(self.edge_count)

    @property
    def run_dimension_label(self) -> str:
        return default_experiment_run_id(
            scenario_slug=self.scenario_slug,
            edge_count=self.edge_count,
            repeat=self.repeat,
            method=self.method,
        )

    def raw_logs_relative_dir(self) -> Path:
        return Path("raw_logs") / self.run_dimension_label


def sanitize_relative_path(value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("relative_path must be non-empty")
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"absolute artifact path is not allowed: {raw!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path: {raw!r}")
    if "\\" in raw or "\x00" in raw:
        raise ValueError(f"unsafe artifact path: {raw!r}")
    return Path(*(sanitize_component(part) for part in path.parts))


def experiment_root(root_dir: str, experiment_id: str) -> Path:
    return Path(str(root_dir)).expanduser() / sanitize_component(experiment_id)


def experiment_run_relative_dir(identity: ExperimentIdentity) -> Path:
    return identity.raw_logs_relative_dir()


def cloud_run_dir(
    root_dir: str,
    experiment_id: str,
    scenario_slug: str,
    edge_count: int | str,
    repeat: int | str,
    method: str,
    run_id: str | None = None,
) -> Path:
    identity = ExperimentIdentity.create(
        experiment_id=experiment_id,
        scenario_slug=scenario_slug,
        edge_count=edge_count,
        repeat=repeat,
        method=method,
        run_id=run_id,
    )
    return (
        experiment_root(root_dir, identity.experiment_id)
        / identity.raw_logs_relative_dir()
        / "cloud"
    )


def edge_run_dir(
    root_dir: str,
    experiment_id: str,
    scenario_slug: str,
    edge_count: int | str,
    repeat: int | str,
    method: str,
    edge_id: int,
    run_id: str | None = None,
) -> Path:
    identity = ExperimentIdentity.create(
        experiment_id=experiment_id,
        scenario_slug=scenario_slug,
        edge_count=edge_count,
        repeat=repeat,
        method=method,
        run_id=run_id,
    )
    resolved_edge_id = normalize_edge_id_for_count(edge_id, identity.edge_count)
    return (
        experiment_root(root_dir, identity.experiment_id)
        / identity.raw_logs_relative_dir()
        / f"edge_{resolved_edge_id}"
    )


def cloud_repository_edge_run_dir(
    root_dir: str,
    experiment_id: str,
    scenario_slug: str,
    edge_count: int | str,
    repeat: int | str,
    method: str,
    edge_id: int,
    run_id: str | None = None,
) -> Path:
    identity = ExperimentIdentity.create(
        experiment_id=experiment_id,
        scenario_slug=scenario_slug,
        edge_count=edge_count,
        repeat=repeat,
        method=method,
        run_id=run_id,
    )
    resolved_edge_id = normalize_edge_id_for_count(edge_id, identity.edge_count)
    return (
        experiment_root(root_dir, identity.experiment_id)
        / identity.raw_logs_relative_dir()
        / f"edge_{resolved_edge_id}"
    )


class ExperimentJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)


@dataclass
class ExperimentArtifactManifest:
    experiment_id: str = ""
    run_id: str = ""
    method: str = ""
    edge_id: int = 0
    scenario_slug: str = ""
    edge_count: int = 0
    repeat: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def add_file(
        self,
        *,
        relative_path: str,
        source_path: Path | None = None,
        size_bytes: int | None = None,
        sha256: str = "",
        content_type: str = "application/octet-stream",
        status: str = "included",
        message: str = "",
        stored_path: str = "",
    ) -> dict[str, Any]:
        safe_path = sanitize_relative_path(relative_path).as_posix()
        if source_path is not None and size_bytes is None and source_path.exists():
            size_bytes = source_path.stat().st_size
        entry = {
            "relative_path": safe_path,
            "size_bytes": int(size_bytes or 0),
            "sha256": str(sha256 or ""),
            "content_type": str(content_type or "application/octet-stream"),
            "status": str(status),
        }
        if message:
            entry["message"] = str(message)
        if stored_path:
            entry["stored_path"] = str(stored_path)
        self.artifacts.append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "method": self.method,
            "edge_id": int(self.edge_id),
            "scenario_slug": self.scenario_slug,
            "edge_count": int(self.edge_count),
            "repeat": int(self.repeat),
            "artifacts": list(self.artifacts),
        }

    def write(self, path: Path) -> None:
        _atomic_write_text(
            Path(path),
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "application/json"
    if suffix in {".log", ".txt"}:
        return "text/plain"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".zip":
        return "application/zip"
    return "application/octet-stream"


def collect_edge_artifacts(
    *,
    method: str,
    run_id: str,
    edge_id: int,
    experiment_id: str,
    scenario_slug: str,
    edge_count: int | str,
    repeat: int | str,
    config: object,
    inference_result_path: Path,
    metrics_path: Path | None,
    cache_path: Path | None,
) -> dict[str, ArtifactContent]:
    identity = ExperimentIdentity.create(
        experiment_id=experiment_id,
        scenario_slug=scenario_slug,
        edge_count=edge_count,
        repeat=repeat,
        method=method,
        run_id=run_id,
    )
    resolved_method = identity.method
    max_bytes = max(1, int(getattr(config, "max_artifact_bytes", 268435456)))
    manifest = ExperimentArtifactManifest(
        experiment_id=identity.experiment_id,
        run_id=identity.run_id,
        method=resolved_method,
        edge_id=int(edge_id),
        scenario_slug=identity.scenario_slug,
        edge_count=identity.edge_count,
        repeat=identity.repeat,
    )
    candidates: list[tuple[str, Path | None, bool]] = [
        (
            "latest_inference_results.jsonl",
            Path(inference_result_path),
            True,
        ),
        (
            "metrics.jsonl" if resolved_method != RECAP_METHOD else "edge_metrics.jsonl",
            Path(metrics_path) if metrics_path is not None else None,
            True,
        ),
    ]
    run_dir = Path(inference_result_path).parent
    candidates.append(
        (
            "edge_summary.json",
            run_dir / "edge_summary.json",
            True,
        )
    )
    trigger_candidates = [
        run_dir / "trigger_manifest.json",
        Path(cache_path) / "trigger_manifest.json" if cache_path is not None else None,
    ]
    trigger_path = next(
        (path for path in trigger_candidates if path is not None and path.is_file()),
        None,
    )
    candidates.append(("trigger_manifest.json", trigger_path, True))
    candidates.extend(
        (path.name, path, True) for path in sorted(run_dir.glob("replay_frames_*.zip"))
    )

    artifacts: dict[str, ArtifactContent] = {}
    seen: set[str] = set()
    for relative_path, source, enabled in candidates:
        if not enabled or relative_path in seen:
            continue
        seen.add(relative_path)
        if source is None or not source.is_file():
            continue
        size = source.stat().st_size
        if size > max_bytes:
            manifest.add_file(
                relative_path=relative_path,
                source_path=source,
                size_bytes=size,
                content_type=content_type_for_path(source),
                status="skipped_too_large",
                message=f"artifact exceeds max_artifact_bytes={max_bytes}",
            )
            continue
        stream_from_path = source.suffix.lower() == ".zip"
        content = None if stream_from_path else source.read_bytes()
        digest = sha256_file(source) if stream_from_path else sha256_bytes(content or b"")
        manifest.add_file(
            relative_path=relative_path,
            source_path=source,
            size_bytes=size,
            sha256=digest,
            content_type=content_type_for_path(source),
        )
        artifacts[relative_path] = source if stream_from_path else (content or b"")

    manifest_path = run_dir / "uploaded_artifacts_manifest.json"
    manifest.write(manifest_path)
    artifacts["uploaded_artifacts_manifest.json"] = manifest_path.read_bytes()
    return artifacts


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
