from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from loguru import logger

from common.experiment_results import (
    EXPERIMENT_METHODS,
    ExperimentIdentity,
    ExperimentJsonlWriter,
    cloud_repository_edge_run_dir,
    cloud_run_dir,
    experiment_root,
    normalize_edge_id_for_count,
    sanitize_relative_path,
    sha256_bytes,
)
from common.video_identity import redact_video_source, resolve_video_identity, video_slug


def detect_log_timezone() -> str:
    configured = str(os.getenv("TZ", "") or "").strip()
    if configured:
        return configured
    timezone_file = Path("/etc/timezone")
    if timezone_file.is_file():
        value = timezone_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "UTC"


def scenario_name_from_video_source(video_source: str) -> str:
    return video_slug(Path(str(video_source or "")).stem) or "unknown_scenario"


def _ordered_methods(methods: object) -> list[str]:
    incoming = list(methods or EXPERIMENT_METHODS) if isinstance(methods, list) else []
    ordered = list(EXPERIMENT_METHODS)
    for method in incoming:
        value = str(method)
        if value not in ordered:
            ordered.append(value)
    return ordered


class CloudExperimentManifestWriter:
    def __init__(
        self,
        *,
        root_dir: str,
        experiment_id: str,
        student_model: str = "",
        teacher_model: str = "",
        log_timezone: str = "",
    ) -> None:
        self.root_dir = str(root_dir)
        self.experiment_id = ExperimentIdentity.create(
            experiment_id=experiment_id,
            scenario_slug="default",
            edge_count=1,
            repeat=1,
            method=EXPERIMENT_METHODS[0],
        ).experiment_id
        self.student_model = str(student_model or "")
        self.teacher_model = str(teacher_model or "")
        self.log_timezone = str(log_timezone or detect_log_timezone())
        self.experiment_dir = experiment_root(self.root_dir, self.experiment_id)
        self.manifest_path = self.experiment_dir / "manifest.yaml"
        self.index_path = self.experiment_dir / "experiment_index.json"
        self._lock = threading.RLock()

    def upsert_cloud_runtime(
        self,
        *,
        method: str,
        scenario_slug: str,
        edge_count: int | str,
        repeat: int | str,
        run_id: str | None = None,
    ) -> None:
        identity = ExperimentIdentity.create(
            experiment_id=self.experiment_id,
            scenario_slug=scenario_slug,
            edge_count=edge_count,
            repeat=repeat,
            method=method,
            run_id=run_id,
        )
        self._upsert(
            identity=identity,
            edge_id=None,
            summary={},
        )

    def upsert_edge_run(
        self,
        *,
        method: str,
        scenario_slug: str,
        edge_count: int | str,
        repeat: int | str,
        run_id: str,
        edge_id: int,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        identity = ExperimentIdentity.create(
            experiment_id=self.experiment_id,
            scenario_slug=scenario_slug,
            edge_count=edge_count,
            repeat=repeat,
            method=method,
            run_id=run_id,
        )
        self._upsert(
            identity=identity,
            edge_id=int(edge_id),
            summary=dict(summary or {}),
        )

    def _upsert(
        self,
        *,
        identity: ExperimentIdentity,
        edge_id: int | None,
        summary: Mapping[str, Any],
    ) -> None:
        with self._lock:
            manifest = self._load()
            video_source = str(summary.get("video_source", "") or "")
            configured_scenario = str(summary.get("scenario_name", "") or identity.scenario_slug)
            configured_slug = str(summary.get("video_slug", "") or "")
            scenario_name = video_slug(configured_scenario)
            resolved_video_slug = video_slug(configured_slug) or identity.scenario_slug
            if video_source:
                video_info = resolve_video_identity(
                    video_source,
                    configured_scenario_name=scenario_name,
                    configured_video_slug=resolved_video_slug,
                )
                video_source = video_info.video_source
                scenario_name = video_info.scenario_name or configured_scenario
                resolved_video_slug = video_info.video_slug
            elif scenario_name or resolved_video_slug:
                scenario_name = scenario_name or resolved_video_slug
                resolved_video_slug = resolved_video_slug or scenario_name

            scenario_name = scenario_name or identity.scenario_slug
            methods = _ordered_methods(manifest.get("methods"))
            if identity.method not in methods:
                methods.append(identity.method)
            manifest["methods"] = methods

            scenarios = list(manifest.get("scenarios") or [])
            scenario = next(
                (
                    item
                    for item in scenarios
                    if str(item.get("scenario_slug", "") or "") == identity.scenario_slug
                ),
                None,
            )
            if scenario is None:
                scenario = {
                    "scenario_name": scenario_name,
                    "scenario_slug": identity.scenario_slug,
                    "video_path": video_source,
                }
                scenarios.append(scenario)
            else:
                scenario.setdefault("scenario_name", scenario_name)
            if video_source:
                existing_source = str(scenario.get("video_path", "") or "")
                scenario["video_path"] = (
                    redact_video_source(existing_source)
                    if existing_source
                    else video_source
                )
            if resolved_video_slug:
                scenario.setdefault("video_slug", resolved_video_slug)
            manifest["scenarios"] = scenarios

            edge_counts = {int(value) for value in list(manifest.get("edge_counts") or [])}
            edge_counts.add(identity.edge_count)
            manifest["edge_counts"] = sorted(edge_counts)

            repeats = {int(value) for value in list(manifest.get("repeats") or [])}
            repeats.add(identity.repeat)
            manifest["repeats"] = sorted(repeats)

            edge_ids_by_count = {
                str(key): list(value or [])
                for key, value in dict(manifest.get("edge_ids_by_count") or {}).items()
            }
            ids_for_count = {
                int(value)
                for value in edge_ids_by_count.get(str(identity.edge_count), [])
            }
            if not ids_for_count:
                ids_for_count.update(range(1, identity.edge_count + 1))
            if edge_id is not None:
                ids_for_count.add(normalize_edge_id_for_count(edge_id, identity.edge_count))
            edge_ids_by_count[str(identity.edge_count)] = sorted(ids_for_count)
            manifest["edge_ids_by_count"] = dict(
                sorted(edge_ids_by_count.items(), key=lambda item: int(item[0]))
            )

            student_model = str(summary.get("student_model", "") or "")
            teacher_model = str(summary.get("teacher_model", "") or "")
            if student_model:
                manifest["student_model"] = student_model
            if teacher_model:
                manifest["teacher_model"] = teacher_model
            self._write(manifest)

    def _load(self) -> dict[str, Any]:
        payload: Any = None
        if self.manifest_path.is_file():
            payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        elif self.index_path.is_file():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        manifest = dict(payload) if isinstance(payload, Mapping) else {}
        if "runs" in manifest:
            raise ValueError("experiment manifest must not define explicit runs")
        manifest["experiment_id"] = self.experiment_id
        manifest["log_timezone"] = str(
            manifest.get("log_timezone") or self.log_timezone or "UTC"
        )
        manifest["methods"] = _ordered_methods(manifest.get("methods"))
        manifest.setdefault("student_model", self.student_model)
        manifest.setdefault("teacher_model", self.teacher_model)
        manifest.setdefault("scenarios", [])
        manifest.setdefault("edge_counts", [])
        manifest.setdefault("repeats", [])
        manifest.setdefault("edge_ids_by_count", {})
        manifest.setdefault(
            "metrics",
            {
                "accuracy_definition": "teacher_supervised_f1",
                "accuracy_file": None,
                "ground_truth_file": None,
            },
        )
        return manifest

    def _write(self, manifest: Mapping[str, Any]) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.manifest_path,
            yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True),
        )
        _atomic_write_text(
            self.index_path,
            json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n",
        )


class CloudExperimentResultRepository:
    def __init__(
        self,
        root_dir: str,
        *,
        max_artifact_bytes: int = 268435456,
        manifest_writer: CloudExperimentManifestWriter | None = None,
    ) -> None:
        self.root_dir = str(root_dir)
        self.max_artifact_bytes = max(1, int(max_artifact_bytes))
        self.manifest_writer = manifest_writer
        self._lock = threading.RLock()
        self._event_writers: dict[tuple[str, str, str, str, str, str], ExperimentJsonlWriter] = {}

    def store_artifacts(self, request: object) -> list[Path]:
        identity = ExperimentIdentity.create(
            experiment_id=getattr(request, "experiment_id", ""),
            scenario_slug=getattr(request, "scenario_slug", ""),
            edge_count=int(getattr(request, "edge_count", 0)),
            repeat=int(getattr(request, "repeat", 0)),
            method=getattr(request, "method", ""),
            run_id=getattr(request, "run_id", ""),
        )
        edge_id = int(getattr(request, "edge_id", 0))
        if edge_id <= 0:
            raise ValueError("edge_id must be a positive integer")
        request_artifacts = list(getattr(request, "artifacts", ()) or ())
        if not request_artifacts:
            raise ValueError("at least one experiment artifact is required")

        run_dir = cloud_repository_edge_run_dir(
            self.root_dir,
            identity.experiment_id,
            identity.scenario_slug,
            identity.edge_count,
            identity.repeat,
            identity.method,
            edge_id,
            identity.run_id,
        )
        summary: dict[str, Any] = {}
        stored_paths: list[Path] = []
        manifest_entries: list[dict[str, Any]] = []
        with self._lock:
            for artifact in request_artifacts:
                self._validate_artifact_identity(
                    artifact,
                    identity=identity,
                    edge_id=edge_id,
                )
                relative_path = sanitize_relative_path(
                    getattr(artifact, "relative_path", "")
                )
                content = bytes(getattr(artifact, "content", b"") or b"")
                declared_size = int(getattr(artifact, "size_bytes", 0))
                if declared_size != len(content):
                    raise ValueError(
                        f"artifact {relative_path} size mismatch: "
                        f"declared={declared_size} actual={len(content)}"
                    )
                if len(content) > self.max_artifact_bytes:
                    raise ValueError(
                        f"artifact {relative_path} exceeds max_artifact_bytes="
                        f"{self.max_artifact_bytes}"
                    )
                digest = sha256_bytes(content)
                declared_digest = str(getattr(artifact, "sha256", "") or "").lower()
                if declared_digest != digest:
                    raise ValueError(f"artifact {relative_path} sha256 mismatch")
                if relative_path.as_posix() == "uploaded_artifacts_manifest.json":
                    manifest_entries.extend(
                        self._client_manifest_metadata_entries(content)
                    )
                    continue
                destination, status = self._store_one(run_dir, relative_path, content, digest)
                stored_paths.append(destination)
                manifest_entries.append(
                    self._manifest_entry(
                        relative_path=relative_path,
                        content=content,
                        digest=digest,
                        artifact=artifact,
                        status=status,
                        stored_path=destination.relative_to(run_dir).as_posix(),
                    )
                )
                if relative_path.as_posix() == "edge_summary.json":
                    try:
                        parsed = json.loads(content.decode("utf-8"))
                        if isinstance(parsed, Mapping):
                            summary = dict(parsed)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        logger.warning("Uploaded edge_summary.json is not valid JSON")
            self._write_uploaded_manifest(
                run_dir,
                identity=identity,
                edge_id=edge_id,
                entries=manifest_entries,
            )
            if self.manifest_writer is not None:
                self.manifest_writer.upsert_edge_run(
                    method=identity.method,
                    scenario_slug=identity.scenario_slug,
                    edge_count=identity.edge_count,
                    repeat=identity.repeat,
                    run_id=identity.run_id,
                    edge_id=edge_id,
                    summary=summary,
                )
            self.record_cloud_event(
                experiment_id=identity.experiment_id,
                scenario_slug=identity.scenario_slug,
                edge_count=identity.edge_count,
                repeat=identity.repeat,
                method=identity.method,
                run_id=identity.run_id,
                event="experiment_result_artifact_received",
                edge_id=edge_id,
                artifact_count=len(request_artifacts),
                stored_paths=[
                    path.relative_to(
                        experiment_root(self.root_dir, identity.experiment_id)
                    ).as_posix()
                    for path in stored_paths
                ],
            )
        return stored_paths

    def record_cloud_event(
        self,
        *,
        experiment_id: str,
        scenario_slug: str,
        edge_count: int | str,
        repeat: int | str,
        method: str,
        run_id: str,
        event: str,
        **payload: Any,
    ) -> None:
        identity = ExperimentIdentity.create(
            experiment_id=experiment_id,
            scenario_slug=scenario_slug,
            edge_count=edge_count,
            repeat=repeat,
            method=method,
            run_id=run_id,
        )
        key = (
            identity.experiment_id,
            identity.scenario_slug,
            str(identity.edge_count),
            str(identity.repeat),
            identity.method,
            identity.run_id,
        )
        writer = self._event_writers.get(key)
        if writer is None:
            writer = ExperimentJsonlWriter(
                cloud_run_dir(
                    self.root_dir,
                    identity.experiment_id,
                    identity.scenario_slug,
                    identity.edge_count,
                    identity.repeat,
                    identity.method,
                    identity.run_id,
                )
                / "cloud_events.jsonl"
            )
            self._event_writers[key] = writer
        writer.write(
            {
                "event": str(event),
                "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
                **payload,
            }
        )

    def _validate_artifact_identity(
        self,
        artifact: object,
        *,
        identity: ExperimentIdentity,
        edge_id: int,
    ) -> None:
        expected = {
            "experiment_id": identity.experiment_id,
            "scenario_slug": identity.scenario_slug,
            "edge_count": identity.edge_count,
            "repeat": identity.repeat,
            "run_id": identity.run_id,
            "method": identity.method,
            "edge_id": edge_id,
        }
        actual = {
            "experiment_id": str(getattr(artifact, "experiment_id", "") or ""),
            "scenario_slug": str(getattr(artifact, "scenario_slug", "") or ""),
            "edge_count": int(getattr(artifact, "edge_count", 0)),
            "repeat": int(getattr(artifact, "repeat", 0)),
            "run_id": str(getattr(artifact, "run_id", "") or ""),
            "method": str(getattr(artifact, "method", "") or ""),
            "edge_id": int(getattr(artifact, "edge_id", 0)),
        }
        if actual != expected:
            raise ValueError(f"artifact identity does not match request: {actual!r}")

    def _store_one(
        self,
        run_dir: Path,
        relative_path: Path,
        content: bytes,
        digest: str,
    ) -> tuple[Path, str]:
        destination = run_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        status = "stored"
        if destination.exists():
            existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if existing_digest == digest:
                return destination, "idempotent"
            status = "overwritten"
            logger.info(
                "Experiment artifact path already exists; overwriting {}",
                destination,
            )
        _atomic_write_bytes(destination, content)
        return destination, status

    @staticmethod
    def _manifest_entry(
        *,
        relative_path: Path,
        content: bytes,
        digest: str,
        artifact: object,
        status: str,
        stored_path: str = "",
    ) -> dict[str, Any]:
        entry = {
            "relative_path": relative_path.as_posix(),
            "stored_path": stored_path,
            "size_bytes": len(content),
            "sha256": digest,
            "content_type": str(
                getattr(artifact, "content_type", "application/octet-stream")
                or "application/octet-stream"
            ),
            "is_final": bool(getattr(artifact, "is_final", False)),
            "status": status,
        }
        return entry

    @staticmethod
    def _client_manifest_metadata_entries(content: bytes) -> list[dict[str, Any]]:
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Uploaded client artifact manifest is not valid JSON")
            return []
        if not isinstance(payload, Mapping):
            return []
        entries: list[dict[str, Any]] = []
        for raw_entry in list(payload.get("artifacts") or []):
            if not isinstance(raw_entry, Mapping):
                continue
            status = str(raw_entry.get("status", "") or "").strip()
            if not status.startswith("skipped"):
                continue
            relative_path = sanitize_relative_path(
                str(raw_entry.get("relative_path", "") or "")
            ).as_posix()
            if relative_path == "uploaded_artifacts_manifest.json":
                continue
            stored_path = str(raw_entry.get("stored_path", "") or "").strip()
            if stored_path:
                stored_path = sanitize_relative_path(stored_path).as_posix()
            entry = {
                "relative_path": relative_path,
                "stored_path": stored_path,
                "size_bytes": max(0, int(raw_entry.get("size_bytes", 0) or 0)),
                "sha256": str(raw_entry.get("sha256", "") or ""),
                "content_type": str(
                    raw_entry.get("content_type", "application/octet-stream")
                    or "application/octet-stream"
                ),
                "is_final": bool(raw_entry.get("is_final", False)),
                "status": status,
            }
            message = str(raw_entry.get("message", "") or "")
            if message:
                entry["message"] = message
            entries.append(entry)
        return entries

    @staticmethod
    def _write_uploaded_manifest(
        run_dir: Path,
        *,
        identity: ExperimentIdentity,
        edge_id: int,
        entries: list[dict[str, Any]],
    ) -> None:
        path = run_dir / "uploaded_artifacts_manifest.json"
        existing_entries: list[dict[str, Any]] = []
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, Mapping):
                    existing_entries = list(existing.get("artifacts") or [])
            except json.JSONDecodeError:
                existing_entries = []
        payload = {
            "experiment_id": identity.experiment_id,
            "scenario_slug": identity.scenario_slug,
            "edge_count": identity.edge_count,
            "repeat": identity.repeat,
            "run_id": identity.run_id,
            "method": identity.method,
            "edge_id": edge_id,
            "artifacts": _deduplicate_manifest_entries(existing_entries + entries),
        }
        _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, content.encode("utf-8"))


def _deduplicate_manifest_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique_by_path: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in entries:
        key = str(entry.get("relative_path", "") or "")
        if key not in unique_by_path:
            order.append(key)
        unique_by_path[key] = entry
    return [unique_by_path[key] for key in order]
