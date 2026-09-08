from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import grpc
from loguru import logger

from common.experiment_results import (
    ArtifactContent,
    ExperimentIdentity,
    content_type_for_path,
    sanitize_relative_path,
    sha256_bytes,
)
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.grpc_options import grpc_message_options


class ExperimentResultUploader:
    def __init__(
        self,
        server_ip: str,
        enabled: bool,
        timeout_sec: float = 30.0,
    ) -> None:
        self.server_ip = str(server_ip or "")
        self.enabled = bool(enabled)
        self.timeout_sec = max(0.1, float(timeout_sec))

    def upload_run_artifacts(
        self,
        *,
        experiment_id: str,
        scenario_slug: str,
        edge_count: int | str,
        repeat: int | str,
        run_id: str,
        method: str,
        edge_id: int,
        artifacts: Mapping[str, ArtifactContent],
    ) -> bool:
        if not self.enabled:
            return False
        identity = ExperimentIdentity.create(
            experiment_id=experiment_id,
            scenario_slug=scenario_slug,
            edge_count=edge_count,
            repeat=repeat,
            method=method,
            run_id=run_id,
        )
        resolved_edge_id = int(edge_id)
        manifest_content = artifacts.get("uploaded_artifacts_manifest.json")
        upload_items = [
            (relative_path, content)
            for relative_path, content in artifacts.items()
            if relative_path != "uploaded_artifacts_manifest.json"
        ]
        if _should_upload_client_manifest(manifest_content):
            upload_items.append(("uploaded_artifacts_manifest.json", manifest_content))
        if not upload_items:
            logger.info("No experiment artifacts selected for upload.")
            return True

        channel = grpc.insecure_channel(
            self.server_ip,
            options=grpc_message_options(),
        )
        try:
            stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
            for index, (relative_path, raw_content) in enumerate(upload_items):
                safe_relative_path = sanitize_relative_path(relative_path)
                if isinstance(raw_content, Path):
                    content = raw_content.read_bytes()
                elif isinstance(raw_content, str):
                    content = raw_content.encode("utf-8")
                else:
                    content = bytes(raw_content)
                artifact = message_transmission_pb2.ExperimentResultArtifact(
                    experiment_id=identity.experiment_id,
                    scenario_slug=identity.scenario_slug,
                    edge_count=identity.edge_count,
                    repeat=identity.repeat,
                    run_id=identity.run_id,
                    method=identity.method,
                    edge_id=resolved_edge_id,
                    relative_path=safe_relative_path.as_posix(),
                    content=content,
                    size_bytes=len(content),
                    sha256=sha256_bytes(content),
                    content_type=content_type_for_path(safe_relative_path),
                    is_final=index == len(upload_items) - 1,
                )
                request = message_transmission_pb2.UploadExperimentResultRequest(
                    experiment_id=identity.experiment_id,
                    scenario_slug=identity.scenario_slug,
                    edge_count=identity.edge_count,
                    repeat=identity.repeat,
                    run_id=identity.run_id,
                    method=identity.method,
                    edge_id=resolved_edge_id,
                    artifacts=[artifact],
                )
                reply = stub.UploadExperimentResult(
                    request,
                    timeout=self.timeout_sec,
                )
                if not bool(getattr(reply, "accepted", False)):
                    logger.warning(
                        "Experiment artifact upload rejected: path={} message={}",
                        safe_relative_path,
                        getattr(reply, "message", ""),
                    )
                    return False
            logger.info(
                "Uploaded {} offline experiment artifact(s): method={} run_id={} edge={}",
                len(upload_items),
                identity.method,
                identity.run_id,
                resolved_edge_id,
            )
            return True
        except Exception as exc:
            logger.warning("Experiment artifact upload failed: {}", exc)
            return False
        finally:
            channel.close()


def _should_upload_client_manifest(content: ArtifactContent | None) -> bool:
    if content is None:
        return False
    if isinstance(content, Path):
        raw = content.read_bytes()
    elif isinstance(content, str):
        raw = content.encode("utf-8")
    else:
        raw = bytes(content)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else None
    if not isinstance(artifacts, list):
        return False
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            continue
        status = str(entry.get("status", "") or "").strip()
        if status.startswith("skipped"):
            return True
    return False
