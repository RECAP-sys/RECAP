from __future__ import annotations

import io
import uuid
import zipfile
from types import SimpleNamespace

from loguru import logger

from common.logging_sanitizer import (
    find_forbidden_log_content,
    format_public_log,
    log_diagnostic_debug,
)
from config.runtime import ClientContinualLearningConfig, ContinualLearningConfig
from edge import transmit
from grpc_server.workspace import prepare_request_workspace


def _capture_logs(action, *, level: str = "DEBUG") -> str:
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level=level,
        format="{level}|{extra}|{message}",
    )
    try:
        action()
    finally:
        logger.remove(sink_id)
    return "".join(messages)


def _zip_payload() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", "{}")
    return buffer.getvalue()


def test_internal_id_logging_is_disabled_by_default() -> None:
    assert ClientContinualLearningConfig().log_internal_ids is False
    assert ContinualLearningConfig().log_internal_ids is False


def test_public_formatter_omits_internal_fields_entirely() -> None:
    message = format_public_log(
        "training accepted",
        {
            "edge": 1,
            "status": "QUEUED",
            "request_id": uuid.uuid4().hex,
            "cache_path": "/home/test/cache/payload.zip",
        },
    )

    assert message == "training accepted: edge=1 status=QUEUED"
    assert find_forbidden_log_content(message) == []


def test_all_public_log_levels_pass_complete_scan() -> None:
    def emit() -> None:
        logger.info("training accepted: edge=1 status=QUEUED")
        logger.warning("runtime fallback selected: split=backbone quality=degraded")
        logger.success("model update complete: version=2 size=12.5 MiB")
        logger.error(
            format_public_log(
                "training failed because request_id was rejected",
                {"reason": "cache_path=/home/test/cache/payload.zip"},
            )
        )

    public_logs = _capture_logs(emit, level="INFO")

    assert all(level in public_logs for level in ("INFO", "WARNING", "SUCCESS", "ERROR"))
    assert find_forbidden_log_content(public_logs) == []


def test_complete_scan_rejects_internal_fields_paths_hashes_and_uuids() -> None:
    forbidden_examples = (
        "request_id=internal-request",
        "artifact=/home/user/models/edge.pth",
        r"weights_path=C:\models\edge.pth",
        r"cache_path=D:\cache\payload.zip",
        "metadata=tmp/training_view.json",
        "sha256=" + "a" * 64,
        "payload=123e4567-e89b-12d3-a456-426614174000",
    )

    for example in forbidden_examples:
        assert find_forbidden_log_content(example), example


def test_diagnostic_fields_only_emit_at_debug_when_enabled() -> None:
    request_id = uuid.uuid4().hex

    disabled = _capture_logs(
        lambda: log_diagnostic_debug(
            False,
            "request details",
            lambda: {"request_id": request_id, "cache_path": "/home/test/cache/data.zip"},
        )
    )
    assert disabled == ""

    enabled = _capture_logs(
        lambda: log_diagnostic_debug(
            True,
            "request details",
            lambda: {"request_id": request_id, "cache_path": "/home/test/cache/data.zip"},
        )
    )
    assert "DEBUG" in enabled
    assert "diagnostic" in enabled
    assert request_id in enabled
    assert "/home/test/cache/data.zip" in enabled


def test_workspace_public_log_has_no_internal_fields_or_paths(tmp_path) -> None:
    payload = _zip_payload()
    public_logs = _capture_logs(
        lambda: prepare_request_workspace(
            tmp_path / "server_workspace",
            edge_id=1,
            request_kind="sample_sync",
            payload_zip=payload,
        ),
        level="INFO",
    )

    assert "Prepared request payload" in public_logs
    assert find_forbidden_log_content(public_logs) == []

    diagnostic_logs = _capture_logs(
        lambda: prepare_request_workspace(
            tmp_path / "diagnostic_workspace",
            edge_id=1,
            request_kind="sample_sync",
            payload_zip=payload,
            log_internal_ids=True,
        )
    )
    assert "DEBUG" in diagnostic_logs
    assert "workspace=" in diagnostic_logs
    assert str(tmp_path) in diagnostic_logs


def test_edge_training_submission_keeps_ids_out_of_public_logs(monkeypatch) -> None:
    request_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex

    class _Stub:
        def __init__(self, _channel) -> None:
            pass

        def submit_training_job(self, _request):
            return SimpleNamespace(
                accepted=True,
                job_id=job_id,
                status="QUEUED",
                queue_position=1,
                message="accepted",
            )

    monkeypatch.setattr(transmit.message_transmission_pb2_grpc, "MessageTransmissionStub", _Stub)
    kwargs = {
        "edge_id": 1,
        "request_id": request_id,
        "job_type": transmit.message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
        "cache_path": "/home/test/cache/payload.zip",
        "payload_zip": b"payload",
        "channel": object(),
    }

    public_logs = _capture_logs(
        lambda: transmit.submit_training_job("127.0.0.1:50051", **kwargs),
        level="INFO",
    )
    assert "accepted=True" in public_logs
    assert find_forbidden_log_content(public_logs) == []

    diagnostic_logs = _capture_logs(
        lambda: transmit.submit_training_job(
            "127.0.0.1:50051",
            **kwargs,
            log_internal_ids=True,
        )
    )
    assert request_id in diagnostic_logs
    assert job_id in diagnostic_logs
    assert "DEBUG" in diagnostic_logs
