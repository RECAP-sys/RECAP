from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from common.experiment_results import collect_edge_artifacts
from edge import experiment_result_uploader
from edge.experiment_result_uploader import ExperimentResultUploader
from grpc_server import message_transmission_pb2

IDENTITY = {
    "experiment_id": "comparison",
    "scenario_slug": "road",
    "edge_count": 1,
    "repeat": 1,
}


class _Channel:
    def close(self) -> None:
        return None


def test_uploader_sends_one_artifact_per_rpc_and_marks_final(monkeypatch) -> None:
    requests = []

    class _Stub:
        def __init__(self, channel) -> None:
            del channel

        def UploadExperimentResult(self, request, timeout):
            assert timeout == 30.0
            requests.append(request)
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=True,
                message="ok",
            )

    monkeypatch.setattr(
        experiment_result_uploader.grpc,
        "insecure_channel",
        lambda *args, **kwargs: _Channel(),
    )
    monkeypatch.setattr(
        experiment_result_uploader.message_transmission_pb2_grpc,
        "MessageTransmissionStub",
        _Stub,
    )
    summary = json.dumps({"offline_result_archival": True}).encode()
    uploader = ExperimentResultUploader("cloud:50051", enabled=True)
    assert uploader.upload_run_artifacts(
        **IDENTITY,
        run_id="pure-r1",
        method="recap",
        edge_id=1,
        artifacts={
            "latest_inference_results.jsonl": b"{}\n",
            "edge_summary.json": summary,
            "uploaded_artifacts_manifest.json": b"ignored",
        },
    )
    assert len(requests) == 2
    assert requests[0].experiment_id == "comparison"
    assert requests[0].scenario_slug == "road"
    assert requests[0].edge_count == 1
    assert requests[0].repeat == 1
    assert not requests[0].artifacts[0].is_final
    assert requests[1].artifacts[0].is_final
    assert json.loads(requests[1].artifacts[0].content)["offline_result_archival"] is True


def test_uploader_sends_client_manifest_when_it_contains_skipped_metadata(
    monkeypatch,
) -> None:
    requests = []

    class _Stub:
        def __init__(self, channel) -> None:
            del channel

        def UploadExperimentResult(self, request, timeout):
            del timeout
            requests.append(request)
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=True,
                message="ok",
            )

    monkeypatch.setattr(
        experiment_result_uploader.grpc,
        "insecure_channel",
        lambda *args, **kwargs: _Channel(),
    )
    monkeypatch.setattr(
        experiment_result_uploader.message_transmission_pb2_grpc,
        "MessageTransmissionStub",
        _Stub,
    )
    manifest = json.dumps(
        {
            "artifacts": [
                {
                    "relative_path": "latest_inference_results.jsonl",
                    "status": "skipped_too_large",
                    "size_bytes": 10_000,
                }
            ]
        }
    ).encode()
    uploader = ExperimentResultUploader("cloud:50051", enabled=True)

    assert uploader.upload_run_artifacts(
        **IDENTITY,
        run_id="run-1",
        method="recap",
        edge_id=1,
        artifacts={
            "edge_summary.json": b"{}\n",
            "uploaded_artifacts_manifest.json": manifest,
        },
    )

    assert [request.artifacts[0].relative_path for request in requests] == [
        "edge_summary.json",
        "uploaded_artifacts_manifest.json",
    ]
    assert not requests[0].artifacts[0].is_final
    assert requests[1].artifacts[0].is_final


def test_uploader_failure_is_reported_without_raising(monkeypatch) -> None:
    class _Stub:
        def __init__(self, channel) -> None:
            del channel

        def UploadExperimentResult(self, request, timeout):
            del request, timeout
            raise RuntimeError("offline")

    monkeypatch.setattr(
        experiment_result_uploader.grpc,
        "insecure_channel",
        lambda *args, **kwargs: _Channel(),
    )
    monkeypatch.setattr(
        experiment_result_uploader.message_transmission_pb2_grpc,
        "MessageTransmissionStub",
        _Stub,
    )
    uploader = ExperimentResultUploader("cloud:50051", enabled=True)
    assert not uploader.upload_run_artifacts(
        **IDENTITY,
        run_id="run-1",
        method="recap",
        edge_id=1,
        artifacts={"latest_inference_results.jsonl": b"{}\n"},
    )


def test_uploader_reads_path_backed_artifact_when_it_is_sent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    requests = []

    class _Stub:
        def __init__(self, channel) -> None:
            del channel

        def UploadExperimentResult(self, request, timeout):
            del timeout
            requests.append(request)
            return message_transmission_pb2.UploadExperimentResultResponse(
                accepted=True,
                message="ok",
            )

    monkeypatch.setattr(
        experiment_result_uploader.grpc,
        "insecure_channel",
        lambda *args, **kwargs: _Channel(),
    )
    monkeypatch.setattr(
        experiment_result_uploader.message_transmission_pb2_grpc,
        "MessageTransmissionStub",
        _Stub,
    )
    archive = tmp_path / "replay_frames_0001.zip"
    archive.write_bytes(b"zip-content")

    uploader = ExperimentResultUploader("cloud:50051", enabled=True)
    assert uploader.upload_run_artifacts(
        **IDENTITY,
        run_id="run-1",
        method="recap",
        edge_id=1,
        artifacts={archive.name: archive},
    )
    assert requests[0].artifacts[0].content == b"zip-content"


def test_collect_artifacts_records_oversized_file_without_uploading_it(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inference = run_dir / "latest_inference_results.jsonl"
    inference.write_bytes(b"12345")
    (run_dir / "edge_summary.json").write_text("{}", encoding="utf-8")
    artifacts = collect_edge_artifacts(
        method="recap",
        run_id="run-1",
        edge_id=1,
        **IDENTITY,
        config=SimpleNamespace(max_artifact_bytes=4),
        inference_result_path=inference,
        metrics_path=None,
        cache_path=None,
    )
    assert "latest_inference_results.jsonl" not in artifacts
    manifest = json.loads(
        (run_dir / "uploaded_artifacts_manifest.json").read_text(encoding="utf-8")
    )
    skipped = next(
        item
        for item in manifest["artifacts"]
        if item["relative_path"] == "latest_inference_results.jsonl"
    )
    assert skipped["status"] == "skipped_too_large"
