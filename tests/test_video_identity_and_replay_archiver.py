from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from common.experiment_results import collect_edge_artifacts
from common.video_identity import redact_video_source, resolve_video_identity, video_slug
from edge.replay_frame_archiver import ReplayFrameArchiver


def test_video_slug_normalizes_filename_and_remote_requires_explicit_name() -> None:
    assert video_slug("City-day.rain #1") == "city_day_rain_1"
    identity = resolve_video_identity("video_data/City-day.rain.mp4")
    assert identity.video_slug == "city_day_rain"
    assert identity.scenario_name == "city_day_rain"
    assert identity.frame_replayable is True
    with pytest.raises(ValueError, match="require"):
        resolve_video_identity("rtsp://camera.example/live")
    remote = resolve_video_identity(
        "rtsp://camera.example/live",
        configured_scenario_name="North Gate",
        remote_frames_saved=True,
    )
    assert remote.video_slug == "north_gate"
    assert remote.frame_replayable is True
    assert (
        redact_video_source(
            "https://camera-user:secret@example.com:8443/live/feed?token=abc#fragment"
        )
        == "https://example.com:8443/live/feed"
    )
    assert remote.video_source == "rtsp://camera.example/live"


def test_replay_frame_archiver_writes_chunked_zip(tmp_path: Path) -> None:
    archiver = ReplayFrameArchiver(
        tmp_path,
        enabled=True,
        queue_size=2,
        archive_chunk_max_bytes=10_000,
    )
    frame = np.full((16, 16, 3), 127, dtype=np.uint8)
    assert archiver.enqueue(7, frame) == "replay_frames/frame_000000007.jpg"
    archives = archiver.close()

    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        assert archive.namelist() == ["replay_frames/frame_000000007.jpg"]

    inference = tmp_path / "latest_inference_results.jsonl"
    inference.write_text("{}\n", encoding="utf-8")
    (tmp_path / "edge_summary.json").write_text("{}\n", encoding="utf-8")
    config = type(
        "Config",
        (),
        {
            "max_artifact_bytes": 100_000,
        },
    )()
    artifacts = collect_edge_artifacts(
        method="recap",
        run_id="recap_camera_001",
        edge_id=1,
        experiment_id="exp_camera_recap_vs_baselines_001",
        scenario_slug="camera",
        edge_count=1,
        repeat=1,
        config=config,
        inference_result_path=inference,
        metrics_path=None,
        cache_path=None,
    )
    assert archives[0].name in artifacts
    assert artifacts[archives[0].name] == archives[0]


def test_replay_archiver_rejects_single_frame_larger_than_zip_limit(
    tmp_path: Path,
) -> None:
    archiver = ReplayFrameArchiver(
        tmp_path,
        enabled=True,
        queue_size=1,
        archive_chunk_max_bytes=700,
    )
    frame = np.full((32, 32, 3), 127, dtype=np.uint8)
    assert archiver.enqueue(1, frame)

    assert archiver.close() == []
    assert "exceeds archive_chunk_max_bytes" in archiver.failures[1]


def test_replay_archiver_does_not_copy_when_queue_capacity_is_reserved(
    tmp_path: Path,
) -> None:
    class _Frame:
        copy_count = 0

        def copy(self):
            self.copy_count += 1
            return self

    archiver = ReplayFrameArchiver(tmp_path, enabled=True, queue_size=1)
    assert archiver._slots.acquire(blocking=False)
    frame = _Frame()
    try:
        assert archiver.enqueue(1, frame) is None
        assert frame.copy_count == 0
    finally:
        archiver._slots.release()
        archiver.close()
