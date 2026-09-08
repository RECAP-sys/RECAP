from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cloud.experiment_result_repository import CloudExperimentManifestWriter


def test_manifest_writer_merges_methods_edges_and_preserves_notes(tmp_path: Path) -> None:
    writer = CloudExperimentManifestWriter(
        root_dir=str(tmp_path),
        experiment_id="comparison",
        student_model="student",
        teacher_model="teacher",
        log_timezone="UTC",
    )
    summary = {"video_source": "road.mp4"}
    writer.upsert_edge_run(
        method="recap",
        scenario_slug="road",
        edge_count=2,
        repeat=1,
        run_id="main-r1",
        edge_id=1,
        summary=summary,
    )
    payload = yaml.safe_load(writer.manifest_path.read_text(encoding="utf-8"))
    payload["scenarios"][0]["notes"] = "keep me"
    payload["custom"] = {"owner": "user"}
    writer.manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    writer.upsert_edge_run(
        method="recap",
        scenario_slug="road",
        edge_count=2,
        repeat=1,
        run_id="main-r1",
        edge_id=2,
        summary=summary,
    )
    result = yaml.safe_load(writer.manifest_path.read_text(encoding="utf-8"))
    assert result["experiment_id"] == "comparison"
    assert "runs" not in result
    assert result["edge_counts"] == [2]
    assert result["repeats"] == [1]
    assert result["edge_ids_by_count"] == {"2": [1, 2]}
    assert result["methods"] == ["recap"]
    assert result["scenarios"][0]["notes"] == "keep me"
    assert result["scenarios"][0]["video_slug"] == "road"
    assert result["scenarios"][0]["scenario_slug"] == "road"
    assert result["custom"] == {"owner": "user"}
    assert writer.index_path.is_file()


def test_manifest_writer_redacts_remote_video_credentials(tmp_path: Path) -> None:
    writer = CloudExperimentManifestWriter(
        root_dir=str(tmp_path),
        experiment_id="remote-comparison",
        student_model="student",
        teacher_model="teacher",
        log_timezone="UTC",
    )
    writer.upsert_edge_run(
        method="recap",
        scenario_slug="north-gate",
        edge_count=1,
        repeat=1,
        run_id="remote-r1",
        edge_id=1,
        summary={
            "video_source": "https://user:secret@example.com/live?token=abc",
            "video_slug": "north_gate",
        },
    )

    result = yaml.safe_load(writer.manifest_path.read_text(encoding="utf-8"))
    assert result["scenarios"][0]["video_path"] == "https://example.com/live"


def test_manifest_writer_rejects_edge_ids_above_declared_edge_count(
    tmp_path: Path,
) -> None:
    writer = CloudExperimentManifestWriter(
        root_dir=str(tmp_path),
        experiment_id="comparison",
        student_model="student",
        teacher_model="teacher",
        log_timezone="UTC",
    )

    with pytest.raises(ValueError, match="edge_id must be <= edge_count"):
        writer.upsert_edge_run(
            method="recap",
            scenario_slug="road",
            edge_count=1,
            repeat=1,
            run_id="road_n1_r01_recap",
            edge_id=2,
            summary={"video_source": "road.mp4"},
        )
