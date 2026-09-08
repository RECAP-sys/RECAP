from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from cloud.orchestration.recent_training_window import RecentTrainingWindowStore
from cloud.orchestration.request_context import RequestContextMixin


def _sample(sample_id: str, frame_id: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "frame_id": int(frame_id),
        "labels": {"boxes": [], "labels": []},
        "feature_ref": {"sample_id": sample_id, "path": f"{sample_id}.npy"},
    }


def test_recent_training_window_appends_dedupes_and_returns_latest_n(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=4)

    first = store.append_samples(
        [_sample("frame-1", 1), _sample("frame-2", 2), _sample("frame-3", 3)],
        sample_source="low_quality",
    )
    assert first.as_dict() == {
        "accepted": 3,
        "replaced": 0,
        "retained": 3,
        "dropped_old": 0,
    }

    replaced = store.append_samples(
        [
            {**_sample("frame-2", 2), "labels": {"boxes": [[1, 2, 3, 4]], "labels": [1]}},
            _sample("frame-4", 4),
        ],
        sample_source="high_quality",
    )

    assert replaced.replaced == 1
    assert store.sample_count() == 4
    assert [sample["sample_id"] for sample in store.latest_samples(2)] == [
        "frame-2",
        "frame-4",
    ]
    assert store.latest_samples(2)[0]["labels"]["labels"] == [1]
    assert "__recent_window_sequence" not in store.latest_samples(1)[0]


def test_recent_training_window_slides_without_consuming_selected_samples(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=3)

    store.append_samples([_sample("frame-1", 1), _sample("frame-2", 2)])
    assert [sample["sample_id"] for sample in store.latest_samples(3)] == [
        "frame-1",
        "frame-2",
    ]

    first_window = store.latest_samples(2)
    second_window = store.latest_samples(2)
    assert first_window == second_window

    slide = store.append_samples([_sample("frame-3", 3), _sample("frame-4", 4)])
    assert slide.dropped_old == 1
    assert [sample["sample_id"] for sample in store.latest_samples(3)] == [
        "frame-2",
        "frame-3",
        "frame-4",
    ]


def test_training_samples_mix_old_anchors_with_recent_samples(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=16)
    store.append_samples([_sample(f"frame-{index}", index) for index in range(16)])

    selected = store.training_samples(8, replay_fraction=0.25)

    assert [sample["sample_id"] for sample in selected[-6:]] == [
        "frame-10",
        "frame-11",
        "frame-12",
        "frame-13",
        "frame-14",
        "frame-15",
    ]
    assert [sample["sample_id"] for sample in selected[:2]] == ["frame-2", "frame-7"]


def test_training_samples_without_replay_returns_latest_samples(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=8)
    store.append_samples([_sample(f"frame-{index}", index) for index in range(8)])

    assert store.training_samples(4, replay_fraction=0.0) == store.latest_samples(4)


def test_training_samples_with_replay_supports_single_sample_windows(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=4)
    store.append_samples([_sample(f"frame-{index}", index) for index in range(4)])

    assert store.training_samples(1, replay_fraction=0.25) == [_sample("frame-3", 3)]


def test_recent_training_window_reset_clears_initial_model_state(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=2)
    store.append_samples([_sample("frame-1", 1), _sample("frame-2", 2)])

    store.reset()

    assert store.latest_samples(2) == []
    assert store.sample_count() == 0


def test_initial_reset_does_not_clear_samples_on_later_session_manifest(tmp_path) -> None:
    class Context(RequestContextMixin):
        pass

    context = Context()
    context.recent_training_window_root = str(tmp_path / "recent_training_windows")
    context.split_contract_root = str(tmp_path / "split_contracts")
    context.edge_model_name = "rfdetr_nano"
    context.training_frame_count = 128
    context.log_internal_ids = False
    context._initial_state_reset_lock = threading.Lock()
    context._initial_state_reset_sessions = {}

    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "0"},
        "front_version": "0",
        "split_config_id": "split-a",
    }
    context._reset_initial_cloud_state_if_needed(
        edge_id=1,
        manifest=manifest,
        model_name="rfdetr_nano",
        fallback_model_version="0",
        allow_without_session=True,
    )
    store = context._recent_training_window_for_manifest(edge_id=1, manifest=manifest)
    store.append_samples(
        [
            {
                "sample_id": "sample-1",
                "frame_id": 1,
                "timestamp_ms": 1000,
                "raw_path": "/tmp/frame.jpg",
            }
        ],
        sample_source="high_quality",
    )

    session_manifest = dict(manifest)
    session_manifest["edge_session_id"] = "session-a"
    context._reset_initial_cloud_state_if_needed(
        edge_id=1,
        manifest=session_manifest,
        model_name="rfdetr_nano",
        fallback_model_version="0",
        allow_without_session=True,
    )

    assert [sample["sample_id"] for sample in store.latest_samples(2)] == ["sample-1"]


def test_initial_reset_clears_window_for_new_session(tmp_path) -> None:
    class Context(RequestContextMixin):
        pass

    context = Context()
    context.recent_training_window_root = str(tmp_path / "recent_training_windows")
    context.split_contract_root = str(tmp_path / "split_contracts")
    context.edge_model_name = "rfdetr_nano"
    context.training_frame_count = 128
    context.log_internal_ids = False
    context._initial_state_reset_lock = threading.Lock()
    context._initial_state_reset_sessions = {}

    manifest = {
        "model": {"model_id": "rfdetr_nano", "model_version": "0"},
        "front_version": "0",
        "split_config_id": "split-a",
        "edge_session_id": "session-a",
    }
    context._reset_initial_cloud_state_if_needed(
        edge_id=1,
        manifest=manifest,
        model_name="rfdetr_nano",
        fallback_model_version="0",
        allow_without_session=True,
    )
    store = context._recent_training_window_for_manifest(edge_id=1, manifest=manifest)
    store.append_samples(
        [
            {
                "sample_id": "sample-1",
                "frame_id": 1,
                "timestamp_ms": 1000,
                "raw_path": "/tmp/frame.jpg",
            }
        ],
        sample_source="high_quality",
    )

    new_session_manifest = dict(manifest)
    new_session_manifest["edge_session_id"] = "session-b"
    context._reset_initial_cloud_state_if_needed(
        edge_id=1,
        manifest=new_session_manifest,
        model_name="rfdetr_nano",
        fallback_model_version="0",
        allow_without_session=True,
    )

    assert store.latest_samples(2) == []


def test_recent_training_window_concurrent_appends_do_not_drop_samples(tmp_path) -> None:
    root = str(tmp_path / "window")
    sample_count = 40

    def append_one(index: int) -> None:
        store = RecentTrainingWindowStore(root, max_samples=sample_count)
        store.append_samples([_sample(f"frame-{index}", index)])

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_one, range(sample_count)))

    store = RecentTrainingWindowStore(root, max_samples=sample_count)
    sample_ids = {str(sample["sample_id"]) for sample in store.latest_samples(sample_count)}
    assert sample_ids == {f"frame-{index}" for index in range(sample_count)}
