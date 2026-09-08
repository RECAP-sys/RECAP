from __future__ import annotations

import json

from cloud.feature_cache import FeatureCacheGC


def test_feature_cache_gc_deletes_orphans_and_retains_recent_window_refs(tmp_path) -> None:
    store_root = tmp_path / "feature_shards"
    window_root = tmp_path / "recent_training_windows"
    store_root.mkdir()
    window_root.mkdir()

    live_path = store_root / "live.safetensors"
    live_index_path = store_root / "live.index.json"
    live_metadata_path = store_root / "live.meta.json"
    orphan_path = store_root / "orphan.safetensors"
    live_path.write_bytes(b"live")
    live_index_path.write_text(
        json.dumps({"metadata_path": str(live_metadata_path)}),
        encoding="utf-8",
    )
    live_metadata_path.write_text("{}", encoding="utf-8")
    orphan_path.write_bytes(b"orphan")
    (window_root / "recent_training_window.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "feature_ref": {
                            "shard_path": str(live_path),
                            "index_path": str(live_index_path),
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = FeatureCacheGC(
        store_root_dir=str(store_root),
        recent_training_window_root_dir=str(window_root),
        dry_run=False,
    ).collect()

    assert result.deleted_files == 1
    assert result.deleted_bytes == len(b"orphan")
    assert live_path.exists()
    assert live_index_path.exists()
    assert live_metadata_path.exists()
    assert not orphan_path.exists()
