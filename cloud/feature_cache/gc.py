from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Mapping
from typing import Any

from loguru import logger

from cloud.feature_cache.shard_reachability import (
    collect_refs_from_pending_annotation,
    collect_refs_from_pending_feature_rebuild,
    collect_refs_from_recent_training_windows,
)
from cloud.feature_cache.types import FeatureCacheGCResult


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


class FeatureCacheGC:
    def __init__(
        self,
        *,
        store_root_dir: str,
        view_root_dir: str | None = None,
        recent_training_window_root_dir: str | None = None,
        staging_root_dir: str | None = None,
        max_live_generations: int = 3,
        dry_run: bool = False,
    ) -> None:
        self.store_root_dir = os.path.abspath(str(store_root_dir))
        self.version_root = self.store_root_dir
        self.view_root_dir = (
            None if view_root_dir in (None, "") else os.path.abspath(str(view_root_dir))
        )
        self.recent_training_window_root_dir = (
            None
            if recent_training_window_root_dir in (None, "")
            else os.path.abspath(str(recent_training_window_root_dir))
        )
        self.staging_root_dir = (
            None if staging_root_dir in (None, "") else os.path.abspath(str(staging_root_dir))
        )
        self.max_live_generations = max(0, int(max_live_generations))
        self.dry_run = bool(dry_run)

    @staticmethod
    def _normalise_live_paths(paths: Iterable[str] | None) -> set[str]:
        return {os.path.abspath(str(path)) for path in list(paths or []) if str(path or "").strip()}

    def _view_live_paths(self) -> set[str]:
        live: set[str] = set()
        if not self.view_root_dir or not os.path.isdir(self.view_root_dir):
            return live
        views = [
            os.path.join(self.view_root_dir, name)
            for name in os.listdir(self.view_root_dir)
            if os.path.isdir(os.path.join(self.view_root_dir, name))
        ]
        views.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        for view_dir in views[: self.max_live_generations]:
            path = os.path.join(view_dir, "metadata_index.json")
            if not os.path.exists(path):
                continue
            try:
                payload = _read_json(path)
            except Exception:
                continue
            for sample in dict(payload.get("samples") or {}).values():
                if not isinstance(sample, Mapping):
                    continue
                ref = sample.get("feature_ref")
                if not isinstance(ref, Mapping):
                    continue
                for key in ("shard_path", "index_path", "shard_dir"):
                    if ref.get(key):
                        live.add(os.path.abspath(str(ref[key])))
                index_path = str(ref.get("index_path") or "")
                if index_path:
                    try:
                        index_payload = _read_json(index_path)
                    except Exception:
                        index_payload = {}
                    metadata_path = index_payload.get("metadata_path") or index_payload.get(
                        "meta_path"
                    )
                    if metadata_path:
                        live.add(os.path.abspath(str(metadata_path)))
                    elif index_path.endswith(".index.json"):
                        live.add(
                            os.path.abspath(
                                index_path[: -len(".index.json")] + ".meta.json"
                            )
                        )
        return live

    def collect(
        self,
        *,
        live_shard_paths: Iterable[str] | None = None,
        dry_run: bool | None = None,
    ) -> FeatureCacheGCResult:
        effective_dry_run = self.dry_run if dry_run is None else bool(dry_run)
        live = self._normalise_live_paths(live_shard_paths)
        live.update(self._view_live_paths())
        if self.recent_training_window_root_dir:
            live.update(
                collect_refs_from_recent_training_windows(
                    self.recent_training_window_root_dir
                )
            )
        if self.staging_root_dir:
            live.update(collect_refs_from_pending_annotation(self.staging_root_dir))
            live.update(collect_refs_from_pending_feature_rebuild(self.staging_root_dir))
        result = FeatureCacheGCResult(dry_run=effective_dry_run)
        if not os.path.isdir(self.version_root):
            logger.info(
                "[FeatureCache][GC] dry_run={} scanned=0 retained=0 deleted=0 deleted_bytes=0",
                effective_dry_run,
            )
            return result
        candidates: list[str] = []
        for root, dirs, files in os.walk(self.version_root):
            for filename in files:
                if filename.endswith((".safetensors", ".index.json", ".meta.json", ".npy")):
                    candidates.append(os.path.abspath(os.path.join(root, filename)))
            for dirname in dirs:
                if dirname.endswith(".tmp"):
                    candidates.append(os.path.abspath(os.path.join(root, dirname)))
        for path in sorted(candidates):
            result.scanned_files += 1
            if path in live or any(
                os.path.commonpath([live_path, path]) == live_path
                for live_path in live
                if os.path.isdir(live_path)
            ):
                result.retained_files += 1
                if len(result.retained_files_preview) < 10:
                    result.retained_files_preview.append(path)
                continue
            size = os.path.getsize(path) if os.path.isfile(path) else 0
            result.orphan_files.append(path)
            if effective_dry_run:
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                result.deleted_files += 1
                result.deleted_bytes += int(size)
            except OSError as exc:
                result.errors[path] = str(exc)
        logger.info(
            "[FeatureCache][GC] dry_run={} scanned={} retained={} orphan={} "
            "deleted={} deleted_bytes={} errors={}",
            effective_dry_run,
            result.scanned_files,
            result.retained_files,
            len(result.orphan_files),
            result.deleted_files,
            result.deleted_bytes,
            len(result.errors),
        )
        return result


__all__ = ["FeatureCacheGC"]
