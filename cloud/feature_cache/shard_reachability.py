from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalise_path(path: object) -> str | None:
    if path in (None, ""):
        return None
    return os.path.abspath(str(path))


def _metadata_path(index_path: str | None) -> str | None:
    if not index_path:
        return None
    try:
        payload = _read_json(index_path)
    except Exception:
        payload = {}
    value = payload.get("metadata_path") or payload.get("meta_path")
    if value:
        return _normalise_path(value)
    if index_path.endswith(".index.json"):
        return _normalise_path(index_path[: -len(".index.json")] + ".meta.json")
    return None


def _paths_from_ref(ref: Mapping[str, object]) -> set[str]:
    paths: set[str] = set()
    for key in ("shard_path", "shard_dir", "index_path"):
        path = _normalise_path(ref.get(key))
        if path:
            paths.add(path)
    meta_path = _metadata_path(str(ref.get("index_path") or ""))
    if meta_path:
        paths.add(meta_path)
    return paths


def _collect_refs_from_payload(payload: object) -> set[str]:
    refs: set[str] = set()
    if isinstance(payload, Mapping):
        feature_ref = payload.get("feature_ref")
        if isinstance(feature_ref, Mapping):
            refs.update(_paths_from_ref(feature_ref))
        for value in payload.values():
            refs.update(_collect_refs_from_payload(value))
    elif isinstance(payload, list):
        for value in payload:
            refs.update(_collect_refs_from_payload(value))
    return refs


def _collect_refs_from_json_files(root_dir: str) -> set[str]:
    refs: set[str] = set()
    if not root_dir or not os.path.isdir(root_dir):
        return refs
    for root, _dirs, files in os.walk(root_dir):
        for filename in files:
            if not filename.endswith((".json", ".jsonl")):
                continue
            path = os.path.join(root, filename)
            try:
                if filename.endswith(".jsonl"):
                    with open(path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                refs.update(_collect_refs_from_payload(json.loads(line)))
                else:
                    refs.update(_collect_refs_from_payload(_read_json(path)))
            except Exception:
                continue
    return refs


def collect_refs_from_recent_training_windows(window_root_dir: str) -> set[str]:
    return _collect_refs_from_json_files(os.path.abspath(str(window_root_dir)))


def collect_refs_from_pending_annotation(staging_root: str) -> set[str]:
    root = os.path.join(os.path.abspath(str(staging_root)), "pending_annotation")
    return _collect_refs_from_json_files(root)


def collect_refs_from_pending_feature_rebuild(staging_root: str) -> set[str]:
    root = os.path.join(os.path.abspath(str(staging_root)), "pending_feature_rebuild")
    return _collect_refs_from_json_files(root)


def collect_refs_from_training_views(view_root_dir: str) -> set[str]:
    return _collect_refs_from_json_files(os.path.abspath(str(view_root_dir)))


def is_shard_reachable(
    feature_ref: Mapping[str, object],
    reachable_paths: set[str] | list[str] | tuple[str, ...],
) -> bool:
    live = {_normalise_path(path) for path in list(reachable_paths or [])}
    live.discard(None)
    ref_paths = _paths_from_ref(feature_ref)
    if ref_paths & live:
        return True
    live_dirs = {path for path in live if path and os.path.isdir(path)}
    for ref_path in ref_paths:
        for live_dir in live_dirs:
            try:
                if os.path.commonpath([live_dir, ref_path]) == live_dir:
                    return True
            except ValueError:
                continue
    return False


__all__ = [
    "collect_refs_from_pending_annotation",
    "collect_refs_from_pending_feature_rebuild",
    "collect_refs_from_recent_training_windows",
    "collect_refs_from_training_views",
    "is_shard_reachable",
]
