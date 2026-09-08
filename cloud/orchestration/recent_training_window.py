from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback.
    fcntl = None


_STATE_FILE = "recent_training_window.json"
_SEQUENCE_KEY = "__recent_window_sequence"
_ADDED_AT_KEY = "__recent_window_added_at"
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class RecentWindowAppendStats:
    accepted: int
    replaced: int
    retained: int
    dropped_old: int

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": int(self.accepted),
            "replaced": int(self.replaced),
            "retained": int(self.retained),
            "dropped_old": int(self.dropped_old),
        }


class RecentTrainingWindowStore:
    """Persistent sliding window of method-eligible training samples."""

    def __init__(self, root_dir: str, *, max_samples: int) -> None:
        self.root_dir = os.path.abspath(str(root_dir))
        self.max_samples = max(1, int(max_samples))
        self.state_path = os.path.join(self.root_dir, _STATE_FILE)

    def append_samples(
        self,
        samples: Sequence[Mapping[str, object]],
        *,
        sample_source: str = "",
    ) -> RecentWindowAppendStats:
        with self._locked():
            state = self._load_state()
            existing = {
                str(sample.get("sample_id") or ""): dict(sample)
                for sample in list(state.get("samples", []) or [])
                if isinstance(sample, Mapping) and str(sample.get("sample_id") or "")
            }
            next_sequence = int(state.get("next_sequence") or 0)
            accepted = 0
            replaced = 0
            now = time.time()
            for raw_sample in list(samples or []):
                if not isinstance(raw_sample, Mapping):
                    continue
                sample = dict(raw_sample)
                sample_id = str(sample.get("sample_id") or "").strip()
                if not sample_id:
                    continue
                if sample_id in existing:
                    replaced += 1
                sample[_SEQUENCE_KEY] = next_sequence
                sample[_ADDED_AT_KEY] = now
                if sample_source and not sample.get("sample_source"):
                    sample["sample_source"] = str(sample_source)
                existing[sample_id] = sample
                next_sequence += 1
                accepted += 1

            ordered = sorted(
                existing.values(),
                key=lambda sample: (
                    int(sample.get(_SEQUENCE_KEY) or 0),
                    str(sample.get("sample_id") or ""),
                ),
            )
            dropped_old = max(0, len(ordered) - self.max_samples)
            retained = ordered[-self.max_samples :]
            self._write_state({"next_sequence": next_sequence, "samples": retained})
            return RecentWindowAppendStats(
                accepted=accepted,
                replaced=replaced,
                retained=len(retained),
                dropped_old=dropped_old,
            )

    def latest_samples(self, count: int | None = None) -> list[dict[str, object]]:
        limit = self.max_samples if count in (None, 0) else max(1, int(count))
        state = self._load_state()
        samples = [
            dict(sample)
            for sample in list(state.get("samples", []) or [])
            if isinstance(sample, Mapping) and str(sample.get("sample_id") or "")
        ]
        samples.sort(
            key=lambda sample: (
                int(sample.get(_SEQUENCE_KEY) or 0),
                str(sample.get("sample_id") or ""),
            )
        )
        selected = samples[-limit:]
        return [_strip_internal_fields(sample) for sample in selected]

    def training_samples(
        self,
        count: int,
        *,
        replay_fraction: float = 0.0,
    ) -> list[dict[str, object]]:
        """Return recent samples plus deterministic anchors from older rounds."""
        limit = max(1, int(count))
        fraction = max(0.0, min(0.999999, float(replay_fraction)))
        state = self._load_state()
        samples = [
            dict(sample)
            for sample in list(state.get("samples", []) or [])
            if isinstance(sample, Mapping) and str(sample.get("sample_id") or "")
        ]
        samples.sort(
            key=lambda sample: (
                int(sample.get(_SEQUENCE_KEY) or 0),
                str(sample.get("sample_id") or ""),
            )
        )
        if len(samples) <= limit or fraction <= 0.0:
            return [_strip_internal_fields(sample) for sample in samples[-limit:]]

        if limit <= 1:
            return [_strip_internal_fields(samples[-1])]

        replay_count = min(limit - 1, max(1, round(limit * fraction)))
        recent_count = limit - replay_count
        older = samples[:-recent_count]
        recent = samples[-recent_count:]
        if len(older) <= replay_count:
            selected = older + recent[-(limit - len(older)) :]
        else:
            step = len(older) / float(replay_count)
            anchors = [
                older[min(len(older) - 1, int((index + 0.5) * step))]
                for index in range(replay_count)
            ]
            selected = anchors + recent
        return [_strip_internal_fields(sample) for sample in selected]

    def sample_count(self) -> int:
        return len(self.latest_samples(self.max_samples))

    def reset(self) -> None:
        with self._locked():
            shutil.rmtree(self.root_dir, ignore_errors=True)

    def _lock_path(self) -> str:
        return f"{self.root_dir}.lock"

    @contextmanager
    def _locked(self):
        lock_path = self._lock_path()
        thread_lock = _thread_lock_for_path(lock_path)
        with thread_lock:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "a+", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> dict[str, Any]:
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _write_state(self, payload: Mapping[str, object]) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".recent_training_window.",
            suffix=".json",
            dir=self.root_dir,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, self.state_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _strip_internal_fields(sample: Mapping[str, object]) -> dict[str, object]:
    clean = dict(sample)
    clean.pop(_SEQUENCE_KEY, None)
    clean.pop(_ADDED_AT_KEY, None)
    return clean


def _thread_lock_for_path(path: str) -> threading.RLock:
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[path] = lock
        return lock


__all__ = ["RecentTrainingWindowStore", "RecentWindowAppendStats"]
