from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from cloud.annotation.types import (
    TeacherAnnotationRequest,
    TeacherAnnotationResult,
    TeacherAnnotationStatus,
    TeacherLabelCacheKey,
)
from common.logging_sanitizer import log_diagnostic_debug, safe_error_summary

def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return cleaned[:120] or "unknown"


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}-{int(time.time() * 1000000)}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalise_label_payload(labels: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(labels or {})
    payload: dict[str, Any] = {
        "boxes": list(source.get("boxes") or source.get("pseudo_boxes") or []),
        "labels": list(source.get("labels") or source.get("pseudo_labels") or []),
    }
    scores = source.get("scores")
    if scores is None:
        scores = source.get("pseudo_scores")
    if scores is not None:
        payload["scores"] = list(scores or [])
    for key, value in source.items():
        if key in payload or key in {
            "pseudo_boxes",
            "pseudo_labels",
            "pseudo_scores",
        }:
            continue
        if key.startswith("label_") and value is not None:
            payload[str(key)] = value
    return payload


class TeacherLabelCache:
    def __init__(
        self,
        root_dir: str,
        *,
        enabled: bool = True,
        log_internal_ids: bool = False,
    ) -> None:
        self.root_dir = os.path.abspath(str(root_dir))
        self.enabled = bool(enabled)
        self.log_internal_ids = bool(log_internal_ids)
        self._lock = threading.Lock()
        if self.enabled:
            os.makedirs(self.version_root, exist_ok=True)

    @property
    def version_root(self) -> str:
        return self.root_dir

    def cache_key_for_request(self, request: TeacherAnnotationRequest) -> TeacherLabelCacheKey:
        return request.cache_key()

    def cache_key_digest(self, request: TeacherAnnotationRequest) -> str:
        return self.cache_key_for_request(request).digest

    def label_path(self, key: TeacherLabelCacheKey) -> str:
        digest = key.digest
        prefix = _sanitize_segment(str(key.image_sha1)[:2] or "xx")
        teacher_dir = _sanitize_segment(key.teacher_model_fingerprint)
        return os.path.join(
            self.version_root,
            teacher_dir,
            prefix,
            f"{digest}.json",
        )

    def metadata_path(self, key: TeacherLabelCacheKey) -> str:
        label_path = self.label_path(key)
        return f"{os.path.splitext(label_path)[0]}.meta.json"

    def exists(self, request: TeacherAnnotationRequest) -> bool:
        return self._read_validated_labels(request, log_errors=False) is not None

    def _metadata_matches(
        self,
        request: TeacherAnnotationRequest,
        metadata: Mapping[str, Any],
    ) -> bool:
        key = self.cache_key_for_request(request)
        if str(metadata.get("cache_key") or "") != key.digest:
            return False
        key_payload = metadata.get("key_payload")
        return isinstance(key_payload, Mapping) and dict(key_payload) == key.payload()

    def read(self, request: TeacherAnnotationRequest) -> dict[str, Any] | None:
        return self._read_validated_labels(request, log_errors=True)

    def _read_validated_labels(
        self,
        request: TeacherAnnotationRequest,
        *,
        log_errors: bool,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        key = self.cache_key_for_request(request)
        label_path = self.label_path(key)
        meta_path = self.metadata_path(key)
        if not os.path.exists(label_path) or not os.path.exists(meta_path):
            return None
        try:
            metadata = _read_json(meta_path)
            if not self._metadata_matches(request, metadata):
                return None
            labels = _read_json(label_path)
        except Exception as exc:
            if log_errors:
                logger.warning(
                    "[TeacherAnnotation][CacheMiss] unreadable cache entry: reason={}.",
                    safe_error_summary(exc),
                )
                log_diagnostic_debug(
                    self,
                    "[TeacherAnnotation][CacheMiss] diagnostics",
                    lambda error=exc: {
                        "cache_key": key.digest,
                        "sample_id": request.sample_id,
                        "label_path": label_path,
                        "metadata_path": meta_path,
                        "error": repr(error),
                    },
                )
            return None
        return _normalise_label_payload(labels)

    def write(
        self,
        request: TeacherAnnotationRequest,
        labels: Mapping[str, Any],
        *,
        source: str = "unknown",
    ) -> bool:
        if not self.enabled:
            return False
        key = self.cache_key_for_request(request)
        label_path = self.label_path(key)
        meta_path = self.metadata_path(key)
        payload = _normalise_label_payload(labels)
        metadata = {
            "cache_key": key.digest,
            "key_payload": key.payload(),
            "sample_id": str(request.sample_id),
            "edge_id": str(request.edge_id),
            "model_id": str(request.model_id),
            "image_path": str(request.image_path),
            "source": str(source),
            "created_at": time.time(),
        }
        with self._lock:
            _atomic_json_dump(label_path, payload)
            _atomic_json_dump(meta_path, metadata)
        return True

    def lookup_many(
        self,
        requests: Sequence[TeacherAnnotationRequest],
    ) -> tuple[list[TeacherAnnotationResult], float]:
        started = time.perf_counter()
        results: list[TeacherAnnotationResult] = []
        for request in list(requests or []):
            labels = self.read(request)
            if labels is None:
                results.append(
                    TeacherAnnotationResult(
                        request=request,
                        status=TeacherAnnotationStatus.CACHE_MISS,
                    )
                )
                continue
            results.append(
                TeacherAnnotationResult(
                    request=request,
                    status=TeacherAnnotationStatus.CACHE_HIT,
                    labels=labels,
                )
            )
        return results, time.perf_counter() - started

    def write_many(
        self,
        entries: Mapping[TeacherAnnotationRequest, Mapping[str, Any]]
        | Sequence[tuple[TeacherAnnotationRequest, Mapping[str, Any]]],
        *,
        source: str = "unknown",
    ) -> float:
        started = time.perf_counter()
        if isinstance(entries, Mapping):
            iterable = list(entries.items())
        else:
            iterable = list(entries or [])
        for request, labels in iterable:
            self.write(request, labels, source=source)
        return time.perf_counter() - started
