from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger

FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "request_id",
        "job_id",
        "session_id",
        "edge_session_id",
        "split_config_id",
        "contract_id",
        "source_contract_id",
        "current_contract_id",
        "feature_layout_id",
        "expected_feature_layout_id",
        "low_quality_feature_layout_id",
        "feature_abi_id",
        "shard_id",
        "view_id",
        "payload_id",
        "workspace_id",
        "workspace_uuid",
        "payload_uuid",
        "graph_signature",
        "split_plan_hash",
        "symbolic_input_schema_hash",
        "runtime_batch_validation_signature",
        "sha1",
        "sha256",
        "cache_key",
        "cache_path",
        "workspace",
        "workspace_path",
        "payload_zip",
        "path",
    }
)

_FIELD_NAME_PARTS = (
    "request_id",
    "job_id",
    "session_id",
    "split_config_id",
    "contract_id",
    "feature_layout_id",
    "feature_abi_id",
    "shard_id",
    "view_id",
    "graph_signature",
    "split_plan_hash",
    "schema_hash",
    "validation_signature",
    "sha1",
    "sha256",
    "cache_key",
    "cache_path",
    "workspace_path",
    "payload_uuid",
    "workspace_uuid",
)

_PATH_VALUE_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:\\|/)[^\s,;)]+|(?:\.{1,2}/)[^\s,;)]+)"
)
_HASH_VALUE_RE = re.compile(r"\b[0-9a-fA-F]{32,64}\b")
_UUID_VALUE_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_LOG_PATH_RE = re.compile(
    r"(?:"
    r"(?<![\w.])/(?:[^\s,;|]+/)*[^\s,;|]+"
    r"|(?<![\w])(?:[A-Za-z]:\\)(?:[^\s,;|]+\\)*[^\s,;|]+"
    r"|(?<![\w.])(?:\.{1,2}[/\\])(?:[^\s,;|]+[/\\])*[^\s,;|]+"
    r")"
)
_LOG_ARTIFACT_PATH_RE = re.compile(
    r"(?i)(?:^|\s|[=(])(?P<path>[^\s,;|]*(?:[/\\])[^\s,;|]*\.(?:pth|pt|json|zip))"
)
_CONTEXTUAL_FIELD_RE = re.compile(
    r"(?i)(?<![\w.])(?P<field>workspace|path)\s*[=:]"
)

DEFAULT_LOG_FORBIDDEN_TOKENS = tuple(
    sorted(
        FORBIDDEN_FIELD_NAMES
        | {
            "sample_id",
            "sample_ids",
            "edge_session_id",
            "runtime_template_key",
            "weights_path",
            "model_path",
            "result_path",
            "temp_path",
            "root_dir",
        }
    )
)
_INTERNAL_TOKEN_RE = re.compile(
    r"(?i)\b(?:"
    + "|".join(
        re.escape(token)
        for token in sorted(DEFAULT_LOG_FORBIDDEN_TOKENS, key=len, reverse=True)
    )
    + r")\b"
)


def should_log_internal_ids(settings: Any) -> bool:
    if isinstance(settings, bool):
        return settings
    return _lookup_bool(settings, "log_internal_ids") or _lookup_bool(
        settings,
        "continual_learning.log_internal_ids",
    )


def should_log_runtime_diagnostics(settings: Any) -> bool:
    return (
        should_log_internal_ids(settings)
        or _lookup_bool(settings, "fixed_split_runtime_diagnostics")
        or _lookup_bool(settings, "continual_learning.fixed_split_runtime_diagnostics")
    )


def summarize_path(path: Any, *, diagnostics: bool = False) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if diagnostics:
        return text
    return os.path.basename(text.rstrip("/\\")) or "<hidden>"


def summarize_hash(value: Any, *, diagnostics: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if diagnostics:
        return text[:8]
    return "<hidden>"


def summarize_internal_id(value: Any, *, diagnostics: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if diagnostics:
        return text[:12]
    return "<hidden>"


def safe_error_summary(exc: BaseException | str | None, *, max_len: int = 180) -> str:
    if exc is None:
        return ""
    if isinstance(exc, BaseException):
        label = type(exc).__name__
        text = str(exc)
    else:
        label = "Error"
        text = str(exc)
    text = _redact_sensitive_text(text).splitlines()[0] if text else ""
    if len(text) > max_len:
        text = f"{text[: max(0, max_len - 3)]}..."
    return f"{label}: {text}" if text else label


def format_public_log(message: str, public_fields: Mapping[str, Any] | None = None) -> str:
    fields = {
        str(key): value
        for key, value in dict(public_fields or {}).items()
        if not is_internal_field_name(str(key))
    }
    if not fields:
        return _redact_sensitive_text(str(message))
    return "{}: {}".format(
        _redact_sensitive_text(str(message)),
        " ".join(f"{key}={_format_public_value(value)}" for key, value in fields.items()),
    )


def format_diagnostic_log(
    message: str,
    diagnostic_fields: Mapping[str, Any] | None = None,
) -> str:
    fields = dict(diagnostic_fields or {})
    if not fields:
        return f"{message}"
    return "{}: {}".format(
        str(message),
        " ".join(f"{key}={_format_diagnostic_value(value)}" for key, value in fields.items()),
    )


def log_diagnostic_debug(
    settings: Any,
    message: str,
    diagnostic_fields: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    *,
    runtime: bool = False,
) -> None:
    enabled = (
        should_log_runtime_diagnostics(settings)
        if runtime
        else should_log_internal_ids(settings)
    )
    if not enabled:
        return
    fields = diagnostic_fields() if callable(diagnostic_fields) else diagnostic_fields
    logger.bind(diagnostic=True).debug(
        "[diagnostics] {}",
        format_diagnostic_log(message, fields),
    )


def is_internal_field_name(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return False
    if normalized in FORBIDDEN_FIELD_NAMES:
        return True
    if normalized.endswith(("_path", "_sha1", "_sha256")):
        return True
    return any(part in normalized for part in _FIELD_NAME_PARTS)


def find_forbidden_log_content(text: str) -> list[str]:
    """Return stable labels for internal fields, hashes, and full paths in public logs."""
    content = str(text or "")
    lowered = content.lower()
    findings: list[str] = []
    for token in DEFAULT_LOG_FORBIDDEN_TOKENS:
        if token in {"workspace", "path"}:
            continue
        if token.lower() in lowered:
            findings.append(f"field:{token}")
    for match in _CONTEXTUAL_FIELD_RE.finditer(content):
        findings.append(f"field:{match.group('field').lower()}")
    if _HASH_VALUE_RE.search(content):
        findings.append("hash:value")
    if _UUID_VALUE_RE.search(content):
        findings.append("uuid:value")
    if _LOG_PATH_RE.search(content) or _LOG_ARTIFACT_PATH_RE.search(content):
        findings.append("path:absolute-or-qualified")
    return sorted(set(findings))


def _lookup_bool(settings: Any, dotted_name: str) -> bool:
    cursor = settings
    for part in dotted_name.split("."):
        if cursor is None:
            return False
        if isinstance(cursor, Mapping):
            cursor = cursor.get(part)
        else:
            cursor = getattr(cursor, part, None)
    return bool(cursor)


def _redact_sensitive_text(text: str) -> str:
    redacted = _PATH_VALUE_RE.sub(lambda match: summarize_path(match.group("path")), str(text))
    redacted = _HASH_VALUE_RE.sub("<hash>", redacted)
    redacted = _UUID_VALUE_RE.sub("<uuid>", redacted)
    return _INTERNAL_TOKEN_RE.sub("<internal>", redacted)


def _format_public_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value)
    return _redact_sensitive_text(text)


def _format_diagnostic_value(value: Any) -> str:
    if isinstance(value, Mapping):
        fields = ", ".join(
            f"{key}: {_format_diagnostic_value(item)}" for key, item in value.items()
        )
        return "{" + fields + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_diagnostic_value(item) for item in value) + "]"
    return str(value)


__all__ = [
    "DEFAULT_LOG_FORBIDDEN_TOKENS",
    "FORBIDDEN_FIELD_NAMES",
    "find_forbidden_log_content",
    "format_diagnostic_log",
    "format_public_log",
    "is_internal_field_name",
    "log_diagnostic_debug",
    "safe_error_summary",
    "should_log_internal_ids",
    "should_log_runtime_diagnostics",
    "summarize_hash",
    "summarize_internal_id",
    "summarize_path",
]
