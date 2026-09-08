from __future__ import annotations

import io
import ntpath
import posixpath
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from loguru import logger

from common.logging_sanitizer import log_diagnostic_debug


def normalize_client_cache_path(path: str) -> str:
    if not path:
        return ""
    normalized = path.replace("\\", "/").strip()
    cleaned = posixpath.normpath(normalized)
    return "" if cleaned == "." else cleaned


def reset_workspace_dir(path: str | Path) -> Path:
    target = Path(path)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _ensure_within_root(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"cache path escapes the server workspace: {resolved_path}") from exc
    return resolved_path


def _resolve_requested_cache_path(workspace_root: Path, client_path: str) -> Path:
    normalized = normalize_client_cache_path(client_path)
    if not normalized:
        raise ValueError("cache_path is required when payload_zip is not provided")

    if posixpath.isabs(normalized) or ntpath.isabs(normalized):
        candidate = Path(normalized)
    else:
        candidate = workspace_root / PurePosixPath(normalized)
    resolved = _ensure_within_root(candidate, workspace_root)
    if not resolved.exists():
        raise FileNotFoundError(f"Resolved cache path does not exist: {resolved}")
    return resolved


def _safe_extract_zip_archive(archive: zipfile.ZipFile, destination: Path) -> None:
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename.replace("\\", "/"))
        if not member.filename:
            continue
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe archive member path: {member.filename}")

        target_path = _ensure_within_root(destination / member_path, destination)
        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, "r") as source, target_path.open("wb") as target:
            shutil.copyfileobj(source, target)


def _safe_extract_zip_bytes(payload_zip: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as archive:
        _safe_extract_zip_archive(archive, destination)


def _uploaded_workspace_dir(
    workspace_root: Path,
    *,
    edge_id: int | str,
    request_kind: str,
) -> Path:
    safe_edge_id = (
        "".join(
            character if str(character).isalnum() else "_" for character in str(edge_id).strip()
        )
        or "unknown"
    )
    request_id = uuid.uuid4().hex
    return workspace_root / request_kind / f"edge_{safe_edge_id}" / request_id


def prepare_request_workspace(
    workspace_root: str | Path | None,
    *,
    edge_id: int | str,
    request_kind: str,
    payload_zip: bytes | None,
    client_cache_path: str = "",
    log_internal_ids: bool = False,
) -> Path:
    root = Path(workspace_root or "./cache/server_workspace").resolve()
    root.mkdir(parents=True, exist_ok=True)

    if payload_zip:
        workspace_dir = _uploaded_workspace_dir(
            root,
            edge_id=edge_id,
            request_kind=request_kind,
        )
        reset_workspace_dir(workspace_dir)
        _safe_extract_zip_bytes(payload_zip, workspace_dir)
        logger.info("Prepared request payload: edge={} kind={}.", edge_id, request_kind)
        log_diagnostic_debug(
            log_internal_ids,
            "prepared uploaded request workspace",
            lambda: {
                "workspace": str(workspace_dir),
                "payload_zip_bytes": len(payload_zip or b""),
            },
        )
        return workspace_dir

    resolved = _resolve_requested_cache_path(root, client_cache_path)
    logger.info("Prepared request payload: edge={} kind={}.", edge_id, request_kind)
    log_diagnostic_debug(
        log_internal_ids,
        "resolved client request workspace",
        lambda: {
            "workspace": str(resolved),
            "client_cache_path": client_cache_path,
        },
    )
    return resolved
