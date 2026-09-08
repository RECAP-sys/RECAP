from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_NON_SLUG = re.compile(r"[^a-z0-9]+")
_UNDERSCORES = re.compile(r"_+")


def video_slug(value: object) -> str:
    normalized = _NON_SLUG.sub("_", str(value or "").strip().lower())
    return _UNDERSCORES.sub("_", normalized).strip("_")


def is_remote_video_source(value: object) -> bool:
    source = str(value or "").strip()
    if not source:
        return False
    return source.isdigit() or "://" in source


def redact_video_source(value: object) -> str:
    source = str(value or "").strip()
    if not source or source.isdigit() or "://" not in source:
        return source
    parsed = urlsplit(source)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{hostname}:{port}" if hostname and port is not None else hostname
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def video_filename_stem(video_source: object) -> str:
    source = str(video_source or "").strip()
    if not source:
        return ""
    if is_remote_video_source(source):
        parsed = urlsplit(source)
        return Path(parsed.path).stem
    return Path(source).stem


@dataclass(frozen=True)
class VideoIdentity:
    video_source: str
    video_slug: str
    scenario_name: str
    frame_replayable: bool


def resolve_video_identity(
    video_source: object,
    *,
    configured_video_slug: object = "",
    configured_scenario_name: object = "",
    remote_frames_saved: bool = False,
) -> VideoIdentity:
    raw_source = str(video_source or "").strip()
    explicit_slug = video_slug(configured_video_slug)
    explicit_scenario = video_slug(configured_scenario_name)
    remote = is_remote_video_source(raw_source)
    if remote and not (explicit_slug or explicit_scenario):
        raise ValueError(
            "RTSP/camera/URI sources require client.source.video_slug or "
            "client.source.scenario_name"
        )
    derived = video_slug(video_filename_stem(raw_source))
    resolved_slug = explicit_slug or explicit_scenario or derived
    if not resolved_slug:
        raise ValueError(f"cannot derive video_slug from video source: {raw_source!r}")
    resolved_scenario = explicit_scenario or resolved_slug
    return VideoIdentity(
        video_source=redact_video_source(raw_source) if remote else raw_source,
        video_slug=resolved_slug,
        scenario_name=resolved_scenario,
        frame_replayable=not remote or bool(remote_frames_saved),
    )
