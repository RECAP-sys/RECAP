#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected YAML mapping in {path}.")
    return dict(payload)


def require_mapping(mapping: Mapping[str, Any], key: str, *, context: str = "config") -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{key} must be a mapping.")
    return dict(value)


def require_text(mapping: Mapping[str, Any], key: str, *, context: str) -> str:
    if key not in mapping or mapping.get(key) is None:
        raise ValueError(f"{context}.{key} is required.")
    value = str(mapping.get(key)).strip()
    if not value:
        raise ValueError(f"{context}.{key} must be a non-empty string.")
    return value


def require_int(mapping: Mapping[str, Any], key: str, *, context: str) -> int:
    return int(require_text(mapping, key, context=context))


def require_float(mapping: Mapping[str, Any], key: str, *, context: str) -> float:
    return float(require_text(mapping, key, context=context))


def require_bool(mapping: Mapping[str, Any], key: str, *, context: str) -> bool:
    if key not in mapping:
        raise ValueError(f"{context}.{key} is required.")
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{context}.{key} must be boolean.")


def resolve_project_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_dir(config: Mapping[str, Any]) -> Path:
    run_cfg = require_mapping(config, "run")
    root = Path(require_text(run_cfg, "output_root", context="run"))
    run_id = require_text(run_cfg, "run_id", context="run")
    return PROJECT_ROOT / root / run_id
