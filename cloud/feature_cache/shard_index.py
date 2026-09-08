from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def read_shard_index(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


__all__ = ["read_shard_index"]
