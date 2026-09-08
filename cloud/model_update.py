from __future__ import annotations

import io
import json
import os
import threading
from collections.abc import Mapping

import torch

from model_management.model_delta_payload import build_state_dict_delta_payload


def _atomic_json_dump(path: str, payload: Mapping[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def serialize_model_update(
    model: torch.nn.Module,
    *,
    model_name: str,
    checkpoint_path: str,
    weights_metadata: Mapping[str, object] | None = None,
    metadata_path: str | None = None,
) -> bytes:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "wb") as handle:
        torch.save(model.state_dict(), handle)

    if weights_metadata is not None:
        if not metadata_path:
            raise ValueError("metadata_path is required when weights_metadata is provided.")
        _atomic_json_dump(metadata_path, weights_metadata)

    base_model_version = "0"
    result_model_version = "1"
    if weights_metadata is not None:
        base_model_version = str(weights_metadata.get("source_base_model_version", "0"))
        result_model_version = str(weights_metadata.get("checkpoint_model_version", "1"))

    payload = build_state_dict_delta_payload(
        model,
        model_name=str(model_name),
        base_model_version=base_model_version,
        result_model_version=result_model_version,
    )
    if weights_metadata is not None:
        payload["weights_metadata"] = dict(weights_metadata)

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return buffer.getvalue()


__all__ = ["serialize_model_update"]
