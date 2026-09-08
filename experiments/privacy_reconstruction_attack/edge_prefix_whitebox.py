from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_weights_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser().resolve()


def configure_edge_prefix_parameters(
    runtime_config: Any,
    edge_prefix_weights: str | Path | None = None,
) -> dict[str, Any]:
    """Apply explicit edge-prefix weights and return reproducibility metadata.

    The split prefix is cut from the lightweight edge model.  Passing
    ``edge_prefix_weights`` therefore overrides ``runtime_config.client.weights_path``
    before the model is built, making the white-box assumption explicit.
    """

    client = runtime_config.client
    configured_weights = getattr(client, "weights_path", None)
    source = "runtime_config"
    if edge_prefix_weights is not None and str(edge_prefix_weights).strip():
        resolved_override = _resolve_weights_path(edge_prefix_weights)
        if resolved_override is None or not resolved_override.is_file():
            raise FileNotFoundError(f"Edge-prefix weights do not exist: {edge_prefix_weights}")
        client.weights_path = str(resolved_override)
        configured_weights = client.weights_path
        source = "cli"
    elif not configured_weights:
        source = "model_zoo_default"

    resolved_path = _resolve_weights_path(configured_weights)
    sha256 = None
    file_size_bytes = None
    if resolved_path is not None:
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Configured edge-prefix weights do not exist: {resolved_path}")
        sha256 = _sha256_file(resolved_path)
        file_size_bytes = int(resolved_path.stat().st_size)

    return {
        "whitebox_edge_prefix": True,
        "model_name": str(getattr(client, "lightweight", "")),
        "source": source,
        "weights_path": str(configured_weights) if configured_weights else None,
        "resolved_weights_path": str(resolved_path) if resolved_path is not None else None,
        "sha256": sha256,
        "file_size_bytes": file_size_bytes,
        "assumption": (
            "attacker can evaluate and differentiate through the exact edge-side "
            "split prefix used to produce the target boundary payload"
        ),
    }


def validate_edge_prefix_matches_manifest(
    current: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    expected = manifest.get("edge_prefix_parameters")
    if not isinstance(expected, Mapping):
        return

    mismatches: list[str] = []
    expected_model = expected.get("model_name")
    current_model = current.get("model_name")
    if expected_model and current_model and str(expected_model) != str(current_model):
        mismatches.append(f"model_name target={expected_model!r} attack={current_model!r}")

    expected_sha = expected.get("sha256")
    current_sha = current.get("sha256")
    if expected_sha and current_sha and str(expected_sha) != str(current_sha):
        mismatches.append("sha256 target and attack edge-prefix weights differ")
    elif expected_sha and not current_sha:
        mismatches.append("target manifest recorded an edge-prefix sha256 but attack did not")

    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(
            "White-box edge-prefix parameters do not match target collection manifest: "
            f"{joined}."
        )
