from __future__ import annotations

import pytest

from cloud.contracts import validate_low_quality_manifest, validate_runtime_contract
from model_management.split_contract import build_runtime_contract, classify_contract_compatibility


def _runtime_contract() -> dict[str, object]:
    return build_runtime_contract(
        logical_split_id="after:test",
        trace_signature="trace-a",
        trace_device_type="cpu",
        runtime_backend="torchlens_native",
        boundary_tensor_labels=["boundary"],
        boundary_schema={
            "boundary": {
                "canonical_id": "boundary",
                "torchlens_label": "boundary",
                "module_path": "fake",
                "op_type": "conv",
                "shape": (1, 2, 3),
                "dtype": "torch.float32",
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            }
        },
        model_id="yolo26n",
        model_version="1",
        input_tensor_shape=[1, 3, 32, 32],
        input_resize_mode="direct_resize",
        feature_layout={"boundary": {"shape": [1, 2, 3], "dtype": "float32"}},
    )


def _low_quality_manifest() -> dict[str, object]:
    runtime_contract = _runtime_contract()
    return {
        "model_id": "yolo26n",
        "model_version": "1",
        "split_config_id": "split-a",
        "input_tensor_shape": [1, 3, 32, 32],
        "input_resize_mode": "direct_resize",
        "runtime_contract": runtime_contract,
        "split_plan": {
            "split_config_id": "split-a",
            "runtime_contract": runtime_contract,
        },
    }


def test_low_quality_manifest_requires_runtime_contract() -> None:
    manifest = _low_quality_manifest()
    manifest.pop("runtime_contract")

    with pytest.raises(RuntimeError, match="runtime_contract"):
        validate_low_quality_manifest(manifest)


@pytest.mark.parametrize("field_name", ("feature_abi_id", "feature_abi_spec"))
def test_low_quality_manifest_requires_matching_feature_abi(field_name: str) -> None:
    manifest = _low_quality_manifest()
    manifest_contract = dict(manifest["runtime_contract"])
    if field_name == "feature_abi_id":
        manifest_contract[field_name] = "different-feature-abi"
    else:
        manifest_contract[field_name] = {"different": "feature-abi-spec"}
    manifest["runtime_contract"] = manifest_contract

    with pytest.raises(RuntimeError, match=field_name):
        validate_low_quality_manifest(manifest)


@pytest.mark.parametrize("field_name", ("feature_abi_id", "feature_abi_spec"))
def test_runtime_contract_requires_current_feature_abi(field_name: str) -> None:
    runtime_contract = _runtime_contract()
    runtime_contract.pop(field_name)

    with pytest.raises(RuntimeError, match=field_name):
        validate_runtime_contract(runtime_contract)


def test_contract_compatibility_does_not_rebuild_missing_feature_abi() -> None:
    current = _runtime_contract()
    missing_abi = dict(current)
    missing_abi.pop("feature_abi_id")

    compatibility = classify_contract_compatibility(missing_abi, current)

    assert compatibility["compatible"] is False
    assert compatibility["reason"] == "missing_edge_feature_abi_id"


@pytest.mark.parametrize(
    ("target", "field_name"),
    (("manifest", "protocol_version"), ("split_plan", "plan_version")),
)
def test_low_quality_manifest_rejects_removed_version_fields(
    target: str,
    field_name: str,
) -> None:
    manifest = _low_quality_manifest()
    if target == "manifest":
        manifest[field_name] = "removed"
    else:
        manifest["split_plan"] = {**dict(manifest["split_plan"]), field_name: "removed"}

    with pytest.raises(RuntimeError, match=field_name):
        validate_low_quality_manifest(manifest)
