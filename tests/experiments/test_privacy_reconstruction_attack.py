from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import model_management.object_detection as object_detection_runtime
from experiments.privacy_reconstruction_attack.attack_dataset import (
    load_attack_samples,
    write_json,
)
from experiments.privacy_reconstruction_attack.boundary_feature_adapter import (
    BoundaryFeatureAdapter,
)
from experiments.privacy_reconstruction_attack.collect_attack_targets import (
    _resolve_split_points,
    configure_object_detection_device,
)
from experiments.privacy_reconstruction_attack.edge_prefix_whitebox import (
    configure_edge_prefix_parameters,
    validate_edge_prefix_matches_manifest,
)
from experiments.privacy_reconstruction_attack.evaluate_privacy_score import (
    _metrics_files,
    _summary_rows,
    evaluate,
)
from experiments.privacy_reconstruction_attack.feature_inversion_attack import (
    _optimise_feature_inversion,
    _payload_feature_loss,
)
from experiments.privacy_reconstruction_attack.reconstruction_metrics import object_metrics


def test_boundary_feature_adapter_single_tensor_zero_distance() -> None:
    adapter = BoundaryFeatureAdapter()
    tensor = torch.randn(1, 3, 4)
    distance = adapter.feature_distance({"payload": tensor}, {"payload": tensor.clone()})
    assert distance.item() == pytest.approx(0.0, abs=1.0e-6)


def test_boundary_feature_adapter_multi_tensor_ignores_metadata_and_nonfloating() -> None:
    adapter = BoundaryFeatureAdapter(tensor_weights={"a": 2.0, "b": 1.0})
    pred = {
        "tensors": {
            "a": torch.ones(1, 2),
            "b": torch.zeros(1, 2),
            "mask": torch.ones(1, 2, dtype=torch.int64),
        },
        "metadata": {"shape": [1, 2]},
    }
    target = {
        "tensors": {
            "a": torch.ones(1, 2),
            "b": torch.ones(1, 2),
            "mask": torch.zeros(1, 2, dtype=torch.int64),
        },
        "metadata": {"shape": [999]},
    }
    distance = adapter.feature_distance(pred, target)
    assert torch.isfinite(distance)
    assert distance.item() > 0.0


def test_boundary_feature_adapter_channel_cosine_for_conv_features() -> None:
    adapter = BoundaryFeatureAdapter(cosine_mode="channel", nmse_weight=0.0)
    pred = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    target = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])

    distance = adapter.feature_distance({"feat": pred}, {"feat": target})

    assert distance.item() == pytest.approx(0.5)


def test_boundary_feature_adapter_mse_for_multi_tensor_payload() -> None:
    adapter = BoundaryFeatureAdapter(
        cosine_weight=0.0,
        nmse_weight=0.0,
        mse_weight=1.0,
        tensor_weights={"a": 2.0, "b": 1.0},
    )
    pred = {"a": torch.zeros(1, 2), "b": torch.zeros(1, 2)}
    target = {"a": torch.ones(1, 2), "b": torch.full((1, 2), 2.0)}

    distance = adapter.feature_distance(pred, target)

    assert distance.item() == pytest.approx(2.0)


def test_feature_inversion_toy_prefix_reduces_feature_loss() -> None:
    class IdentityRuntimeAdapter:
        def to_runtime_input(self, image: torch.Tensor) -> torch.Tensor:
            return image

    class IdentityPrefix:
        def edge_forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"payload": image}

    target = torch.full((1, 3, 4, 4), 0.75)
    adapter = BoundaryFeatureAdapter(cosine_weight=0.0, nmse_weight=0.0, mse_weight=1.0)

    result = _optimise_feature_inversion(
        target_payload={"payload": target},
        splitter=IdentityPrefix(),
        runtime_adapter=IdentityRuntimeAdapter(),
        adapter=adapter,
        input_shape=tuple(target.shape),
        inversion_cfg={
            "iterations": 60,
            "learning_rate": 0.1,
            "adam_eps": 1.0e-8,
            "amsgrad": False,
            "init": "gray",
            "tv_weight": 0.0,
            "l2_weight": 0.0,
            "log_every_n_steps": 100,
        },
        device=torch.device("cpu"),
    )

    assert result.final_feature_loss < result.initial_feature_loss


def test_feature_inversion_prefers_differentiable_training_prefix() -> None:
    class IdentityRuntimeAdapter:
        def to_runtime_input(self, image: torch.Tensor) -> torch.Tensor:
            return image

    class TrainingPrefix:
        def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"payload": image * 2.0}

    class Splitter:
        def _ensure_runtime(self) -> SimpleNamespace:
            return SimpleNamespace(
                segments=SimpleNamespace(training_prefix=TrainingPrefix())
            )

        def edge_forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"payload": image.detach()}

    image = torch.full((1, 3, 4, 4), 0.25, requires_grad=True)
    target = {"payload": torch.ones_like(image)}
    adapter = BoundaryFeatureAdapter(cosine_weight=0.0, nmse_weight=0.0, mse_weight=1.0)

    loss = _payload_feature_loss(
        splitter=Splitter(),
        runtime_adapter=IdentityRuntimeAdapter(),
        adapter=adapter,
        image=image,
        target_payload=target,
        require_grad=True,
    )
    loss.backward()

    assert loss.requires_grad
    assert image.grad is not None
    assert image.grad.abs().sum().item() > 0.0


def test_object_metrics_marks_empty_original_teacher_as_nan() -> None:
    metrics = object_metrics(
        {"boxes": [], "labels": [], "scores": []},
        {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
    )
    assert metrics["ObjectF1"] != metrics["ObjectF1"]


def test_object_metrics_class_aware_iou_matching() -> None:
    metrics = object_metrics(
        {"boxes": [[0, 0, 10, 10]], "labels": [3], "scores": [1.0]},
        {"boxes": [[1, 1, 9, 9]], "labels": [3], "scores": [0.8]},
        iou_threshold=0.5,
    )
    assert metrics["ObjectPrecision"] == pytest.approx(1.0)
    assert metrics["ObjectRecall"] == pytest.approx(1.0)
    assert metrics["ObjectF1"] == pytest.approx(1.0)


def test_object_metrics_normalizes_boxes_when_image_sizes_differ() -> None:
    metrics = object_metrics(
        {
            "boxes": [[0, 0, 100, 100]],
            "labels": [3],
            "scores": [1.0],
            "image_size": [100, 100],
        },
        {
            "boxes": [[0, 0, 50, 50]],
            "labels": [3],
            "scores": [0.8],
            "image_size": [50, 50],
        },
        iou_threshold=0.5,
    )
    assert metrics["ObjectF1"] == pytest.approx(1.0)


def test_summary_rows_ignores_nan_object_f1() -> None:
    rows = [
        {
            "method": "drag",
            "privacy_leakage_score": 0.8,
            "ObjectF1": float("nan"),
            "L_actual": float("nan"),
            "MSE": 1.0,
            "PSNR": 2.0,
            "SSIM": 0.5,
            "LPIPS": float("nan"),
            "FeatureDistanceFinal": 3.0,
        },
        {
            "method": "drag",
            "privacy_leakage_score": 0.8,
            "ObjectF1": 0.25,
            "L_actual": 0.4,
            "MSE": 2.0,
            "PSNR": 4.0,
            "SSIM": 0.7,
            "LPIPS": float("nan"),
            "FeatureDistanceFinal": 5.0,
        },
    ]
    summary = _summary_rows(rows)
    assert summary[0]["num_samples"] == 2
    assert summary[0]["valid_object_f1_samples"] == 1
    assert summary[0]["ObjectF1_mean"] == pytest.approx(0.25)


def test_configure_object_detection_device_updates_runtime_global(monkeypatch) -> None:
    monkeypatch.setattr(object_detection_runtime, "device", torch.device("cuda:0"))
    configure_object_detection_device(torch.device("cpu"))
    assert object_detection_runtime.device == torch.device("cpu")


def test_attack_sample_discovery_accepts_custom_split_names(tmp_path) -> None:
    sample_dir = tmp_path / "custom_split" / "sample_a"
    write_json(
        sample_dir / "metadata.json",
        {
            "sample_id": "sample_a",
            "split_name": "custom_split",
            "split_point": "after:layer",
            "privacy_leakage_score": 0.5,
            "boundary_payload_path": "boundary_payload.pt.gz",
        },
    )
    samples = load_attack_samples(tmp_path)
    assert [sample.split_name for sample in samples] == ["custom_split"]


def test_metrics_file_discovery_accepts_custom_split_names(tmp_path) -> None:
    metrics_path = tmp_path / "custom_split" / "sample_a" / "metrics.json"
    write_json(metrics_path, {"method": "drag", "privacy_leakage_score": 0.5})
    assert _metrics_files(tmp_path) == [metrics_path]


def test_evaluate_accepts_generic_attack_dir_and_keeps_drag_csv_alias(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    attack_dir = tmp_path / "attack"
    write_json(attack_dir / "manifest.json", {"method": "whitebox_feature_inversion"})
    write_json(
        attack_dir / "split_score_0_8" / "sample_a" / "metrics.json",
        {
            "method": "whitebox_feature_inversion",
            "privacy_leakage_score": 0.8,
            "SSIM": 0.5,
            "L_actual": 0.25,
        },
    )

    output_dir = tmp_path / "results"
    evaluate(
        SimpleNamespace(
            config=str(config_path),
            attack_dir=str(attack_dir),
            drag_dir=None,
            output_dir=str(output_dir),
        )
    )

    assert (output_dir / "per_sample.csv").exists()
    assert (output_dir / "drag_per_sample.csv").exists()
    assert "whitebox_feature_inversion" in (output_dir / "summary_by_score.csv").read_text()


def test_auto_split_resolver_picks_unique_nearest_candidates() -> None:
    candidates = [
        SimpleNamespace(candidate_id="after:a", edge_parameter_ratio=0.19),
        SimpleNamespace(candidate_id="after:b", edge_parameter_ratio=0.39),
        SimpleNamespace(candidate_id="after:c", edge_parameter_ratio=0.61),
        SimpleNamespace(candidate_id="after:d", edge_parameter_ratio=0.81),
    ]
    splitter = SimpleNamespace(enumerate_candidates=lambda max_candidates=None: candidates)
    config = {
        "privacy_score_split_points": [
            {"name": "split_score_0_8", "privacy_leakage_score": 0.8, "split_point": "auto"},
            {"name": "split_score_0_6", "privacy_leakage_score": 0.6, "split_point": "auto"},
            {"name": "split_score_0_4", "privacy_leakage_score": 0.4, "split_point": "auto"},
            {"name": "split_score_0_2", "privacy_leakage_score": 0.2, "split_point": "auto"},
        ],
        "split_resolution": {"require_unique": True},
    }
    resolved = _resolve_split_points(splitter, config)
    assert [item.split_point for item in resolved] == ["after:a", "after:b", "after:c", "after:d"]


def test_first_compute_split_resolver_skips_non_compute_candidates() -> None:
    candidates = [
        SimpleNamespace(
            candidate_id="after:to_1_2",
            edge_parameter_ratio=0.0,
            legacy_layer_index=2,
        ),
        SimpleNamespace(
            candidate_id="after:conv2d_1_3",
            edge_parameter_ratio=0.01,
            legacy_layer_index=3,
        ),
        SimpleNamespace(
            candidate_id="after:batchnorm_1_4",
            edge_parameter_ratio=0.02,
            legacy_layer_index=4,
        ),
    ]
    splitter = SimpleNamespace(enumerate_candidates=lambda max_candidates=None: candidates)
    config = {
        "privacy_score_split_points": [
            {
                "name": "split_first_compute",
                "privacy_leakage_score": "auto",
                "split_point": "first_compute",
            },
        ],
        "split_resolution": {"require_unique": True},
    }

    resolved = _resolve_split_points(splitter, config)

    assert resolved[0].split_point == "after:conv2d_1_3"
    assert resolved[0].privacy_leakage_score == pytest.approx(0.99)
    assert resolved[0].actual_privacy_leakage_score == pytest.approx(0.99)


def test_edge_prefix_weights_override_records_sha256(tmp_path) -> None:
    weights = tmp_path / "edge-prefix.pth"
    weights.write_bytes(b"edge prefix parameters")
    runtime_config = SimpleNamespace(
        client=SimpleNamespace(lightweight="toy_edge", weights_path=None)
    )

    info = configure_edge_prefix_parameters(runtime_config, weights)

    assert runtime_config.client.weights_path == str(weights.resolve())
    assert info["whitebox_edge_prefix"] is True
    assert info["source"] == "cli"
    assert info["model_name"] == "toy_edge"
    assert info["resolved_weights_path"] == str(weights.resolve())
    assert info["sha256"] == (
        "ab0ae2c30a76a787db77fe65a354403087d4eae6645af26c6bb37611c68759a9"
    )


def test_edge_prefix_manifest_mismatch_is_rejected() -> None:
    current = {
        "whitebox_edge_prefix": True,
        "model_name": "toy_edge",
        "sha256": "current",
    }
    manifest = {
        "edge_prefix_parameters": {
            "whitebox_edge_prefix": True,
            "model_name": "toy_edge",
            "sha256": "target",
        }
    }

    with pytest.raises(RuntimeError, match="edge-prefix weights differ"):
        validate_edge_prefix_matches_manifest(current, manifest)
