from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import torch
from loguru import logger

from config import load_runtime_config
from experiments.privacy_reconstruction_attack.attack_dataset import (
    AttackSample,
    group_samples_by_split,
    load_attack_samples,
    load_experiment_config,
    load_rgb_image,
    load_tensor,
    read_json,
    save_rgb_image,
    save_tensor_image,
    write_json,
)
from experiments.privacy_reconstruction_attack.boundary_feature_adapter import (
    BoundaryFeatureAdapter,
)
from experiments.privacy_reconstruction_attack.collect_attack_targets import (
    _build_trace_splitter,
    configure_object_detection_device,
)
from experiments.privacy_reconstruction_attack.edge_prefix_whitebox import (
    configure_edge_prefix_parameters,
    validate_edge_prefix_matches_manifest,
)
from experiments.privacy_reconstruction_attack.reconstruction_metrics import (
    OptionalLPIPS,
    evaluate_reconstruction,
)
from experiments.privacy_reconstruction_attack.runtime_input_adapter import (
    RuntimeInputAdapter,
    freeze_module,
    prediction_on_reconstruction,
)
from model_management.object_detection import Object_Detection
from model_management.payload import BoundaryPayload

METHOD_NAME = "whitebox_feature_inversion"


@dataclass(frozen=True)
class FeatureInversionResult:
    image: torch.Tensor
    initial_feature_loss: float
    final_feature_loss: float
    final_total_loss: float
    num_iterations: int
    init_label: str


def _metadata_input_shape(sample: AttackSample) -> tuple[int, ...]:
    value = sample.metadata.get("input_size")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Sample {sample.sample_id} is missing metadata.input_size.")
    try:
        shape = tuple(int(dim) for dim in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sample {sample.sample_id} has invalid metadata.input_size: {value!r}."
        ) from exc
    if len(shape) != 4 or int(shape[1]) != 3 or any(int(dim) <= 0 for dim in shape):
        raise ValueError(
            f"Sample {sample.sample_id} has invalid metadata.input_size: {value!r}; "
            "expected positive BCHW with three RGB channels."
        )
    return shape


def _total_variation(image: torch.Tensor) -> torch.Tensor:
    height_tv = image[..., 1:, :].sub(image[..., :-1, :]).square().mean()
    width_tv = image[..., :, 1:].sub(image[..., :, :-1]).square().mean()
    return height_tv + width_tv


def _feature_distance_adapter(
    config: Mapping[str, Any],
    inversion_cfg: Mapping[str, Any],
    *,
    device: torch.device,
) -> BoundaryFeatureAdapter:
    feature_loss = str(inversion_cfg.get("feature_loss", "mse")).lower()
    if feature_loss == "mse":
        adapter_cfg: dict[str, Any] = {
            "cosine_weight": 0.0,
            "nmse_weight": 0.0,
            "mse_weight": 1.0,
            "cosine_mode": "flat",
            "eps": 1.0e-8,
            "tensor_weights": {},
        }
    elif feature_loss in {"configured", "feature_distance"}:
        adapter_cfg = dict(config.get("feature_distance") or {})
    else:
        raise ValueError(
            "whitebox_feature_inversion.feature_loss must be 'mse' or 'configured', "
            f"got {feature_loss!r}."
        )

    for key in (
        "cosine_weight",
        "nmse_weight",
        "mse_weight",
        "cosine_mode",
        "eps",
        "tensor_weights",
    ):
        if key in inversion_cfg:
            adapter_cfg[key] = inversion_cfg[key]
    return BoundaryFeatureAdapter.from_config(adapter_cfg, device=device)


def _initial_image(
    shape: tuple[int, ...],
    inversion_cfg: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    init = str(inversion_cfg.get("init", "gray_noise")).lower()
    seed = inversion_cfg.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    if init in {"gray", "grey", "constant"}:
        image = torch.full(shape, 0.5, device=device, dtype=torch.float32)
    elif init in {"gray_noise", "grey_noise"}:
        image = torch.full(shape, 0.5, device=device, dtype=torch.float32)
        noise_std = float(inversion_cfg.get("init_noise_std", 0.01))
        if noise_std > 0.0:
            image = image + noise_std * torch.randn(
                shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
    elif init in {"random", "random_noise", "noise"}:
        image = torch.rand(
            shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported whitebox_feature_inversion.init: {init!r}.")
    return image.clamp(0.0, 1.0), init


def _payload_feature_loss(
    *,
    splitter: Any,
    runtime_adapter: Any,
    adapter: BoundaryFeatureAdapter,
    image: torch.Tensor,
    target_payload: BoundaryPayload,
    require_grad: bool = False,
) -> torch.Tensor:
    pred_payload = _differentiable_prefix_payload(
        splitter,
        runtime_adapter.to_runtime_input(image),
    )
    loss = adapter.feature_distance(pred_payload, target_payload)
    if require_grad and not loss.requires_grad:
        raise RuntimeError(
            "White-box feature inversion requires a differentiable edge prefix, "
            "but the feature loss has no gradient."
        )
    return loss


def _differentiable_prefix_payload(splitter: Any, runtime_input: torch.Tensor) -> Any:
    if hasattr(splitter, "_ensure_runtime"):
        runtime = splitter._ensure_runtime()
        segments = getattr(runtime, "segments", None)
        training_prefix = getattr(segments, "training_prefix", None)
        if callable(training_prefix):
            return training_prefix(runtime_input)
    return BoundaryFeatureAdapter().model_forward_to_payload(splitter, runtime_input)


def _optimise_feature_inversion(
    *,
    target_payload: BoundaryPayload,
    splitter: Any,
    runtime_adapter: Any,
    adapter: BoundaryFeatureAdapter,
    input_shape: tuple[int, ...],
    inversion_cfg: Mapping[str, Any],
    device: torch.device,
    split_name: str = "",
    sample_id: str = "",
) -> FeatureInversionResult:
    iterations = max(0, int(inversion_cfg.get("iterations", 1000)))
    image, init_label = _initial_image(input_shape, inversion_cfg, device=device)
    image = image.detach().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [image],
        lr=float(inversion_cfg.get("learning_rate", 1.0e-2)),
        eps=float(inversion_cfg.get("adam_eps", 1.0e-3)),
        amsgrad=bool(inversion_cfg.get("amsgrad", True)),
    )
    tv_weight = float(inversion_cfg.get("tv_weight", 1.0e-4))
    l2_weight = float(inversion_cfg.get("l2_weight", 1.0e-5))
    log_every = max(1, int(inversion_cfg.get("log_every_n_steps", 100)))

    with torch.no_grad():
        initial_feature_loss = float(
            _payload_feature_loss(
                splitter=splitter,
                runtime_adapter=runtime_adapter,
                adapter=adapter,
                image=image,
                target_payload=target_payload,
            )
            .detach()
            .cpu()
        )

    final_total_loss = initial_feature_loss
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        feature_loss = _payload_feature_loss(
            splitter=splitter,
            runtime_adapter=runtime_adapter,
            adapter=adapter,
            image=image,
            target_payload=target_payload,
            require_grad=True,
        )
        loss = (
            feature_loss
            + tv_weight * _total_variation(image)
            + l2_weight * image.square().mean()
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            image.clamp_(0.0, 1.0)
        final_total_loss = float(loss.detach().cpu())
        if iteration == 1 or iteration % log_every == 0 or iteration == iterations:
            logger.info(
                "[FeatureInversion] split={} sample={} iter={} feature_loss={:.6g}",
                split_name,
                sample_id,
                iteration,
                float(feature_loss.detach().cpu()),
            )

    with torch.no_grad():
        final_feature_loss = float(
            _payload_feature_loss(
                splitter=splitter,
                runtime_adapter=runtime_adapter,
                adapter=adapter,
                image=image,
                target_payload=target_payload,
            )
            .detach()
            .cpu()
        )
    return FeatureInversionResult(
        image=image.detach().clone(),
        initial_feature_loss=initial_feature_loss,
        final_feature_loss=final_feature_loss,
        final_total_loss=final_total_loss,
        num_iterations=iterations,
        init_label=init_label,
    )


def _save_model_input_reference(
    sample: AttackSample,
    *,
    sample_out: Path,
    runtime_adapter: RuntimeInputAdapter,
    device: torch.device,
) -> None:
    model_input = load_tensor(sample.path("model_input_tensor_path"), device=device)
    reference = runtime_adapter.from_runtime_input(model_input, clamp=True).detach()
    save_tensor_image(sample_out / "model_input_reference.png", reference)


def _run_one_sample(
    sample: AttackSample,
    *,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    adapter: BoundaryFeatureAdapter,
    teacher: Object_Detection,
    inversion_cfg: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    lpips_metric: OptionalLPIPS,
    edge_prefix_parameters: Mapping[str, Any],
) -> None:
    target_payload = adapter.load_payload(sample.path("boundary_payload_path"))
    result = _optimise_feature_inversion(
        target_payload=target_payload,
        splitter=splitter,
        runtime_adapter=runtime_adapter,
        adapter=adapter,
        input_shape=_metadata_input_shape(sample),
        inversion_cfg=inversion_cfg,
        device=device,
        split_name=sample.split_name,
        sample_id=sample.sample_id,
    )

    sample_out = output_dir / sample.split_name / sample.sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    _save_model_input_reference(
        sample,
        sample_out=sample_out,
        runtime_adapter=runtime_adapter,
        device=device,
    )
    reference_rgb = load_rgb_image(sample_out / "model_input_reference.png")
    save_tensor_image(sample_out / "recon.png", result.image)
    raw_rgb = load_rgb_image(sample.path("raw_image_path"))
    save_rgb_image(sample_out / "raw.png", raw_rgb)
    recon_rgb = load_rgb_image(sample_out / "recon.png")

    original_teacher = read_json(sample.path("teacher_prediction_path"))
    teacher_threshold = metrics_cfg.get("teacher_threshold")
    teacher_threshold_value = None if teacher_threshold is None else float(teacher_threshold)
    recon_teacher = prediction_on_reconstruction(
        teacher,
        result.image,
        threshold=teacher_threshold_value,
    )
    metrics = evaluate_reconstruction(
        reference_rgb,
        recon_rgb,
        original_teacher_prediction=original_teacher,
        recon_teacher_prediction=recon_teacher,
        feature_distance_final=result.final_feature_loss,
        lpips_metric=lpips_metric,
        object_iou_threshold=float(metrics_cfg.get("object_iou_threshold", 0.5)),
        psnr_norm_max=float(metrics_cfg.get("psnr_norm_max", 40.0)),
        lpips_norm_max=float(metrics_cfg.get("lpips_norm_max", 1.0)),
    )
    metrics.update(
        {
            "method": METHOD_NAME,
            "reconstruction_mode": METHOD_NAME,
            "split_name": sample.split_name,
            "split_point": sample.split_point,
            "sample_id": sample.sample_id,
            "frame_index": int(sample.metadata.get("frame_index", 0)),
            "privacy_leakage_score": float(sample.privacy_leakage_score),
            "metric_reference": "model_input_reference",
            "FeatureDistanceInitial": float(result.initial_feature_loss),
            "FeatureDistanceFinal": float(result.final_feature_loss),
            "feature_inversion_init": result.init_label,
            "feature_inversion_feature_loss": str(inversion_cfg.get("feature_loss", "mse")),
            "feature_inversion_total_loss_final": float(result.final_total_loss),
            "num_iterations": int(result.num_iterations),
            "whitebox_edge_prefix": bool(
                edge_prefix_parameters.get("whitebox_edge_prefix", False)
            ),
            "edge_prefix_parameters": dict(edge_prefix_parameters),
            "recon_teacher_prediction": recon_teacher,
        }
    )
    write_json(sample_out / "metrics.json", metrics)


def run_feature_inversion(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
    configure_object_detection_device(device)
    config = load_experiment_config(args.config)
    inversion_cfg = dict(config.get("whitebox_feature_inversion") or {})
    if not bool(inversion_cfg.get("enabled", True)):
        logger.info("[FeatureInversion] disabled by config")
        return

    manifest = read_json(Path(args.targets_dir) / "manifest.json")
    runtime_config = load_runtime_config(manifest.get("yaml_path") or "./config/config.yaml")
    edge_prefix_parameters = configure_edge_prefix_parameters(
        runtime_config,
        args.edge_prefix_weights,
    )
    validate_edge_prefix_matches_manifest(edge_prefix_parameters, manifest)
    logger.info(
        "[FeatureInversion] white-box edge prefix model={} source={} sha256={}",
        edge_prefix_parameters.get("model_name"),
        edge_prefix_parameters.get("source"),
        edge_prefix_parameters.get("sha256"),
    )

    student = Object_Detection(runtime_config.client, "small inference")
    teacher = Object_Detection(runtime_config.server, "large inference")
    student.model.to(device).eval()
    teacher.model.to(device).eval()
    freeze_module(student.model)
    freeze_module(teacher.model)

    adapter = _feature_distance_adapter(config, inversion_cfg, device=device)
    metrics_cfg = dict(config.get("metrics") or {})
    lpips_metric = OptionalLPIPS(device=device)
    if not lpips_metric.available:
        logger.warning(
            "[FeatureInversion] LPIPS unavailable; metric will be skipped: {}",
            lpips_metric.error,
        )

    grouped = group_samples_by_split(load_attack_samples(args.targets_dir))
    output_dir = Path(args.output_dir)
    write_json(
        output_dir / "manifest.json",
        {
            "method": METHOD_NAME,
            "targets_dir": str(Path(args.targets_dir).resolve()),
            "target_manifest": str((Path(args.targets_dir) / "manifest.json").resolve()),
            "config": str(Path(args.config).resolve()),
            "edge_prefix_parameters": edge_prefix_parameters,
            "feature_inversion": dict(inversion_cfg),
        },
    )

    for split_name, samples in grouped.items():
        max_samples = int(
            inversion_cfg.get("max_samples_per_split", len(samples)) or len(samples)
        )
        selected = samples[: max(0, max_samples)]
        if not selected:
            continue
        target_shape = _metadata_input_shape(selected[0])
        sample_input = torch.zeros(target_shape, device=device, dtype=torch.float32)
        splitter = _build_trace_splitter(
            student,
            sample_input,
            split_point=selected[0].split_point,
            device=device,
        )
        runtime_adapter = RuntimeInputAdapter(
            student,
            target_shape,
            device=device,
        )
        for sample in selected:
            logger.info(
                "[FeatureInversion] running split={} sample={}",
                split_name,
                sample.sample_id,
            )
            _run_one_sample(
                sample,
                splitter=splitter,
                runtime_adapter=runtime_adapter,
                adapter=adapter,
                teacher=teacher,
                inversion_cfg=inversion_cfg,
                metrics_cfg=metrics_cfg,
                output_dir=output_dir,
                device=device,
                lpips_metric=lpips_metric,
                edge_prefix_parameters=edge_prefix_parameters,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run white-box feature inversion reconstruction."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--edge-prefix-weights",
        default=None,
        help="Exact edge-side lightweight checkpoint used to generate the target payloads.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    run_feature_inversion(build_parser().parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
