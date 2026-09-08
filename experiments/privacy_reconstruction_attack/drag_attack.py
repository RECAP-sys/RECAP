from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import torch
import torch.nn.functional as F
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
from experiments.privacy_reconstruction_attack.linear_feature_decoder import (
    LinearFeatureDecodeResult,
    reconstruct_from_linear_conv_feature,
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

METHOD_NAME = "drag_linear_clean"


def _require_diffusers() -> tuple[Any, Any]:
    try:
        from diffusers import DDIMScheduler, StableDiffusionPipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "DRAG linear clean requires diffusers. Install the project dependencies "
            "or keep drag_linear_clean.enabled=false."
        ) from exc
    return DDIMScheduler, StableDiffusionPipeline


def _torch_dtype(name: object) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    text = str(name or "float16").replace("torch.", "")
    dtype = getattr(torch, text, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name!r}")
    return dtype


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


def _encode_prompt(pipe: Any, device: torch.device) -> torch.Tensor:
    if hasattr(pipe, "encode_prompt"):
        result = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return result[0] if isinstance(result, tuple) else result
    return pipe._encode_prompt("", device, 1, False)


def _decode_latents(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling = float(getattr(getattr(pipe.vae, "config", None), "scaling_factor", 0.18215))
    decoded = pipe.vae.decode(latents / scaling).sample
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def _encode_image_latents(
    pipe: Any,
    image: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    scaling = float(getattr(getattr(pipe.vae, "config", None), "scaling_factor", 0.18215))
    vae_input = image.to(device=pipe.device, dtype=dtype).mul(2.0).sub(1.0)
    posterior = pipe.vae.encode(vae_input).latent_dist
    return posterior.mean * scaling


def _scheduler_step(
    scheduler: Any,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    kwargs = {}
    try:
        if "eta" in inspect.signature(scheduler.step).parameters:
            kwargs["eta"] = float(eta)
    except (TypeError, ValueError):
        pass
    return scheduler.step(noise_pred, timestep, latents, **kwargs).prev_sample


def _load_pipeline(config: Mapping[str, Any], device: torch.device) -> Any:
    DDIMScheduler, StableDiffusionPipeline = _require_diffusers()
    dtype = _torch_dtype(config.get("torch_dtype", "float16"))
    if device.type == "cpu" and dtype == torch.float16:
        logger.warning("[DRAGLinearClean] CPU execution requested with float16; using float32.")
        dtype = torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        str(config.get("checkpoint", "stable-diffusion-v1-5/stable-diffusion-v1-5")),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.vae.eval()
    pipe.unet.eval()
    for module in (pipe.vae, pipe.unet, getattr(pipe, "text_encoder", None)):
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return pipe


def _linear_initial_image(
    sample: AttackSample,
    *,
    target_payload: BoundaryPayload,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    adapter: BoundaryFeatureAdapter,
    regularization: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, LinearFeatureDecodeResult] | None:
    decoded = reconstruct_from_linear_conv_feature(
        target_payload,
        detector=runtime_adapter.detector,
        runtime_adapter=runtime_adapter,
        target_shape=_metadata_input_shape(sample),
        regularization=regularization,
    )
    if decoded is None:
        return None
    image = decoded.image.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_payload = adapter.model_forward_to_payload(
            splitter,
            runtime_adapter.to_runtime_input(image),
        )
        feature_loss = float(adapter.feature_distance(pred_payload, target_payload).detach().cpu())
    return image, feature_loss, decoded


def _initial_latents_from_image(
    pipe: Any,
    image: torch.Tensor,
    *,
    drag_cfg: Mapping[str, Any],
    prompt_embeds: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = int(drag_cfg.get("image_size", image.shape[-1]))
    vae_image = F.interpolate(
        image,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    dtype = prompt_embeds.dtype
    init_latents = _encode_image_latents(pipe, vae_image, dtype=dtype)
    num_steps = int(drag_cfg.get("num_inference_steps", 15))
    pipe.scheduler.set_timesteps(num_steps, device=device)
    strength = float(drag_cfg.get("strength", 0.4))
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"drag_linear_clean.strength must be in [0, 1], got {strength}.")
    if strength <= 0.0:
        return init_latents, pipe.scheduler.timesteps[:0]
    init_timestep = min(max(int(num_steps * strength), 1), num_steps)
    t_start = max(num_steps - init_timestep, 0)
    timesteps = pipe.scheduler.timesteps[t_start:]
    seed = drag_cfg.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        init_latents.shape,
        device=device,
        dtype=init_latents.dtype,
        generator=generator,
    )
    latents = pipe.scheduler.add_noise(init_latents, noise, timesteps[:1])
    return latents, timesteps


def _initial_random_latents(
    pipe: Any,
    *,
    drag_cfg: Mapping[str, Any],
    prompt_embeds: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = int(drag_cfg.get("image_size", 512))
    vae_blocks = getattr(pipe.vae.config, "block_out_channels", [1, 1, 1, 1])
    vae_scale_factor = 2 ** (len(vae_blocks) - 1)
    latent_channels = int(getattr(pipe.unet.config, "in_channels", 4))
    num_steps = int(drag_cfg.get("num_inference_steps", 15))
    pipe.scheduler.set_timesteps(num_steps, device=device)
    seed = drag_cfg.get("fallback_seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn(
        (
            1,
            latent_channels,
            image_size // vae_scale_factor,
            image_size // vae_scale_factor,
        ),
        device=device,
        dtype=prompt_embeds.dtype,
        generator=generator,
    )
    latents = latents * float(getattr(pipe.scheduler, "init_noise_sigma", 1.0))
    return latents, pipe.scheduler.timesteps


def _guided_diffusion_sample(
    sample: AttackSample,
    *,
    pipe: Any,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    adapter: BoundaryFeatureAdapter,
    drag_cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, float, str, float, str]:
    target_payload = adapter.load_payload(sample.path("boundary_payload_path"))
    prompt_embeds = _encode_prompt(pipe, device)
    linear_init = _linear_initial_image(
        sample,
        target_payload=target_payload,
        splitter=splitter,
        runtime_adapter=runtime_adapter,
        adapter=adapter,
        regularization=float(drag_cfg.get("linear_pseudoinverse_regularization", 1.0e-5)),
        device=device,
    )
    if linear_init is None:
        latents, timesteps = _initial_random_latents(
            pipe,
            drag_cfg=drag_cfg,
            prompt_embeds=prompt_embeds,
            device=device,
        )
        init_feature_loss = float("nan")
        init_label = (
            f"random_latent:steps={int(drag_cfg.get('num_inference_steps', 15))}:"
            f"seed={drag_cfg.get('fallback_seed')}"
        )
        latent_init_name = "random_latent"
    else:
        init_image, init_feature_loss, decoded = linear_init
        init_label = (
            f"{decoded.label}:{decoded.module_name}:reg={decoded.regularization:g}:"
            f"feature_mse={decoded.feature_mse:.6g}"
        )
        latents, timesteps = _initial_latents_from_image(
            pipe,
            init_image,
            drag_cfg=drag_cfg,
            prompt_embeds=prompt_embeds,
            device=device,
        )
        latent_init_name = "linear_pseudoinverse"
    guidance_rate = float(drag_cfg.get("guidance_rate", 0.2))
    max_grad_norm = float(drag_cfg.get("max_grad_norm", 0.02))
    eta = float(drag_cfg.get("eta", 1.0))
    l2_weight = float(drag_cfg.get("l2_regularization_x", 0.01))
    log_every = max(1, int(drag_cfg.get("log_every_n_steps", 5)))
    final_feature_loss = init_feature_loss

    for step_index, timestep in enumerate(timesteps, start=1):
        latents = latents.detach().requires_grad_(True)
        image = _decode_latents(pipe, latents).float()
        attack_image = F.interpolate(
            image,
            size=(runtime_adapter.height, runtime_adapter.width),
            mode="bilinear",
            align_corners=False,
        )
        pred_payload = adapter.model_forward_to_payload(
            splitter,
            runtime_adapter.to_runtime_input(attack_image),
        )
        feature_loss = adapter.feature_distance(pred_payload, target_payload)
        loss = feature_loss + l2_weight * image.square().mean()
        grad = torch.autograd.grad(loss, latents)[0]
        grad_norm = grad.float().norm()
        if max_grad_norm > 0.0:
            scale = min(1.0, max_grad_norm / (float(grad_norm.detach().cpu()) + 1.0e-12))
            grad = grad * scale
        latents = (latents - guidance_rate * grad).detach()
        final_feature_loss = float(feature_loss.detach().cpu())
        with torch.no_grad():
            noise_pred = pipe.unet(latents, timestep, encoder_hidden_states=prompt_embeds).sample
            latents = _scheduler_step(pipe.scheduler, noise_pred, timestep, latents, eta)
        if (
            step_index == 1
            or step_index % log_every == 0
            or step_index == len(timesteps)
        ):
            logger.info(
                "[DRAGLinearClean] split={} sample={} step={} feature_loss={:.6g}",
                sample.split_name,
                sample.sample_id,
                step_index,
                final_feature_loss,
            )

    with torch.no_grad():
        image = _decode_latents(pipe, latents).float()
        image = F.interpolate(
            image,
            size=(runtime_adapter.height, runtime_adapter.width),
            mode="bilinear",
            align_corners=False,
        )
    return image.detach(), final_feature_loss, init_label, init_feature_loss, latent_init_name


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
    pipe: Any,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    adapter: BoundaryFeatureAdapter,
    teacher: Object_Detection,
    drag_cfg: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    lpips_metric: OptionalLPIPS,
    edge_prefix_parameters: Mapping[str, Any],
) -> None:
    try:
        result = _guided_diffusion_sample(
            sample,
            pipe=pipe,
            splitter=splitter,
            runtime_adapter=runtime_adapter,
            adapter=adapter,
            drag_cfg=drag_cfg,
            device=device,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "DRAG linear clean ran out of memory. Reduce num_inference_steps, "
                "image_size, torch_dtype, or max_samples_per_split."
            ) from exc
        raise
    image, final_feature_loss, init_label, init_feature_loss, latent_init_name = result

    sample_out = output_dir / sample.split_name / sample.sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    _save_model_input_reference(
        sample,
        sample_out=sample_out,
        runtime_adapter=runtime_adapter,
        device=device,
    )
    reference_rgb = load_rgb_image(sample_out / "model_input_reference.png")
    save_tensor_image(sample_out / "recon.png", image)
    raw_rgb = load_rgb_image(sample.path("raw_image_path"))
    save_rgb_image(sample_out / "raw.png", raw_rgb)
    recon_rgb = load_rgb_image(sample_out / "recon.png")
    original_teacher = read_json(sample.path("teacher_prediction_path"))
    teacher_threshold = metrics_cfg.get("teacher_threshold")
    teacher_threshold_value = None if teacher_threshold is None else float(teacher_threshold)
    recon_teacher = prediction_on_reconstruction(
        teacher,
        image,
        threshold=teacher_threshold_value,
    )
    metrics = evaluate_reconstruction(
        reference_rgb,
        recon_rgb,
        original_teacher_prediction=original_teacher,
        recon_teacher_prediction=recon_teacher,
        feature_distance_final=final_feature_loss,
        lpips_metric=lpips_metric,
        object_iou_threshold=float(metrics_cfg.get("object_iou_threshold", 0.5)),
        psnr_norm_max=float(metrics_cfg.get("psnr_norm_max", 40.0)),
        lpips_norm_max=float(metrics_cfg.get("lpips_norm_max", 1.0)),
    )
    metrics.update(
        {
            "method": METHOD_NAME,
            "reconstruction_mode": METHOD_NAME,
            "drag_latent_init": latent_init_name,
            "drag_strength": float(drag_cfg.get("strength", 0.4)),
            "init_label": init_label,
            "init_feature_loss": float(init_feature_loss),
            "linear_decoder_label": init_label
            if latent_init_name == "linear_pseudoinverse"
            else None,
            "linear_init_feature_loss": float(init_feature_loss)
            if latent_init_name == "linear_pseudoinverse"
            else None,
            "split_name": sample.split_name,
            "split_point": sample.split_point,
            "sample_id": sample.sample_id,
            "frame_index": int(sample.metadata.get("frame_index", 0)),
            "privacy_leakage_score": float(sample.privacy_leakage_score),
            "metric_reference": "model_input_reference",
            "whitebox_edge_prefix": bool(
                edge_prefix_parameters.get("whitebox_edge_prefix", False)
            ),
            "edge_prefix_parameters": dict(edge_prefix_parameters),
            "recon_teacher_prediction": recon_teacher,
        }
    )
    write_json(sample_out / "metrics.json", metrics)


def run_drag_linear_clean(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
    configure_object_detection_device(device)
    config = load_experiment_config(args.config)
    drag_cfg = dict(config.get("drag_linear_clean") or {})
    if not bool(drag_cfg.get("enabled", True)):
        logger.info("[DRAGLinearClean] disabled by config")
        return
    manifest = read_json(Path(args.targets_dir) / "manifest.json")
    runtime_config = load_runtime_config(manifest.get("yaml_path") or "./config/config.yaml")
    edge_prefix_parameters = configure_edge_prefix_parameters(
        runtime_config,
        args.edge_prefix_weights,
    )
    validate_edge_prefix_matches_manifest(edge_prefix_parameters, manifest)
    logger.info(
        "[DRAGLinearClean] white-box edge prefix model={} source={} sha256={}",
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
    adapter = BoundaryFeatureAdapter.from_config(
        config.get("feature_distance") or {},
        device=device,
    )
    metrics_cfg = dict(config.get("metrics") or {})
    lpips_metric = OptionalLPIPS(device=device)
    if not lpips_metric.available:
        logger.warning(
            "[DRAGLinearClean] LPIPS unavailable; metric will be skipped: {}",
            lpips_metric.error,
        )
    pipe = _load_pipeline(drag_cfg, device)

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
            "latent_init": "linear_pseudoinverse_or_random_latent",
        },
    )
    for split_name, samples in grouped.items():
        max_samples = int(drag_cfg.get("max_samples_per_split", len(samples)) or len(samples))
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
                "[DRAGLinearClean] running split={} sample={}",
                split_name,
                sample.sample_id,
            )
            _run_one_sample(
                sample,
                pipe=pipe,
                splitter=splitter,
                runtime_adapter=runtime_adapter,
                adapter=adapter,
                teacher=teacher,
                drag_cfg=drag_cfg,
                metrics_cfg=metrics_cfg,
                output_dir=output_dir,
                device=device,
                lpips_metric=lpips_metric,
                edge_prefix_parameters=edge_prefix_parameters,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DRAG with clean linear-pseudoinverse latent initialization."
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
    run_drag_linear_clean(build_parser().parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
