from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from loguru import logger

import model_management.model_zoo as model_zoo
from cloud.orchestration.runtime_stage import (
    FIXED_SPLIT_DYNAMIC_BATCH_MIN as _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
)
from cloud.orchestration.runtime_stage import (
    cloud_fixed_split_trace_batch_size as _cloud_fixed_split_trace_batch_size,
)
from cloud.orchestration.runtime_stage import (
    negotiate_cached_split_runtime_batch_size as _negotiate_cached_split_runtime_batch_size,
)
from cloud.training import (
    FixedSplitRetrainEngine,
    FixedSplitTrainingContext,
    FixedSplitTrainingPlan,
    ProxyEvalConfig,
    get_training_adapter,
)
from cloud.training.types import FixedSplitTrainingResult
from model_management.split_model_adapters import (
    build_split_training_loss,
    get_split_runtime_model,
)
from model_management.universal_model_split import (
    SplitRetrainProfile,
    UniversalModelSplitter,
    collect_suffix_trainable_parameters,
)


class FixedSplitTrainingStage:
    def __init__(self, engine: FixedSplitRetrainEngine | None = None) -> None:
        self.engine = engine or FixedSplitRetrainEngine()

    def run(self, context: FixedSplitTrainingContext) -> FixedSplitTrainingResult:
        return self.engine.run(context)


class TrainingStageMixin:
    def _resolve_fixed_split_learning_rate(
        self,
        model_name: str,
    ) -> float:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family == "tinynext":
            learning_rate = self.tinynext_fixed_split_learning_rate
        elif model_family == "rfdetr":
            learning_rate = self.rfdetr_fixed_split_learning_rate
        else:
            learning_rate = self.wrapper_fixed_split_learning_rate

        return float(learning_rate)

    def _resolve_fixed_split_target_steps_per_round(
        self,
        model_name: str,
    ) -> int | None:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family == "tinynext":
            return max(1, int(self.tinynext_fixed_split_target_steps_per_round))
        if model_family == "rfdetr":
            return max(1, int(self.rfdetr_fixed_split_target_steps_per_round))
        if model_family == "yolo":
            return max(1, int(self.yolo_fixed_split_target_steps_per_round))
        return None

    @staticmethod
    def _resolve_fixed_split_training_label(
        model_name: str,
    ) -> str:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family == "tinynext":
            return "TinyNeXt"
        if model_family == "rfdetr":
            return "RF-DETR"
        if model_family == "yolo":
            return str(model_name)
        return str(model_name)

    @staticmethod
    def _fixed_split_optimizer_overrides(
        model_name: str,
    ) -> dict[str, object]:
        model_family = model_zoo.get_model_family(str(model_name))
        if model_family in {"rfdetr", "yolo", "tinynext"}:
            return {
                "optimizer_name": "adamw",
                "weight_decay": 1e-4,
                "grad_clip_norm": 1.0,
                "shuffle_samples": True,
            }
        return {}

    @staticmethod
    def _count_manifest_training_samples(manifest: Mapping[str, object]) -> int:
        count = 0
        for sample in manifest.get("samples", []):
            if not isinstance(sample, Mapping):
                continue
            if str(sample.get("sample_id", "")).strip():
                count += 1
        return count

    def _resolve_fixed_split_runtime_batch_size(
        self,
        model_name: str,
        *,
        num_train_samples: int,
    ) -> int:
        configured_batch_size = max(
            _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
            int(self.batch_size),
        )
        target_steps = self._resolve_fixed_split_target_steps_per_round(model_name)
        if target_steps is None:
            return configured_batch_size
        effective_batch_size = min(
            configured_batch_size,
            max(
                _FIXED_SPLIT_DYNAMIC_BATCH_MIN,
                math.ceil(max(0, int(num_train_samples)) / target_steps),
            ),
        )
        return int(effective_batch_size)

    def _negotiate_cached_split_runtime_batch_size(
        self,
        *,
        current_model_name: str,
        training_cache_path: str,
        all_sample_ids: Sequence[object],
        gt_annotations: Mapping[object, object],
        prepared_splitter: UniversalModelSplitter | None,
        prepared_candidate,
        configured_batch_size: int,
        preloaded_records: Mapping[object, Mapping[str, object]] | None,
        manifest: Mapping[str, object],
    ) -> int:
        if prepared_splitter is None or not training_cache_path or not all_sample_ids:
            return max(1, int(configured_batch_size))

        model_family = model_zoo.get_model_family(str(current_model_name))
        split_plan = dict(manifest.get("split_plan", {}) or {})
        trace_batch_size = _cloud_fixed_split_trace_batch_size(
            split_plan,
            model_family=model_family,
            default=self.trace_batch_size,
        )
        return _negotiate_cached_split_runtime_batch_size(
            model_name=str(current_model_name),
            training_cache_path=training_cache_path,
            all_sample_ids=all_sample_ids,
            gt_annotations=gt_annotations,
            splitter=prepared_splitter,
            candidate=prepared_candidate,
            configured_batch_size=int(configured_batch_size),
            trace_batch_size=int(trace_batch_size),
            preloaded_records=preloaded_records,
        )

    def _run_fixed_split_retrain(
        self,
        model: torch.nn.Module,
        *,
        current_model_name: str,
        bundle_info: dict[str, object],
        manifest: dict[str, object],
        bundle_cache_path: str,
        training_cache_path: str,
        frame_dir: str,
        gt_annotations: dict[str, dict],
        validation_gt_annotations: dict[str, dict],
        validation_sample_ids: list[str],
        num_epoch: int,
        prepared_trace_sample_input: object | None,
        prepared_splitter: UniversalModelSplitter | None,
        prepared_candidate,
        effective_batch_size: int,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None,
        proxy_eval_frame_cache: dict[str, np.ndarray | None] | None = None,
        preloaded_records: Mapping[str, Mapping[str, object]] | None = None,
    ) -> FixedSplitTrainingResult:
        split_runtime_model = get_split_runtime_model(model)
        if prepared_splitter is not None:
            suffix_params = collect_suffix_trainable_parameters(
                prepared_splitter,
                update_requires_grad=False,
            )
            trainable_param_count = sum(int(parameter.numel()) for parameter in suffix_params)
            if trainable_param_count <= 0:
                raise RuntimeError(
                    f"[FixedSplitCL] {current_model_name} has no trainable split-tail parameters."
                )
            logger.info(
                "[FixedSplitCL] {} split retrain enabled {} suffix parameter(s).",
                self._resolve_fixed_split_training_label(current_model_name),
                trainable_param_count,
            )
        else:
            logger.info(
                "[FixedSplitCL] {} split retrain will resolve suffix parameters after tracing.",
                self._resolve_fixed_split_training_label(current_model_name),
            )
        effective_num_epoch = max(1, int(num_epoch))
        effective_learning_rate = self.default_split_learning_rate
        if (
            model_zoo.is_wrapper_model(current_model_name)
            or model_zoo.get_model_family(current_model_name) == "tinynext"
        ):
            effective_learning_rate = self._resolve_fixed_split_learning_rate(
                current_model_name,
            )
            logger.info(
                "[FixedSplitCL] Using {} fixed-split learning rate {}.",
                self._resolve_fixed_split_training_label(current_model_name),
                effective_learning_rate,
            )

        bs = max(1, int(effective_batch_size))
        model_family = model_zoo.get_model_family(current_model_name)
        training_label = self._resolve_fixed_split_training_label(current_model_name)
        target_steps_per_round = self._resolve_fixed_split_target_steps_per_round(
            current_model_name,
        )
        if target_steps_per_round is not None:
            logger.info(
                "[FixedSplitCL] {} effective batch size {} resolved from configured "
                "batch size {} with target_steps_per_round={} and samples={}.",
                training_label,
                bs,
                int(self.batch_size),
                int(target_steps_per_round),
                len(bundle_info["all_sample_ids"]),
            )
        if prepared_trace_sample_input is None and prepared_splitter is not None:
            logger.info(
                "[FixedSplitCL] Split retrain will reuse the bound runtime template "
                "and skip retracing inside universal_split_retrain."
            )

        retrain_profile = SplitRetrainProfile()
        optimizer_overrides = self._fixed_split_optimizer_overrides(current_model_name)
        split_retrain_kwargs = {
            "model": split_runtime_model,
            "sample_input": prepared_trace_sample_input,
            "cache_path": training_cache_path,
            "all_indices": bundle_info["all_sample_ids"],
            "gt_annotations": gt_annotations,
            "device": self.device,
            "learning_rate": effective_learning_rate,
            "loss_fn": build_split_training_loss(model),
            "splitter": prepared_splitter,
            "chosen_candidate": prepared_candidate,
            "batch_size": bs,
            "preloaded_records": preloaded_records,
            "retrain_profile": retrain_profile,
        }
        split_retrain_kwargs.update(optimizer_overrides)

        def _fixed_proxy_evaluator(
            *,
            stage_label: str,
            max_samples: int | None,
        ) -> dict[str, float | int | str | None]:
            del stage_label
            return self._evaluate_fixed_split_proxy_metrics(
                model,
                frame_dir=frame_dir,
                gt_annotations=validation_gt_annotations,
                model_name=current_model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=proxy_eval_frame_cache,
                max_samples=max_samples,
                inference_batch_size=int(split_retrain_kwargs["batch_size"]),
                split_cache_path=training_cache_path,
                splitter=prepared_splitter,
                split_candidate=prepared_candidate,
                preloaded_records=preloaded_records,
            )

        def _tinynext_proxy_evaluator(
            *,
            stage_label: str,
            max_samples: int | None,
        ) -> dict[str, float | int | str | None]:
            return self._evaluate_tinynext_proxy_metrics(
                model,
                frame_dir=frame_dir,
                gt_annotations=validation_gt_annotations,
                model_name=current_model_name,
                sample_metadata_by_id=sample_metadata_by_id,
                frame_cache=proxy_eval_frame_cache,
                max_samples=max_samples,
                inference_batch_size=int(split_retrain_kwargs["batch_size"]),
                stage_label=stage_label,
                split_cache_path=training_cache_path,
                splitter=prepared_splitter,
                split_candidate=prepared_candidate,
                preloaded_records=preloaded_records,
            )

        proxy_config = ProxyEvalConfig(
            enabled=bool(validation_gt_annotations),
            eval_final=True,
            interval_epochs=max(1, int(getattr(self, "proxy_eval_interval_epochs", 10))),
            max_eval_samples=self.proxy_eval_max_samples,
            max_dets=max(10, int(getattr(self, "proxy_eval_max_dets", 500))),
            min_delta=max(0.0, float(self.proxy_eval_min_delta)),
            patience=max(0, int(self.proxy_eval_patience)),
            validation_fraction=float(getattr(self, "proxy_eval_validation_fraction", 0.2)),
        )
        plan = FixedSplitTrainingPlan(
            model_name=current_model_name,
            model_family=model_family,
            total_samples=len(bundle_info["all_sample_ids"]),
            epochs=effective_num_epoch,
            effective_batch_size=bs,
            learning_rate=effective_learning_rate,
            proxy_eval_config=proxy_config,
            training_label=training_label,
            optimizer_name=str(split_retrain_kwargs.get("optimizer_name", "adam")),
            weight_decay=float(split_retrain_kwargs.get("weight_decay", 0.0)),
            grad_clip_norm=split_retrain_kwargs.get("grad_clip_norm"),
            shuffle_samples=bool(split_retrain_kwargs.get("shuffle_samples", False)),
        )
        adapter = get_training_adapter(current_model_name, model_family)
        result = FixedSplitTrainingStage().run(
            FixedSplitTrainingContext(
                model=model,
                plan=plan,
                adapter=adapter,
                training_kwargs=split_retrain_kwargs,
                gt_annotations=gt_annotations,
                validation_gt_annotations=validation_gt_annotations,
                validation_sample_ids=validation_sample_ids,
                fixed_proxy_evaluator=_fixed_proxy_evaluator,
                tinynext_proxy_evaluator=_tinynext_proxy_evaluator,
                retrain_profile=retrain_profile,
                logger=logger,
            )
        )
        self._fixed_split_proxy_evaluator().set_detection_model_eval_mode(model)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
