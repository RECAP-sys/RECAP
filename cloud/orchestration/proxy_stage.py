from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch
from loguru import logger

from cloud.training import FixedSplitProxyEvaluator
from model_management.universal_model_split import UniversalModelSplitter

__all__ = ["FixedSplitProxyEvaluator"]


class ProxyStageMixin:
    def _proxy_eval_frame_cache(self) -> dict[str, np.ndarray | None] | None:
        return self._fixed_split_proxy_evaluator().new_frame_cache()

    def _fixed_split_proxy_evaluator(self) -> FixedSplitProxyEvaluator:
        return FixedSplitProxyEvaluator(
            device=self.device,
            default_batch_size=self.batch_size,
            max_samples=self.proxy_eval_max_samples,
            frame_cache_enabled=self.proxy_eval_frame_cache_enabled,
            max_dets=self.proxy_eval_max_dets,
        )

    def _evaluate_fixed_split_proxy_metrics(
        self,
        model: torch.nn.Module,
        *,
        frame_dir: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        model_name: str,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
        frame_cache: dict[str, np.ndarray | None] | None = None,
        max_samples: int | None = None,
        inference_batch_size: int | None = None,
        split_cache_path: str | None = None,
        splitter: UniversalModelSplitter | None = None,
        split_candidate=None,
        preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
        priority_sample_ids: Iterable[object] | None = None,
        random_fill_seed: object | None = None,
    ) -> dict[str, float | int | None]:
        return self._fixed_split_proxy_evaluator().evaluate_detection(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            model_name=model_name,
            sample_metadata_by_id=sample_metadata_by_id,
            frame_cache=frame_cache,
            max_samples=max_samples,
            inference_batch_size=inference_batch_size,
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
            priority_sample_ids=priority_sample_ids,
            random_fill_seed=random_fill_seed,
        )

    def _evaluate_tinynext_proxy_metrics(
        self,
        model: torch.nn.Module,
        *,
        frame_dir: str,
        gt_annotations: Mapping[str, Mapping[str, object]],
        model_name: str,
        sample_metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
        frame_cache: dict[str, np.ndarray | None] | None = None,
        max_samples: int | None = None,
        inference_batch_size: int | None = None,
        stage_label: str,
        split_cache_path: str | None = None,
        splitter: UniversalModelSplitter | None = None,
        split_candidate=None,
        preloaded_records: Mapping[object, Mapping[str, object]] | None = None,
        priority_sample_ids: Iterable[object] | None = None,
        random_fill_seed: object | None = None,
    ) -> dict[str, float | int | None]:
        return self._fixed_split_proxy_evaluator().evaluate_tinynext(
            model,
            frame_dir=frame_dir,
            gt_annotations=gt_annotations,
            model_name=model_name,
            sample_metadata_by_id=sample_metadata_by_id,
            frame_cache=frame_cache,
            max_samples=max_samples,
            inference_batch_size=inference_batch_size,
            stage_label=stage_label,
            split_cache_path=split_cache_path,
            splitter=splitter,
            split_candidate=split_candidate,
            preloaded_records=preloaded_records,
            priority_sample_ids=priority_sample_ids,
            random_fill_seed=random_fill_seed,
            logger=logger,
        )
