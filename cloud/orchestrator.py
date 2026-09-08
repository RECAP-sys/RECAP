from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cloud.orchestration.fixed_split_pipeline import FixedSplitPipeline

if TYPE_CHECKING:
    from model_management.object_detection import Object_Detection


class CloudFixedSplitOrchestrator:
    """Thin cloud-side facade for fixed-split continual learning RPC calls."""

    def __init__(
        self,
        config: Any,
        large_object_detection: "Object_Detection",
        *,
        gpu_lease_client=None,
        worker_id: str = "",
        teacher_annotation_service=None,
        teacher_annotation_metadata: dict[str, Any] | None = None,
    ):
        self._pipeline = FixedSplitPipeline(
            config,
            large_object_detection,
            gpu_lease_client=gpu_lease_client,
            worker_id=worker_id,
            teacher_annotation_service=teacher_annotation_service,
            teacher_annotation_metadata=teacher_annotation_metadata,
        )
        self.config = config
        self.large_od = large_object_detection
        self.max_concurrent_jobs = self._pipeline.max_concurrent_jobs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or "_pipeline" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        if name in {"config", "large_od", "max_concurrent_jobs"}:
            object.__setattr__(self, name, value)
        setattr(self._pipeline, name, value)

    def close(self) -> None:
        self._pipeline.close()

    def training_queue_state(self) -> tuple[int, int]:
        return self._pipeline.training_queue_state()

    def sync_samples(
        self,
        edge_id: int,
        protocol_version: str,
        sync_type: str,
        payload_zip: bytes,
        model_id: str = "",
        model_version: str = "",
        split_config_id: str = "",
    ) -> tuple[bool, str, int]:
        return self._pipeline.sync_samples(
            edge_id=edge_id,
            protocol_version=protocol_version,
            sync_type=sync_type,
            payload_zip=payload_zip,
            model_id=model_id,
            model_version=model_version,
            split_config_id=split_config_id,
        )

    def get_ground_truth_and_retrain(
        self,
        edge_id: int,
        frame_indices: list[int],
        cache_path: str,
    ) -> tuple[bool, str, str]:
        return self._pipeline.get_ground_truth_and_retrain(edge_id, frame_indices, cache_path)

    def get_ground_truth_and_fixed_split_retrain(
        self,
        edge_id: int,
        bundle_cache_path: str,
        *,
        num_epoch: int | None = None,
    ) -> tuple[bool, str, str]:
        return self._pipeline.get_ground_truth_and_fixed_split_retrain(
            edge_id=edge_id,
            bundle_cache_path=bundle_cache_path,
            num_epoch=num_epoch,
        )


CloudContinualLearner = CloudFixedSplitOrchestrator
