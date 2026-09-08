from __future__ import annotations

from cloud.feature_cache.gc import FeatureCacheGC
from cloud.feature_cache.materializer import FeatureCacheMaterializer
from cloud.feature_cache.planner import FeatureCachePlanner
from cloud.feature_cache.shard_reachability import (
    collect_refs_from_pending_annotation,
    collect_refs_from_pending_feature_rebuild,
    collect_refs_from_recent_training_windows,
    collect_refs_from_training_views,
    is_shard_reachable,
)
from cloud.feature_cache.shard_reader import FeatureShardPayloadCache, ShardFeatureBatchReader
from cloud.feature_cache.shard_store import FeatureShardStore
from cloud.feature_cache.shard_validator import (
    ShardFeatureRefValidator,
    ValidationResult,
    feature_layouts_abi_compatible,
    shard_feature_layout_from_metadata,
)
from cloud.feature_cache.types import (
    NPY_MEMMAP_SHARD,
    SAFETENSORS_SHARD,
    FeatureCacheGCResult,
    FeatureCacheKey,
    FeatureCachePreparePlan,
    FeatureCachePrepareResult,
    FeatureCacheStats,
    FeatureShardMetadata,
    FeatureShardRef,
    LabelRef,
    SampleTrainingRef,
    TrainingCacheView,
)

__all__ = [
    "FeatureCacheGC",
    "FeatureCacheGCResult",
    "FeatureCacheKey",
    "FeatureCacheMaterializer",
    "FeatureCachePlanner",
    "FeatureCachePreparePlan",
    "FeatureCachePrepareResult",
    "FeatureCacheStats",
    "FeatureShardMetadata",
    "FeatureShardPayloadCache",
    "FeatureShardRef",
    "FeatureShardStore",
    "LabelRef",
    "NPY_MEMMAP_SHARD",
    "SAFETENSORS_SHARD",
    "SampleTrainingRef",
    "ShardFeatureRefValidator",
    "ShardFeatureBatchReader",
    "TrainingCacheView",
    "ValidationResult",
    "collect_refs_from_pending_annotation",
    "collect_refs_from_pending_feature_rebuild",
    "collect_refs_from_recent_training_windows",
    "collect_refs_from_training_views",
    "feature_layouts_abi_compatible",
    "is_shard_reachable",
    "shard_feature_layout_from_metadata",
]
