"""Edge-affine worker pool for real multi-device cloud training."""

from cloud.workers.assignment_store import EdgeAssignment, EdgeAssignmentStore
from cloud.workers.edge_assignment import worker_id_for_edge, workspace_for_worker
from cloud.workers.gpu_lease_manager import GpuLeaseManager

__all__ = [
    "EdgeAssignment",
    "EdgeAssignmentStore",
    "GpuLeaseManager",
    "worker_id_for_edge",
    "workspace_for_worker",
]
