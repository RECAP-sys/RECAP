import argparse
import os
import threading
from concurrent import futures
from pathlib import Path

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import grpc
from loguru import logger

from cloud.annotation import (
    TeacherAnnotationService,
    TeacherAnnotationWorker,
    TeacherLabelCache,
)
from cloud.annotation.remote_service import SharedTeacherAnnotationRpcServer
from cloud.edge_registry import EdgeRegistry
from cloud.experiment_result_repository import (
    CloudExperimentManifestWriter,
    CloudExperimentResultRepository,
)
from cloud.orchestration.teacher_stage import TeacherAnnotationMixin
from cloud.workers.assignment_store import EdgeAssignmentStore
from cloud.workers.edge_worker_pool import EdgeWorkerPool
from cloud.workers.gpu_lease_manager import GpuLeaseManager
from cloud.workers.lease_service import GpuLeaseService
from cloud.workers.mps_runtime import ensure_mps_runtime
from common.experiment_results import (
    RECAP_METHOD,
    ExperimentIdentity,
    normalize_scenario_slug,
)
from common.logging_sanitizer import log_diagnostic_debug
from common.video_identity import resolve_video_identity
from config import load_runtime_config
from grpc_server import message_transmission_pb2_grpc
from grpc_server.continual_backends import EdgeWorkerRoutedContinualLearningBackend
from grpc_server.rpc_server import MessageTransmissionServicer
from tools.grpc_options import grpc_message_options

__all__ = ["CloudServer"]


def _runtime_source(runtime_config) -> object | None:
    client = getattr(runtime_config, "client", None)
    return getattr(client, "source", None)


def _scenario_slug_from_inputs(*, scenario: str | None, runtime_config) -> str:
    explicit = str(scenario or "").strip()
    if explicit:
        return normalize_scenario_slug(explicit)
    source = _runtime_source(runtime_config)
    if source is None:
        return "default"
    scenario_name = str(getattr(source, "scenario_name", "") or "")
    video_slug = str(getattr(source, "video_slug", "") or "")
    video_path = str(getattr(source, "video_path", "") or "")
    try:
        identity = resolve_video_identity(
            video_path,
            configured_scenario_name=scenario_name,
            configured_video_slug=video_slug,
        )
        return normalize_scenario_slug(identity.scenario_name or identity.video_slug)
    except ValueError:
        for value in (scenario_name, video_slug, Path(video_path).stem):
            if str(value or "").strip():
                return normalize_scenario_slug(str(value))
    return "default"


def _create_experiment_identity(
    *,
    experiment_id: str | None,
    scenario: str | None,
    edge_count: int | str | None,
    repeat: int | str | None,
    method: str,
    runtime_config,
) -> ExperimentIdentity:
    return ExperimentIdentity.create(
        experiment_id=str(experiment_id or "default_experiment"),
        scenario_slug=_scenario_slug_from_inputs(
            scenario=scenario,
            runtime_config=runtime_config,
        ),
        edge_count=1 if edge_count is None else edge_count,
        repeat=1 if repeat is None else repeat,
        method=method,
    )


class _SharedTeacherAnnotationAdapter(TeacherAnnotationMixin):
    """Reuse the fixed-split label mapping for the process-shared teacher."""

    def __init__(self, config, teacher_detector) -> None:
        self.config = config
        self.large_od = teacher_detector
        self.weight_folder = str(Path(__file__).parent / "model_management" / "models")
        self.teacher_annotation_metadata: dict[str, object] = {}
        self._teacher_weights_fingerprint_cache: str | None = None
        self._metadata_lock = threading.Lock()
        self._metadata_cache: dict[str, object] | None = None

    def build_labels(self, request, frame, prediction):
        return self._teacher_labels_from_request_prediction(request, frame, prediction)

    def metadata(self) -> dict[str, object]:
        if self._metadata_cache is None:
            with self._metadata_lock:
                if self._metadata_cache is None:
                    self._metadata_cache = {
                        "teacher_model_name": self._teacher_model_name(),
                        "teacher_weights_fingerprint": (
                            self._teacher_weights_fingerprint()
                        ),
                        "teacher_label_schema": self._teacher_label_schema(),
                        "teacher_num_classes": self._teacher_num_classes(),
                        "teacher_class_names": self._teacher_class_names(),
                    }
        return dict(self._metadata_cache)


def __getattr__(name: str):
    if name == "CloudContinualLearner":
        from cloud.orchestrator import CloudContinualLearner

        return CloudContinualLearner
    raise AttributeError(name)


class CloudServer:
    def __init__(
        self,
        config,
        *,
        experiment_id: str = "",
        scenario: str = "",
        edge_count: int | str = 1,
        repeat: int | str = 1,
        yaml_path: str = "./config/config.yaml",
        runtime_config=None,
    ):
        self.config = config
        self.runtime_config = runtime_config
        self.yaml_path = str(yaml_path)
        self.server_id = config.server_id
        self.edge_registry = EdgeRegistry()
        self.large_object_detection = None
        self.continual_backend = None
        self.worker_pool = None
        self.gpu_lease_manager = None
        self.gpu_lease_service = None
        self.shared_teacher_rpc_server = None
        self.shared_teacher_annotation_worker = None
        self.shared_teacher_annotation_service = None
        self.grpc_server = None
        self.experiment_result_repository = None
        self.experiment_manifest_writer = None
        self._experiment_log_sink = None
        self._closing = False
        self.log_internal_ids = False
        edge_affine = getattr(config, "edge_affine_workers", None)
        if edge_affine is None or not bool(getattr(edge_affine, "enabled", False)):
            raise ValueError(
                "RECAP cloud continual learning requires "
                "server.edge_affine_workers.enabled=true."
            )
        self.experiment_identity = _create_experiment_identity(
            experiment_id=experiment_id,
            scenario=scenario,
            edge_count=edge_count,
            repeat=repeat,
            method=RECAP_METHOD,
            runtime_config=runtime_config,
        )
        self.run_id = self.experiment_identity.run_id
        self._init_edge_affine_backend(edge_affine, run_id=self.run_id)
        self._init_experiment_results_if_enabled()

    def _init_experiment_results_if_enabled(self) -> None:
        experiment_config = getattr(self.config, "experiment_results", None)
        if experiment_config is None or not bool(getattr(experiment_config, "enabled", False)):
            return
        root_dir = str(getattr(experiment_config, "root_dir", "results/experiments"))
        identity = self.experiment_identity
        self.experiment_manifest_writer = CloudExperimentManifestWriter(
            root_dir=root_dir,
            experiment_id=identity.experiment_id,
            student_model=str(getattr(self.config, "edge_model_name", "") or ""),
            teacher_model=str(getattr(self.config, "golden", "") or ""),
        )
        self.experiment_result_repository = CloudExperimentResultRepository(
            root_dir,
            max_artifact_bytes=int(getattr(experiment_config, "max_artifact_bytes", 268435456)),
            manifest_writer=self.experiment_manifest_writer,
        )
        self.experiment_manifest_writer.upsert_cloud_runtime(
            method=RECAP_METHOD,
            scenario_slug=identity.scenario_slug,
            edge_count=identity.edge_count,
            repeat=identity.repeat,
            run_id=identity.run_id,
        )
    def _init_edge_affine_backend(self, edge_affine, *, run_id: str) -> None:
        run_id = str(run_id or "").strip()
        if not run_id:
            raise ValueError("experiment run_id must be non-empty")
        self.run_id = run_id
        workspace_root = str(getattr(self.config, "workspace_root", "./cache/server_workspace"))
        assignment_store = EdgeAssignmentStore(
            Path(workspace_root) / "worker_assignments.json",
            run_id=run_id,
            mode=str(getattr(edge_affine, "mode", "edge_affine_single_gpu_mps")),
            worker_workspace_root=str(edge_affine.edge_workers.workspace_root),
        )
        lease_cfg = edge_affine.gpu_lease
        self.gpu_lease_manager = GpuLeaseManager(
            memory_usage_threshold=float(lease_cfg.memory_usage_threshold),
            reserve_memory_gb=float(lease_cfg.reserve_memory_gb),
            max_active_gpu_workers=lease_cfg.max_active_gpu_workers,
            default_estimated_job_memory_gb=float(lease_cfg.default_estimated_job_memory_gb),
            lease_ttl_sec=float(lease_cfg.lease_ttl_sec),
            teacher_reserved_memory_gb=float(lease_cfg.teacher_reserved_memory_gb),
        )
        self.gpu_lease_service = GpuLeaseService(
            listen_address="127.0.0.1:0",
            manager=self.gpu_lease_manager,
        )
        self.gpu_lease_service.start()
        mps_env = ensure_mps_runtime(
            edge_affine.mps,
            max_active_gpu_workers=self.gpu_lease_manager.max_active_gpu_workers,
        )
        if bool(getattr(edge_affine.mps, "enabled", False)):
            os.environ.update(mps_env.as_env())
        teacher_annotation_address = self._init_shared_teacher_annotation_backend()
        self.worker_pool = EdgeWorkerPool(
            yaml_path=self.yaml_path,
            run_id=run_id,
            mode=str(edge_affine.mode),
            assignment_store=assignment_store,
            edge_workers_config=edge_affine.edge_workers,
            worker_service_config=edge_affine.worker,
            mps_env=mps_env,
            lease_address=self.gpu_lease_service.listen_address,
            teacher_annotation_address=teacher_annotation_address,
            log_internal_ids=self.log_internal_ids,
        )
        self.continual_backend = EdgeWorkerRoutedContinualLearningBackend(
            worker_pool=self.worker_pool,
            edge_registry=self.edge_registry,
            gpu_lease_manager=self.gpu_lease_manager,
        )
        logger.info(
            "edge_affine_worker_pool enabled=true mode={} assignment={} lazy_start={} "
            "lazy_cuda_init={} mps_enabled={} gpu_device={} memory_usage_threshold={} "
            "reserve_memory_gb={} max_active_gpu_workers={} default_estimated_job_memory_gb={}",
            edge_affine.mode,
            edge_affine.edge_workers.assignment,
            edge_affine.edge_workers.lazy_start,
            edge_affine.edge_workers.lazy_cuda_init,
            edge_affine.mps.enabled,
            lease_cfg.device,
            lease_cfg.memory_usage_threshold,
            lease_cfg.reserve_memory_gb,
            self.gpu_lease_manager.max_active_gpu_workers,
            lease_cfg.default_estimated_job_memory_gb,
        )

    def _init_shared_teacher_annotation_backend(self) -> str:
        cl_cfg = getattr(self.config, "continual_learning", None)
        teacher_cfg = (
            getattr(cl_cfg, "teacher_annotation", None) if cl_cfg is not None else None
        )
        if not (
            bool(getattr(teacher_cfg, "async_enabled", True))
            and bool(getattr(teacher_cfg, "cache_enabled", True))
        ):
            return ""

        from cloud.workers.edge_worker_service import LazyObjectDetection

        self.large_object_detection = LazyObjectDetection(self.config, "large inference")
        adapter = _SharedTeacherAnnotationAdapter(
            self.config,
            self.large_object_detection,
        )
        cache = TeacherLabelCache(
            str(getattr(teacher_cfg, "cache_root_dir", "./cache/teacher_label_cache")),
            enabled=True,
            log_internal_ids=self.log_internal_ids,
        )
        self.shared_teacher_annotation_worker = TeacherAnnotationWorker(
            label_cache=cache,
            batch_inference=self.large_object_detection.large_inference_batch,
            single_inference=self.large_object_detection.large_inference,
            label_builder=adapter.build_labels,
            max_queue_size=int(getattr(teacher_cfg, "worker_max_queue_size", 4096)),
            worker_batch_size=int(getattr(teacher_cfg, "worker_batch_size", 8)),
            max_retries=int(getattr(teacher_cfg, "worker_max_retries", 2)),
            oom_retry_enabled=bool(getattr(teacher_cfg, "oom_retry_enabled", True)),
            min_worker_batch_size=int(getattr(teacher_cfg, "min_worker_batch_size", 1)),
            log_internal_ids=self.log_internal_ids,
        )
        self.shared_teacher_annotation_service = TeacherAnnotationService(
            label_cache=cache,
            worker=self.shared_teacher_annotation_worker,
            log_internal_ids=self.log_internal_ids,
        )
        self.shared_teacher_rpc_server = SharedTeacherAnnotationRpcServer(
            service=self.shared_teacher_annotation_service,
            metadata_provider=adapter.metadata,
        )
        self.shared_teacher_rpc_server.start()
        logger.info(
            "shared teacher annotation service enabled endpoint={} model={} batch_size={} ",
            self.shared_teacher_rpc_server.listen_address,
            getattr(self.config, "golden", "rtdetr_x"),
            int(getattr(teacher_cfg, "worker_batch_size", 8)),
        )
        return self.shared_teacher_rpc_server.listen_address

    def start_server(self):
        listen_address = str(getattr(self.config, "listen_address", "[::]:50051")).strip()
        workspace_root = str(
            getattr(self.config, "workspace_root", "./cache/server_workspace")
        ).strip()
        grpc_max_workers = max(
            4,
            int(getattr(self.config, "grpc_max_workers", 8)),
        )
        logger.info(
            "cloud server effective startup config: pid={}, golden={}, "
            "edge_model_name={}, listen_address={}, "
            "grpc_max_workers={}, run_id={}",
            os.getpid(),
            getattr(self.config, "golden", "unknown"),
            getattr(self.config, "edge_model_name", "unknown"),
            listen_address,
            grpc_max_workers,
            getattr(self, "run_id", ""),
        )
        experiment_config = getattr(self.config, "experiment_results", None)
        if experiment_config is not None:
            identity = getattr(self, "experiment_identity", None)
            logger.info(
                "experiment results: experiment_id={} scenario={} edges={} repeat={} "
                "root={} method={} run_id={}",
                getattr(identity, "experiment_id", ""),
                getattr(identity, "scenario_slug", ""),
                getattr(identity, "edge_count", ""),
                getattr(identity, "repeat", ""),
                getattr(experiment_config, "root_dir", ""),
                RECAP_METHOD,
                getattr(self, "run_id", ""),
            )
        log_diagnostic_debug(
            self.log_internal_ids,
            "cloud server startup paths",
            lambda: {"workspace_root": workspace_root},
        )
        experiment_method = RECAP_METHOD
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=grpc_max_workers),
            options=grpc_message_options(),
        )
        self.grpc_server = server
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                id=self.server_id,
                workspace_root=workspace_root,
                edge_registry=self.edge_registry,
                continual_backend=self.continual_backend,
                log_internal_ids=self.log_internal_ids,
                experiment_result_repository=self.experiment_result_repository,
                experiment_id=str(getattr(self.experiment_identity, "experiment_id", "") or ""),
                experiment_scenario_slug=str(
                    getattr(self.experiment_identity, "scenario_slug", "") or ""
                ),
                experiment_edge_count=int(
                    getattr(self.experiment_identity, "edge_count", 1) or 1
                ),
                experiment_repeat=int(getattr(self.experiment_identity, "repeat", 1) or 1),
                experiment_method=experiment_method,
                experiment_run_id=str(getattr(self, "run_id", "") or ""),
            ),
            server,
        )
        server.add_insecure_port(listen_address)
        server.start()
        logger.info(
            "cloud server is listening on {} (pid={}, edge_model_name={})",
            listen_address,
            os.getpid(),
            getattr(self.config, "edge_model_name", "unknown"),
        )
        if self.experiment_result_repository is not None:
            self.experiment_result_repository.record_cloud_event(
                experiment_id=str(self.experiment_identity.experiment_id),
                scenario_slug=str(self.experiment_identity.scenario_slug),
                edge_count=int(self.experiment_identity.edge_count),
                repeat=int(self.experiment_identity.repeat),
                method=experiment_method,
                run_id=str(getattr(self, "run_id", "") or ""),
                event="cloud_server_started",
                mode="main",
            )
        try:
            server.wait_for_termination()
        finally:
            self.close()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.grpc_server is not None:
            self.grpc_server.stop(0)
            self.grpc_server = None
        close_backend = getattr(self.continual_backend, "close", None)
        if callable(close_backend):
            close_backend()
        if self.worker_pool is not None:
            self.worker_pool.close()
        if self.shared_teacher_rpc_server is not None:
            self.shared_teacher_rpc_server.shutdown()
            self.shared_teacher_rpc_server = None
        if self.shared_teacher_annotation_worker is not None:
            self.shared_teacher_annotation_worker.stop()
            self.shared_teacher_annotation_worker = None
        if self.gpu_lease_service is not None:
            self.gpu_lease_service.shutdown()
        if self.gpu_lease_manager is not None:
            self.gpu_lease_manager.close()
        training_job_manager = getattr(self, "training_job_manager", None)
        if training_job_manager is not None:
            training_job_manager.close()
        if self._experiment_log_sink is not None:
            logger.remove(self._experiment_log_sink)
            self._experiment_log_sink = None


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


if __name__ == "__main__":
    from tools.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument(
        "--yaml_path",
        default="./config/config.yaml",
        help="input the path of *.yaml",
    )
    parser.add_argument("--listen_address", default=None, help="override server.listen_address")
    parser.add_argument("--workspace_root", default=None, help="override server.workspace_root")
    parser.add_argument(
        "--grpc_max_workers",
        type=int,
        default=None,
        help="override server.grpc_max_workers",
    )
    parser.add_argument("--experiment_id", default=None, help="experiment id")
    parser.add_argument("--scenario", default=None, help="experiment scenario slug/name")
    parser.add_argument("--edge_count", type=int, default=None, help="number of edge devices")
    parser.add_argument("--repeat", default=None, help="repeat index, e.g. 1 or r01")
    parser.add_argument(
        "--experiment_results_root",
        default=None,
        help="override cloud experiment result repository root",
    )
    parser.add_argument(
        "--edge_affine_workers_enabled",
        type=str,
        default=None,
        help="override server.edge_affine_workers.enabled",
    )
    parser.add_argument(
        "--edge_affine_worker_mode",
        default=None,
        help="override server.edge_affine_workers.mode",
    )
    args = parser.parse_args()
    config = load_runtime_config(args.yaml_path)
    experiment_run = config.experiment_run
    server_config = config.server
    if args.experiment_results_root is not None:
        config.experiment_results.root_dir = args.experiment_results_root
    if args.listen_address is not None:
        server_config.listen_address = args.listen_address
    if args.workspace_root is not None:
        server_config.workspace_root = args.workspace_root
    if args.grpc_max_workers is not None:
        server_config.grpc_max_workers = args.grpc_max_workers
    if args.edge_affine_workers_enabled is not None:
        server_config.edge_affine_workers.enabled = _parse_bool(args.edge_affine_workers_enabled)
    if args.edge_affine_worker_mode is not None:
        server_config.edge_affine_workers.mode = args.edge_affine_worker_mode
    cloud_server = CloudServer(
        server_config,
        experiment_id=(
            args.experiment_id
            if args.experiment_id is not None
            else experiment_run.experiment_id
        ),
        scenario=args.scenario if args.scenario is not None else experiment_run.scenario,
        edge_count=(
            args.edge_count
            if args.edge_count is not None
            else experiment_run.edge_count
        ),
        repeat=args.repeat if args.repeat is not None else experiment_run.repeat,
        yaml_path=args.yaml_path,
        runtime_config=config,
    )
    cloud_server.start_server()
