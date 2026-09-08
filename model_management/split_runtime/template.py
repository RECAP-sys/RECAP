from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .runtime_cache import stable_hash, symbolic_input_schema
from .torchlens_native_runtime import SplitRuntime, SplitSpec, prepare_split_runtime


@dataclass(frozen=True)
class FixedSplitRuntimeTemplateKey:
    model_name: str
    model_family: str
    graph_signature: str | None
    split_plan_hash: str
    symbolic_input_schema_hash: str
    canonical_split_key: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "graph_signature": self.graph_signature,
            "split_plan_hash": self.split_plan_hash,
            "symbolic_input_schema_hash": self.symbolic_input_schema_hash,
            "canonical_split_key": self.canonical_split_key,
        }


def fixed_split_runtime_template_key(
    *,
    model_name: str,
    model_family: str,
    split_spec: SplitSpec,
    example_inputs: Any,
    graph_signature: str | None = None,
    split_plan_hash: str | None = None,
    canonical_split_key: str | None = None,
) -> FixedSplitRuntimeTemplateKey:
    schema_hash = stable_hash(
        symbolic_input_schema(example_inputs, batch_symbol=split_spec.batch_symbol)
    )
    structural_split_hash = split_plan_hash or stable_hash(
        {
            "boundary": str(split_spec.boundary),
            "batch_symbol": str(split_spec.batch_symbol),
        }
    )
    return FixedSplitRuntimeTemplateKey(
        model_name=str(model_name),
        model_family=str(model_family),
        graph_signature=graph_signature,
        split_plan_hash=str(structural_split_hash),
        symbolic_input_schema_hash=schema_hash,
        canonical_split_key=str(canonical_split_key or split_spec.boundary or ""),
    )


@dataclass(frozen=True)
class FixedSplitRuntimeTemplate:
    cache_key: FixedSplitRuntimeTemplateKey
    # TorchLens native executable runtimes are not rebindable across model
    # instances. This runtime is kept only as a read-only trace/layout artifact.
    runtime: SplitRuntime
    split_spec: SplitSpec
    model_name: str
    model_family: str
    graph_signature: str | None
    symbolic_input_schema_hash: str
    split_plan_hash: str
    mode: str = "generated_eager"
    runtime_device: str = ""
    candidate_descriptor: Mapping[str, object] | None = None
    runtime_contract: Mapping[str, object] | None = None
    boundary_tensor_labels: tuple[str, ...] = ()
    boundary_schema: Mapping[str, object] | None = None


@dataclass(frozen=True)
class FixedSplitRuntimeTemplateLookup:
    template: FixedSplitRuntimeTemplate
    cache_status: str
    wait_time_sec: float = 0.0
    cold_build_time_sec: float = 0.0


@dataclass
class _InflightTemplateBuild:
    event: threading.Event
    template: FixedSplitRuntimeTemplate | None = None
    error: BaseException | None = None


class FixedSplitRuntimeTemplateCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._templates: dict[Hashable, FixedSplitRuntimeTemplate] = {}
        self._inflight: dict[Hashable, _InflightTemplateBuild] = {}

    def get_or_create_lookup(
        self,
        cache_key: Hashable,
        builder: Callable[[], FixedSplitRuntimeTemplate],
        *,
        log_label: str | None = None,
        diagnostics: bool = False,
    ) -> FixedSplitRuntimeTemplateLookup:
        label = _template_cache_log_label(cache_key, log_label)
        with self._lock:
            cached = self._templates.get(cache_key)
            if cached is not None:
                logger.info("[FixedSplitCL] Runtime template hit: {}.", label)
                _log_template_cache_diagnostics(
                    "runtime template cache hit",
                    cache_key,
                    diagnostics=diagnostics,
                )
                return FixedSplitRuntimeTemplateLookup(template=cached, cache_status="hit")

            inflight = self._inflight.get(cache_key)
            if inflight is None:
                inflight = _InflightTemplateBuild(event=threading.Event())
                self._inflight[cache_key] = inflight
                build_owner = True
                logger.info("[FixedSplitCL] Runtime template miss: {}.", label)
                _log_template_cache_diagnostics(
                    "runtime template cache miss",
                    cache_key,
                    diagnostics=diagnostics,
                )
            else:
                build_owner = False
                logger.info("[FixedSplitCL] Runtime template wait: {}.", label)
                _log_template_cache_diagnostics(
                    "waiting for runtime template build",
                    cache_key,
                    diagnostics=diagnostics,
                )

        if not build_owner:
            wait_started = time.perf_counter()
            inflight.event.wait()
            wait_time = time.perf_counter() - wait_started
            if inflight.error is not None:
                raise inflight.error
            if inflight.template is None:
                raise RuntimeError("TorchLens runtime template build completed without a template.")
            return FixedSplitRuntimeTemplateLookup(
                template=inflight.template,
                cache_status="wait",
                wait_time_sec=wait_time,
            )

        started = time.perf_counter()
        try:
            template = builder()
        except BaseException as exc:
            with self._lock:
                inflight.error = exc
                self._inflight.pop(cache_key, None)
                inflight.event.set()
            raise

        elapsed = time.perf_counter() - started
        with self._lock:
            self._templates[cache_key] = template
            inflight.template = template
            self._inflight.pop(cache_key, None)
            inflight.event.set()
        logger.info("[FixedSplitCL] Runtime prepared: {} elapsed={:.2f}s.", label, elapsed)
        _log_template_cache_diagnostics(
            "runtime template cold build completed",
            cache_key,
            diagnostics=diagnostics,
        )
        return FixedSplitRuntimeTemplateLookup(
            template=template,
            cache_status="miss",
            cold_build_time_sec=elapsed,
        )

    def get_or_create(
        self,
        cache_key: Hashable,
        builder: Callable[[], FixedSplitRuntimeTemplate],
        *,
        log_label: str | None = None,
        diagnostics: bool = False,
    ) -> FixedSplitRuntimeTemplate:
        return self.get_or_create_lookup(
            cache_key,
            builder,
            log_label=log_label,
            diagnostics=diagnostics,
        ).template

    def clear(self) -> None:
        with self._lock:
            self._templates.clear()
            self._inflight.clear()


def bind_request_runtime_from_template(
    template: FixedSplitRuntimeTemplate,
    *,
    model: Any | None = None,
    example_inputs: Any | None = None,
    device: str | None = None,
) -> SplitRuntime:
    del device
    if model is None or model is getattr(template.runtime, "model", None):
        return template.runtime
    if example_inputs is None:
        raise RuntimeError(
            "TorchLens runtime cannot be rebound by deepcopy. Provide example_inputs "
            "so bind_request_runtime_from_template can prepare a request-local "
            "SplitRuntime for the supplied model."
        )
    return prepare_split_runtime(
        model,
        example_inputs,
        template.split_spec,
        mode=template.mode,
    )


_PROCESS_FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE = FixedSplitRuntimeTemplateCache()


def get_fixed_split_runtime_template_cache() -> FixedSplitRuntimeTemplateCache:
    return _PROCESS_FIXED_SPLIT_RUNTIME_TEMPLATE_CACHE


def _template_cache_log_label(cache_key: Hashable, explicit: str | None) -> str:
    if explicit:
        return explicit
    model_name = getattr(cache_key, "model_name", None)
    split = getattr(cache_key, "canonical_split_key", None)
    if model_name and split:
        return f"model={model_name} split={split}"
    return "model=unknown split=unknown"


def _log_template_cache_diagnostics(
    message: str,
    cache_key: Hashable,
    *,
    diagnostics: bool,
) -> None:
    if not diagnostics:
        return
    if hasattr(cache_key, "as_dict"):
        payload = cache_key.as_dict()
    else:
        payload = repr(cache_key)
    logger.debug("[FixedSplitCL][diagnostics] {} key={}", message, payload)


__all__ = [
    "FixedSplitRuntimeTemplate",
    "FixedSplitRuntimeTemplateCache",
    "FixedSplitRuntimeTemplateKey",
    "FixedSplitRuntimeTemplateLookup",
    "bind_request_runtime_from_template",
    "fixed_split_runtime_template_key",
    "get_fixed_split_runtime_template_cache",
]
