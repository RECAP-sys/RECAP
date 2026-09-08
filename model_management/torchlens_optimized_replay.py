from __future__ import annotations

import types
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchlens.intervention.types import (
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)
from torchlens.split import generated as generated_replay
from torchlens.split.boundary import ReplayBoundary
from torchlens.utils.rng import execute_with_restored_rng_autocast

from model_management.split_runtime import SplitRuntime


@dataclass(frozen=True)
class _TensorSlot:
    index: int


@dataclass(frozen=True)
class _ContainerSpec:
    kind: str
    children: tuple[Any, ...]


@dataclass(frozen=True)
class _ConstantValue:
    value: Any


def _output_template(value: Any, paths: list[tuple[Any, ...]], path=()) -> Any:
    if isinstance(value, torch.Tensor):
        slot = _TensorSlot(len(paths))
        paths.append(tuple(path))
        return slot
    if isinstance(value, dict):
        return _ContainerSpec(
            "dict",
            tuple(
                (key, _output_template(item, paths, (*path, key)))
                for key, item in value.items()
            ),
        )
    if isinstance(value, list):
        return _ContainerSpec(
            "list",
            tuple(
                _output_template(item, paths, (*path, index))
                for index, item in enumerate(value)
            ),
        )
    if isinstance(value, tuple):
        return _ContainerSpec(
            "tuple",
            tuple(
                _output_template(item, paths, (*path, index))
                for index, item in enumerate(value)
            ),
        )
    return _ConstantValue(value)


def _restore_output(template: Any, tensors: tuple[torch.Tensor, ...]) -> Any:
    if isinstance(template, _TensorSlot):
        return tensors[template.index]
    if isinstance(template, _ConstantValue):
        return template.value
    if template.kind == "dict":
        return {
            key: _restore_output(child, tensors)
            for key, child in template.children
        }
    values = tuple(_restore_output(child, tensors) for child in template.children)
    return list(values) if template.kind == "list" else values


def _as_tensor_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    values = tuple(value)
    if not all(isinstance(item, torch.Tensor) for item in values):
        raise TypeError("Optimized TorchLens replay expected tensor-only traced outputs.")
    return values


def _first_batch_size(inputs: tuple[Any, ...]) -> int | None:
    for value in inputs:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _runtime_batch_expression(
    inputs: tuple[torch.Tensor, ...],
    reference_batch_size: int | None,
) -> str:
    if reference_batch_size is None:
        raise RuntimeError("Optimized TorchLens replay could not infer a batch size.")
    for index, value in enumerate(inputs):
        if value.ndim == 0 or int(value.shape[0]) % reference_batch_size:
            continue
        multiplier = int(value.shape[0]) // reference_batch_size
        expression = f"inputs[{index}].shape[0]"
        return expression if multiplier == 1 else f"({expression}//{multiplier})"
    raise RuntimeError("Optimized TorchLens replay could not locate a batched tensor.")


class _GeneratedModule(nn.Module):
    pass


class _StaticSegmentBuilder:
    def __init__(self, runtime: SplitRuntime, module: _GeneratedModule) -> None:
        self.runtime = runtime
        self.graph = runtime.trace_graph
        self.plan = runtime.plan
        self.module = module
        self.namespace: dict[str, Any] = {
            "execute_with_context": execute_with_restored_rng_autocast,
            "slice_output": generated_replay._slice_output_by_path,
            "reconstruct_output": generated_replay._reconstruct_model_output,
            "trace_graph": self.graph,
        }
        self._tensor_names: dict[int, str] = {}
        self._value_count = 0

    def value_expression(self, value: Any) -> str:
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            name = self._tensor_names.get(tensor_id)
            if name is None:
                name = f"_tensor_{len(self._tensor_names)}"
                self._tensor_names[tensor_id] = name
                if isinstance(value, nn.Parameter):
                    self.module.register_parameter(name, value)
                else:
                    stored = value.detach() if value.requires_grad else value
                    self.module.register_buffer(name, stored, persistent=False)
            return f"self.{name}"
        name = f"_value_{self._value_count}"
        self._value_count += 1
        self.namespace[name] = value
        return name

    def component_expression(
        self,
        component: Any,
        node: Any,
        env: dict[str, Any],
        param_state: dict[str, int],
        path: tuple[Any, ...],
    ) -> str:
        if isinstance(component, ParentRef):
            label = generated_replay._final_label_for_ref(
                self.graph,
                component.parent_label,
            )
            return f"env[{label!r}]"
        if isinstance(component, LiteralTensor):
            parameter = generated_replay._next_matching_param(
                node,
                component.value,
                param_state,
            )
            return self.value_expression(
                parameter if parameter is not None else component.value
            )
        if isinstance(component, LiteralValue):
            dynamic_value = generated_replay._dynamic_batch_literal(
                component.value,
                node,
                self.graph,
                env,
                path,
            )
            if dynamic_value is None:
                return self.value_expression(component.value)
            runtime_batch = generated_replay._runtime_batch_from_env(env, self.graph)
            if runtime_batch is None or dynamic_value % runtime_batch:
                raise RuntimeError("Could not generate a dynamic batch expression.")
            multiplier = dynamic_value // runtime_batch
            batch_expression = (
                f"env[{generated_replay._RUNTIME_BATCH_ENV_KEY!r}]"
            )
            return (
                batch_expression
                if multiplier == 1
                else f"({batch_expression}*{multiplier})"
            )
        if isinstance(component, Unsupported):
            if component.value_type == "ellipsis":
                return "Ellipsis"
            raise RuntimeError(component.reason)
        if isinstance(component, tuple):
            if generated_replay._looks_like_template_dict(component):
                items = (
                    f"{self.value_expression(key)}:"
                    f"{self.component_expression(value, node, env, param_state, (*path, key))}"
                    for key, value in component
                )
                return "{" + ",".join(items) + "}"
            items = [
                self.component_expression(
                    value,
                    node,
                    env,
                    param_state,
                    (*path, index),
                )
                for index, value in enumerate(component)
            ]
            body = ",".join(items)
            return f"({body}{',' if len(items) == 1 else ''})"
        return self.value_expression(component)

    def append_node(self, node: Any, env: dict[str, Any], lines: list[str]) -> None:
        label = node.torchlens_label
        if node.is_input:
            return
        if label in env and node.target is None and not node.parents:
            return
        if (
            env.get(generated_replay._LIVE_PARAM_SOURCES_ENV_KEY, False)
            and node.replay_source_policy == "live_param"
        ):
            live_source = generated_replay._live_source_value(node, self.graph)
            if live_source is not None:
                lines.append(f"env[{label!r}]={self.value_expression(live_source)}")
                generated_replay._execute_node(
                    node,
                    self.graph,
                    env,
                    split_point=self.plan.split_id,
                )
                return
        activation = getattr(node.layer, "activation", None)
        if generated_replay._can_reuse_trace_activation(node, activation, env):
            lines.append(f"env[{label!r}]={self.value_expression(activation)}")
            generated_replay._execute_node(
                node,
                self.graph,
                env,
                split_point=self.plan.split_id,
            )
            return
        if node.is_output or node.target is None:
            parent = generated_replay._final_label_for_ref(
                self.graph,
                node.parents[0],
            )
            lines.append(f"env[{label!r}]=env[{parent!r}]")
            generated_replay._execute_node(
                node,
                self.graph,
                env,
                split_point=self.plan.split_id,
            )
            return

        param_state = {"index": 0}
        args = ",".join(
            self.component_expression(
                value,
                node,
                env,
                param_state,
                ("args", index),
            )
            for index, value in enumerate(node.args)
        )
        kwargs = ",".join(
            f"{key!r}:"
            f"{self.component_expression(value, node, env, param_state, ('kwargs', key))}"
            for key, value in node.kwargs.items()
        )
        target = self.value_expression(node.target)
        autocast_state = getattr(node.layer, "func_autocast_state", None)
        if generated_replay._node_needs_replay_context(node, autocast_state):
            call = (
                f"execute_with_context({target},tuple([{args}]),{{{kwargs}}},"
                f"rng_states={self.value_expression(getattr(node.layer, 'func_rng_states', None))},"
                f"autocast_state={self.value_expression(autocast_state)})"
            )
        else:
            call = f"{target}(*[{args}],**{{{kwargs}}})"
        output_path = tuple(node.layer.output_path or ())
        if output_path:
            call = (
                f"slice_output({call},{self.value_expression(output_path)})"
            )
        lines.append(f"env[{label!r}]={call}")
        generated_replay._execute_node(
            node,
            self.graph,
            env,
            split_point=self.plan.split_id,
        )

    def install_forward(self, lines: list[str]) -> None:
        source = "def forward(self,*inputs):\n    " + "\n    ".join(lines)
        exec(source, self.namespace)
        self.module.forward = types.MethodType(self.namespace["forward"], self.module)


@dataclass
class TorchScriptSplitReplay:
    prefix: torch.jit.ScriptModule
    suffix: torch.jit.ScriptModule
    boundary_labels: tuple[str, ...]
    boundary_specs: dict[str, Any]
    boundary_metadata: dict[str, Any]
    output_template: Any

    def run_prefix(self, *inputs: Any) -> ReplayBoundary:
        tensors = _as_tensor_tuple(self.prefix(*inputs))
        metadata = dict(self.boundary_metadata)
        metadata["batch_size"] = _first_batch_size(inputs)
        return ReplayBoundary(
            dict(zip(self.boundary_labels, tensors, strict=True)),
            self.boundary_specs,
            metadata,
        )

    def run_suffix(self, boundary: ReplayBoundary) -> Any:
        values = tuple(boundary.tensors[label] for label in self.boundary_labels)
        outputs = _as_tensor_tuple(self.suffix(*values))
        return _restore_output(self.output_template, outputs)


def build_torchscript_split_replay(
    runtime: SplitRuntime,
    sample_inputs: tuple[Any, ...],
) -> TorchScriptSplitReplay:
    if not sample_inputs or not all(
        isinstance(value, torch.Tensor) for value in sample_inputs
    ):
        raise TypeError("TorchScript split replay currently requires positional tensor inputs.")

    with torch.no_grad():
        reference_boundary = runtime.segments.prefix.forward(*sample_inputs)
        reference_output = runtime.segments.suffix.forward(reference_boundary)
        boundary_inputs = tuple(
            reference_boundary.tensors[label]
            for label in runtime.plan.boundary_nodes
        )

        prefix_module = _GeneratedModule()
        prefix_builder = _StaticSegmentBuilder(runtime, prefix_module)
        prefix_env = generated_replay._input_env(runtime.trace_graph, sample_inputs)
        prefix_lines = [
            "env={}",
            f"env[{generated_replay._RUNTIME_BATCH_ENV_KEY!r}]="
            f"{_runtime_batch_expression(sample_inputs, reference_boundary.batch_size)}",
        ]
        prefix_env[generated_replay._RUNTIME_BATCH_ENV_KEY] = (
            reference_boundary.batch_size
        )
        for label, value in prefix_env.items():
            if label not in runtime.trace_graph.input_nodes:
                prefix_lines.append(
                    f"env[{label!r}]={prefix_builder.value_expression(value)}"
                )
        for label, value in zip(
            runtime.trace_graph.input_nodes,
            sample_inputs,
            strict=True,
        ):
            del value
            input_index = runtime.trace_graph.input_nodes.index(label)
            prefix_lines.append(f"env[{label!r}]=inputs[{input_index}]")
        for label in runtime.plan.prefix_nodes:
            prefix_builder.append_node(
                runtime.trace_graph.get(label),
                prefix_env,
                prefix_lines,
            )
        boundary_values = ",".join(
            f"env[{label!r}].detach()" for label in runtime.plan.boundary_nodes
        )
        prefix_lines.append(f"return ({boundary_values},)")
        prefix_builder.install_forward(prefix_lines)

        suffix_module = _GeneratedModule()
        suffix_builder = _StaticSegmentBuilder(runtime, suffix_module)
        suffix_env = dict(reference_boundary.tensors)
        suffix_env[generated_replay._RUNTIME_BATCH_ENV_KEY] = (
            reference_boundary.batch_size
        )
        if reference_boundary.metadata.get(
            generated_replay._LIVE_PARAM_SOURCES_METADATA_KEY,
            runtime.plan.use_live_param_sources,
        ):
            suffix_env[generated_replay._LIVE_PARAM_SOURCES_ENV_KEY] = True
        suffix_lines = ["env={}"]
        suffix_lines.extend(
            f"env[{label!r}]=inputs[{index}]"
            for index, label in enumerate(runtime.plan.boundary_nodes)
        )
        suffix_lines.append(
            f"env[{generated_replay._RUNTIME_BATCH_ENV_KEY!r}]="
            f"{_runtime_batch_expression(boundary_inputs, reference_boundary.batch_size)}"
        )
        if suffix_env.get(generated_replay._LIVE_PARAM_SOURCES_ENV_KEY, False):
            suffix_lines.append(
                f"env[{generated_replay._LIVE_PARAM_SOURCES_ENV_KEY!r}]=True"
            )
        for label in runtime.plan.suffix_nodes:
            suffix_builder.append_node(
                runtime.trace_graph.get(label),
                suffix_env,
                suffix_lines,
            )
        for label in runtime.trace_graph.output_nodes:
            suffix_builder.append_node(
                runtime.trace_graph.get(label),
                suffix_env,
                suffix_lines,
            )
        suffix_lines.append("output=reconstruct_output(trace_graph,env)")
        output_paths: list[tuple[Any, ...]] = []
        template = _output_template(reference_output, output_paths)
        if not output_paths:
            raise TypeError("TorchScript split replay requires at least one tensor output.")
        output_expressions: list[str] = []
        for path in output_paths:
            expression = "output"
            for component in path:
                expression += f"[{suffix_builder.value_expression(component)}]"
            output_expressions.append(expression)
        suffix_lines.append(f"return ({','.join(output_expressions)},)")
        suffix_builder.install_forward(suffix_lines)

    with warnings.catch_warnings(), torch.inference_mode():
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        traced_prefix = torch.jit.trace(
            prefix_module,
            sample_inputs,
            strict=False,
            check_trace=False,
        )
        traced_suffix = torch.jit.trace(
            suffix_module,
            boundary_inputs,
            strict=False,
            check_trace=False,
        )

    return TorchScriptSplitReplay(
        prefix=traced_prefix,
        suffix=traced_suffix,
        boundary_labels=tuple(runtime.plan.boundary_nodes),
        boundary_specs=runtime.plan.boundary_specs,
        boundary_metadata=dict(reference_boundary.metadata),
        output_template=template,
    )


__all__ = ["TorchScriptSplitReplay", "build_torchscript_split_replay"]
