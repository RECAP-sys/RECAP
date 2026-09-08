# RECAP

RECAP is a drift-aware, resource-constrained edge–cloud video analytics system. The repository contains one end-to-end implementation: fixed split planning, edge feature/sample caching, Lyapunov resource-aware triggering, teacher annotation, and cloud split-tail retraining.

## Quick start

Install the project environment:

```shell
python -m pip install --upgrade uv
uv sync --all-extras
```

Start the cloud service:

```shell
python cloud_server.py --yaml_path ./config/config.yaml
```

Start an edge client:

```shell
python edge_client.py --yaml_path ./config/config.yaml --headless
```

The runtime configuration is in [config/config.yaml](config/config.yaml). Experiment identity can be overridden with `--experiment_id`, `--scenario`, `--edge_count`, and `--repeat`; edge identity can be overridden with `--edge_id` and `--cache_path`.

## Architecture

- `edge/`: frame filtering, inference, drift detection, sample collection, feature shards, and model-update handling.
- `cloud/`: teacher annotation, feature cache, orchestration, edge-affine workers, and split-tail training.
- `model_management/`: detector adapters, fixed split planning, split runtime, and inference utilities.
- `grpc_server/`: the RECAP edge–cloud contract and asynchronous continual-learning jobs.
- `common/`: experiment identity, artifact handling, video identity, and logging utilities.
- `experiments/`, `tools/`, and `scripts/`: RECAP evaluation, privacy, drift, and motivation experiments.

## Tests

Run the lightweight suite with:

```shell
pytest
```

Generated protobuf modules live under `grpc_server/`. If the protocol changes, regenerate them with:

```shell
uv run --no-project --with grpcio-tools==1.80.0 python -m grpc_tools.protoc \
  -I ./grpc_server/protos --python_out=./grpc_server --pyi_out=./grpc_server \
  --grpc_python_out=./grpc_server ./grpc_server/protos/message_transmission.proto
```
