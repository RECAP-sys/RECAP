# Privacy Reconstruction Attack Offline Experiment

This directory contains an isolated offline experiment for measuring whether
RECAP privacy leakage scores correspond to actual reconstruction leakage.
It does not modify online inference, continual-learning training, or baseline
scheduling.

The experiment evaluates split points with target privacy leakage scores
`0.8`, `0.6`, `0.4`, and `0.2`. When a split point is set to `auto`, the target
collector enumerates TorchLens split candidates and selects the nearest unique
candidate using RECAP's existing score definition:

```text
privacy_leakage_score = 1 - edge_parameter_ratio
```

Resolved split IDs are written to `resolved_split_points.json` and reused by the
reconstruction script.

## Reconstruction Methods

Two reconstruction methods are supported:

- `whitebox_feature_inversion`: Torch_SFL-style white-box feature inversion.
  It directly optimizes an input image so the exact edge-side split prefix
  reproduces the leaked boundary payload.
- `drag_linear_clean`: the DRAG-style baseline used for the previous
  linear-clean quantification figure.

`drag_linear_clean` is the DRAG-style reconstruction used for the previous
linear-clean quantification figure. It first tries to compute a clean
linear-pseudoinverse image from a white-box split prefix. If that closed-form
initialization is unavailable, it falls back to the legacy random Stable
Diffusion latent initialization, then updates the latents so that
the decoded image matches the leaked boundary payload:

```text
boundary payload -> linear image or random latent -> DRAG guidance -> reconstruction
```

The attacker is given the cloud-visible boundary payload and the exact
edge-side lightweight model checkpoint from which the split prefix is cut. Pass
that checkpoint with `--edge-prefix-weights`; the collection and reconstruction
manifests record the resolved path and SHA-256 hash.

## Full Run

```bash
# 1. Collect attack targets
python experiments/privacy_reconstruction_attack/collect_attack_targets.py \
  --yaml_path ./config/config.yaml \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --video_path ./video_data/road.mp4 \
  --output_dir ./outputs/privacy_reconstruction/targets \
  --num_frames 100 \
  --frame_stride 5 \
  --edge-prefix-weights ./model_management/models/rf-detr-nano.pth \
  --device cuda:0

# 2. Run DRAG reconstruction
python experiments/privacy_reconstruction_attack/drag_attack.py \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --targets_dir ./outputs/privacy_reconstruction/targets \
  --output_dir ./outputs/privacy_reconstruction/drag \
  --edge-prefix-weights ./model_management/models/rf-detr-nano.pth \
  --device cuda:0

# 2b. Or run white-box feature inversion reconstruction
python experiments/privacy_reconstruction_attack/feature_inversion_attack.py \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --targets_dir ./outputs/privacy_reconstruction/targets \
  --output_dir ./outputs/privacy_reconstruction/feature_inversion \
  --edge-prefix-weights ./model_management/models/rf-detr-nano.pth \
  --device cuda:0

# 3. Summarize evaluation
python experiments/privacy_reconstruction_attack/evaluate_privacy_score.py \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --attack_dir ./outputs/privacy_reconstruction/feature_inversion \
  --output_dir ./outputs/privacy_reconstruction/results

# 4. Plot the manuscript figure (Figure 9)
python experiments/privacy_reconstruction_attack/plot_attack_comparison_paper.py \
  --results-root ./outputs/privacy_reconstruction \
  --output-dir ./outputs/privacy_reconstruction/figures
```

## Outputs

Target collection writes one subdirectory per split and one subdirectory per
sample. Each sample contains `raw_frame.png`, `boundary_payload.pt.gz`,
`model_input_tensor.pt`, `boundary_feature.pt`, prediction JSON files, and
`metadata.json`.

The reconstruction scripts write `recon.png`, `raw.png`, `metrics.json`, and a
model-input `model_input_reference.png` for each reconstructed sample. Evaluation
writes `per_sample.csv`, compatibility `drag_per_sample.csv`, `summary_by_score.csv`,
and `score_correlation.json`.

The plotting script generates reconstruction grids and score-vs-leakage curves:
`reconstruction_grid.png/pdf`, `score_vs_object_f1.png/pdf`, and
`score_vs_actual_leakage.png/pdf`.
