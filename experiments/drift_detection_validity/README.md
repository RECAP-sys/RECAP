# RECAP Real Weather Drift Detection Test

This offline experiment evaluates whether RECAP's unlabeled drift signals
detect harmful real-weather shifts on exactly two weather-named videos:

- `rainy.mp4`: rainy
- `snowy.mp4`: snowy

Scene metadata must be explicit and ordered as `rainy` and `snowy`;
the runner does not infer missing scene names from file names.

## Evaluation

For each configured scene, the script samples real video frames uniformly after
the configured start and end margins. The student detector output is compared
against teacher pseudo-labels on the same real frame with IoU=0.5 and
class-aware matching.

The rainy scene defines the clean baseline. Snowy windows are labeled
as harmful drift only when their teacher-pseudo-label F1 drops beyond
`window.harmful_f1_drop_threshold`. RECAP drift scores are computed from
student-side unlabeled signals and evaluated against those harmful-window
labels. Online trigger replay uses only the unlabeled signal values; teacher
pseudo-labels are used only for offline evaluation.

The reported detection precision, recall, and F1 are student-vs-teacher
pseudo-label consistency metrics, not human-ground-truth accuracy. The drift
metrics report whether RECAP detects the harmful consistency drop.

## Run

Full two-scene evaluation:

```bash
python experiments/drift_detection_validity/run_all.py --config experiments/drift_detection_validity/configs/drift_detection_validity.yaml
```

## Outputs

Outputs are written under `results/drift_detection_validity/<run_id>/`:

- `records/real_weather_frame_metrics.csv`: frame-level precision, recall, F1,
  TP/FP/FN, box counts, and confidence summaries.
- `records/real_weather_scene_summary.csv`: scene-level micro and mean metrics.
- `records/real_weather_predictions.json`: normalized student and teacher
  predictions for sampled frames.
- `records/frame_signals.csv`: frame-level RECAP drift signals.
- `records/window_metrics.csv`: window-level harmful-drift labels and signal
  summaries.
- `analysis/signal_validity_summary.csv`: correlation/AUC/best-threshold
  validity of each unlabeled signal.
- `analysis/online_trigger_method_summary.csv`: online trigger precision,
  recall, trigger-F1, misses, false triggers, and delay.
- `figures/real_weather_scene_metric_summary.png`: scene-level metric bar plot.
- `figures/real_weather_drift_detection_effectiveness.png`: F1 drop and
  RECAP drift-score overlay.
- `figures/*_student_teacher_examples.png`: visual examples for each scene.
- `plots/figure_signal_validity_summary.{png,svg,pdf}`: Signal validity summary.
- `plots/figure_online_trigger_summary.{png,svg,pdf}`: Online trigger summary.
- `real_weather_scene_report.html`: compact HTML report linking the plots.
