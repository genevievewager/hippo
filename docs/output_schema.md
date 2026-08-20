# Output schema

Concise tree (see root README for the public summary):

```text
outputs/<run>/
├── behavior.csv
├── units.csv
├── spikes_ground_truth.csv
├── spikes_sorted.csv
├── summary.json
├── rates.npy
├── anatomy_regions.csv
├── ratinabox_group_metadata.csv
├── neural_backend_metadata.json
├── trajectory_metadata.json
├── trajectory/
├── decoder_comparison/
├── models/
│   └── best_realtime_decoders.json
├── deployment_decoder_selection/
├── deployment_bundles/
├── live_sessions/
├── realtime_decoding/
├── latency_profiling/
├── decoding/
└── figures/
    └── output.pdf
```

## Simulation outputs

| File | Description |
|------|-------------|
| `behavior.csv` | Position, speed, head direction over time |
| `anatomy_regions.csv` | Region geometry and channel mapping |
| `units.csv` | Per-unit metadata (type, region, channel, place field, `ratinabox_class`) |
| `spikes_ground_truth.csv` | True spike times |
| `spikes_sorted.csv` | Re-extracted spike times after degradation |
| `ratinabox_group_metadata.csv` | Per-population RiaB group summary |
| `rates.npy` | Firing rates (Hz), shape `(n_units, n_steps)` |
| `summary.json` | Run statistics (includes `behavior_dt`) |
| `neural_backend_metadata.json` | Rate model / feedforward metadata |

## Trajectory snapshots

| Path | Contents |
|------|----------|
| `trajectory/active.json` | Active insertion (name, AP/ML/DV, paths) |
| `trajectory/active_trajectory.yaml` | Snapshot of the config used |
| `trajectory/anatomy_regions_used.csv` | Region–depth table driving capture |
| `trajectory/cell_capture.yaml` | Cell-capture rules copy |
| `trajectory_metadata.json` | Coords + uncertainty flags at trial root |

## Decoder comparison

```text
decoder_comparison/
  sorted/                 # deployment-relevant
    models/manifold_transforms/
    models/feature_transforms/
    models/neural_feature_extractors/
  ground_truth/           # oracle / non-deployable (optional)
  dynamic/<method>/       # LDS/GPFA association tables when run
```

Key files (per source):

- `decoder_comparison_metrics.csv` — one row per F×E×D×W×target, including `config_id`
- `decoder_comparison_metrics.json`
- `predictions/<config_id>.parquet` — held-out test traces (see below)
- `decoded_examples/best_{target}_predictions.csv` — overall winner only (legacy)
- `best_decoder_by_target.csv` / related selection tables (`best_config_id` on the winner)
- `models/manifold_transforms/...`  # fitted `E` (UI Latent Representations + comparison reuse)
- `models/feature_transforms/...`   # fitted `F` extractors when used
- `models/neural_feature_extractors/...`
- `models/*.joblib`

### `config_id`

Deterministic, filesystem-safe identity (not a UUID) from:

`spike_source`, `target_name`, `decoder_name`, `feature_set`, `feature_mode`, `embedding_type`, `decode_window_s`, `manifold_n_components`, `n_neighbors`, `n_landmarks`

The same configuration in the same experiment always resolves to the same id. Failed/skipped fits still receive an id but **no** parquet.

### Held-out prediction parquet

Path: `decoder_comparison/<spike_source>/predictions/<config_id>.parquet` (snappy).

Always includes `config_id`, `time`, `split=held_out_test`, and causal window columns. Target-specific columns:

| Target | Columns |
|--------|---------|
| `position` | `true_x`, `true_y`, `pred_x`, `pred_y`, `error_cm` (Euclidean) |
| `speed`, `acceleration`, `distance_to_wall` | `true`, `pred`, `residual` where **residual = pred − true** |
| `head_direction` | `true_deg`, `pred_deg`, `circular_error_deg` (shortest arc), plus sin/cos |
| categoricals | `true`, `pred`, optional `proba_<class>`; class order in parquet metadata `class_labels` |

These are **offline held-out test** traces, not realtime replay. Legacy runs without this directory still have aggregate metrics; Decoder Benchmark diagnostics will ask you to re-run.

## Deployment registry

| Path | Contents |
|------|----------|
| `models/best_realtime_decoders.json` | Lab-transplant / deployable registry (sorted only) |
| `deployment_decoder_selection/all_sorted_window_scores.csv` | Full score table |
| `deployment_decoder_selection/best_decoder_by_target_sorted.csv` | Sorted best table |
| `deployment_decoder_selection/best_realtime_decoders.json` | Selection copy |

The registry references saved transforms and decoder `.joblib` artifacts under `decoder_comparison/sorted/`.

## Live bundles and sessions

Packed from the Streamlit **Live Deployment** page (not by `run_decoder.py`):

```text
deployment_bundles/<target>__<decoder>__w####ms/
  config.json, metadata.json, unit_order.json, feature_config.json
  decoder.joblib
  embedding/              # optional fitted E
live_sessions/session_YYYYMMDD_HHMMSS/
  deployment_config.json, predictions.csv, runtime_metrics.csv
  unit_mapping.json, events.log
```

See [realtime_deployment.md](realtime_deployment.md#live-runtime).

## Realtime / latency

```text
realtime_decoding/
  sorted/{target}_{policy}/   # deployable registry replay
  dynamic/                    # LDS (and other RT dynamic) replays when run
  quadrants/                  # UI Realtime Replay (three realtime-capable E)
  quadrant_comparison.json    # sidecar for quadrant figures
latency_profiling/
  latency_summary.csv/.json + stage tables
decoding/                     # optional temporal W×L comparison
```

## Figures layout

```text
figures/
  trajectory/                 # probe_trajectory_*, channel_region_map, unit_count_*
  behavior/                   # fig_behavior_dynamics
  features/                   # fig_neural_drivers; UI fig_feature_panel_*
  manifolds/                  # UI fig_winner_counts_*, fig_winner_manifold_*
  neural/                     # population / tuning / feedforward panels
  sorting/
  report/
  decoder_comparison/         # fig_decoding_performance, fig_manifold_*, fig_isomap_*, …
  realtime_decoding/          # fig_closed_loop; UI fig_quadrant_*
  deployment_decoder_selection/  # fig_deployment
  latency/                    # fig_latency
  dynamic/<method>/           # latent trajectory / association figures
  temporal_decoding/          # fig_temporal_wl
  output.pdf
```

Figures live in subfolders; compiled PDF is `figures/output.pdf`. See [visualizations.md](visualizations.md) for the paper figure catalog.
