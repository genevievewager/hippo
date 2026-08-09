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
  ground_truth/           # oracle / non-deployable (optional)
  dynamic/<method>/       # LDS/GPFA association tables when run
  models/                 # transforms + .joblib under source subdirs
```

Key files (per source):

- `decoder_comparison_metrics.csv`
- `best_decoder_by_target.csv` / related selection tables
- `models/manifold_transforms/...`
- `models/*.joblib`

## Deployment registry

| Path | Contents |
|------|----------|
| `models/best_realtime_decoders.json` | Lab-transplant / deployable registry (sorted only) |
| `deployment_decoder_selection/all_sorted_window_scores.csv` | Full score table |
| `deployment_decoder_selection/best_decoder_by_target_sorted.csv` | Sorted best table |
| `deployment_decoder_selection/best_realtime_decoders.json` | Selection copy |

The registry references saved transforms and decoder `.joblib` artifacts under `decoder_comparison/sorted/`.

## Realtime / latency

```text
realtime_decoding/
  sorted/{target}_{policy}/   # deployable replay
  dynamic/                    # LDS (and other RT dynamic) replays when run
latency_profiling/
  latency_summary.csv/.json + stage tables
decoding/                     # optional temporal W×L comparison
```

## Figures layout

```text
figures/
  trajectory/                 # probe_trajectory_*, channel_region_map, unit_count_*
  behavior/                   # fig_behavior_dynamics
  features/                   # fig_neural_drivers
  neural/                     # population / tuning / feedforward panels
  sorting/
  report/
  decoder_comparison/         # fig_decoding_performance, fig_manifold_*, fig_isomap_*, …
  realtime_decoding/          # fig_closed_loop
  deployment_decoder_selection/  # fig_deployment
  latency/                    # fig_latency
  dynamic/<method>/           # latent trajectory / association figures
  temporal_decoding/          # fig_temporal_wl
  output.pdf
```

Figures live in subfolders; compiled PDF is `figures/output.pdf`. See [visualizations.md](visualizations.md) for the paper figure catalog.
