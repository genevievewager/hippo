# Visualizations

Public entry point:

```bash
python run_visualizations.py \
    --experiment outputs/<run> \
    --all \
    --compile-pdf
```

Visualization **reads saved results only**, does not retrain models, and is safe to run independently of decoding.

With only `--experiment`, the script detects available outputs and plots all of them.

## Saved artifacts that may feed figures

```text
decoder_comparison_metrics.csv
best_decoder_by_target.csv
best_lab_deployable_decoders.csv
closed_loop_trigger_comparison.csv
cross_run_decoder_summary.csv
decoder_control_summary.csv
population_ablation_summary.csv
sorted_information_loss_summary.csv
```

## CLI flags

| Flag | Purpose |
|------|---------|
| `--experiment` / `--input` | Experiment directory |
| `--output` | Figures directory (default `<experiment>/figures`) |
| `--all` | Every available figure type (default when only `--experiment` is set) |
| `--include-simulation` | Behavior / neural / probe figures |
| `--include-decoder` / `--include-comparison` | Decoder comparison figures |
| `--include-realtime` | Closed-loop / realtime figures |
| `--compile-pdf` | Compile `figures/**/*.png` into `figures/output.pdf` |
| `--rate-bin-size` | Bin size for rate/population plots (default 0.250 s) |

## Paper figure set

Default regeneration writes a small set of seaborn multi-panel `fig_*.png` files:

| Paper fig | Stem | Content |
|-----------|------|---------|
| Fig 1 | `fig_behavior_dynamics`, `fig_neural_drivers` | Spatial overview + covariates + neural drivers |
| Fig 2 | `fig_spikes_on_trajectory_by_class`, `fig_population_tuning` | Spikes-on-trajectory + 3×3 tuning overview |
| Fig 3 | `fig_circuit_feedforward`, `fig_population_structure`, `fig_population_activity` | Feedforward circuit + circuit-node spike raster + mean-rate traces |
| Fig 4 | `fig_decoding_performance` | Causal decoding vs window / best decoder |
| Fig 5 | `fig_manifold_decoding` | Feature × window heatmaps (cell = best decoder) |
| Fig 5a′ | `fig_manifold_decoder_window_threeway` | Full manifold × decoder × W table |
| Fig 5b–c | `fig_deployable_decoder_x_window_heatmaps`, `fig_manifold_vs_spikes_onepager` | Ideal W/decoder selection + counts vs manifold |
| Fig 6 | `fig_latent_geometry_<feature>` | All embeddings × one recovered feature per page |
| Fig 7 | `fig_isomap_diagnostics` | Trustworthiness, connectivity, residual variance, geodesic/knn |
| Fig 8 | `fig_isomap_story` | Counts/PCA/Isomap story + distilled vs teacher |
| Fig 9 | `fig_closed_loop` | Closed-loop position, confusions, triggers |
| Fig 10 | `fig_deployment` | Deployment winner summary + window×decoder heatmaps |
| Fig 11 | `fig_latency` | Causal-update latency budget |
| Fig 12 (suppl.) | `fig_temporal_wl` | Temporal W×L heatmaps (when `decoding/` exists) |

Fig 6 is a **suite**: one dense page per behavioral variable
(`position`, `speed`, `acceleration`, `head_direction`, `distance_to_wall`,
`spatial_context`, `movement_state`, `wall_distance_bin`). Each page shows every
embedding mode present in the sorted comparison at the **best-performing window**
(and k / n_neighbors) for decoding that variable — not a shared W.

Compiled `figures/output.pdf` follows: simulation → decoding → manifolds/Isomap → realtime → deployment → latency.

## Regenerating manifold / Isomap panels

Default `--profile manifolds` already includes region/layer PCA and classic + distilled Isomap. If an older run was counts/PCA-only, Isomap panels show an explicit empty state until you re-decode, then re-plot:

```bash
python run_decoder.py \
  --input outputs/ratinabox_005 \
  --output outputs/ratinabox_005 \
  --skip-visualization

python run_visualizations.py \
  --experiment outputs/ratinabox_005 \
  --all \
  --compile-pdf
```

## Probe trajectory figures

See [anatomy_and_trajectory.md](anatomy_and_trajectory.md). Standalone:

```bash
python -m hippo.visualization.probe_trajectory \
  --trajectory lab_npx2_default \
  --output outputs/ratinabox_006 \
  --make-3d
```

## Future / deferred

Not in the current suite: 3D interactive embeddings, unit-ablation movies, animated Bayesian place-map GIFs.
