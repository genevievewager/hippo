# D. Lab-matched sim capability (read-only inventory)

## §3.2 matched conditions — what the simulator accepts without code edits

| Condition | Accepts via public args/config? | Function / config key | Wrapper needed? | Can't be done? |
| --------- | ------------------------------- | --------------------- | --------------- | -------------- |
| Per-region unit counts (MEC/SUB fractions) | **Partial.** Absolute counts/densities yes; lab fraction targeting no single knob. | `SimConfig.ratinabox_params` (`n_mec_grid_cells`, `n_mec_hd_cells`, `n_mec_speed_cells`, `n_sub_bvc_cells`, …); trajectory `cell_capture.*.unit_density_per_mm` + `candidate_cell_classes` in `configs/trajectories/lab_npx2_default_cell_capture.yaml`; `apply_trajectory_to_config(..., cell_capture_config=)`. Region inclusion list: `configs/quadrant_n5.yaml` → `unit_inclusion.regions`. | Wrapper to set densities / ratinabox_params before `generate_dataset`. Mapping lab Region→sim band is ours. | Exact MEC/SUB **fraction** matching is not a first-class API — only absolute n / density. Note: `hippocampal_formation_transition` units are captured but `include_in_decoder=False` in Report 1 outputs. |
| Session / segment duration | **Yes.** | `SimConfig.session_duration_s` (default 600); also `configs/quadrant_n5.yaml` `session.session_s`; `generate_seed_dataset` reads cfg session. | Thin wrapper when regenerating. | Segment-style mid-session room cuts: **not** modeled (sim is one continuous session). Matching lab retained length = set `session_duration_s` (or chunk post hoc, which we already do). |
| Arena boundary polygon | **No** (square only). | `SimConfig.arena_size_cm` (default 100); RatInABox env `scale` + solid square bounds in `hippo_sim.behavior.simulate_behavior`. | N/A for polygon. | Arbitrary lab `preprocessing.boundary` polygons **cannot** be passed in without code edits. Confound: square vs lab polygon. |
| External position trajectories | **No** public loader. | Behavior is always `simulate_behavior(config)` (RatInABox random exploration). No arg to inject `behavior.csv` / lab `postions_dataset.csv`. | Would require new wrapper that bypasses/replaces `simulate_behavior` — that is new pipeline code, not a public hook. | Replay of lab trajectories **not** supported through public API. |

## Other public parameters (preview for §6.1; no calibration_space.yaml yet)

### `SimConfig` fields (`hippo_sim.config.SimConfig`)

| Name | Default | Controls |
| ---- | ------- | -------- |
| `output_dir` | `outputs/` | Where sim artifacts are written |
| `seed` | 42 | RNG seed for the run |
| `session_duration_s` | 600 | Session length (s) |
| `behavior_dt` | 0.05 | Behavior / rate update step (s) |
| `sample_rate_hz` | 30000 | Recording sample rate |
| `n_channels` | 384 | Probe channels |
| `site_pitch_um` | 20 | Site spacing |
| `arena_size_cm` | 100 | Square arena side (cm) |
| `thigmotaxis` | 0.8 | Wall-hugging strength in RatInABox agent |
| `region_segments` | `REGION_SEGMENTS` | Schematic anatomy bands (overridden by trajectory) |
| `rate_params` | `RATE_PARAMS` | Per-cell-type rate / theta / place / HD / speed / boundary weights |
| `ripple_params` | `RIPPLE_PARAMS` | Ripple event rate/duration/amplitude |
| `recording_params` | `RECORDING_PARAMS` | Template span, amplitude range, noise, collisions, motion drift |
| `sorting_params` | `SORTING_PARAMS` | Detection threshold, miss/FP/merge/contamination/jitter |
| `ratinabox_params` | `RATINABOX_PARAMS` | Population sizes (`n_*_cells`), rate_scale, speed encoding, feedforward, INT anti-weights |
| `trajectory_meta` | `{}` | Filled by `apply_trajectory_to_config` |
| `anatomy_table` | `[]` | Probe anatomy rows |
| `cell_capture_config` | `{}` | Capture densities / class mixes |
| `probe_type` | `neuropixels_1.0` | Probe label |

### CLI / `apply_trajectory_to_config` (`run_simulation.py`, `hippo_sim.pipeline`)

| Name | Default | Controls |
| ---- | ------- | -------- |
| `--trajectory-config` | lab default yaml | Probe insertion / anatomy regions file |
| `--trajectory-export` | optional | Neuropixels trajectory explorer export |
| `--anatomy-regions-file` | optional | Explicit region-depth CSV |
| `--cell-capture-config` | optional | Per-band `unit_density_per_mm` + class mix |
| `--fallback-schematic-anatomy` | flag | Schematic bands if no export |
| `--include-non-hippocampal-regions` | flag | Keep VIS etc. |

### Entry points

- `run_simulation.generate_dataset(config)` → `hippo_sim.pipeline.run_pipeline`
- `realtime.quadrant_n5_run.generate_seed_dataset(cfg, seed_index, output_dir=...)` (quadrant_n5 frozen cfg + seed streams)

Report 4 currently **reads copied Report 1 sim outputs**; regeneration with matched params is deferred until calibration (§6.1).
