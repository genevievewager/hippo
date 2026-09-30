# Architecture

Hippocampal BCI simulation and decoding testbed. The UI orchestrates backend
functions; it does not own scientific logic.

---

## Repository layout

```text
hippo/                 core library (dataset, anatomy, partitions, probe viz)
hippo_sim/             simulation backends
realtime/              decoding, representations, quadrant pipeline
visualization/         publication / report plots
ui/                    Streamlit app
agents/                pressure-test and experiment agents (do not import into core)
analysis/              offline experiments / workstreams (may import core; never imported by core)
configs/               tracked configs (no data, no secrets)
scripts/               shell helpers and sync utilities
tests/                 pytest suite
docs/                  notes and design docs
run_*.py               public CLI entry points (repo root)
```

### Import and data rules

1. `hippo/`, `realtime/`, and `hippo_sim/` never import from `analysis/`,
   `scripts/`, or `agents/`.
2. `analysis/` may import `hippo`, `hippo_sim`, `realtime`, and `visualization`.
3. Data and run outputs live outside git. Prefer `HIPPO_DATA_ROOT` for the data
   root; experiment outputs go under a local `outputs/` (or an explicit run
   directory), never into the package trees.
4. Code moves from `analysis/` into `hippo/` / `realtime/` only once a second
   consumer needs it. Do not prematurely “promote” one-off notebooks or
   offline scripts.

Active pipeline window ≠ window benchmark grid. Sweeps must be explicit and
separately hashed. See configuration layers below.

---

## Pipeline overview

```text
Behavior
    ↓
Simulation / recorded spikes
    ↓
Causal Feature Construction (F, W)     ← ObservationConfig owns W
    ↓
Neural Representation (E)
    ↓
Decoder (D)
    ↓
Evaluation
    ↓
Quadrant Comparison                    ← scientific spine
    ↓
Realtime Replay
    ↓
Deployment / Closed Loop (C)
```

Public CLI remains:

```text
run_simulation.py → run_decoder.py → run_visualizations.py
```

Streamlit follows the same backends:

```text
Experiment Setup
  → Neural Simulation
  → Feature Construction
  → Latent Representations
  → Decoder Benchmark
  → Quadrant Comparison
  → Realtime Replay
  → Live Deployment
```

---

## Scientific spine: representation quadrants

A **quadrant** is a scientific category (linearity × temporal type). A
**method** is one implementation. Metadata lives in
`realtime/representation_registry.py` (`RepresentationSpec`). Do not hardcode
method names in many places when the spec already has `linearity`,
`temporal_type`, `realtime_capable`, and `offline_only`.

```text
                    STATIC                          DYNAMIC
LINEAR              linear_static                   linear_dynamic
                    PCA-family (default:            LDS (default: global_lds)
                    global_pca)

NONLINEAR           nonlinear_static                nonlinear_dynamic
                    Diffusion maps + Nyström        not implemented
                    (default: diffusion_nystrom)    (placeholders only)
```

Implemented methods (registry; aliases resolved to canonical names):

| Quadrant | Implementations |
| -------- | --------------- |
| Linear static | `identity` / `counts`, `global_pca`, `region_pca`, `layer_pca`, `cell_type_pca`; also `rate_model_pca`, `pls` (ambiguous / supervised) |
| Nonlinear static | `diffusion_nystrom`, `global_isomap` (offline-only), `global_isomap_distilled` |
| Linear dynamic | `global_lds` (causal / realtime), `gpfa` (offline RTS smoother — **not** nonlinear-dynamic) |
| Nonlinear dynamic | unimplemented (`lfads`, `switching_lds`, `recurrent_slds` placeholders) |

`layer_pca` remains an internal/anatomy method and is excluded from the public
UI quadrant pickers.

Feature modes, windows, targets, decoders, and simulation parameters stay
available as experimental factors around this 2×2. They must not create a
competing configuration source or hide the quadrant comparison.

---

## Configuration layers (do not conflate)

| Layer | What it is | Where it lives |
| ----- | ---------- | -------------- |
| **Scientific analysis** | `F`, `E`, `D`, `W`, target, causal, train/test | `ObservationConfig` + `AnalysisConfig`; `pipeline_run.json` |
| **Benchmark grid** | Intentional F×E×D×W sweep | `ComparisonRunConfig` / Decoder Benchmark Targeted–Full |
| **Visualization** | figure selection, plot toggles, PDF flags | viz services / `run_visualizations.py` |
| **UI state** | tabs, expanders, job flags, active dataset path | `ui/state.py` session keys |
| **Runtime / deployment** | live bundle, closed-loop policy, Replay I/O | deployment artifacts under the experiment dir |

The UI may **modify** the canonical analysis configuration. The UI must not
**become** the source of scientific logic (no duplicate train/test, window,
or feature construction inside Streamlit callbacks).

---

## Window ownership

**Active pipeline window:** committed on Feature Construction, stored on
`PipelineRun.observation`, displayed downstream as inherited. Changing it
stale-marks representation → decoder → replay.

**Window benchmark grid:** an explicit sweep (Decoder Benchmark Targeted/Full,
Quadrant Comparison Advanced, optional Realtime Replay sweep). Each `W` is a
separate observation hash / experiment. Selecting a sweep must be a visible
checkbox, not a silent second slider that mutates the pipeline.

Downstream pages may:

* display the active window,
* compare results that were precomputed across multiple windows,
* run an explicit labeled sweep,

and must not silently replace upstream features or representations.

---

## Canonical configuration flow

```text
Experiment Setup     → active dataset, spike source (UI + dataset files)
Neural Simulation    → spikes, behavior, units
Feature Construction → commit ObservationConfig (W, F, update_dt, source)
                       FeatureDataset cache keyed by observation hash
Latent Representations → RepresentationResult inherits observation
Decoder Benchmark    → pipeline mode: inherited W
                       benchmark mode: explicit window grid
Quadrant Comparison  → one ObservationConfig; branches are E only
Realtime Replay      → model's saved observation; reject offline-only E
Live Deployment      → frozen bundle; same observation contract
```

Helpers:

* `get_active_analysis_config()` / `set_active_analysis_config()` in `ui/state.py`
* `AnalysisConfig.resolve(observation, overlay)` ignores overlay `window_s`
* `validate_analysis_config()` / `assert_*` in `realtime/pipeline_invariants.py`

---

## CLI and UI

| Task | Shared backend |
| ---- | -------------- |
| Decoder search | `realtime.decoder_comparison.run_decoder_comparison` |
| Quadrant comparison | `realtime.quadrant_experiment.run_controlled_quadrant_experiment` |
| Realtime replay | `realtime.evaluate_realtime` |
| Visualizations | `visualization/` via `run_visualizations.py` / `ui/services/visualizations.py` |

UI adapters (`ui/services/*`) build configs and call those functions. They
must not reimplement fitting, splitting, or metrics.

---

## Historical inconsistencies (audit)

These existed before the architecture tightening. Several are now guarded;
others remain as listed technical debt.

1. `KEY_SELECTED_ANALYSIS_CONFIG` existed but was unused; scientific knobs
   lived as per-page widgets (`quad_target`, `bench_windows`, …).
2. `ObservationConfig` already owned `W`, but Decoder Benchmark / Realtime
   Replay could still present independent window selectors (gated to cached
   `W`, but easy to treat as “the” pipeline window).
3. Shared Streamlit key `pipeline_bench_sweep_windows` leaked sweep UI state
   across pages.
4. `ComparisonRunConfig` (grid) and pipeline observation were easy to
   conflate.
5. `ui.services.realtime.run_quadrant_comparison` (realtime-capable replay)
   is a different experiment from `quadrant_experiment` (controlled 2×2).
6. Manifold explorer still has per-quadrant window checkboxes in session
   state (benchmark-style, not active pipeline).
7. Representation metadata was centralized in the registry; some UI pages
   still used legacy quadrant id aliases (`static_linear`, …).
8. Expensive LDS causal filtering could look “stuck” at ~40% because
   progress only advanced after decoder eval (performance, not config).

---

## Performance notes (no major rewrite)

Separate computational dependencies from visual ones:

* Feature matrices are cached by `ObservationConfig.hash()`.
* Changing decoder or plot options must not refit `F` or `E`.
* `AnalysisConfig.hash()` changing invalidates decoder-level cached
  *identity*, not the observation cache.
* Quadrant comparison sets `reuse_transforms=True` and reuses
  `decoder_comparison/` caches.

Obvious remaining cost: large causal LDS filtering; decoder grids; Isomap.
Do not “fix” those by silently dropping methods.
