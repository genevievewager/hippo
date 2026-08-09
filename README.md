# Hippocampal Neuropixels Simulation and Decoding

End-to-end hippocampal BCI simulation and deployment testbed: open-field behavior → hippocampal population rates → Neuropixels acquisition degradation → spike sorting → causal neural features → static or dynamic neural representations → behavioral decoding → deployment selection → realtime replay → closed-loop policy.

```text
Behavior
   ↓
Hippocampal population simulation
   ↓
Neuropixels acquisition degradation
   ↓
Spike sorting
   ↓
Causal neural features
   ↓
Static OR dynamic neural representation
   ↓
Behavioral decoder
   ↓
Deployment selection
   ↓
Realtime replay
   ↓
Closed-loop policy
```

The complete BCI design space spans

```text
F × E × D × W × C
```

where:

* `F` = neural observation construction from spikes
* `E` = population-state representation applied to that observation
* `D` = behavioral decoder
* `W` = causal spike integration window
* `C` = closed-loop rule

Architecturally:

```text
spikes → F → neural observation → E → latent/state → D → prediction → C (policy)
```

The core decoder benchmark searches `F × E × D × W`. Closed-loop rule `C` is then evaluated on decoded predictions (and during registry replay), not as an additional axis of the same Cartesian decoder grid. Selection uses **sorted spikes only**; ground-truth spikes are oracle / diagnostic / non-deployable. Figures are a separate inspection step and never retrain models.

**Purpose.** Identify a maximally compatible hippocampal BCI decoder under causal and realtime constraints: compare static manifolds and dynamic latents under the same decoder zoo, select deployable configurations on sorted-spike held-out performance (with shortest-near-optimal windows and realtime-compatible representations), export a lab transplant registry, and evaluate closed-loop policies on that registry. Calibration, sorting-degradation summaries, controls, ablations, and cross-run generalization are implemented as additional validation / alternate selection paths (see [Deployment selection](#deployment-selection)).

Equations use plain Unicode / code formatting (no LaTeX math plugin required).

---

## Scientific questions

1. Which neural features retain behaviorally useful information after Neuropixels-like recording degradation and spike sorting?
2. Do anatomically structured or nonlinear population representations improve decoding over raw population activity?
3. Do dynamic latent-state representations outperform static neural manifolds under causal constraints?
4. How much causal neural history is optimal for different behavioral variables?
5. Which high-performing configurations remain deployable after sorted-spike evaluation, shortest-near-optimal window selection, and realtime-compatibility constraints (with optional calibration / robustness / cross-run analyses)?
6. Which decoded neural variables remain robust enough under recording degradation and across runs to support closed-loop BCI operation?

Results answering (3) are **not yet fixed** in this README; LDS/GPFA are implemented and runnable (see [Current results](#current-results)). Longitudinal plasticity via a slowly varying observation mapping `C_t` remains a future direction (dynamic latent state ≠ neural plasticity).

---

## Public workflow

Three scripts **or** the Streamlit UI. The happy path searches `F × E × D × W` inside one comparison (and evaluates closed-loop `C` on the resulting predictions / registry replay); you do not pick windows or models by hand for normal use. `E` may be a **static manifold** (`x_t → z_t`) or a **dynamic latent state** (`z_(t−1), x_t → z_t`).
```text
Simulate → Decode / Search / Gate / Export → Visualize
```

### Installation

```bash
cd ~/projects/hippo
python3 -m venv .hippo
source .hippo/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

Python 3.10+ (developed on 3.12). See `requirements.txt`.

### Streamlit UI

Same backends as the CLI (`generate_dataset`, decoder comparison, visualizations). The UI does **not** reimplement the science.

```bash
streamlit run ui/app.py
```

```text
Experiment Setup
→ Neural Simulation
→ Feature Explorer
→ Manifold Explorer
→ Static vs Dynamic
→ Decoder Benchmark
→ Realtime Replay
```

Use **Experiment Setup** to generate or load a dataset (sets the shared **Active Dataset**), then continue through the pages above.

### CLI happy path

```bash
# 1. Simulate
python run_simulation.py \
    --output outputs/ratinabox_001 \
    --trajectory hpc_optimal \
    --seed 1

# 2. Search + select + replay
python run_decoder.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001 \
    --skip-visualization

# 3. Visualize saved results
python run_visualizations.py \
    --experiment outputs/ratinabox_001 \
    --all \
    --compile-pdf
```

| Step | Script | Role |
|------|--------|------|
| 1 | `run_simulation.py` | Behavior + hippocampal rates + Neuropixels degradation + sorting |
| 2 | `run_decoder.py` | Compare `F×E×D×W`, select deployable registry, closed-loop replay (default `--profile manifolds`) |
| 3 | `run_visualizations.py` | Read saved outputs only; write figures / PDF |

List trajectories with `python run_simulation.py --list-trajectories`. Default trajectory when `--trajectory` is omitted is `lab_npx2_default` (lab NP2.0). `hpc_optimal` provides the broader CA1→CA2→CA3→DG→Subiculum→MEC simulation stack used in the example above.

What Step 2 does under the hood (sorted spikes by default):

1. Compare candidate `F × E × D × W` configurations → `decoder_comparison/sorted/` (also scores default closed-loop trigger rules `C` on decoded predictions)
2. Write the deployable registry → `models/best_realtime_decoders.json` and `deployment_decoder_selection/`
3. Replay causal closed-loop decoding from that registry → `realtime_decoding/sorted/`
4. Optional in-decode figures (prefer Step 3 instead)

### Important output

```text
models/best_realtime_decoders.json
```

This is the **lab-transplant / deployment registry**. It stores per-target deployable decoder, `W`, feature mode, and paths to saved transforms and `.joblib` decoder artifacts under `decoder_comparison/sorted/`.

Also check before transplant / realtime review:

| Path | Contents |
|------|----------|
| `deployment_decoder_selection/all_sorted_window_scores.csv` | Full target × decoder × window scores |
| `realtime_decoding/sorted/` | Causal closed-loop replay from the registry |
| `latency_profiling/` | Stage latencies vs the 50 ms budget |
| `figures/output.pdf` | Compiled inspection PDF after Step 3 |

### Task table

| Goal | Command |
|------|---------|
| Launch UI | `streamlit run ui/app.py` |
| Simulate dataset | `run_simulation.py` |
| Search representations / decoders / windows | `run_decoder.py` |
| Generate figures / PDF | `run_visualizations.py` |

### Public vs advanced

| Usage | What you run |
|-------|----------------|
| **Public / standard** | UI **or** the three-script happy path above |
| **Advanced / developer** | Profile overrides, GT diagnostics, temporal `W×L`, static+dynamic grids, `run_decoder_comparison.py`, `run_BCI.py` |

Advanced flags (not needed for the happy path):

- `--profile quick` — coarse `W`, counts + global/region PCA smoke
- `--profile standard` — denser `W` pool, counts + PCA
- `--profile manifolds` — **default**; PCA family + classic/distilled Isomap
- `--profile full` — densest `W` incl. 25 ms + full model zoo (temporal still opt-in)
- `--include-ground-truth-diagnostics` — oracle GT comparisons (non-deployable)
- `--enable-temporal-manifold` — explicit latent-history `W×L` comparison
- static vs dynamic grids via `run_decoder_comparison.py` (`--dynamic-latents global_lds gpfa`)

Full flag tables, profiles, detached/`nohup` examples, and developer grids: [`docs/cli_reference.md`](docs/cli_reference.md).

---

## System architecture / simulation pipeline

```text
RatInABox behavior
→ behavioral/neural covariates
→ hippocampal rate model
→ Poisson spike generation
→ Neuropixels observation model
→ Kilosort-like re-extraction
→ causal decoding
```

### Timing

Behavior is sampled at **20 Hz / 50 ms frames** (`behavior_dt = 0.05` in `hippo_sim/config.py`). Decoder update cadence and causal windows are separate:

| Symbol | Default (public) | Meaning |
|--------|------------------|---------|
| `behavior_dt` | **0.050 s** (20 Hz) | Behavioral / rate frame interval |
| `update_dt` | **0.050 s** (20 Hz) | Decoder prediction cadence (`--update-dt`; **0.025 supported**) |
| `W` | searched | Causal spike integration window `[t−W, t)` |

Profile `W` grids (from `realtime/workflow_profiles.py` / `adaptive_windows.py`):

| Profile | Candidate `W` (s) |
|---------|-------------------|
| `manifolds`, `quick` | 0.050, 0.250, 0.500, 1.000 |
| `standard` | 0.050, 0.100, 0.250, 0.500, 1.000 |
| `full`, `feature_robustness` | 0.025, 0.050, 0.100, 0.250, 0.500, 1.000 |

UI dynamic-latent defaults and some developer examples use `update_dt=0.025` (40 Hz). The public CLI default remains 20 Hz.

### Simulation outputs (core files)

| File | Description |
|------|-------------|
| `behavior.csv` | Position, speed, head direction over time |
| `units.csv` | Per-unit metadata (type, region, channel, rate model) |
| `spikes_ground_truth.csv` | True spike times (oracle / diagnostic) |
| `spikes_sorted.csv` | Sorted spikes after degradation (deployment-relevant) |
| `rates.npy` | Firing rates (Hz), `(n_units, n_steps)` |
| `summary.json` | Run statistics including `behavior_dt` |
| `anatomy_regions.csv` | Region–depth / channel mapping |

Full artifact tree: [`docs/output_schema.md`](docs/output_schema.md).

---

## Hippocampal neural model

Activity is built in three stages: RatInABox receptive fields → cell-type dynamics overlays → trisynaptic / entorhinal feedforward.

| Population | Role |
|------------|------|
| CA1 phase-precessing place cells | Place + theta phase precession |
| CA3 place cells / recurrence | Autoassociative / recurrent gain |
| DG sparse place cells | Narrow fields + sparsity |
| CA2 place cells | Intermediate CA place coding |
| MEC grid cells | Multi-module spatial grid |
| MEC head-direction cells | Preferred head direction |
| MEC speed cells | Linear speed encoding |
| Subiculum boundary-vector cells | Wall distance / angle |
| Local interneuron pools (`INT_*`) | Home-region inhibition |

```text
MEC → DG → CA3 → CA1
MEC → CA1
MEC → CA2
CA3 → CA2
local INT_X → home region X
```

**Stage A** — RatInABox (or documented fallback) supplies nonnegative rates from shared behavior `(p(t), θ(t), v(t))`.

**Stage B** — within-cell-type overlays (parameters from `RATE_PARAMS`):

```text
R ← R_RiaB
R ← R · (1 + w_θ cos φ_θ(t))                 # theta (skipped for CA1_place_pp)
R ← R · (1 + w_speed · f_speed(t))           # speed gain
R ← R + w_ripple · A · r(t)                  # sharp-wave ripple envelope
R ← 0.1·b  if  R/max_t(R) < θ_sparse else R  # DG sparsity
R ← R + w_rec · mean_{j∈G} R_j(t)            # CA3 recurrent
R ← max(R, 0)
```

**Stage C** — region-mean feedforward (optional max-normalization `R̃`), then local INT pools that inhibit only their home region. Core trisynaptic / entorhinal drives:

```text
DG  ← DG  + w_MEC→DG  · R̃_MEC
CA3 ← CA3 + w_DG→CA3  · R̃_DG
CA2 ← CA2 + w_MEC→CA2 · R̃_MEC + w_CA3→CA2 · R̃_CA3
CA1 ← CA1 + w_CA3→CA1 · R̃_CA3 + w_MEC→CA1 · R̃_MEC
```

Default feedforward weights (tunable hypotheses; e.g. `w_mec_to_dg=0.20`, `w_dg_to_ca3=0.25`, `w_ca3_to_ca1=0.20`, `w_mec_to_ca1=0.15`) live in `ratinabox_params.feedforward`. An optional `w_mec_to_sub` path is applied when Subiculum is present.

Full rate equations, cell-type parameter matrices, recurrent/feedforward tables, population counts, and schematic anatomy: [`docs/neural_model.md`](docs/neural_model.md).

---

## Neuropixels recording and spike sorting

```text
ground-truth spikes
→ multi-channel Neuropixels templates
→ recording noise / amplitude drift / collisions
→ misses / jitter / contamination / merges
→ sorted spikes
```

**Deployable model selection uses sorted spikes only.** Ground-truth spikes are oracle / diagnostic / non-deployable (optional via `--include-ground-truth-diagnostics`). Later sections reuse this rule rather than restating it.

Modeled degradation sources (summary): multi-channel template spread, additive/correlated noise and burst noise, motion-related amplitude drift, spike collisions, detection misses (`miss_rate=0.12`), spike-time jitter (0.3 ms), contamination (`0.08`), and merges (`merge_prob=0.04`). Sample rate 30 kHz.

Probe context:

| Context | Probe | Pitch | Channels |
|---------|-------|------:|---------:|
| Schematic (`--no-trajectory`) | NP1.0 defaults in `hippo_sim/config.py` | 20 µm | 384 |
| Default lab trajectory | NP2.0 (`lab_npx2_default`) | 15 µm (confirm from channel map) | 384 |

Full recording/sorting parameter tables: [`docs/neuropixels_model.md`](docs/neuropixels_model.md).

---

## Anatomy and trajectory modeling

Trajectories are **config-driven**. AP/ML/DV/angle live in YAML under `configs/trajectories/`.

| Item | Notes |
|------|-------|
| Default lab trajectory | `lab_npx2_default` — NP2.0; AP −3.967, ML 3.758, DV ≈ 3.0 mm, angle 330h/80V |
| Uncertainty | DV / angle convention may remain uncertain until NTE or histology registration |
| Capture priors | Configurable in `*_cell_capture.yaml` |
| Decoder allowlist | Non-hippocampal regions excluded from decoding by default |
| `hpc_optimal` | Broader CA1→CA2→CA3→DG→Subiculum→MEC simulation stack |

```bash
python run_simulation.py --list-trajectories
python run_simulation.py --output outputs/ratinabox_001 --trajectory hpc_optimal --seed 1
```

Each trial snapshots coordinates under `trajectory/` (do not keep a separate global coords tree). Trajectory output bundles, NTE figure generation, future-insertion instructions, and full flag lists: [`docs/anatomy_and_trajectory.md`](docs/anatomy_and_trajectory.md).

---

## Causal decoding framework

At each update time `t`:

```text
x_t^(W) ∈ R^N

(x_t^(W))_i =
    number of spikes from unit i in [t − W, t)
```

**Spikes at time ≥ t are never used.**

| Symbol | Meaning |
|--------|---------|
| `W` | Causal spike evidence / integration window |
| `z_t` | Latent neural representation |
| `L` | Explicit latent-history length (optional temporal stage) |
| `τ` | Prediction lag (`ŷ_{t+τ}`) |
| `update_dt` | Decoder update cadence |
| `train_frac` | Contiguous train fraction (default **0.70**; remainder test) |

`W`, `L`, and `update_dt` are distinct: `W` accumulates recent spikes into the current observation; `L` stacks recent latents for an explicit history decoder; `update_dt` sets how often predictions are emitted. Optional rate features: `r_t^(W) = x_t^(W) / W`.

Decoder zoo, targets, metrics, and temporal `W×L` grids: [`docs/decoding_methods.md`](docs/decoding_methods.md).

---

## Static, dynamic, and temporal representations

### Static manifold

Current observation maps to a latent representation (frozen map fit on training data):

```text
x_t → z_t → ŷ_t
```

Examples: `counts`/`identity`, `global_pca`, `region_pca`, `layer_pca`, `global_isomap`, `global_isomap_distilled`.

### Dynamic latent state

Current state depends on the previous latent state and the current observation:

```text
z_(t−1), x_t → z_t → ŷ_t
```

The latent representation is dynamic; the downstream behavioral decoder may remain a fixed static map `z_t → ŷ_t`.

Examples: LDS (`global_lds`, causal / realtime); GPFA (`gpfa`, offline / acausal benchmark).

### Explicit temporal history

Recent latent vectors are stacked for the decoder (optional `--enable-temporal-manifold` path):

```text
x_t → z_t
        ↓
[z_(t−L+1), ..., z_t]
        ↓
decoder
        ↓
ŷ_(t+τ)
```

| Method | Temporal information |
|--------|----------------------|
| static manifold | none |
| static latent + history `L` | decoder input history |
| LDS | latent state transition |
| GPFA | temporal latent model |
| future adaptive LDS | dynamic state + changing observation mapping |

**Dynamic latent state ≠ neural plasticity/adaptation.** A normal LDS has a dynamic latent state but a fixed observation mapping `C`. Future plasticity modeling may introduce a slowly varying `C_t`.

Offline-only modes (`global_isomap`, `gpfa`) are evaluated in comparison but cannot auto-win closed-loop deployment. Commands and outputs: [`docs/dynamic_latents.md`](docs/dynamic_latents.md).

### Isomap (summary)

Isomap is a nonlinear **representation**, not a behavioral decoder. It is fit on training data only. Standard Isomap is offline: sklearn inductive `transform` is available for held-out coordinates, but the method is tagged non-realtime for closed-loop auto-deployment. Distilled Isomap approximates the embedding parametrically for realtime use when latency and held-out distortion gates pass. Nonlinear representation does **not** imply a nonlinear behavioral decoder. Evaluate with held-out decoding and geometry metrics (trustworthiness, residual variance, geodesic diagnostics), not visual appearance.

```text
sqrt(counts) → StandardScaler → optional pre-PCA → Isomap → z_t
```

`--profile manifolds` enables classic + distilled Isomap by default (lean neighbor/dim grids). Algorithm details, diagnostics, distillation, CLI grids, troubleshooting: [`docs/manifolds.md`](docs/manifolds.md).

---

## Search space

```text
spikes → F → neural observation → E → latent/state → D → ŷ → C
```

| Dimension | Meaning | Implemented examples |
|-----------|---------|----------------------|
| `F` | Observation / feature construction from spikes | `counts`, `rates`, `sqrt_counts`, `log1p_counts`, `zscore_counts`, `region_normalized_counts`, `cell_type_normalized_counts`; named sets such as `counts_dynamics` / `counts_regional` / `full_population_state` under `--profile feature_robustness` |
| `E` | Population-state representation | `identity`, `global_pca`, `region_pca`, `layer_pca`, `cell_type_pca`, `rate_model_pca`, `pls`, `bayesian_place_tuning`, `global_isomap`, `global_isomap_distilled`, `global_lds`, `gpfa` |
| `D` | Behavioral decoder | ridge, random forest, Bayesian place decoder, logistic regression, … |
| `W` | Causal integration window | profile candidate windows (see Timing) |
| `C` | Closed-loop rule (evaluated downstream) | spatial-context, wall-distance-bin, distance-to-wall, speed, movement, head-direction triggers |

The decoder benchmark’s Cartesian search is `F × E × D × W`. Trigger rules `C` are scored on decoded outputs (see `closed_loop_trigger_comparison.csv`) and exercised again during registry replay; they do not expand the decoder grid into a full five-way Cartesian product. Default closed-loop primary target for replay is `--closed-loop-target position` (overridable). Default day-to-day profiles typically hold `F=counts` while sweeping embeddings `E`.

| Target | Primary metric | Better |
|--------|----------------|--------|
| position | `mean_position_error_cm` | lower |
| speed | `r2` | higher |
| acceleration | `r2` | higher |
| head_direction | `mean_circular_error_deg` | lower |
| distance_to_wall | `r2` | higher |
| spatial_context | `balanced_accuracy` | higher |
| movement_state | `balanced_accuracy` | higher |
| wall_distance_bin | `balanced_accuracy` | higher |

Secondary metrics (where relevant): median / 90th-percentile position error, MAE, RMSE, Pearson correlation, macro-F1, confusion matrices.

Hyperparameters, quick/full model zoos, and Bayesian place-decoder likelihood:

```text
log P(bin | x_t)  ∝  log P(bin)
                   +  Σ_i [ x_{t,i} · log λ_i(bin)  −  λ_i(bin) · W ]
```

Full decoder documentation: [`docs/decoding_methods.md`](docs/decoding_methods.md).

---

## Deployment selection

Highest offline score is not sufficient. Deployable winners must use **sorted spikes** (see rule above).

### Public registry path (`models/best_realtime_decoders.json`)

Mandatory steps used by the public `run_decoder.py` registry + closed-loop replay:

```text
candidate F × E × D × W configurations (sorted spikes)
        ↓
held-out sorted-spike performance
        ↓
shortest-near-optimal window selection
        ↓
prefer realtime-compatible E (remap offline Isomap / reject GPFA for closed-loop)
        ↓
best_realtime_decoders.json
        ↓
causal realtime replay + closed-loop policy C
```

**`shortest_near_optimal`** (default): shortest `W` within 5% of the best metric (≤ 1.05× best if lower-is-better; ≥ 0.95× best if higher-is-better). Alternative: `--selection-policy best_accuracy`.

Per-target `W` is chosen independently from the profile grid. A shared 250 ms window is allowed only if it wins empirically — it is not hard-coded. Offline-only representations (`global_isomap`, `gpfa`) cannot become closed-loop deployment winners. Closed-loop replay loads saved artifacts and does not retrain when comparison artifacts exist.

Latency microbenchmarks and a full causal-update latency profile are **recorded** during / after comparison; the public registry selection is driven by sorted metrics + shortest-near-optimal + realtime-compatible remapping, not by requiring every stage to pass a total-update budget.

### Additional validation (implemented; not mandatory for the public registry)

During comparison, calibration metrics and per-candidate `passes_realtime_gate` flags are computed. A parallel lab-deployable table (`best_lab_deployable_decoders.csv` / profiles) can further filter categorical calibration, microbench latency gates, sorting-robustness labels, and negative-control beats when those columns are present.

| Analysis | When it runs | Role |
|----------|--------------|------|
| Classifier calibration (ECE / Brier) | Every comparison (categorical) | Metrics always; hard filter in lab-deployable selection |
| Per-candidate compute / history gate | Every comparison | Flags always; hard filter in lab-deployable selection |
| Closed-loop trigger score table `C` | Default comparison | Downstream policy evaluation, not decoder-grid selection |
| Sorted-vs-GT information loss | Only if GT diagnostics are compared | Degradation robustness summary |
| Negative controls / population ablation | Opt-in comparison flags | Sanity / interpretability |
| Cross-run generalization | Multi-run `run_decoder_comparison.py --inputs …` | Optional generalization selection |

Details: [`docs/realtime_deployment.md`](docs/realtime_deployment.md).

---

## Current results

### Deployable decoding benchmark

No validated, named-run decoder accuracy tables are frozen in this README yet. Populate later from saved artifacts such as `decoder_comparison/sorted/decoder_comparison_metrics.csv` and `models/best_realtime_decoders.json` after a reproducible public run.

### Latency (run-specific engineering profile: `ratinabox_004`, 50 ms / 20 Hz budget)

| Component | Approximate latency |
|-----------|--------------------:|
| counts | 0.003 ms |
| global PCA | 0.089 ms |
| region PCA | 0.507 ms |
| distilled Isomap | ~0.10 ms |
| classic Isomap | ~10 ms |
| spike binning | ~0.36 ms |
| position / speed / movement decoder | ~0.8–1.0 ms each |
| RF spatial-context decoder | ~55 ms |
| duplicated primary RF decode | ~54 ms |
| closed-loop policy | ~0.01 ms |
| **total update** | **~112 ms** |

Interpretation for this profiled run only: distilled Isomap itself is fast enough for realtime; the budget overrun is dominated by RF classifier heads; total update exceeds the 50 ms / 20 Hz target. These are implementation measurements for that run, not universal biological conclusions.

### Static vs dynamic

> Static-vs-dynamic numeric results are currently TBD. LDS and GPFA are implemented and runnable, but this README does not claim that dynamic latents outperform static manifolds until those experiments are run.

Produce local numbers by including both families in a comparison (see [`docs/dynamic_latents.md`](docs/dynamic_latents.md)), then inspect `decoder_comparison_metrics.csv` grouped by `representation_family`, `embedding_type`, `causal_status`, target, window, and latent dimension. Optionally open the Streamlit **Static vs Dynamic** page or `figures/dynamic/` panels.

Until that analysis is run and recorded here, treat LDS/GPFA as an implemented, runnable pathway—not a claim about relative performance.

---

## Results interpretation rules

```text
manifold ≠ decoder
dynamic latent ≠ adaptive decoder
dynamic latent ≠ neural plasticity
Isomap ≠ nonlinear behavioral decoder
offline best ≠ deployable best
ground-truth best ≠ sorted-spike best
visual separation ≠ held-out decoding improvement
```

---

## Outputs

```text
outputs/<run>/
├── behavior.csv
├── units.csv
├── spikes_ground_truth.csv
├── spikes_sorted.csv
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

| Path | Role |
|------|------|
| Simulation CSVs / `rates.npy` / `summary.json` | Raw trial data |
| `trajectory/` | Snapshotted insertion coords + capture rules |
| `decoder_comparison/` | Metrics, transforms, `.joblib` models (`sorted/` deployable; `ground_truth/` oracle) |
| `models/best_realtime_decoders.json` | Deployable registry |
| `deployment_decoder_selection/` | Window×decoder score tables |
| `realtime_decoding/` | Causal closed-loop replay |
| `latency_profiling/` | Stage latencies |
| `decoding/` | Optional temporal `W×L` results |
| `figures/` | Panels + `output.pdf` (no loose top-level PNGs) |

`decoder_comparison/dynamic/` and `figures/dynamic/` appear when LDS/GPFA association or latent figures are generated. Full artifact schema: [`docs/output_schema.md`](docs/output_schema.md).

---

## Visualizations

```bash
python run_visualizations.py \
    --experiment outputs/<run> \
    --all \
    --compile-pdf
```

Visualization reads saved results, does not retrain models, and is safe to run independently.

| Paper fig | Content |
|-----------|---------|
| 1–3 | Behavior / neural drivers; spikes-on-trajectory & tuning; circuit structure |
| 4–5 | Causal decoding performance; manifold × window / decoder tables |
| 6–8 | Latent geometry suite; Isomap diagnostics; Isomap + distillation story |
| 9–11 | Closed-loop replay; deployment winners; latency budget |
| 12 (suppl.) | Temporal `W×L` heatmaps when `decoding/` exists |

Complete catalog and stems: [`docs/visualizations.md`](docs/visualizations.md).

---

## Documentation map

| Document | Contents |
|----------|----------|
| [`docs/neural_model.md`](docs/neural_model.md) | Rate equations, populations, feedforward weights |
| [`docs/neuropixels_model.md`](docs/neuropixels_model.md) | Recording / sorting parameters |
| [`docs/anatomy_and_trajectory.md`](docs/anatomy_and_trajectory.md) | Trajectories, NTE, capture allowlist |
| [`docs/decoding_methods.md`](docs/decoding_methods.md) | Decoder zoo, targets, temporal decoding |
| [`docs/manifolds.md`](docs/manifolds.md) | Isomap details, distillation, diagnostics |
| [`docs/dynamic_latents.md`](docs/dynamic_latents.md) | LDS / GPFA commands and outputs |
| [`docs/realtime_deployment.md`](docs/realtime_deployment.md) | Gating, registry, latency workflow |
| [`docs/visualizations.md`](docs/visualizations.md) | Figure catalog |
| [`docs/output_schema.md`](docs/output_schema.md) | Artifact tree |
| [`docs/cli_reference.md`](docs/cli_reference.md) | Profiles and CLI flags |
| [`docs/developer.md`](docs/developer.md) | Modules, utilities, tests |

---

## Limitations

- Simulated anatomy and rate models are configurable hypotheses, not recovered biology.
- Approximate trajectory coordinates are not histology confirmation.
- Recording / sorting degradation is a model of Neuropixels + Kilosort-like effects.
- Ground-truth spike results are non-deployable oracle diagnostics.
- Realtime methods cannot use future neural data (no future spikes / future smoothing).
- Biological claims require validation on experimental recordings.
- Dynamic latent state does not itself solve neural plasticity / drift.
- Sorted spikes lose some ground-truth metadata by construction.
- Supervised / task-conditioned embeddings (when added) reflect supplied labels.

Public-workflow smoke (simulate → decode → visualize):

```bash
bash scripts/smoke_test_public_workflow.sh
```

Broader developer troubleshooting (RatInABox install, empty PDF, timestamp mismatch): [`docs/developer.md`](docs/developer.md).

Module map, `run_BCI.py`, archived helpers, and component tests also live there.

---

## License

Research / educational use. Earlier analytic models that informed notation live under `previous_models/` (see [`docs/developer.md`](docs/developer.md)).
