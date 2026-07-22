# Hippocampal Neuropixels Simulation and Decoding

Simulates realistic hippocampal single-unit activity during 10-minute open-field navigation, recorded through a Neuropixels 1.0 single-shank probe (384 channels), with Neuropixels-like recording degradation and Kilosort-like spike re-extraction. Decodes latent behavioral variables from **causal** population spike counts, with optional low-dimensional manifold features.

**Purpose:** a research-engineering loop to find a **maximally compatible hippocampal BCI** — search embeddings `E` × decoders `D` × windows `W` on sorted spikes, gate what clears realtime constraints, then transplant the winners to the lab. Figures are a separate inspect step (never required to retrain).

Equations below use plain Unicode / code formatting so they render in Cursor, GitHub, and terminals without a LaTeX math plugin.

---

## Public workflow

Three scripts. Decode searches manifolds and decoders inside one comparison; you do not pick `W` or `(E, D)` by hand for the happy path.

```text
Simulate → Search (E × D × W) → Gate (sorted + realtime + latency) → Export registry
                                    ↘ Inspect with figures anytime (separate command)
```

```bash
source .hippo/bin/activate
pip install -r requirements.txt

# 1. Simulate
python run_simulation.py \
    --output outputs/ratinabox_001 \
    --seed 1

# 2. Decode — default --profile manifolds: sorted spikes, lean grid over
#    counts / global·region·layer PCA / classic+distilled Isomap, quick decoder
#    zoo, closed-loop replay, deployable registry. Skip figures here.
python run_decoder.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001 \
    --skip-visualization

# 3. Visualize anytime (reads saved outputs only; never retrains)
python run_visualizations.py \
    --experiment outputs/ratinabox_001 \
    --all \
    --compile-pdf
```

**Lab transplant artifact** (after Step 2): `models/best_realtime_decoders.json` plus the referenced manifold transforms and decoder `.joblib` files under `decoder_comparison/sorted/`.

| Goal | Command |
|------|---------|
| Simulate data | `run_simulation.py` |
| Search embeddings × decoders, gate, export registry | `run_decoder.py` (default `--profile manifolds`) |
| Inspect figures / PDF | `run_visualizations.py --experiment ... --all --compile-pdf` |

**Advanced** (not needed for the happy path): `--profile standard` or `quick` (faster counts+PCA smoke), `--profile full` (dense grids + full model zoo), `--include-ground-truth-diagnostics` (oracle GT, non-deployable), `--enable-temporal-manifold` (W×L Table 8), or explicit `--feature-modes` / `--max-models` overrides.

### Detached simulation (survives terminal close)

```bash
mkdir -p outputs/run_001
nohup .hippo/bin/python run_simulation.py --output outputs/run_001 \
  > outputs/run_001/simulation.log 2>&1 &
echo $! > outputs/run_001/simulation.pid
tail -f outputs/run_001/simulation.log
```

---

## Installation

```bash
cd ~/projects/hippo
python3 -m venv .hippo
source .hippo/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

Python 3.10+ (developed on 3.12). See `requirements.txt`.

---

## Simulation pipeline

1. **Behavior** — RatInABox square open field (thigmotaxis, stalls, smooth turns) at **20 Hz** (50 ms steps)
2. **Features** — theta phase, ripple, speed (used by hippocampal dynamics overlays)
3. **Neural rates** — RatInABox receptive fields → cell-type overlays → trisynaptic / EC feedforward (MEC→DG→CA3→CA1, INT→CA1)
4. **Spikes** — ground-truth Poisson spike times
5. **Recording** — multi-channel templates, noise, motion amplitude drift, collisions
6. **Sorting** — Kilosort-like re-extraction with misses, jitter, contamination

### Simulation outputs

| File | Description |
|------|-------------|
| `behavior.csv` | Position, speed, head direction over time |
| `anatomy_regions.csv` | Region geometry and channel mapping |
| `units.csv` | Per-unit metadata (type, region, channel, place field, `ratinabox_class`) |
| `spikes_ground_truth.csv` | True spike times |
| `spikes_sorted.csv` | Re-extracted spike times after degradation |
| `ratinabox_group_metadata.csv` | Per-population RiaB group summary |
| `rates.npy` | Firing rates (Hz), shape `(n_units, n_steps)` |
| `summary.json` | Run statistics |

---

## Table 1: Neural rate model (RatInABox + hippocampal circuit overlays)

Neural activity is generated in three stacked stages. Source of truth:
`hippo_sim/ratinabox_neural_backend.py`, `hippo_sim/feedforward.py`,
`hippo_sim/hippocampal_populations.py`, `hippo_sim/config.py`.

### Stage A — RatInABox receptive fields

For each population group \(G\), RatInABox (or a documented fallback) supplies a
nonnegative rate \(R_i^{\mathrm{RiaB}}(t)\) from the shared behavioral state
\((p(t),\theta(t),v(t))\):

| Group | Class | Primary tuning |
|-------|-------|----------------|
| CA1_place_pp | PhasePrecessingPlaceCells | place + theta phase precession |
| CA3 / CA2 / DG place | PlaceCells | Gaussian / thresholded place fields (widths differ) |
| MEC_grid | GridCells | multi-module grid |
| MEC_hd | HeadDirectionCells | preferred head direction |
| Sub_bvc | BoundaryVectorCells | wall distance / angle |
| MEC_speed | SpeedCells_fallback | linear speed encoding |
| CA1_int | synthetic | constructed in Stage C |

### Stage B — within-cell-type dynamics overlays

With cell-type parameters \(c\) from `RATE_PARAMS`:

```text
R ← R_RiaB
R ← R · (1 + w_θ cos φ_θ(t))                 # theta (skipped for CA1_place_pp)
R ← R · (1 + w_speed · f_speed(t))           # speed gain
R ← R + w_ripple · A · r(t)                  # sharp-wave ripple envelope
R ← 0.1·b  if  R/max_t(R) < θ_sparse else R  # DG sparsity
R ← R + w_rec · mean_{j∈G} R_j(t)            # CA3 recurrent (autoassociative)
R ← max(R, 0)
```

where \(f_speed = \max(0,v-v_{thresh})/(30-v_{thresh})\) and \(r(t)\in[0,1]\) is a sparse SWR burst envelope.

### Stage C — trisynaptic / entorhinal feedforward

Define region means (after Stage B):

```text
R̄_A(t) = (1/|A|) Σ_{i∈A} R_i(t),   A ∈ {MEC, DG, CA3, CA2, CA1, INT}
```

Optional normalization \(\tilde R_A = R̄_A / \max_t R̄_A\) (default on) makes weights dimensionless. Then, in order:

```text
DG  ← DG  + w_MEC→DG  · R̃_MEC
CA3 ← CA3 + w_DG→CA3  · R̃_DG          # CA3↔CA3 recurrence already in Stage B
CA2 ← CA2 + w_MEC→CA2 · R̃_MEC + w_CA3→CA2 · R̃_CA3
CA1 ← CA1 + w_CA3→CA1 · R̃_CA3 + w_MEC→CA1 · R̃_MEC
```

Interneurons are then built from post-feedforward CA1:

```text
R_i^int(t) = g_i · b_int · (1 + w_θ cos 2π f_θ t) · (1 − w_anti · R̃_CA1(t))
           + w_ripple · A_int · r(t)
```

and inhibit CA1:

```text
CA1 ← max( CA1 − w_INT→CA1 · R̃_INT , 0 )
```

**Default weights** (tunable hypotheses in `ratinabox_params.feedforward`):

| Synapse | Symbol | Default |
|---------|--------|---------|
| MEC → DG | `w_mec_to_dg` | 0.20 |
| DG → CA3 | `w_dg_to_ca3` | 0.25 |
| CA3 → CA1 | `w_ca3_to_ca1` | 0.20 |
| MEC → CA1 (direct) | `w_mec_to_ca1` | 0.15 |
| INT → CA1 | `w_int_to_ca1` | 0.30 |
| MEC → CA2 | `w_mec_to_ca2` | 0.10 |
| CA3 → CA2 | `w_ca3_to_ca2` | 0.10 |
| CA3 recurrent (Stage B) | `w_recurrent` | 0.15 |

Disable with `apply_feedforward: false` in `ratinabox_params`. Metadata is written to `neural_backend_metadata.json` under `feedforward`.

### Overlay / feedforward parameters (principal cells)

| Parameter | CA1_pyr | CA2_pyr | CA3_pyr | DG_granule | Units |
|-----------|---------|---------|---------|------------|-------|
| b (baseline) | 0.5 | 0.4 | 0.3 | 0.05 | Hz |
| A (amplitude) | 12.0 | 10.0 | 8.0 | 15.0 | Hz |
| w_speed | 0.3 | 0.25 | 0.2 | 0.5 | — |
| v_thresh | 2.0 | 2.0 | 2.0 | 3.0 | cm/s |
| w_θ | 0.25 | 0.15 | 0.1 | 0.05 | — |
| f_θ | 8.0 | 8.0 | 8.0 | 8.0 | Hz |
| w_ripple | 0.5 | 0.1 | 0.6 | 0.0 | — |
| w_recurrent | — | — | 0.15 | — | — |
| θ_sparse | — | — | — | 0.3 | — |

| Recording / sorting parameter | Value |
|-------------------------------|-------|
| Channels | 384 |
| Site pitch | 20 µm |
| Sample rate | 30 kHz |
| Template span | 3–10 channels |
| Amplitude range | 20–200 µV |
| Noise σ | 15 µV |
| Miss rate | 12% |
| Jitter | 0.3 ms |
| Contamination rate | 0.08 |
| Merge probability | 0.04 |

---

## Table 2: Anatomical Regions and Probe Geometry

Neuropixels 1.0 single-shank, 384 channels, 20 µm pitch, 1D depth axis (dorsal hippocampus + ventral afferent bands).

| Region | Layer | Depth start (µm) | Depth end (µm) | Cell types | Density (units/channel) |
|--------|-------|------------------|----------------|------------|-------------------------|
| CA1 | oriens | 0 | 200 | CA1_int | 2.0 |
| CA1 | pyramidal | 200 | 400 | CA1_pyr | 8.0 |
| CA1 | radiatum | 400 | 600 | CA1_pyr | 5.0 |
| CA2 | pyramidal | 600 | 800 | CA2_pyr | 6.0 |
| CA3 | pyramidal | 800 | 1300 | CA3_pyr | 7.0 |
| DG | granule | 1300 | 1650 | DG_granule | 10.0 |
| DG | hilus | 1650 | 1850 | DG_granule | 3.0 |
| Subiculum | pyramidal | 1850 | 2050 | Sub_bvc | 4.0 |
| MEC | layer2 | 2050 | 2350 | MEC_grid | 5.0 |
| MEC | layer3 | 2350 | 2600 | MEC_hd | 4.0 |

Exact mapping is written to `anatomy_regions.csv` per run. MEC/Subiculum bands are co-recorded afferent populations (not a claim that one shank spans all of these in vivo).

---

## Neural activity (RatInABox)

Maximally hippocampal config: anatomically mapped RatInABox classes, cell-type dynamics overlays, and **trisynaptic / entorhinal feedforward** (MEC→DG→CA3→CA1 plus INT→CA1). Population table:

| Group | n | RiaB class | Region / cell type | Dynamics |
|-------|---|------------|--------------------|----------|
| CA1_place_pp | 60 | PhasePrecessingPlaceCells | CA1_pyr | phase precession (built-in), ripples, speed gain |
| CA3_place | 40 | PlaceCells | CA3_pyr | theta, ripples, recurrent |
| DG_place | 50 | PlaceCells (narrow) | DG_granule | theta, sparsity, speed gain |
| CA2_place | 20 | PlaceCells | CA2_pyr | theta, weak ripples |
| MEC_grid | 40 | GridCells | MEC_grid | theta, speed gain |
| MEC_hd | 30 | HeadDirectionCells | MEC_hd | theta |
| Sub_bvc | 30 | BoundaryVectorCells | Sub_bvc | theta |
| MEC_speed | 20 | SpeedCells_fallback | MEC_speed | speed tuning |
| CA1_int | 15 | synthetic | CA1_int | theta, ripples, anti-CA1-pyr |

Source of truth: `hippo_sim/hippocampal_populations.py`. Saves `rate_model` and `ratinabox_class` in `units.csv`.

```bash
python run_simulation.py \
    --output outputs/ratinabox_003 \
    --seed 1
```

---

## Table 3: Causal Decoding — Timing and Features

**Scientific framing.** The simulator knows true behavior. The decoder only sees spike counts. `spike_source=sorted` is the **deployment-relevant** source after Neuropixels degradation and sorting. `spike_source=ground_truth` is an **oracle** upper bound (optional diagnostics only — never deployable). Window search asks how much recent history each target needs; model search asks which estimator extracts latent variables best; closed-loop replay tests realtime triggers using sorted-selected models.

### Causal observation

At each behavioral frame time `t`:

```text
x_t^(W) ∈ R^N

(x_t^(W))_i  =  # { spikes of unit i with times in [t − W, t) }
```

Half-open interval: spikes with time `≥ t` are **never** included.

Optional rate features:

```text
r_t^(W)  =  x_t^(W) / W
```

### Timing parameters

| Symbol / CLI | Default | Meaning |
|--------------|---------|---------|
| Behavior timestamps | from `behavior.csv` | Prediction grid (prefer actual frames) |
| `update_dt` / `dt_update` | **0.050 s** (20 Hz) | One prediction per behavioral frame |
| `decode_window` / `W` | searched | Causal integration window |
| `train_frac` | 0.70 | Contiguous train fraction (rest = test) |
| Alignment tolerance | 0.005 s | Max allowed decode↔behavior mismatch |

### Default window search

| W (s) | 0.050 | 0.100 | 0.250 | 0.500 | 1.000 |
|-------|-------|-------|-------|-------|-------|
| Duration | 50 ms | 100 ms | 250 ms | 500 ms | 1 s |

Short `W`: lower latency, fewer spikes. Long `W`: more reliable counts, higher effective latency.

### Roles kept separate

| Symbol | Role | Not the same as |
|--------|------|-----------------|
| `W` | Spike evidence for **current** neural state | Model memory |
| `z_t` | Manifold coordinate of current observation | Temporal dynamics |
| `L` | History of recent latents (optional temporal stage) | Integration window |
| `τ` | Prediction lag `ŷ_{t+τ}` | Update interval |

---

## Table 4: Decoding Targets and Metrics

| Target | Family | Output | Primary metric | Direction |
|--------|--------|--------|----------------|-----------|
| `position` | continuous | `(x, y)` cm | `mean_position_error_cm` | lower |
| `speed` | continuous | cm/s | `r2` | higher |
| `acceleration` | continuous | cm/s² | `r2` | higher |
| `head_direction` | continuous / circular | rad | `mean_circular_error_deg` | lower |
| `distance_to_wall` | continuous | cm | `r2` | higher |
| `spatial_context` | categorical | class label | `balanced_accuracy` | higher |
| `movement_state` | categorical | class label | `balanced_accuracy` | higher |
| `wall_distance_bin` | categorical | class label | `balanced_accuracy` | higher |

Secondary metrics (also computed where relevant): median / 90th-percentile position error, MAE, RMSE, Pearson correlation, macro-F1, confusion matrices.

---

## Table 5: Decoder Model Zoo

Source: `realtime/decoder_models.py`. Continuous targets use regressors (position / HD via `MultiOutputRegressor` where needed). Categorical targets use classifiers. All sklearn pipelines that need scaling include `StandardScaler` where applicable.

### Quick mode (`--max-models quick`, default)

| Name | Task | Default hyperparameters |
|------|------|-------------------------|
| `ridge` | regression | `α = 1.0` |
| `pca_ridge` | regression | PCA retain 95% variance → Ridge `α = 1.0` |
| `random_forest_regressor` | regression | 100 trees, `max_depth=12` |
| `logistic_regression` | classification | `max_iter=1000`, `class_weight=balanced` |
| `random_forest_classifier` | classification | 100 trees, `max_depth=12`, `class_weight=balanced` |
| `bayesian_place_decoder` | position / wall distance (added in quick when applicable) | see Table 6 |
| `bayesian_place_decoder_derived_context` | spatial_context / wall_distance_bin (when applicable) | see Table 6 |

### Full mode (`--max-models full`)

Adds: `elastic_net` (`α=0.1`, `l1_ratio=0.5`), `pls_regression` (10 components), `hist_gradient_boosting_regressor` / `hist_gradient_boosting_classifier` (`max_depth=8`), `knn_regressor` / `knn_classifier` (`k=15`, distance weights), `linear_svm_classifier`, `bayesian_place_decoder_smoothed`.

### Selection policies (under the hood)

After comparing all `(model, W, feature_mode)` configurations per target:

| Policy | Rule |
|--------|------|
| `best_accuracy` | Best primary metric over all windows |
| `shortest_near_optimal` (default) | Shortest `W` with metric within **5%** of best (≤ 1.05× best if lower-is-better; ≥ 0.95× best if higher-is-better) |

Closed-loop replay loads the selected `.joblib` and does **not** retrain when comparison artifacts exist.

---

## Table 6: Bayesian Place Decoder

Source: `realtime/bayesian_decoder.py`.

Poisson likelihood place-map decoder on a 2D grid.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_bins` | 20 | Bins per spatial axis |
| `occupancy_floor` | `1e-3` | Floor on occupancy / prior |
| `smooth` | `false` (`true` for `_smoothed` variant) | Causal exponential posterior smoothing |
| `smooth_alpha` | 0.7 | Smoothing weight on previous posterior |

```text
log P(bin | x_t)  ∝  log P(bin)
                   +  Σ_i [ x_{t,i} · log λ_i(bin)  −  λ_i(bin) · W ]
```

(with implementation details in code; smoothing uses **past** posteriors only).

Derived-context variant maps the decoded place posterior to categorical spatial / wall-distance labels.

---

## Table 7: Manifold Feature Modes (static decoding)

Source: `realtime/manifold_features.py`.

Architecture for manifold-informed static decoding:

```text
x_t^(W)  →  E(·)  →  z_t  →  decoder  →  ŷ_t
```

Encoder `E` is fit on **training** activity only, then frozen for test / realtime.

| Feature mode | Transform | Grouping column | Latent dim |
|--------------|-----------|-----------------|------------|
| `counts` | `z = x` | — | `N` |
| `rates` | `z = x / W` | — | `N` |
| `global_pca` | PCA on all units | — | `k` (default 3) |
| `region_pca` | PCA per region, concatenate | `region` | `k` per group |
| `layer_pca` | PCA per layer | `layer` | `k` per group |
| `cell_type_pca` | PCA per cell type | `cell_type` | `k` per group |
| `rate_model_pca` | PCA per rate model | `rate_model` (fallback `ratinabox_class`) | `k` per group |
| `global_isomap` | Isomap on all units (√counts + scale + optional pre-PCA) | — | `d` (offline only) |

| Setting | Default |
|---------|---------|
| Quick feature modes | `counts`, `global_pca`, `region_pca` |
| Default `k` list | `(3,)` (override with `--manifold-components-list 2 3 5`) |
| Isomap neighbors | `--isomap-neighbors` (default `10` when `global_isomap` requested) |
| Fit data | train split only |
| Realtime | load saved transform from comparison `models/` (**not** standard Isomap) |

**Note:** Static manifold PCA in `manifold_features.py` is applied to raw (or rate) count vectors. The optional **temporal** PCA path (`realtime/manifolds/pca.py`) uses `√counts` + `StandardScaler` before PCA — see Table 8. Static / temporal Isomap both use √counts + standardization + optional pre-PCA and are tagged `offline_analysis_only`.

---

## Table 8: Temporal Manifold Decoding (optional)

Enabled with `--enable-temporal-manifold` on `run_decoder.py`. Config reference: `configs/temporal_decoding.yaml`.

Under `--profile standard` (or when temporal is enabled without `full`), temporal mode is **lean**: PCA-only latents, `L ∈ {1,5,20}`, core models, and **W inherited from Step 1** (best/recommended windows plus short/long flanks). Use `--profile full` for the dense research grid (`raw`+`pca`, full L list, shuffle/average controls). Default decode profile `manifolds` leaves temporal off unless you pass `--enable-temporal-manifold`.

```text
x_t^(W)  →  z_t  →  Z_t^(L) = [z_{t−L+1}, …, z_t]  →  ŷ_{t+τ}
```

| Parameter | Standard (lean) | Full |
|-----------|-----------------|------|
| `W` | inherited from Step 1 + flanks | explicit `--decode-windows` grid |
| Representations | `pca` | `raw`, `pca` (+ `isomap` via `--representations`) |
| PCA preprocess | `√x` + standardize | same |
| Isomap | opt-in offline | `√x` + standardize + optional pre-PCA; `realtime_compatible=false` |
| Default latent dim (temporal PCA) | 16 | 16 |
| Default Isomap latent dim | 8 (`--isomap-latent-dim`) | same |
| `L` (frames) | 1, 5, 20 | 1, 2, 5, 10, 20 |
| Temporal models | `raw_static`, `static_latent`, `flattened_history` | + `averaged_history`, `shuffled_sequence` |
| Split | train / val / test contiguous + leakage gap | Selection on **validation** only |

Planned (not in baseline): causal GRU, TCN, autoencoder, CEBRA.

---

## Isomap nonlinear manifold analysis

Isomap is an **offline** nonlinear manifold encoder. It is **not** a behavioral decoder.

Architecture (same as other manifold methods):

```text
x_t^(W)  →  E_Isomap(·)  →  z_t  →  D(·)  →  ŷ_t
```

or with temporal dynamics kept separate from Isomap geometry:

```text
x_t^(W)  →  z_t^Isomap  →  Z_t^(L) = [z_{t−L+1}, …, z_t]  →  D_temporal  →  ŷ_{t+τ}
```

| Concept | Role |
|---------|------|
| Manifold extraction `E` | Maps high-D neural activity → low-D coordinates `z_t` |
| Behavioral decoder `D` | Maps `z_t` (or raw `x_t`) → behavior / memory targets |
| Isomap | One possible **offline** nonlinear choice for `E` |

Do **not** assume a nonlinear manifold requires a nonlinear decoder. Always compare linear and nonlinear `D` on the same Isomap coordinates (and on PCA / raw).

### Scientific motivation

Hippocampal population activity may lie on a curved, low-dimensional manifold. PCA is a useful linear baseline, but it may need many components to approximate a manifold whose intrinsic dimension is small. Isomap estimates **geodesic** distances along a neighbor graph and embeds them with classical MDS, preserving global nonlinear geometry better than Euclidean PCA when the geometry is curved.

### Algorithm (sklearn `manifold.Isomap`)

Given neural observations `X ∈ R^{T×N}` (rows = time, columns = units):

1. Build a `k`-nearest-neighbor graph on **training** observations only.
2. Weight edges by pairwise distances between neighbors.
3. Estimate geodesic distances via shortest paths on the graph.
4. Apply classical multidimensional scaling (MDS) to the geodesic matrix.
5. Return coordinates `Z ∈ R^{T×d}`.

Implementation: `realtime/manifolds/isomap.py` (`IsomapManifoldEncoder`), registered as `"isomap"`. Static decoding feature mode: `global_isomap` in `realtime/manifold_features.py`.

### Preprocessing (training-only)

Recommended default:

```text
spike counts → √counts → StandardScaler (train fit) → optional PCA precompression → Isomap
```

Config keys (`configs/temporal_decoding.yaml` → `manifold.methods.isomap`):

| Key | Default | Notes |
|-----|---------|-------|
| `n_neighbors` | `[5,10,20,30,50]` | Too small → disconnected graph; too large → shortcut edges |
| `latent_dims` / `n_components` | `[2,3,4,6,8]` | Embedding dimension `d` |
| `pre_pca.enabled` | `true` | Reduces noise / cost when `N` is large |
| `pre_pca.n_components` | `50` | Or set variance threshold |
| `require_connected_graph` | `true` | Reject disconnected configs |
| `allow_largest_component_only` | `false` | Optional recovery if largest component ≥ 95% |
| `realtime_compatible` | `false` | Standard Isomap is offline-only |

**Leakage rules:** never fit scaling, PCA, neighbor graphs, or Isomap on validation/test. Held-out coordinates use the training model’s out-of-sample `transform`. Do not concatenate train+val+test before fitting merely for a continuous visualization. Tag any full-session embedding as `exploratory_full_session_embedding` / `not_valid_for_held_out_performance`.

### Graph diagnostics

After fit, diagnostics report:

- connected-component count and largest-component fraction
- min / median / max / mean node degree
- `mean_degree / n_train` (dense-graph flag)
- duplicate-observation fraction
- geodesic-distance finite fraction

Disconnected graphs raise `DisconnectedGraphError` (or are recorded with `exclusion_reason`) unless `allow_largest_component_only` is enabled.

### Geometry metrics

- Trustworthiness at several `k`
- k-NN overlap / continuity proxy
- Geodesic vs embedding distance correlation (sampled pairs)
- Residual variance `1 − R²(D_G, D_Z)`

Hyperparameters are selected from **held-out validation** decoding / geometry metrics — not from visualization quality.

### Linear vs nonlinear decoders

Recommended Phase-1 comparisons:

```text
raw + Ridge / Random Forest
PCA + Ridge / Random Forest
Isomap + Ridge / Random Forest / k-NN
```

Full model zoo also includes `rbf_svr` / `rbf_svc` and small MLPs (`--max-models full`). Position uses multi-output regressors and Euclidean position error.

This separates **nonlinear representation** benefit from **nonlinear output mapping** benefit.

### Temporal decoding on Isomap

Isomap geometry and temporal history are kept separate: encode each causal window to `z_t`, then build `Z_t^(L)`. Supported temporal models (existing Table 8 path): `static_latent`, `flattened_history`, plus controls `averaged_history` / `shuffled_sequence`. Pass `--representations isomap` with `--enable-temporal-manifold`.

### Real-time compatibility

| Method | Tag | Auto-deploy in closed-loop? |
|--------|-----|-------------------------------|
| `isomap` / `global_isomap` | `offline_analysis_only` | **No** |
| `isomap_distilled` / `global_isomap_distilled` | parametric approx. of Isomap | **Yes** if latency ≤ 50 ms **and** held-out distortion OK |
| `pca` / `counts` | realtime-capable | Yes (existing path) |

Best-decoder selection prefers realtime-compatible feature modes for closed-loop recommendations even when offline Isomap wins on accuracy.

Parametric distillation (`realtime/manifolds/isomap_distillation.py`, static mode `global_isomap_distilled`) trains Ridge / kernel Ridge / small MLP so `E_θ(x_t) ≈ z_t^Isomap`. This is **not** exact Isomap. Enable with `--enable-isomap-distillation`, or use `--profile manifolds` (distillation on by default).

Every causal-update stage (spike binning, feature transforms including distilled Isomap, each decoder head, closed-loop policy) is timed under `latency_profiling/` and plotted in `figures/latency/` in the compiled PDF.

### Causal update latency (`ratinabox_004`, 50 ms / 20 Hz budget)

| Stage / feature | Mean latency | Realtime? |
|-----------------|-------------:|:---------:|
| counts | 0.003 ms | yes |
| global_pca | 0.089 ms | yes |
| region_pca | 0.507 ms | yes |
| classic Isomap (`global_isomap`) | ~10 ms | **no** |
| distilled Isomap (`global_isomap_distilled`) | ~0.10 ms | **yes** |
| spike binning | 0.36 ms | — |
| decode position / speed / movement | ~0.8–1.0 ms each | — |
| decode spatial_context (RF) | ~55 ms | — |
| decode_primary (same RF again) | ~54 ms | — |
| closed-loop policy | 0.01 ms | — |
| **total update** | **~112 ms** | over budget |

Distilled Isomap is ~100× faster than classic Isomap and well under the update budget. The budget miss is from random-forest classifier heads (context + primary both ~55 ms), not the manifold front-end.

### Commands

```bash
# Static: PCA vs Isomap + linear/RF decoders
python run_decoder.py \
    --input outputs/ratinabox_004 \
    --output outputs/ratinabox_004 \
    --feature-modes counts global_pca global_isomap \
    --manifold-components-list 2 3 4 6 8 \
    --isomap-neighbors 5 10 20 30 50 \
    --max-models quick \
    --skip-visualization

# Temporal Isomap (offline; inherits W from Step 1 when profile=standard)
python run_decoder.py \
    --input outputs/ratinabox_004 \
    --output outputs/ratinabox_004 \
    --enable-temporal-manifold \
    --representations pca isomap \
    --isomap-neighbors 10 \
    --isomap-latent-dim 8 \
    --latent-history-frames 1 2 5 10 20

# Staged BCI workflow (developer): fit datasets + partitions + decode
python run_BCI.py \
    --stage decode-temporal \
    --input outputs/ratinabox_004 \
    --output outputs/ratinabox_004 \
    --representations pca isomap \
    --isomap-neighbors 10 20 \
    --enable-temporal-manifold

# Figures (reads saved outputs only)
python run_visualizations.py --experiment outputs/ratinabox_004 --all --compile-pdf

# Tests
python -m pytest tests/test_isomap_manifold.py tests/test_isomap_decoders.py \
    tests/test_isomap_distillation.py tests/test_isomap_temporal.py -q
```

### Outputs

| Path | Contents |
|------|----------|
| `decoder_comparison/*/models/manifold_transforms/global_isomap_k*_nn*_w*ms/` | Fitted Isomap + preprocessing (`meta.json`, `isomap.joblib`, `scaler.joblib`, `pre_pca.joblib`, diagnostics) |
| `decoder_comparison/*/decoder_comparison_metrics.csv` | Rows include `manifold_method`-like fields: `n_neighbors`, `graph_connected`, `trustworthiness`, `residual_variance`, `decoder_nonlinear`, `realtime_compatible` |
| `decoding/comparison/*/manifolds/isomap/` | Temporal Isomap encoders per `W` |
| `decoding/comparison/*/all_configurations.csv` | Temporal W×L results with Isomap geometry columns |
| Distilled models (optional API) | `IsomapDistilledEncoder.save(...)` → `meta.json` + `distiller.joblib` |

### Limitations and troubleshooting

| Issue | Guidance |
|-------|----------|
| Disconnected graph | Increase `n_neighbors`; check duplicate population vectors |
| Shortcut / collapsed geometry | Decrease `n_neighbors`; inspect dense-graph flag |
| Slow / high memory | Enable pre-PCA; reduce `T` via coarser update; sample distance pairs |
| Too few units / samples | Isomap needs enough neighbors; prefer PCA or raw |
| Unstable `n_neighbors` | Report stability across neighbors; select on validation metrics |
| Poor held-out transform | Expected limitation of inductive Isomap; consider distillation only as approximation |
| Realtime wants Isomap | Use PCA/counts for closed loop; Isomap informs dimensionality / partition choice only |

Do **not** conclude Isomap is superior because a 2-D plot looks more curved. Use held-out decoding and geometry metrics.

---

## Decoder workflow (public script)

`run_decoder.py` (default `--profile manifolds`) runs under the hood:

1. **Compare** models × windows × feature modes on **sorted spikes only** → `decoder_comparison/sorted/`
2. **Write deployable registry** → `models/best_realtime_decoders.json` + `deployment_decoder_selection/`
3. **Replay** closed-loop causal decoding from that registry → `realtime_decoding/sorted/`
4. **Optional** in-decode figures (prefer Step 3 `run_visualizations.py` instead)

### Deployment vs oracle

| Source | Role |
|--------|------|
| **Sorted spikes** (Neuropixels / Open Ephys / Kilosort-like) | **Only** valid model-selection source for realtime deployment |
| **Ground-truth spikes** | Optional **oracle / non-deployable** diagnostic (`--include-ground-truth-diagnostics`) |

Deployable models must survive missed spikes, jitter, contamination, amplitude drift, collisions, noise, and unit-quality variability. Therefore realtime never loads ground-truth-selected models.

### Causal window selection

- Update rate is fixed at **20 Hz / 50 ms** (`update_dt=0.050`).
- Under `--profile manifolds`, the **causal integration window `W`** is selected **independently per target** from the lean grid:

```text
[0.050, 0.250, 0.500, 1.000]
```

- `--profile standard` uses the full pool `[0.050, 0.100, 0.250, 0.500, 1.000]`.
- A common 250 ms window is allowed **only if it wins empirically** for that target — it is **not hard-coded**.
- After selection, if every target picks the same `W`, the workflow prints a warning and you should inspect `all_sorted_window_scores.csv`.

```bash
# Happy path — manifold search, gate, export (figures via Step 3)
python run_decoder.py \
    --input outputs/ratinabox_005 \
    --output outputs/ratinabox_005 \
    --skip-visualization

python run_visualizations.py \
    --experiment outputs/ratinabox_005 \
    --all \
    --compile-pdf

# Advanced: faster counts+PCA smoke
python run_decoder.py \
    --input outputs/ratinabox_005 \
    --output outputs/ratinabox_005 \
    --profile standard \
    --skip-visualization

# Advanced: oracle GT diagnostics (still selects deployable models from sorted)
python run_decoder.py \
    --input outputs/ratinabox_005 \
    --output outputs/ratinabox_005 \
    --include-ground-truth-diagnostics \
    --skip-visualization

# Advanced: dense temporal Table 8
python run_decoder.py \
    --input outputs/ratinabox_005 \
    --output outputs/ratinabox_005 \
    --profile full \
    --enable-temporal-manifold \
    --skip-visualization
```

| Flag | Purpose |
|------|---------|
| `--profile` | `manifolds` (default) / `standard` / `quick` / `full` |
| `--deployment-only` | Default: sorted-only deployable selection |
| `--include-ground-truth-diagnostics` | Also run GT oracle comparisons (non-deployable) |
| `--compare-sources` | Deprecated alias for GT diagnostics |
| `--closed-loop-target` | Target for realtime selection / triggers |
| `--selection-policy` | `shortest_near_optimal` or `best_accuracy` |
| `--decode-windows` | Override `W` grid (else from profile) |
| `--feature-modes` | Override observation / manifold modes |
| `--max-models` | `quick` or `full` model zoo |
| `--enable-isomap-distillation` | Add classic + distilled Isomap (on by default in `manifolds`) |
| `--enable-temporal-manifold` | Also run Table 8 W×L comparison |
| `--skip-comparison` | Reuse existing `decoder_comparison/` |
| `--skip-visualization` | Skip in-decode figure generation (preferred; use Step 3) |
| `--compile-pdf` | Optional in-decode PDF (prefer `run_visualizations.py`) |

### Deployment outputs to check before realtime

| Path | Contents |
|------|----------|
| `models/best_realtime_decoders.json` | **Deployable** per-target decoder, `W`, feature mode, artifact paths |
| `deployment_decoder_selection/all_sorted_window_scores.csv` | Full target × decoder × window score table |
| `deployment_decoder_selection/best_decoder_by_target_sorted.csv` | Sorted best table copy |
| `figures/deployment_decoder_selection/fig_deployment.png` | Publication deployment selection (winners + window×decoder heatmaps) |
| `figures/decoder_comparison/fig_decoding_performance.png` | Causal decoding performance (Fig 4) |
| `figures/decoder_comparison/fig_manifold_decoding.png` | Manifold vs counts (Fig 5) |
| `figures/decoder_comparison/fig_latent_geometry.png` | PCA / Isomap embeddings (Fig 6) |
| `figures/decoder_comparison/fig_isomap_diagnostics.png` | Isomap geometry diagnostics (Fig 7) |
| `figures/decoder_comparison/fig_isomap_story.png` | Isomap decoding + distillation (Fig 8) |
| `figures/realtime_decoding/fig_closed_loop.png` | Closed-loop realtime (Fig 9) |
| `figures/latency/fig_latency.png` | Latency budget (Fig 11) |

Step 1 prints manifold vs counts interpretations for sorted spikes (`manifold improves / reduces / comparable`).

---

## Visualizations

`run_visualizations.py` is the only public plotting entry. It **never** retrains decoders. With only `--experiment`, it detects available outputs and plots all of them.

```bash
python run_visualizations.py --experiment outputs/ratinabox_002 --all --compile-pdf
```

### Paper figure set

Default regeneration writes a **small set of seaborn multi-panel** `fig_*.png` files (matching the neural publication style) instead of dozens of single-panel decoder/manifold PNGs:

| Paper fig | Stem | Content |
|-----------|------|---------|
| Fig 1 | `fig_behavior_overview`, `fig_behavior_dynamics` (+ report) | Behavioral spatial overview + dynamics |
| Fig 2 | `fig_behavior_features`, `fig_neural_drivers`, `fig_circuit_population` | Covariates, drivers, circuit population |
| Fig 3 | `fig_cell_class_population`, `fig_population_structure`, `fig_spike_raster_summary` | Population structure + spikes |
| Fig 4 | `fig_decoding_performance` | Causal decoding vs window / best decoder |
| Fig 5 | `fig_manifold_decoding` | Counts vs PCA / region PCA (/ Isomap) |
| Fig 6 | `fig_latent_geometry` (+ `fig_latent_geometry_<feature>`) | All embeddings × one recovered feature per page (best W per mode; position = x→hue, y→brightness) |
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
(and k / n_neighbors) for decoding that variable — not a shared W. The canonical
`fig_latent_geometry.png` is a copy of the position page for PDF ordering.

Compiled `figures/output.pdf` follows: simulation → decoding → manifolds/Isomap → realtime → deployment → latency.

### Regenerating manifold / Isomap panels

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

`--profile manifolds` uses coarse `W`, latent dims `{3,8}`, Isomap neighbors `{10,30}`, and the quick decoder zoo so compute stays bounded while covering all realtime-relevant embeddings.

Future / deferred visualizations (not in this suite): 3D interactive embeddings, unit-ablation movies, animated Bayesian place-map GIFs.

Includes when present:

- behavior trajectory / occupancy / feature traces
- neural publication multi-panels (`fig_circuit_population`, …)
- decoder / manifold / Isomap / realtime / deployment / latency `fig_*` panels
- sectioned `figures/output.pdf`

---

## Output layout

```text
outputs/<run>/
  behavior.csv
  units.csv
  spikes_ground_truth.csv
  spikes_sorted.csv
  summary.json
  decoder_comparison/           # metrics, models, best_decoder_by_target.*
    sorted/ | ground_truth/
  realtime_decoding/            # decoded CSV, closed-loop events, latency
    {spike_source}/{target}_{policy}/
  decoding/                     # optional temporal W×L results
  figures/                      # subfolders + output.pdf only (no loose PNGs)
    behavior/                   # fig_behavior_overview, fig_behavior_dynamics
    features/                   # fig_behavior_features, fig_neural_drivers
    neural/                     # fig_circuit_population, fig_cell_class_*, …
    sorting/
    report/
    decoder_comparison/         # fig_decoding_performance, fig_manifold_*, fig_isomap_*, fig_latent_*
    realtime_decoding/          # fig_closed_loop
    deployment_decoder_selection/  # fig_deployment
    latency/                    # fig_latency
    temporal_decoding/          # fig_temporal_wl (when temporal comparison was run)
    output.pdf
```

---

## Shared code modules

| Module | Role |
|--------|------|
| `hippo_sim/` | Behavior, rates, spikes, recording, sorting |
| `realtime/data_loading.py` | Load simulation outputs |
| `realtime/timing.py` | `dt_update`, `W`, `L`, `τ`, timestamp validation |
| `realtime/spike_features.py` | Causal spike-count matrices |
| `realtime/decoder_models.py` | Model zoo |
| `realtime/bayesian_decoder.py` | Bayesian place decoder |
| `realtime/manifold_features.py` | Static manifold feature modes |
| `realtime/manifolds/` | Temporal raw / PCA / Isomap encoders (+ distillation) |
| `realtime/temporal/` | History sequences, controls, W×L comparison |
| `realtime/workflow.py` | Full decode orchestration |
| `realtime/workflow_profiles.py` | `quick` / `standard` / `full` lean defaults |
| `realtime/adaptive_windows.py` | Coarse→refine W; temporal W inheritance |
| `realtime/decoder_comparison.py` | Multi-model / window search |
| `realtime/deployment_selection.py` | Sorted-only deployable registry + window-score table |
| `realtime/best_decoder_selection.py` | Selection policies |
| `realtime/evaluate_realtime.py` | Closed-loop replay |
| `realtime/workflow.py` | Orchestration for public decoder script |
| `visualization/experiment_viz.py` | Unified figure generation |
| `visualization/pdf.py` | PDF compilation |

---

## Advanced / developer utilities

Most users should **not** need these. Tuning and best-decoder selection already run inside `run_full_decoder_workflow.py`.

| Utility | Role |
|---------|------|
| `run_decoder_comparison.py` | Comparison step only |
| `run_realtime_decoding.py` | Replay step only (manual or `--use-best-decoder`) |
| `run_BCI.py` | Staged simulate / manifolds / partitions / decode |

---

## Tests

```bash
python -m pytest tests/ -q
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Missing RatInABox | `pip install -r requirements.txt` inside `.hippo` |
| No figures / empty PDF | Ensure outputs exist; `python run_visualizations.py --experiment ... --all --compile-pdf` |
| Timestamp mismatch | Behavior is 20 Hz (`behavior_dt=0.05`); check `summary.json` |
| Want plots without re-decoding | Use `run_visualizations.py` only |
| Re-run only comparison for debugging | `run_decoder_comparison.py` (developer) |
| README equations look like raw LaTeX | This README uses Unicode/code blocks on purpose (no `\( \)` / `\[ \]`) |

---

## Limitations

- Simulated anatomy and rate models are configurable hypotheses, not recovered biology.
- Sorted spikes lose some ground-truth metadata.
- Supervised / task-conditioned embeddings (when added) reflect supplied labels.
- Realtime methods must not use future spikes or future smoothing.
- Results from simulation need experimental validation.

---

## Previous models

Adapted from:

- `previous_models/CA1.m` — Gaussian place fields, ensemble drift, Poisson spikes
- `previous_models/hw2simulationmethodinneuroscience.ipynb` — rate equation notation, drift parameters

## License

Research / educational use.
