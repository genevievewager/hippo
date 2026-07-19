# Hippocampal Neuropixels Simulation and Decoding

Simulates realistic hippocampal single-unit activity during 10-minute open-field navigation, recorded through a Neuropixels 1.0 single-shank probe (384 channels), with Neuropixels-like recording degradation and Kilosort-like spike re-extraction. Decodes latent behavioral variables from **causal** population spike counts, with optional low-dimensional manifold features.

---

## Public workflow

Three user-facing scripts. Hyperparameter search (decoder model, causal window \(W\), feature mode) runs **inside** the decoder workflow; you do not pick windows by hand for the standard path.

```bash
source .hippo/bin/activate
pip install -r requirements.txt

# 1. Simulate
python run_simulation.py \
    --output outputs/ratinabox_002 \
    --seed 2 \
    --neural-backend ratinabox_neurons

# 2. Decode (compare → select best → closed-loop replay → optional figures)
python run_full_decoder_workflow.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002 \
    --compare-sources \
    --closed-loop-target spatial_context \
    --selection-policy shortest_near_optimal \
    --compile-pdf

# 3. Visualize anytime (reads saved outputs only; never retrains)
python run_visualizations.py \
    --experiment outputs/ratinabox_002 \
    --all \
    --compile-pdf
```

| Goal | Command |
|------|---------|
| Simulate data | `run_simulation.py` |
| Full decoder workflow (tune + select + replay) | `run_full_decoder_workflow.py` |
| Generate all available visualizations | `run_visualizations.py --experiment ... --all --compile-pdf` |

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
2. **Features** — place, head direction, speed, acceleration, boundary, theta phase, ripple
3. **Rate equations** — CA1, CA2, CA3 pyramidal + DG granule with drift (or RatInABox neurons)
4. **Spikes** — ground-truth Poisson spike times
5. **Recording** — multi-channel templates, noise, motion amplitude drift, collisions
6. **Sorting** — Kilosort-like re-extraction with misses, jitter, contamination

### Simulation outputs

| File | Description |
|------|-------------|
| `behavior.csv` | Position, speed, head direction over time |
| `anatomy_regions.csv` | Region geometry and channel mapping |
| `units.csv` | Per-unit metadata (type, region, channel, place field) |
| `spikes_ground_truth.csv` | True spike times |
| `spikes_sorted.csv` | Re-extracted spike times after degradation |
| `summary.json` | Run statistics |

---

## Table 1: Rate Equations, Parameters, and Driver Features

Source of truth: `hippo_sim/rate_equations.py`, `hippo_sim/config.py`.

### Dynamics

\[
\tau_i \frac{dR_i}{dt} = -R_i + \lambda_i^{\mathrm{target}}(t)
\]

### Rate equations by cell type

| Cell type | Rate equation | Driver features |
|-----------|---------------|-----------------|
| **CA1_pyr** | \(\tau\,dR/dt = -R + \mathrm{target}\) where \(\mathrm{target} = [b + A\cdot f_{\mathrm{place}}\cdot(1 + w_{\mathrm{hd}}\cdot f_{\mathrm{hd}})\cdot(1 + w_{\mathrm{speed}}\cdot f_{\mathrm{speed}})\cdot(1 + 0.2\cdot f_{\mathrm{acc}})\cdot f_\theta + w_{\mathrm{bnd}}\cdot A\cdot f_{\mathrm{bnd}} + w_{\mathrm{ripple}}\cdot A\cdot f_{\mathrm{ripple}}]\cdot \xi_{\mathrm{state}}\cdot g\) | place, HD, speed, accel, boundary, theta, ripple |
| **CA2_pyr** | Same as CA1 with lower ripple/theta weights and sharper place fields (\(\sigma = 8\) cm) | place, HD, speed, accel, boundary, theta |
| **CA3_pyr** | Same as CA1 plus recurrent term \(+\, w_{\mathrm{rec}}\cdot \bar R_{\mathrm{pop}}\) | place, HD, speed, recurrent, ripple |
| **DG_granule** | Same structure with sparsity gate: if \(f_{\mathrm{place}}\cdot(1 + w_{\mathrm{speed}}\cdot f_{\mathrm{speed}}) < \theta_{\mathrm{sparse}}\) then \(\mathrm{target} \approx 0.1\cdot b\) | place, speed, boundary |

### Driver feature definitions

| Feature | Type | Mathematical definition |
|---------|------|-------------------------|
| **Place** (exteroceptive) | Allocentric | \(f_{\mathrm{place}} = \exp(-\|p - p_0\|^2 / 2\sigma^2)\) |
| **Head direction** (proprioceptive) | Egocentric | \(f_{\mathrm{hd}} = \exp(\kappa\cos(\theta - \theta_{\mathrm{pref}})) / (\exp(\kappa)/I_0(\kappa))\) |
| **Speed** (proprioceptive) | Egocentric | \(f_{\mathrm{speed}} = \max(0,\, v - v_{\mathrm{thresh}}) / (30 - v_{\mathrm{thresh}})\) |
| **Acceleration** (proprioceptive) | Egocentric | \(f_{\mathrm{acc}} = \mathrm{clip}(\|dv/dt\| / 50,\, 0,\, 1)\) |
| **Boundary** (exteroceptive) | Allocentric | \(f_{\mathrm{bnd}} = \exp(-d_{\mathrm{wall}}^2 / (2\cdot 15^2))\) |
| **Theta phase** (internal) | Internal | \(f_\theta = 1 + w_\theta\cos(\phi_\theta(t))\) |
| **Ripple** (internal) | CA1-biased | Sparse 80 ms bursts, \(\sin(\pi\cdot\mathrm{phase})\) envelope |

### Drift terms

| Process | Equation | Timescale |
|---------|----------|-----------|
| Place-field drift | \(p_0 \leftarrow p_0 + \mathcal{N}(0, \sigma_{\mathrm{drift}})\) every 30 s | slow (minutes) |
| State drift (arousal) | OU: \(d\xi = -\xi/\tau\,dt + \sigma\,dW\), multiply target by \(\exp(\xi)\) | \(\tau = 120\) s |
| Gain drift | OU on per-unit gain \(g\) | \(\tau = 180\) s |

### Parameter values

| Parameter | CA1_pyr | CA2_pyr | CA3_pyr | DG_granule | Units |
|-----------|---------|---------|---------|------------|-------|
| \(\tau\) | 0.05 | 0.05 | 0.06 | 0.08 | s |
| \(b\) (baseline) | 0.5 | 0.4 | 0.3 | 0.05 | Hz |
| \(A\) (amplitude) | 12.0 | 10.0 | 8.0 | 15.0 | Hz |
| \(\sigma_{\mathrm{place}}\) | 10.0 | 8.0 | 12.0 | 6.0 | cm |
| \(w_{\mathrm{hd}}\) | 0.4 | 0.35 | 0.2 | 0.1 | — |
| \(\kappa_{\mathrm{hd}}\) | 2.0 | 2.5 | 1.5 | 1.0 | — |
| \(w_{\mathrm{speed}}\) | 0.3 | 0.25 | 0.2 | 0.5 | — |
| \(v_{\mathrm{thresh}}\) | 2.0 | 2.0 | 2.0 | 3.0 | cm/s |
| \(w_\theta\) | 0.25 | 0.15 | 0.1 | 0.05 | — |
| \(f_\theta\) | 8.0 | 8.0 | 8.0 | 8.0 | Hz |
| \(w_{\mathrm{ripple}}\) | 0.5 | 0.1 | 0.6 | 0.0 | — |
| \(w_{\mathrm{boundary}}\) | 0.2 | 0.15 | 0.1 | 0.3 | — |
| \(w_{\mathrm{recurrent}}\) | — | — | 0.15 | — | — |
| \(\theta_{\mathrm{sparse}}\) | — | — | — | 0.3 | — |

| Drift parameter | Value | Units |
|-----------------|-------|-------|
| Behavior / rate update rate | 20 | Hz |
| Place drift \(\sigma\) | **0.1** | cm/min |
| Place update interval | 30 | s |
| State drift \(\tau\) | 120 | s |
| State drift \(\sigma\) | 0.15 | — |
| Gain drift \(\tau\) | 180 | s |
| Gain drift \(\sigma\) | 0.1 | — |

| Recording / sorting parameter | Value |
|-------------------------------|-------|
| Channels | 384 |
| Site pitch | 20 µm |
| Sample rate | 30 kHz |
| Template span | 3–10 channels |
| Amplitude range | 20–200 µV |
| Noise \(\sigma\) | 15 µV |
| Miss rate | 12% |
| Jitter | 0.3 ms |
| Contamination rate | 0.08 |
| Merge probability | 0.04 |

---

## Table 2: Anatomical Regions and Probe Geometry

Neuropixels 1.0 single-shank, 384 channels, 20 µm pitch, 1D depth axis (dorsal hippocampus).

| Region | Layer | Depth start (µm) | Depth end (µm) | Channels (approx.) | Cell types | Density (units/channel) |
|--------|-------|------------------|----------------|--------------------|------------|-------------------------|
| CA1 | oriens | 0 | 200 | 1–10 | CA1_pyr | 2.0 |
| CA1 | pyramidal | 200 | 400 | 11–20 | CA1_pyr | 8.0 |
| CA1 | radiatum | 400 | 600 | 21–30 | CA1_pyr | 5.0 |
| CA2 | pyramidal | 600 | 800 | 31–40 | CA2_pyr | 6.0 |
| CA3 | pyramidal | 800 | 1400 | 41–70 | CA3_pyr | 7.0 |
| DG | granule | 1400 | 1800 | 71–90 | DG_granule | 10.0 |
| DG | hilus | 1800 | 2000 | 91–100 | DG_granule | 3.0 |

Exact mapping is written to `anatomy_regions.csv` per run.

---

## Neural activity backends

### 1. Custom hippocampal rate equations

```bash
python run_simulation.py \
    --output outputs/run_custom_001 \
    --seed 1 \
    --neural-backend custom_rate_equations
```

### 2. RatInABox neurons (recommended)

Uses RatInABox `PlaceCells`, head-direction, boundary-vector, and speed cells on the same trajectory; Poisson spikes feed the same recording/sorting/decoding pipeline. Saves `rate_model` and `ratinabox_class` in `units.csv`.

```bash
python run_simulation.py \
    --output outputs/ratinabox_003 \
    --seed 1 \
    --neural-backend ratinabox_neurons
```

---

## Table 3: Causal Decoding — Timing and Features

**Scientific framing.** The simulator knows true behavior. The decoder only sees spike counts. `spike_source=ground_truth` is ideal neural information; `spike_source=sorted` is information after Neuropixels degradation and sorting. Window search asks how much recent history is needed; model search asks which estimator extracts latent variables best; closed-loop replay tests realtime triggers.

### Causal observation

At each behavioral frame time \(t\):

\[
x_t^{(W)} \in \mathbb{R}^{N},\qquad
\bigl(x_t^{(W)}\bigr)_i
=
\#\{\text{spikes of unit } i \text{ with times in } [t-W,\, t)\}
\]

Half-open interval: spikes with time \(\ge t\) are **never** included.

Optional rate features: \(r_t^{(W)} = x_t^{(W)} / W\).

### Timing parameters

| Symbol / CLI | Default | Meaning |
|--------------|---------|---------|
| Behavior timestamps | from `behavior.csv` | Prediction grid (prefer actual frames) |
| `update_dt` / `dt_update` | **0.050 s** (20 Hz) | One prediction per behavioral frame |
| `decode_window` / \(W\) | searched | Causal integration window |
| `train_frac` | 0.70 | Contiguous train fraction (rest = test) |
| Alignment tolerance | 0.005 s | Max allowed decode↔behavior mismatch |

### Default window search

| \(W\) (s) | 0.050 | 0.100 | 0.250 | 0.500 | 1.000 |
|-----------|-------|-------|-------|-------|-------|
| Duration | 50 ms | 100 ms | 250 ms | 500 ms | 1 s |

Short \(W\): lower latency, fewer spikes. Long \(W\): more reliable counts, higher effective latency.

### Roles kept separate

| Symbol | Role | Not the same as |
|--------|------|-----------------|
| \(W\) | Spike evidence for **current** neural state | Model memory |
| \(z_t\) | Manifold coordinate of current observation | Temporal dynamics |
| \(L\) | History of recent latents (optional temporal stage) | Integration window |
| \(\tau\) | Prediction lag \(\hat y_{t+\tau}\) | Update interval |

---

## Table 4: Decoding Targets and Metrics

| Target | Family | Output | Primary metric | Direction |
|--------|--------|--------|----------------|-----------|
| `position` | continuous | \((x,y)\) cm | `mean_position_error_cm` | lower |
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
| `ridge` | regression | \(\alpha = 1.0\) |
| `pca_ridge` | regression | PCA retain 95% variance → Ridge \(\alpha=1.0\) |
| `random_forest_regressor` | regression | 100 trees, `max_depth=12` |
| `logistic_regression` | classification | `max_iter=1000`, `class_weight=balanced` |
| `random_forest_classifier` | classification | 100 trees, `max_depth=12`, `class_weight=balanced` |
| `bayesian_place_decoder` | position / wall distance (added in quick when applicable) | see Table 6 |
| `bayesian_place_decoder_derived_context` | spatial_context / wall_distance_bin (when applicable) | see Table 6 |

### Full mode (`--max-models full`)

Adds: `elastic_net` (\(\alpha=0.1\), `l1_ratio=0.5`), `pls_regression` (10 components), `hist_gradient_boosting_{regressor,classifier}` (`max_depth=8`), `knn_{regressor,classifier}` (\(k=15\), distance weights), `linear_svm_classifier`, `bayesian_place_decoder_smoothed`.

### Selection policies (under the hood)

After comparing all \((\mathrm{model},\, W,\, \mathrm{feature\_mode})\) configurations per target:

| Policy | Rule |
|--------|------|
| `best_accuracy` | Best primary metric over all windows |
| `shortest_near_optimal` (default) | Shortest \(W\) with metric within **5%** of best (≤ 1.05× best if lower-is-better; ≥ 0.95× best if higher-is-better) |

Closed-loop replay loads the selected `.joblib` and does **not** retrain when comparison artifacts exist.

---

## Table 6: Bayesian Place Decoder

Source: `realtime/bayesian_decoder.py`.

Poisson likelihood place-map decoder on a 2D grid.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_bins` | 20 | Bins per spatial axis |
| `occupancy_floor` | \(10^{-3}\) | Floor on occupancy / prior |
| `smooth` | `false` (`true` for `_smoothed` variant) | Causal exponential posterior smoothing |
| `smooth_alpha` | 0.7 | Smoothing weight on previous posterior |

\[
\log P(\mathrm{bin}\mid x_t)
\propto
\log P(\mathrm{bin})
+
\sum_i \bigl[ x_{t,i}\log\lambda_i(\mathrm{bin}) - \lambda_i(\mathrm{bin})\,W \bigr]
\]

(with implementation details in code; smoothing uses **past** posteriors only).

Derived-context variant maps the decoded place posterior to categorical spatial / wall-distance labels.

---

## Table 7: Manifold Feature Modes (static decoding)

Source: `realtime/manifold_features.py`.

Architecture for manifold-informed static decoding:

\[
x_t^{(W)}
\;\xrightarrow{\;E\;}
z_t
\;\xrightarrow{\;\mathrm{decoder}\;}
\hat y_t
\]

Encoder \(E\) is fit on **training** activity only, then frozen for test / realtime.

| Feature mode | Transform | Grouping column | Latent dim |
|--------------|-----------|-----------------|------------|
| `counts` | \(z = x\) | — | \(N\) |
| `rates` | \(z = x / W\) | — | \(N\) |
| `global_pca` | PCA on all units | — | \(k\) (default 3) |
| `region_pca` | PCA per region, concatenate | `region` | \(k\) per group |
| `layer_pca` | PCA per layer | `layer` | \(k\) per group |
| `cell_type_pca` | PCA per cell type | `cell_type` | \(k\) per group |
| `rate_model_pca` | PCA per rate model | `rate_model` (fallback `ratinabox_class`) | \(k\) per group |

| Setting | Default |
|---------|---------|
| Quick feature modes | `counts`, `global_pca`, `region_pca` |
| Default \(k\) list | `(3,)` (override with `--manifold-components-list 2 3 5`) |
| Fit data | train split only |
| Realtime | load saved transform from comparison `models/` |

**Note:** Static manifold PCA in `manifold_features.py` is applied to raw (or rate) count vectors. The optional **temporal** PCA path (`realtime/manifolds/pca.py`) uses \(\sqrt{\mathrm{counts}}\) + `StandardScaler` before PCA — see Table 8.

---

## Table 8: Temporal Manifold Decoding (optional)

Enabled with `--enable-temporal-manifold` on `run_full_decoder_workflow.py`. Config reference: `configs/temporal_decoding.yaml`.

\[
x_t^{(W)}
\;\rightarrow\;
z_t
\;\rightarrow\;
Z_t^{(L)} = [z_{t-L+1},\ldots,z_t]
\;\rightarrow\;
\hat y_{t+\tau}
\]

| Parameter | Defaults | Meaning |
|-----------|----------|---------|
| \(W\) | same window grid as static | Spike integration |
| Representations | `raw`, `pca` | Latent encoder |
| PCA preprocess | \(\sqrt{x}\) + standardize | Temporal PCA path |
| Default latent dim (temporal PCA) | 16 | From config |
| \(L\) (frames) | 1, 2, 5, 10, 20 | History length |
| \(L\) duration @ 20 Hz | 50, 100, 250, 500, 1000 ms | |
| \(\tau\) | 0.0 (searchable) | Prediction lag |
| Temporal models | `raw_static`, `static_latent`, `flattened_history` | Phase-1 |
| Controls | `averaged_history`, `shuffled_sequence` | |
| Split | train / val / test contiguous + leakage gap | Selection on **validation** only |

Planned (not in baseline): causal GRU, TCN, autoencoder, CEBRA.

---

## Decoder workflow (public script)

`run_full_decoder_workflow.py` runs under the hood:

1. **Compare** models × windows × feature modes → `decoder_comparison/`
2. **Select** best / shortest-near-optimal config per target
3. **Replay** closed-loop causal decoding → `realtime_decoding/`
4. **Optional** figures + `figures/output.pdf`

```bash
python run_full_decoder_workflow.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002 \
    --compare-sources \
    --spike-source sorted \
    --decode-windows 0.05 0.1 0.25 0.5 1.0 \
    --feature-modes counts global_pca region_pca \
    --manifold-components-list 3 \
    --closed-loop-target spatial_context \
    --selection-policy shortest_near_optimal \
    --compile-pdf
```

| Flag | Purpose |
|------|---------|
| `--compare-sources` | Run both `ground_truth` and `sorted` |
| `--closed-loop-target` | Target for realtime selection / triggers |
| `--selection-policy` | `shortest_near_optimal` or `best_accuracy` |
| `--decode-windows` | Values of \(W\) to search |
| `--feature-modes` | Observation / manifold modes |
| `--max-models` | `quick` or `full` model zoo |
| `--enable-temporal-manifold` | Also run Table 8 W×L comparison |
| `--skip-visualization` | Skip figure generation |
| `--compile-pdf` | Write `figures/output.pdf` |

---

## Visualizations

`run_visualizations.py` is the only public plotting entry. It **never** retrains decoders. With only `--experiment`, it detects available outputs and plots all of them.

```bash
python run_visualizations.py --experiment outputs/ratinabox_002 --all --compile-pdf
```

Includes when present:

- behavior trajectory / occupancy / feature traces
- neural driver features, ground-truth and sorted rasters, probe geometry
- `decoder_comparison/` figures
- `realtime_decoding/` closed-loop figures
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
  figures/                      # all PNGs
    decoder_comparison/
    realtime_decoding/
    output.pdf
```

---

## Shared code modules

| Module | Role |
|--------|------|
| `hippo_sim/` | Behavior, rates, spikes, recording, sorting |
| `realtime/data_loading.py` | Load simulation outputs |
| `realtime/timing.py` | `dt_update`, \(W\), \(L\), \(\tau\), timestamp validation |
| `realtime/spike_features.py` | Causal spike-count matrices |
| `realtime/decoder_models.py` | Model zoo |
| `realtime/bayesian_decoder.py` | Bayesian place decoder |
| `realtime/manifold_features.py` | Static manifold feature modes |
| `realtime/manifolds/` | Temporal raw / PCA encoders |
| `realtime/temporal/` | History sequences, controls, W×L comparison |
| `realtime/decoder_comparison.py` | Multi-model / window search |
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
| `run_full_workflow.py` | Staged simulate / manifolds / partitions / decode |

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
