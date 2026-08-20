# Decoding methods

Decoder zoo source: `realtime/decoder_models.py`. Continuous targets use regressors (position / HD via `MultiOutputRegressor` where needed). Categorical targets use classifiers. Sklearn pipelines that need scaling include `StandardScaler` where applicable.

## Targets and metrics

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

## Quick mode (`--max-models quick`, default)

| Name | Task | Default hyperparameters |
|------|------|-------------------------|
| `ridge` | regression | `α = 1.0` |
| `pca_ridge` | regression | PCA retain 95% variance → Ridge `α = 1.0` |
| `random_forest_regressor` | regression | 100 trees, `max_depth=12` |
| `logistic_regression` | classification | `max_iter=1000`, `class_weight=balanced` |
| `random_forest_classifier` | classification | 100 trees, `max_depth=12`, `class_weight=balanced` |
| `bayesian_place_decoder` | position / wall distance (added in quick when applicable) | see below |
| `bayesian_place_decoder_derived_context` | spatial_context / wall_distance_bin (when applicable) | see below |

## Full mode (`--max-models full`)

Adds: `elastic_net` (`α=0.1`, `l1_ratio=0.5`), `pls_regression` (10 components), `hist_gradient_boosting_regressor` / `hist_gradient_boosting_classifier` (`max_depth=8`), `knn_regressor` / `knn_classifier` (`k=15`, distance weights), `linear_svm_classifier`, `bayesian_place_decoder_smoothed`, plus RBF SVR/SVC and small MLPs in some paths.

## Selection policies

After comparing all `(model, W, feature_mode)` configurations per target:

| Policy | Rule |
|--------|------|
| `best_accuracy` | Best primary metric over all windows |
| `shortest_near_optimal` (default) | Shortest `W` with metric within **5%** of best (≤ 1.05× best if lower-is-better; ≥ 0.95× best if higher-is-better) |

Closed-loop replay loads the selected `.joblib` and does **not** retrain when comparison artifacts exist.

## Bayesian place decoder

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

Smoothing uses **past** posteriors only. The derived-context variant maps the decoded place posterior to categorical spatial / wall-distance labels.

## Static manifold feature modes

Source: `realtime/manifold_features.py`.

Architecture:

```text
x_t^(W)  →  E(·)  →  z_t  →  decoder  →  ŷ_t
```

Encoder `E` is fit on **training** activity only, then frozen for test / realtime.

| Feature mode | Transform | Grouping column | Latent dim |
|--------------|-----------|-----------------|------------|
| `counts` | `z = x` | — | `N` |
| `rates` | `z = x / W` | — | `N` |
| `global_pca` | PCA on all units | — | `k` |
| `region_pca` | PCA per region, concatenate | `region` | `k` per group |
| `layer_pca` | PCA per layer | `layer` | `k` per group |
| `cell_type_pca` | PCA per cell type | `cell_type` | `k` per group |
| `rate_model_pca` | PCA per rate model | `rate_model` (fallback `ratinabox_class`) | `k` per group |
| `global_isomap` | Isomap on all units (√counts + scale + optional pre-PCA) | — | `d` (offline only) |
| `global_isomap_distilled` | parametric approx. of Isomap | — | `d` (realtime if gates pass) |
| `diffusion_nystrom` | landmark diffusion maps + Nyström | — | `d` (realtime; `P99 < 25 ms`) |
| `global_lds` | dynamic LDS latent | — | `k` (realtime / causal) |
| `gpfa` | GPFA-style latent | — | `k` (offline / acausal) |

| Setting | Default |
|---------|---------|
| Quick feature modes | `counts`, `global_pca`, `region_pca` (legacy F+E composite; typically `F=counts`) |
| Manifolds profile modes | counts + global/region PCA + classic/distilled Isomap + `diffusion_nystrom` |
| Observation `F` types (`feature_representations`) | `counts`, `rates`, `sqrt_counts`, `log1p_counts`, `zscore_counts`, `region_normalized_counts`, `cell_type_normalized_counts` |
| Default `k` list | `(3,)` (override with `--manifold-components-list`) |
| Manifolds profile `k` | `{3, 8}` |
| Isomap neighbors | default `10`; manifolds profile `{10, 30}` |
| Fit data | train split only |
| Offline-only modes | `global_isomap`, `gpfa` |

**Note:** Static manifold PCA in `manifold_features.py` is applied to raw (or rate) count vectors. The optional **temporal** PCA path (`realtime/manifolds/pca.py`) uses `√counts` + `StandardScaler` before PCA. Static / temporal Isomap both use √counts + standardization + optional pre-PCA and are tagged `offline_analysis_only`.

## Temporal manifold decoding (optional)

Enabled with `--enable-temporal-manifold` on `run_decoder.py`. Config reference: `configs/temporal_decoding.yaml`.

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

Default decode profile `manifolds` leaves temporal off unless you pass `--enable-temporal-manifold`. Under lean temporal mode, `W` is inherited from Step 1 (best/recommended windows plus short/long flanks). Use `--profile full` for the dense research grid.

Planned (not in baseline): causal GRU, TCN, autoencoder, CEBRA.

## Causal timing roles

| Symbol | Role | Not the same as |
|--------|------|-----------------|
| `W` | Spike evidence for **current** neural state | Model memory |
| `z_t` | Manifold / latent coordinate of current observation | Temporal dynamics model |
| `L` | History of recent latents (optional temporal stage) | Integration window |
| `τ` | Prediction lag `ŷ_{t+τ}` | Update interval |
| `update_dt` | Decoder update cadence | Behavior sampling / `W` |

See the root README timing table for defaults and supported values.
