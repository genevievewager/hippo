# Manifold representations and Isomap

Isomap is a **nonlinear neural representation**, not a behavioral decoder.

Architecture:

```text
x_t^(W)  →  E_Isomap(·)  →  z_t  →  D(·)  →  ŷ_t
```

or with temporal history kept separate from Isomap geometry:

```text
x_t^(W)  →  z_t^Isomap  →  Z_t^(L) = [z_{t−L+1}, …, z_t]  →  D_temporal  →  ŷ_{t+τ}
```

| Concept | Role |
|---------|------|
| Manifold extraction `E` | Maps high-D neural activity → low-D coordinates `z_t` |
| Behavioral decoder `D` | Maps `z_t` (or raw `x_t`) → behavior / memory targets |
| Isomap | One possible **offline** nonlinear choice for `E` |

Do **not** assume a nonlinear manifold requires a nonlinear decoder. Always compare linear and nonlinear `D` on the same Isomap coordinates (and on PCA / raw).

## Scientific motivation

Hippocampal population activity may lie on a curved, low-dimensional manifold. PCA is a useful linear baseline, but it may need many components to approximate a manifold whose intrinsic dimension is small. Isomap estimates **geodesic** distances along a neighbor graph and embeds them with classical MDS.

## Pipeline (training-only fit)

```text
sqrt(counts)
→ StandardScaler
→ optional pre-PCA
→ Isomap
→ z_t
```

Held-out coordinates use the training model’s out-of-sample `transform` (sklearn inductive Isomap). Do not concatenate train+val+test before fitting merely for a continuous visualization. Tag any full-session embedding as `exploratory_full_session_embedding` / `not_valid_for_held_out_performance`.

## Algorithm (sklearn `manifold.Isomap`)

Given neural observations `X ∈ R^{T×N}` (rows = time, columns = units):

1. Build a `k`-nearest-neighbor graph on **training** observations only.
2. Weight edges by pairwise distances between neighbors.
3. Estimate geodesic distances via shortest paths on the graph.
4. Apply classical multidimensional scaling (MDS) to the geodesic matrix.
5. Return coordinates `Z ∈ R^{T×d}`.

Implementation: `realtime/manifolds/isomap.py` (`IsomapManifoldEncoder`), registered as `"isomap"`. Static decoding feature mode: `global_isomap` in `realtime/manifold_features.py`.

## Config keys

From `configs/temporal_decoding.yaml` → `manifold.methods.isomap`:

| Key | Default | Notes |
|-----|---------|-------|
| `n_neighbors` | `[5,10,20,30,50]` | Too small → disconnected graph; too large → shortcut edges |
| `latent_dims` / `n_components` | `[2,3,4,6,8]` | Embedding dimension `d` |
| `pre_pca.enabled` | `true` | Reduces noise / cost when `N` is large |
| `pre_pca.n_components` | `50` | Or set variance threshold |
| `require_connected_graph` | `true` | Reject disconnected configs |
| `allow_largest_component_only` | `false` | Optional recovery if largest component ≥ 95% |
| `realtime_compatible` | `false` | Standard Isomap is offline-only |

`--profile manifolds` uses lean grids: latent dims `{3,8}`, Isomap neighbors `{10,30}`.

## Graph diagnostics

After fit, diagnostics report:

- connected-component count and largest-component fraction
- min / median / max / mean node degree
- `mean_degree / n_train` (dense-graph flag)
- duplicate-observation fraction
- geodesic-distance finite fraction

Disconnected graphs raise `DisconnectedGraphError` (or are recorded with `exclusion_reason`) unless `allow_largest_component_only` is enabled.

## Geometry metrics

- Trustworthiness at several `k`
- k-NN overlap / continuity proxy
- Geodesic vs embedding distance correlation (sampled pairs)
- Residual variance `1 − R²(D_G, D_Z)`

Hyperparameters are selected from **held-out validation** decoding / geometry metrics — not from visualization quality.

## Linear vs nonlinear decoders

Recommended comparisons:

```text
raw + Ridge / Random Forest
PCA + Ridge / Random Forest
Isomap + Ridge / Random Forest / k-NN
diffusion_nystrom + Ridge / Random Forest
```

This separates **nonlinear representation** benefit from **nonlinear output mapping** benefit.

## Real-time compatibility

| Method | Tag | Auto-deploy in closed-loop? |
|--------|-----|-------------------------------|
| `isomap` / `global_isomap` | `offline_analysis_only` | **No** |
| `isomap_distilled` / `global_isomap_distilled` | parametric approx. of Isomap | **Yes** if latency ≤ 50 ms **and** held-out distortion OK |
| `diffusion_nystrom` | landmark diffusion maps + Nyström | **Yes** if `P99(T_operation) < 25 ms` |
| `pca` / `counts` | realtime-capable | Yes |

Distilled-Isomap’s 50 ms check is the decoder **update** budget (`update_dt`, typically 50 ms / 20 Hz). Diffusion Nyström’s 25 ms `realtime_qualified` flag is **operation** compute (`P99(T_operation)`), distinct from window length `W`. See [realtime_deployment.md](realtime_deployment.md).

Best-decoder selection prefers realtime-compatible feature modes for closed-loop recommendations even when offline Isomap wins on accuracy.

### Distilled Isomap

Parametric distillation (`realtime/manifolds/isomap_distillation.py`, static mode `global_isomap_distilled`) trains Ridge / kernel Ridge / small MLP so `E_θ(x_t) ≈ z_t^Isomap`. This is **not** exact Isomap. Enable with `--enable-isomap-distillation`, or use `--profile manifolds` (distillation on by default).

## Temporal decoding on Isomap

Isomap geometry and temporal history are kept separate: encode each causal window to `z_t`, then build `Z_t^(L)`. Supported temporal models: `static_latent`, `flattened_history`, plus controls `averaged_history` / `shuffled_sequence`. Pass `--representations isomap` with `--enable-temporal-manifold`.

## Commands

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

# Temporal Isomap (offline; inherits W from Step 1 when profile inherits windows)
python run_decoder.py \
    --input outputs/ratinabox_004 \
    --output outputs/ratinabox_004 \
    --enable-temporal-manifold \
    --representations pca isomap \
    --isomap-neighbors 10 \
    --isomap-latent-dim 8 \
    --latent-history-frames 1 2 5 10 20

# Figures (reads saved outputs only)
python run_visualizations.py --experiment outputs/ratinabox_004 --all --compile-pdf

# Tests
python -m pytest tests/test_isomap_manifold.py tests/test_isomap_decoders.py \
    tests/test_isomap_distillation.py tests/test_isomap_temporal.py -q
```

## Outputs

| Path | Contents |
|------|----------|
| `decoder_comparison/*/models/manifold_transforms/global_isomap_k*_nn*_w*ms/` | Fitted Isomap + preprocessing (`meta.json`, `isomap.joblib`, `scaler.joblib`, `pre_pca.joblib`, diagnostics) |
| `decoder_comparison/*/decoder_comparison_metrics.csv` | Rows include `n_neighbors`, `graph_connected`, `trustworthiness`, `residual_variance`, `decoder_nonlinear`, `realtime_compatible` |
| `decoding/comparison/*/manifolds/isomap/` | Temporal Isomap encoders per `W` |
| `decoding/comparison/*/all_configurations.csv` | Temporal W×L results with Isomap geometry columns |
| Distilled models (optional API) | `IsomapDistilledEncoder.save(...)` → `meta.json` + `distiller.joblib` |

## Troubleshooting

| Issue | Guidance |
|-------|----------|
| Disconnected graph | Increase `n_neighbors`; check duplicate population vectors |
| Shortcut / collapsed geometry | Decrease `n_neighbors`; inspect dense-graph flag |
| Slow / high memory | Enable pre-PCA; reduce `T` via coarser update; sample distance pairs |
| Too few units / samples | Isomap needs enough neighbors; prefer PCA or raw |
| Unstable `n_neighbors` | Report stability across neighbors; select on validation metrics |
| Poor held-out transform | Expected limitation of inductive Isomap; consider distillation only as approximation |
| Realtime wants Isomap | Use PCA/counts/distilled for closed loop; classic Isomap informs dimensionality / geometry only |

Do **not** conclude Isomap is superior because a 2-D plot looks more curved. Use held-out decoding and geometry metrics.

See also: [`diffusion_nystrom.md`](diffusion_nystrom.md) for the deployable nonlinear alternative (landmark diffusion maps + Nyström).
