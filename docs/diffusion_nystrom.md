# Diffusion Maps + Nyström (deployable nonlinear manifold)

`diffusion_nystrom` is a **realtime-compatible** nonlinear embedding `E` in the existing `F × E × D × W` search. It does **not** replace PCA, Isomap, or counts.

## Standard nonlinear manifold issue

Classic nonlinear methods (Isomap, full diffusion maps) build a graph or kernel on **all** training observations and eigendecompose an `N×N` operator. That is too slow to rebuild, and often too slow even to extend, inside a 25 ms / 40 Hz decode loop.

## Solution

Fit the nonlinear diffusion geometry **offline** on a compact landmark set `L` (`M ≪ N`), then project every new observation with Nyström extension:

```text
neural window
→ feature extraction
→ fixed training-time normalization
→ query-to-landmark kernel k(x_t, L)
→ Nyström diffusion coordinates z_t
→ decoder
→ behavioral prediction ŷ_t
```

Conceptually:

```text
x_t → k(x_t, L) → p(x_t, L) → z_t → ŷ_t
```

The expensive eigendecomposition happens **offline only**. Replay and live inference call `transform_one(x_t)` and never `fit()`.

Query-side local bandwidth `σ_x = d_k(x, L)` adapts the kernel to local density. It does **not** retrain or replace the fitted diffusion operator (landmarks, landmark scales, eigenvectors, eigenvalues, `α`, and the feature scaler stay frozen for the deployment session).

## Offline vs online

| Stage | Work |
|-------|------|
| Offline `fit` | landmark selection → landmark kernel → density normalization (`α`) → symmetric diffusion operator `A` → eigendecomposition → drop trivial `λ≈1` → precompute Nyström projection |
| Online `transform_one` | scale `x` with the **training** scaler → distances to fixed landmarks → query kernel → normalize → `z = p @ projection_matrix` |

Decode **window** (e.g. 500 ms of spikes) is not computation **latency**. A long `W` can still emit a prediction every 25 ms if the operation finishes in a few milliseconds.

Realtime qualification: `P99(T_operation) < 25 ms` (`realtime_qualified`). Headroom is `25 ms − P99` when positive.

## Defaults

| Parameter | Default |
|-----------|---------|
| `n_landmarks` | 512 |
| `landmark_method` | `minibatch_kmeans` (`random`, `kmeans` also supported) |
| `n_components` | 10 nontrivial diffusion dimensions |
| `local_scale_k` | 10 |
| `alpha` | 1.0 |
| `diffusion_time` | 1 |
| `dtype` | `float32` |

Implementation: `realtime/manifolds/diffusion_nystrom.py` (`DiffusionNystrom`). Feature-mode wrapper: `diffusion_nystrom` in `realtime/manifold_features.py`.

## Commands

```bash
# Offline comparison including diffusion_nystrom
python run_decoder.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001 \
    --feature-modes counts global_pca diffusion_nystrom \
    --n-landmarks 512 \
    --landmark-method minibatch_kmeans \
    --diffusion-components 10 \
    --skip-visualization

# Landmark-count vs accuracy/latency table (128…2048)
python run_decoder.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001 \
    --feature-modes diffusion_nystrom \
    --n-landmarks 128 256 512 1024 2048 \
    --benchmark-diffusion-landmarks \
    --skip-visualization

# Comparison-only grid
python run_decoder_comparison.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001/decoder_comparison/sorted \
    --manifolds counts global_pca diffusion_nystrom \
    --n-landmarks 512

# Realtime replay loads the fitted bundle (never calls fit)
python run_decoder.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001 \
    --skip-comparison \
    --skip-visualization
```

`--profile manifolds` includes `diffusion_nystrom` with the other RT-relevant embeddings.

## Outputs

| Path | Contents |
|------|----------|
| `decoder_comparison/*/models/manifold_transforms/*diffusion_nystrom*_nl*_w*ms/` | Fitted embedding (`meta.json`, landmarks, projection, scaler) |
| `.../diagnostics/` | Eigenvalue spectrum, coordinates, landmark coverage, OOD summary |
| `decoder_comparison/*/diffusion_landmark_benchmark/` | Landmark-count tradeoff CSV/JSON/PNG |
| `realtime_decoding/*/latency/` | Per-update stage latencies, P99, deadline misses, `realtime_qualified` |

OOD flags (`nearest_landmark_distance`, `sigma_x`, `max_kernel_weight`, `kernel_entropy`, `effective_n_landmarks`) are logged; they do **not** trigger retraining.
