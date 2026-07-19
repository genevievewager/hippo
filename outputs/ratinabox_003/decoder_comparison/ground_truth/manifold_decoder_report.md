# Manifold-informed decoder report

Region-specific PCA tests whether each hippocampal subregion contains a distinct low-dimensional code for behavioral variables. If region-specific manifolds outperform global PCA, this suggests that preserving anatomical structure helps decoding.

## 1. Manifold vs raw counts

- Targets where manifold improves decoding (>5%): position, acceleration, movement_state
- Targets where manifold is comparable (±5%): speed, head_direction, distance_to_wall, spatial_context, wall_distance_bin
- Targets where manifold reduces decoding (>5%): none

| target | spike_source | interpretation | counts | manifold | diff |
|---|---|---|---:|---:|---:|
| position | ground_truth | manifold improves decoding | 5.0614 | 4.1329 | 0.9285 |
| speed | ground_truth | manifold comparable to raw counts | 0.6615 | 0.6652 | 0.0037 |
| acceleration | ground_truth | manifold improves decoding | 0.0207 | 0.0305 | 0.0099 |
| head_direction | ground_truth | manifold comparable to raw counts | 13.0278 | 12.6959 | 0.3319 |
| distance_to_wall | ground_truth | manifold comparable to raw counts | 0.9066 | 0.9051 | -0.0015 |
| spatial_context | ground_truth | manifold comparable to raw counts | 0.9362 | 0.9188 | -0.0174 |
| movement_state | ground_truth | manifold improves decoding | 0.6938 | 0.7510 | 0.0573 |
| wall_distance_bin | ground_truth | manifold comparable to raw counts | 0.8577 | 0.8691 | 0.0114 |

## 2. Targets benefiting most from manifold features

- **position** (ground_truth): manifold improves decoding using `region_pca` (k=3.0)
- **head_direction** (ground_truth): manifold comparable to raw counts using `region_pca` (k=3.0)
- **movement_state** (ground_truth): manifold improves decoding using `region_pca` (k=3.0)
- **wall_distance_bin** (ground_truth): manifold comparable to raw counts using `region_pca` (k=3.0)
- **acceleration** (ground_truth): manifold improves decoding using `region_pca` (k=3.0)

## 3. Region / layer manifold strength

### region
- CA2: mean explained variance (selected components) = 0.516
- CA1: mean explained variance (selected components) = 0.409
- CA3: mean explained variance (selected components) = 0.404
- DG: mean explained variance (selected components) = 0.326


## 4. Components sufficient for best manifold setups

- position: `region_pca` with k=3.0 (window=0.5s, metric=4.1329)
- speed: `region_pca` with k=3.0 (window=0.1s, metric=0.6652)
- acceleration: `region_pca` with k=3.0 (window=0.1s, metric=0.0305)
- head_direction: `region_pca` with k=3.0 (window=0.1s, metric=12.6959)
- distance_to_wall: `region_pca` with k=3.0 (window=0.5s, metric=0.9051)
- spatial_context: `region_pca` with k=3.0 (window=0.5s, metric=0.9188)
- movement_state: `region_pca` with k=3.0 (window=0.1s, metric=0.7510)
- wall_distance_bin: `region_pca` with k=3.0 (window=0.25s, metric=0.8691)

## 5. Sorted vs ground-truth manifold decoding

Single spike source only; cross-source comparison unavailable.

## 6. Realtime suitability

PCA manifold transforms are linear and typically realtime-compatible; deploy using the selected transform fitted on training data only.
