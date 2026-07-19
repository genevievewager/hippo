# Manifold-informed decoder report

Region-specific PCA tests whether each hippocampal subregion contains a distinct low-dimensional code for behavioral variables. If region-specific manifolds outperform global PCA, this suggests that preserving anatomical structure helps decoding.

## 1. Manifold vs raw counts

- Targets where manifold improves decoding (>5%): acceleration, movement_state, wall_distance_bin
- Targets where manifold is comparable (±5%): position, speed, head_direction, spatial_context
- Targets where manifold reduces decoding (>5%): distance_to_wall

| target | spike_source | interpretation | counts | manifold | diff |
|---|---|---|---:|---:|---:|
| position | sorted | manifold comparable to raw counts | 7.1007 | 7.0178 | 0.0829 |
| speed | sorted | manifold comparable to raw counts | 0.5567 | 0.5726 | 0.0159 |
| acceleration | sorted | manifold improves decoding | 0.0168 | 0.0241 | 0.0073 |
| head_direction | sorted | manifold comparable to raw counts | 16.9939 | 16.4150 | 0.5790 |
| distance_to_wall | sorted | manifold reduces decoding | 0.7817 | 0.7283 | -0.0534 |
| spatial_context | sorted | manifold comparable to raw counts | 0.8648 | 0.8658 | 0.0010 |
| movement_state | sorted | manifold improves decoding | 0.6088 | 0.7054 | 0.0966 |
| wall_distance_bin | sorted | manifold improves decoding | 0.7707 | 0.8436 | 0.0730 |

## 2. Targets benefiting most from manifold features

- **head_direction** (sorted): manifold comparable to raw counts using `region_pca` (k=3.0)
- **movement_state** (sorted): manifold improves decoding using `region_pca` (k=3.0)
- **position** (sorted): manifold comparable to raw counts using `region_pca` (k=3.0)
- **wall_distance_bin** (sorted): manifold improves decoding using `region_pca` (k=3.0)
- **speed** (sorted): manifold comparable to raw counts using `region_pca` (k=3.0)

## 3. Region / layer manifold strength

### region
- CA2: mean explained variance (selected components) = 0.432
- CA3: mean explained variance (selected components) = 0.378
- CA1: mean explained variance (selected components) = 0.308
- DG: mean explained variance (selected components) = 0.281


## 4. Components sufficient for best manifold setups

- position: `region_pca` with k=3.0 (window=1.0s, metric=7.0178)
- speed: `region_pca` with k=3.0 (window=0.25s, metric=0.5726)
- acceleration: `region_pca` with k=3.0 (window=0.1s, metric=0.0241)
- head_direction: `region_pca` with k=3.0 (window=0.25s, metric=16.4150)
- distance_to_wall: `region_pca` with k=3.0 (window=1.0s, metric=0.7283)
- spatial_context: `region_pca` with k=3.0 (window=1.0s, metric=0.8658)
- movement_state: `region_pca` with k=3.0 (window=0.25s, metric=0.7054)
- wall_distance_bin: `region_pca` with k=3.0 (window=1.0s, metric=0.8436)

## 5. Sorted vs ground-truth manifold decoding

Single spike source only; cross-source comparison unavailable.

## 6. Realtime suitability

PCA manifold transforms are linear and typically realtime-compatible; deploy using the selected transform fitted on training data only.
