# Manifold-informed decoder report

Region-specific PCA tests whether each hippocampal subregion contains a distinct low-dimensional code for behavioral variables. If region-specific manifolds outperform global PCA, this suggests that preserving anatomical structure helps decoding.

## 1. Manifold vs raw counts

- Targets where manifold improves decoding (>5%): position, acceleration, movement_state, wall_distance_bin
- Targets where manifold is comparable (±5%): speed, head_direction, spatial_context
- Targets where manifold reduces decoding (>5%): distance_to_wall

| target | spike_source | interpretation | counts | manifold | diff |
|---|---|---|---:|---:|---:|
| position | sorted | manifold improves decoding | 13.1058 | 10.1629 | 2.9429 |
| speed | sorted | manifold comparable to raw counts | 0.5567 | 0.5756 | 0.0189 |
| acceleration | sorted | manifold improves decoding | 0.0168 | 0.0257 | 0.0089 |
| head_direction | sorted | manifold comparable to raw counts | 16.9939 | 16.3691 | 0.6248 |
| distance_to_wall | sorted | manifold reduces decoding | 0.6462 | 0.5979 | -0.0483 |
| spatial_context | sorted | manifold comparable to raw counts | 0.7701 | 0.7914 | 0.0213 |
| movement_state | sorted | manifold improves decoding | 0.6088 | 0.7054 | 0.0966 |
| wall_distance_bin | sorted | manifold improves decoding | 0.7194 | 0.7756 | 0.0562 |

## 2. Targets benefiting most from manifold features

- **position** (sorted): manifold improves decoding using `region_pca` (k=3.0)
- **head_direction** (sorted): manifold comparable to raw counts using `region_pca` (k=2.0)
- **movement_state** (sorted): manifold improves decoding using `region_pca` (k=3.0)
- **wall_distance_bin** (sorted): manifold improves decoding using `region_pca` (k=3.0)
- **spatial_context** (sorted): manifold comparable to raw counts using `region_pca` (k=3.0)

## 3. Region / layer manifold strength

### region
- CA2: mean explained variance (selected components) = 0.335
- CA3: mean explained variance (selected components) = 0.285
- CA1: mean explained variance (selected components) = 0.218
- DG: mean explained variance (selected components) = 0.209


## 4. Components sufficient for best manifold setups

- position: `region_pca` with k=3.0 (window=0.25s, metric=10.1629)
- speed: `region_pca` with k=2.0 (window=0.25s, metric=0.5756)
- acceleration: `region_pca` with k=2.0 (window=0.1s, metric=0.0257)
- head_direction: `region_pca` with k=2.0 (window=0.25s, metric=16.3691)
- distance_to_wall: `region_pca` with k=3.0 (window=0.25s, metric=0.5979)
- spatial_context: `region_pca` with k=3.0 (window=0.25s, metric=0.7914)
- movement_state: `region_pca` with k=3.0 (window=0.25s, metric=0.7054)
- wall_distance_bin: `region_pca` with k=3.0 (window=0.25s, metric=0.7756)

## 5. Sorted vs ground-truth manifold decoding

Single spike source only; cross-source comparison unavailable.

## 6. Realtime suitability

PCA manifold transforms are linear and typically realtime-compatible; deploy using the selected transform fitted on training data only.
