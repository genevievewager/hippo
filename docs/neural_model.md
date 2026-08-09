# Hippocampal neural rate model

Source of truth: `hippo_sim/ratinabox_neural_backend.py`, `hippo_sim/feedforward.py`,
`hippo_sim/hippocampal_populations.py`, `hippo_sim/config.py`.

Neural activity is generated in three stacked stages.

## Stage A — RatInABox receptive fields

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
| interneuron | synthetic | constructed in Stage C (legacy global label) |
| INT_CA1 / INT_CA3 / INT_DG / INT_CA2 / INT_SUB | synthetic | local pools constructed in Stage C |

## Stage B — within-cell-type dynamics overlays

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

## Stage C — trisynaptic / entorhinal feedforward

Define region means (after Stage B):

```text
R̄_A(t) = (1/|A|) Σ_{i∈A} R_i(t),   A ∈ {MEC, DG, CA3, CA2, CA1, SUB, INT_*}
```

Optional normalization \(\tilde R_A = R̄_A / \max_t R̄_A\) (default on) makes weights dimensionless. Then, in order:

```text
DG  ← DG  + w_MEC→DG  · R̃_MEC
CA3 ← CA3 + w_DG→CA3  · R̃_DG          # CA3↔CA3 recurrence already in Stage B
CA2 ← CA2 + w_MEC→CA2 · R̃_MEC + w_CA3→CA2 · R̃_CA3
CA1 ← CA1 + w_CA3→CA1 · R̃_CA3 + w_MEC→CA1 · R̃_MEC
```

When Subiculum is present and `w_mec_to_sub ≠ 0`, an additional MEC → SUB drive is applied.

Local interneurons are then built from each post-feedforward home principal:

```text
R_i^{INT_X}(t) = g_i · b_int · (1 + w_θ cos 2π f_θ t) · (1 − w_anti · R̃_X(t))
               + w_ripple · A_int · r(t)
```

and each pool inhibits only its home region:

```text
X ← max( X − w_INT→X · R̃_INT_X , 0 )   for X ∈ {CA1, CA3, DG, CA2, SUB}
```

## Circuit architecture

```text
MEC → DG → CA3 → CA1
MEC → CA1
MEC → CA2
CA3 → CA2
local INT_X → home region X
```

## Default feedforward weights

Tunable hypotheses in `ratinabox_params.feedforward`:

| Synapse | Symbol | Default |
|---------|--------|---------|
| MEC → DG | `w_mec_to_dg` | 0.20 |
| DG → CA3 | `w_dg_to_ca3` | 0.25 |
| CA3 → CA1 | `w_ca3_to_ca1` | 0.20 |
| MEC → CA1 (direct) | `w_mec_to_ca1` | 0.15 |
| INT_CA1 → CA1 | `w_int_to_ca1` | 0.30 |
| INT_CA3 → CA3 | `w_int_to_ca3` | 0.25 |
| INT_DG → DG | `w_int_to_dg` | 0.25 |
| INT_CA2 → CA2 | `w_int_to_ca2` | 0.20 |
| INT_SUB → SUB | `w_int_to_sub` | 0.20 |
| MEC → CA2 | `w_mec_to_ca2` | 0.10 |
| CA3 → CA2 | `w_ca3_to_ca2` | 0.10 |
| CA3 recurrent (Stage B) | `w_recurrent` | 0.15 |

Disable with `apply_feedforward: false` in `ratinabox_params`. Metadata is written to `neural_backend_metadata.json` under `feedforward`.

## Overlay / rate parameters (principal cells)

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

## Population counts (maximally hippocampal config)

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
| INT_CA1 | 6 | synthetic | INT_CA1 | theta, ripples, anti-CA1-pyr |
| INT_CA3 | 3 | synthetic | INT_CA3 | theta, ripples, anti-CA3-pyr |
| INT_DG | 3 | synthetic | INT_DG | theta, ripples, anti-DG |
| INT_CA2 | 1 | synthetic | INT_CA2 | theta, ripples, anti-CA2-pyr |
| INT_SUB | 2 | synthetic | INT_SUB | theta, ripples, anti-Sub |

Saves `rate_model` and `ratinabox_class` in `units.csv`.

## Schematic anatomy (when `--no-trajectory`)

Neuropixels 1.0 single-shank defaults from `hippo_sim/config.py`: 384 channels, 20 µm pitch.

| Region | Layer | Depth start (µm) | Depth end (µm) | Cell types | Density (units/channel) |
|--------|-------|------------------|----------------|------------|-------------------------|
| CA1 | oriens | 0 | 200 | INT_CA1 | 2.0 |
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

See also the root [README](../README.md) for the concise scientific overview.
