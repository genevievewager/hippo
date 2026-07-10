# Hippocampal Neuropixels Simulation

Simulates realistic hippocampal single-unit activity during 10-minute open-field navigation, recorded through a Neuropixels 1.0 single-shank probe (384 channels), with Neuropixels-like recording degradation and Kilosort-like spike re-extraction.

## Quick start

```bash
source .hippo/bin/activate
pip install -r requirements.txt
python run_simulation.py --output outputs/run_001 --seed 1
```

### Detached run (survives terminal close)

```bash
mkdir -p outputs/run_001
nohup .hippo/bin/python run_simulation.py --output outputs/run_001 \
  > outputs/run_001/simulation.log 2>&1 &
echo $! > outputs/run_001/simulation.pid
tail -f outputs/run_001/simulation.log
```

## Pipeline

1. **Behavior** — RatInABox square open field (thigmotaxis, stalls, smooth turns) at **20 Hz** (50 ms steps)
2. **Features** — place, head direction, speed, acceleration, boundary, theta phase, ripple
3. **Rate equations** — CA1, CA2, CA3 pyramidal + DG granule with drift
4. **Spikes** — ground-truth Poisson spike times
5. **Recording** — multi-channel templates, noise, motion amplitude drift, collisions
6. **Sorting** — Kilosort-like re-extraction with misses, jitter, contamination

## Outputs

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

### Rate equations by cell type

| Cell type | Rate equation | Driver features |
|-----------|---------------|-----------------|
| **CA1_pyr** | `τ dR/dt = −R + target` where `target = [b + A·f_place·(1 + w_hd·f_hd)·(1 + w_speed·f_speed)·(1 + 0.2·f_acc)·f_θ + w_bnd·A·f_bnd + w_ripple·A·f_ripple] · ξ_state · gain` | place, HD, speed, accel, boundary, theta, ripple |
| **CA2_pyr** | Same as CA1 with lower ripple/theta weights and sharper place fields (σ = 8 cm) | place, HD, speed, accel, boundary, theta |
| **CA3_pyr** | Same as CA1 plus recurrent term `+ w_rec · R̄_pop` | place, HD, speed, recurrent, ripple |
| **DG_granule** | Same structure with sparsity gate: if `f_place·(1 + w_speed·f_speed) < θ_sparse` then `target ≈ 0.1·b` | place, speed, boundary |

### Driver feature definitions

| Feature | Type | Mathematical definition |
|---------|------|-------------------------|
| **Place** (exteroceptive) | Allocentric | `f_place = exp(−‖p − p₀‖² / 2σ²)` |
| **Head direction** (proprioceptive) | Egocentric | `f_hd = exp(κ cos(θ − θ_pref)) / (exp(κ)/I₀(κ))` |
| **Speed** (proprioceptive) | Egocentric | `f_speed = max(0, v − v_thresh) / (30 − v_thresh)` |
| **Acceleration** (proprioceptive) | Egocentric | `f_acc = clip(|dv/dt| / 50, 0, 1)` |
| **Boundary** (exteroceptive) | Allocentric | `f_bnd = exp(−d_wall² / 2σ_bnd²)` |
| **Theta phase** (proprioceptive) | Internal | `f_θ = 1 + w_θ cos(2π f_θ t)` |
| **Ripple** (internal) | CA1-biased | Sparse 80 ms bursts, `sin(π·phase)` envelope |

### Drift terms

| Process | Equation | Timescale |
|---------|----------|-----------|
| Place-field drift | `p₀ ← p₀ + N(0, σ_drift)` every 30 s | slow (minutes) |
| State drift (arousal) | OU: `dξ = −ξ/τ · dt + σ · dW`, `target × exp(ξ)` | τ = 120 s |
| Gain drift | OU on per-unit gain `g` | τ = 180 s |

### Parameter values

| Parameter | CA1_pyr | CA2_pyr | CA3_pyr | DG_granule | Units |
|-----------|---------|---------|---------|------------|-------|
| τ | 0.05 | 0.05 | 0.06 | 0.08 | s |
| b (baseline) | 0.5 | 0.4 | 0.3 | 0.05 | Hz |
| A (amplitude) | 12.0 | 10.0 | 8.0 | 15.0 | Hz |
| σ_place | 10.0 | 8.0 | 12.0 | 6.0 | cm |
| w_hd | 0.4 | 0.35 | 0.2 | 0.1 | — |
| κ_hd | 2.0 | 2.5 | 1.5 | 1.0 | — |
| w_speed | 0.3 | 0.25 | 0.2 | 0.5 | — |
| v_thresh | 2.0 | 2.0 | 2.0 | 3.0 | cm/s |
| w_θ | 0.25 | 0.15 | 0.1 | 0.05 | — |
| f_θ | 8.0 | 8.0 | 8.0 | 8.0 | Hz |
| w_ripple | 0.5 | 0.1 | 0.6 | 0.0 | — |
| w_boundary | 0.2 | 0.15 | 0.1 | 0.3 | — |
| w_recurrent | — | — | 0.15 | — | — |
| θ_sparse | — | — | — | 0.3 | — |

| Drift parameter | Value | Units |
|-----------------|-------|-------|
| Behavior / rate update rate | 20 | Hz |
| Place drift σ | 0.8 | cm/min |
| Place update interval | 30 | s |
| State drift τ | 120 | s |
| State drift σ | 0.15 | — |
| Gain drift τ | 180 | s |
| Gain drift σ | 0.1 | — |

| Recording parameter | Value |
|---------------------|-------|
| Channels | 384 |
| Site pitch | 20 µm |
| Sample rate | 30 kHz |
| Template span | 3–10 channels |
| Amplitude range | 20–200 µV |
| Noise σ | 15 µV |
| Miss rate | 12% |
| Jitter | 0.3 ms |

---

## Table 2: Anatomical Regions and Probe Geometry

Neuropixels 1.0 single-shank, 384 channels, 20 µm pitch, 1D depth axis (dorsal hippocampus).

| Region | Layer | Depth start (µm) | Depth end (µm) | Channels | Cell types | Density (units/channel) |
|--------|-------|------------------|----------------|----------|------------|-------------------------|
| CA1 | oriens | 0 | 200 | 1–10 | CA1_pyr | 2.0 |
| CA1 | pyramidal | 200 | 400 | 11–20 | CA1_pyr | 8.0 |
| CA1 | radiatum | 400 | 600 | 21–30 | CA1_pyr | 5.0 |
| CA2 | pyramidal | 600 | 800 | 31–40 | CA2_pyr | 6.0 |
| CA3 | pyramidal | 800 | 1400 | 41–70 | CA3_pyr | 7.0 |
| DG | granule | 1400 | 1800 | 71–90 | DG_granule | 10.0 |
| DG | hilus | 1800 | 2000 | 91–100 | DG_granule | 3.0 |

Channel counts are approximate; exact mapping is written to `anatomy_regions.csv` per run.

## Previous models

Adapted from:
- `previous_models/CA1.m` — Gaussian place fields, ensemble drift, Poisson spikes
- `previous_models/hw2simulationmethodinneuroscience.ipynb` — rate equation notation, drift parameters

## Neural activity backends

The simulator supports two neural activity backends.

### 1. Custom hippocampal rate equations

This backend uses explicit CA1, CA2, CA3, and DG rate equations driven by behavioral features including position, head direction, speed, acceleration, boundary distance, theta phase, ripple state, and drift.

Run:

```bash
python run_simulation.py \
    --output outputs/run_custom_001 \
    --seed 1 \
    --neural-backend custom_rate_equations
```

### 2. RatInABox neurons

This backend uses RatInABox neural classes to generate spatially and/or velocity-modulated firing rates from the same RatInABox trajectory. The rates are converted into ground-truth Poisson spike trains so that the rest of the Neuropixels recording, sorting, visualization, and decoding pipeline remains unchanged.

Run:

```bash
python run_simulation.py \
    --output outputs/run_ratinabox_001 \
    --seed 1 \
    --neural-backend ratinabox_neurons
```

The RatInABox backend saves `rate_model` and `ratinabox_class` columns in `units.csv` and `spikes_ground_truth.csv`, allowing spike rasters to be sorted by the neural model that generated each unit.

## License

Research / educational use.

## Visualization Suite

The simulation includes a visualization suite for inspecting behavior, neural driver features, ground-truth Poisson spike trains, sorted-spike degradation, and hippocampal probe geometry.

Run:

```bash
python run_visualizations.py \
    --input outputs/run_001 \
    --output outputs/run_001/figures
```

Main outputs include:

* behavioral trajectory and occupancy maps
* behavioral feature traces over time
* neural driver feature traces
* ground-truth spike rasters sorted by cell class and rate equation
* population activity by cell class
* sorted versus ground-truth spike comparisons
* simulated Neuropixels probe geometry
* a combined report summary figure

## Real-Time Closed-Loop Decoding

This module simulates online decoding from hippocampal Neuropixels spike activity. The decoder updates every 25 ms and uses a causal 250 ms spike-history window. At each time point, it estimates the animal's position, spatial context, movement state, and speed from sorted spikes only. The estimated state can be used to trigger a simulated closed-loop event.

The decoder is causal: it only uses spikes from the current and past window, never future spikes.

Example:

```bash
python run_realtime_decoding.py \
    --input outputs/run_001 \
    --output outputs/run_001/realtime_decoding \
    --spike-source sorted \
    --update-dt 0.025 \
    --decode-window 0.250
```

To compare ideal ground-truth spike decoding against sorted-spike decoding:

```bash
python run_realtime_decoding.py \
    --input outputs/run_001 \
    --output outputs/run_001/realtime_decoding \
    --compare-sources
```
