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
    --output outputs/ratinabox_002 \
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
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/figures
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

## Real-Time Decoding Workflow

The decoder pipeline turns hippocampal spike activity into latent behavioral state estimates using **causal population spike-count features**. It has two computation modes and one plotting mode.

```text
Simulation output
      ↓
causal spike-count features
      ↓
decoder model(s)
      ↓
decoded latent behavioral state
      ↓
closed-loop trigger and/or evaluation
      ↓
visualization
```

At each decoder update time `t`, the decoder constructs a causal population spike-count vector:

`X(t) = spike counts from [t - decode_window, t)`

The decoder never uses future spikes. The default `update_dt` is 25 ms. The default causal history `decode_window` is 250 ms.

Short windows reduce latency but contain fewer spikes. Long windows improve spike-count reliability but increase effective decoding latency. The decoder comparison module tests this tradeoff directly.

**Scientific framing:** The simulator knows the true behavior. The decoder only sees hippocampal spike counts. `spike_source=ground_truth` represents ideal neural information; `spike_source=sorted` represents information available after Neuropixels degradation and spike sorting. Comparing causal windows tells us how much recent neural history is needed to decode behavioral, exteroceptive, and proprioceptive variables. Comparing decoder models tells us which model class best extracts those latent variables from population activity. Closed-loop replay tests whether decoded state estimates are accurate enough to trigger interventions in real time.

### Step 1: Choose whether to optimize or replay

If you do not know the best decoder model and `decode_window` yet, run **decoder comparison** first.

If you already know the settings and want to simulate **closed-loop replay**, run **single-run realtime decoding**.

### Step 2A: Decoder comparison and causal window optimization

**Script:** `run_decoder_comparison.py`

Use this before closed-loop replay when you want to determine which decoder and causal history window work best. It tests multiple decoder models and multiple `decode_window` values, evaluates latent behavioral / exteroceptive / proprioceptive targets, reports the best model/window per target, and reports the shortest near-optimal causal window. Use `--compare-sources` to compare ground-truth versus sorted spikes.

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/decoder_comparison \
    --compare-sources \
    --decode-windows 0.025 0.050 0.100 0.250 0.500 1.000
```

Single spike source only:

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/decoder_comparison \
    --spike-source sorted \
    --decode-windows 0.025 0.050 0.100 0.250 0.500 1.000
```

### Step 2B: Single realtime closed-loop replay

**Script:** `run_realtime_decoding.py`

Use this when you already know the decoder settings you want to test. It uses one `spike_source` (or compares ground-truth versus sorted with `--compare-sources`), one causal `decode_window`, trains/evaluates the selected decoder setup, replays the session causally, optionally generates closed-loop events, and saves decoded state estimates.

```bash
python run_realtime_decoding.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/realtime_decoding \
    --spike-source sorted \
    --update-dt 0.025 \
    --decode-window 0.250
```

Compare ground-truth versus sorted spikes at one window (no closed-loop trigger comparison across many models):

```bash
python run_realtime_decoding.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/realtime_decoding \
    --compare-sources \
    --update-dt 0.025 \
    --decode-window 0.250
```

### Step 3: Decoder visualization

**Script:** `run_decoder_visualization.py`

Use this only after computation. It reads saved CSV/JSON/prediction outputs, makes figures, and does not retrain decoders or recompute comparisons.

```bash
python run_decoder_visualization.py \
    --realtime-dir outputs/ratinabox_002/realtime_decoding \
    --comparison-dir outputs/ratinabox_002/decoder_comparison
```

Either `--realtime-dir` or `--comparison-dir` (or both) must be provided.

### Recommended order

1. Run the simulation.
2. Run decoder comparison to identify the best model and causal spike-history window.
3. Run single-run realtime closed-loop decoding using the chosen settings.
4. Run decoder visualization to generate report figures.

```bash
python run_simulation.py \
    --output outputs/ratinabox_002 \
    --seed 2 \
    --neural-backend ratinabox_neurons
```

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/decoder_comparison \
    --compare-sources \
    --decode-windows 0.025 0.050 0.100 0.250 0.500 1.000
```

```bash
python run_realtime_decoding.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/realtime_decoding \
    --spike-source sorted \
    --update-dt 0.025 \
    --decode-window 0.250
```

```bash
python run_decoder_visualization.py \
    --realtime-dir outputs/ratinabox_002/realtime_decoding \
    --comparison-dir outputs/ratinabox_002/decoder_comparison
```

### Which script should I run?

| Goal | Script |
|---|---|
| Test one realtime decoder setup | `run_realtime_decoding.py` |
| Compare many decoders and windows | `run_decoder_comparison.py` |
| Compare sorted spikes to ground-truth spikes | `--compare-sources` with either computation script |
| Generate figures from saved decoder results | `run_decoder_visualization.py` |
| Choose the best causal history window | `run_decoder_comparison.py` |
| Simulate closed-loop triggers | `run_realtime_decoding.py` |

### Output layout

```text
outputs/run_001/
    realtime_decoding/          # single-run closed-loop replay
        sorted/                 # one spike_source: CSVs, JSON, models
        ground_truth/           # when --compare-sources
        comparison/             # side-by-side figures (after visualization)
        figures/                # single-source replay figures (after visualization)

    decoder_comparison/         # model/window optimization
        sorted/                 # metrics, models, decoded_examples per source
        ground_truth/           # when --compare-sources
        source_comparison_metrics.csv
        figures/                # summary + per-source figures (after visualization)
```

**Terminology:** `decode_window` = causal spike-history window; `update_dt` = decoder update interval; `spike_source` = `ground_truth` or `sorted`; **single-run realtime decoding** = one decoder/window configuration; **decoder comparison** = many decoder/window configurations; **closed-loop replay** = causal decoding plus trigger logic. Both computation scripts use causal realtime-compatible features.

### Shared code modules

Both computation scripts share:

| Module | Role |
|---|---|
| `realtime/data_loading.py` | Load simulation outputs |
| `realtime/spike_features.py` | Causal spike-count feature construction |
| `realtime/decoder_models.py` | Decoder model zoo (sklearn pipelines) |
| `realtime/train_decoder.py` | Training for single-run replay |
| `realtime/decoder_comparison.py` | Multi-model/window evaluation |
| `visualization/decoder_plots.py` | All decoder figures (plot-only script) |

# Github 
```bash
cd ~/projects/hippo
git status
git add -A
git commit -m "Update project files"
git push
```