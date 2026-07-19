# Hippocampal Neuropixels Simulation and Decoding

Simulate hippocampal single-unit activity during open-field navigation, record it through a Neuropixels-like probe, spike-sort with realistic degradation, then decode behavior from **causal** spike counts.

## Public workflow

```bash
source .hippo/bin/activate
pip install -r requirements.txt

# 1. Simulate
python run_simulation.py \
    --output outputs/ratinabox_002 \
    --seed 2 \
    --neural-backend ratinabox_neurons

# 2. Decode (compare windows/models → best closed-loop replay → optional figures)
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
| Run full decoder workflow | `run_full_decoder_workflow.py` |
| Generate all available visualizations | `run_visualizations.py --experiment ... --all --compile-pdf` |

---

## Installation

```bash
cd ~/projects/hippo
python3 -m venv .hippo
source .hippo/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

Python 3.10+ (developed on 3.12). Dependencies: see `requirements.txt`.

---

## Causal decoding (core rule)

At each behavioral frame time \(t\), features are spike counts from **\([t - W,\, t)\)** only — never spikes at or after \(t\).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `update_dt` | 0.050 s (20 Hz) | One prediction per behavioral frame |
| `decode_window` (`W`) | searched over 0.05–1.0 s | Causal neural integration window |
| `spike_source` | `sorted` | `sorted` or `ground_truth` spikes |
| `--compare-sources` | off | Run both spike sources and compare |

Pipeline inside `run_full_decoder_workflow.py`:

1. Decoder comparison across models and windows (`decoder_comparison/`)
2. Best decoder/window selection (`--selection-policy`)
3. Causal closed-loop realtime replay (`realtime_decoding/`)
4. Optional figures + `figures/output.pdf` (`--compile-pdf`)

---

## Visualizations

`run_visualizations.py` is the only public plotting entry point. Pass `--experiment` and it detects what exists:

- simulation → trajectory, occupancy, features, rasters, probe geometry
- `decoder_comparison/` → window/model comparison figures
- `realtime_decoding/` → closed-loop replay figures
- `--compile-pdf` → sectioned `figures/output.pdf`

```bash
# Everything available (default when only --experiment is set)
python run_visualizations.py --experiment outputs/ratinabox_002 --all --compile-pdf

# Subsets
python run_visualizations.py --experiment outputs/ratinabox_002 --include-simulation
python run_visualizations.py --experiment outputs/ratinabox_002 --include-comparison --include-realtime
```

Figures always land under `outputs/<run>/figures/` (with `decoder_comparison/` and `realtime_decoding/` subfolders). Plotting **never** retrains decoders.

---

## Simulation backends

```bash
# RatInABox neurons (default recommendation)
python run_simulation.py --output outputs/ratinabox_002 --seed 2 --neural-backend ratinabox_neurons

# Custom CA1/CA2/CA3/DG rate equations
python run_simulation.py --output outputs/run_custom_001 --seed 1 --neural-backend custom_rate_equations
```

Behavior is generated at **20 Hz** in a square open field. Pipeline: behavior → rates → ground-truth spikes → Neuropixels-like recording → sorting simulation.

### Custom rate equation (matches `hippo_sim/rate_equations.py`)

\[
\tau \frac{dR}{dt} = -R + \lambda^{\mathrm{target}}(t)
\]

with multiplicative place × HD × speed × acceleration × theta drive, plus boundary and ripple terms; CA3 adds recurrent population mean; DG applies a sparsity gate. Place-field drift SD = **0.1 cm/min**. Full parameter tables live in `hippo_sim/config.py`.

---

## Output layout

```text
outputs/<run>/
  behavior.csv
  units.csv
  spikes_ground_truth.csv
  spikes_sorted.csv
  summary.json
  decoder_comparison/          # from run_full_decoder_workflow.py
  realtime_decoding/           # closed-loop replay
  figures/                     # from workflow and/or run_visualizations.py
    decoder_comparison/
    realtime_decoding/
    output.pdf                 # with --compile-pdf
```

---

## Decoder CLI (public)

```bash
python run_full_decoder_workflow.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002 \
    --compare-sources \
    --spike-source sorted \
    --decode-windows 0.05 0.1 0.25 0.5 1.0 \
    --closed-loop-target spatial_context \
    --selection-policy shortest_near_optimal \
    --compile-pdf
```

Useful options:

| Flag | Purpose |
|------|---------|
| `--compare-sources` | Ground-truth vs sorted |
| `--closed-loop-target` | Target for realtime replay (e.g. `spatial_context`) |
| `--selection-policy` | `shortest_near_optimal` or `best_accuracy` |
| `--decode-windows` | Causal windows \(W\) to search |
| `--feature-modes` | e.g. `counts global_pca region_pca` |
| `--skip-visualization` | Skip figure generation |
| `--compile-pdf` | Write `figures/output.pdf` |
| `--enable-temporal-manifold` | Optional W×L temporal comparison |

---

## Advanced / developer utilities

Lower-level scripts and modules exist for debugging and custom experiments. **Most users should not need them.**

| Utility | Role |
|---------|------|
| `run_decoder_comparison.py` | Comparison step only |
| `run_realtime_decoding.py` | Replay step only |
| `run_full_workflow.py` | Staged simulate / manifolds / partitions / decode |
| `realtime/`, `visualization/`, `hippo/` | Importable package APIs |

Manifold partitions, temporal history (`L`), and communication analyses are documented in package modules and optional flags; they are not required for the standard workflow above.

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
| No figures / empty PDF | Ensure simulation (and optionally decoder) outputs exist; re-run `run_visualizations.py --all --compile-pdf` |
| Timestamp / alignment errors | Behavior is 20 Hz (`behavior_dt=0.05`); check `summary.json` |
| Want plots without re-decoding | Use `run_visualizations.py` only |

---

## Limitations

- Simulated anatomy and rate models are configurable hypotheses, not recovered biology.
- Sorted spikes lose some ground-truth metadata (e.g. true cell identity).
- Realtime-compatible methods must not use future information.
- Results from simulation need experimental validation.

## License

Research / educational use.
