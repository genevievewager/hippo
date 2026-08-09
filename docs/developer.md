# Developer reference

Most users should use the public scripts in the root README:

```text
run_simulation.py → run_decoder.py → run_visualizations.py
```

or `streamlit run ui/app.py` (same backends; no science reimplementation in the UI).

## Shared modules

| Module | Role |
|--------|------|
| `hippo_sim/` | Behavior, rates, spikes, recording, sorting |
| `hippo/anatomy/` | Trajectory import (NTE), cell capture, trajectory figures |
| `hippo/visualization/` | Publication probe-trajectory plots (`probe_trajectory.py`) |
| `realtime/data_loading.py` | Load simulation outputs |
| `realtime/timing.py` | `dt_update`, `W`, `L`, `τ`, timestamp validation |
| `realtime/spike_features.py` | Causal spike-count matrices |
| `realtime/decoder_models.py` | Model zoo |
| `realtime/bayesian_decoder.py` | Bayesian place decoder |
| `realtime/manifold_features.py` | Static manifold feature modes (+ dynamic embedding factory hooks) |
| `realtime/dynamic_latents/` | Dynamic latent models (LDS, GPFA), registry, metrics, figures |
| `realtime/manifolds/` | Temporal raw / PCA / Isomap encoders (+ distillation) |
| `realtime/temporal/` | History sequences, controls, W×L comparison |
| `realtime/workflow.py` | Full decode orchestration |
| `realtime/workflow_profiles.py` | `quick` / `standard` / `full` / `manifolds` / `feature_robustness` |
| `realtime/adaptive_windows.py` | Coarse→refine W; temporal W inheritance |
| `realtime/decoder_comparison.py` | Multi-model / window search |
| `realtime/deployment_selection.py` | Sorted-only deployable registry + window-score table |
| `realtime/best_decoder_selection.py` | Selection policies |
| `realtime/evaluate_realtime.py` | Closed-loop replay |
| `visualization/experiment_viz.py` | Unified figure generation |
| `visualization/pdf.py` | PDF compilation |
| `ui/` | Streamlit pages calling the same backends |

## Advanced / non-public utilities

| Utility | Role |
|---------|------|
| `run_decoder_comparison.py` | Comparison / F×E×D×W×C grid only (no registry/replay) |
| `archive/run_realtime_decoding.py` | Archived replay-only helper |
| `run_BCI.py` | Staged simulate / manifolds / partitions / decode |

Tuning, deployable registry export, and best-decoder closed-loop replay already run inside `run_decoder.py` → `realtime/workflow.py::run_full_decoder_workflow`.

### Staged BCI workflow example

```bash
python run_BCI.py \
    --stage decode-temporal \
    --input outputs/ratinabox_004 \
    --output outputs/ratinabox_004 \
    --representations pca isomap \
    --isomap-neighbors 10 20 \
    --enable-temporal-manifold
```

## Tests

```bash
python -m pytest tests/ -q

# Short public-workflow smoke (simulate → decode → visualize)
bash scripts/smoke_test_public_workflow.sh
```

Component-focused tests include (non-exhaustive):

- `tests/test_workflow_profiles.py`
- `tests/test_isomap_manifold.py`, `tests/test_isomap_decoders.py`, `tests/test_isomap_distillation.py`, `tests/test_isomap_temporal.py`
- `tests/test_dynamic_latents.py`
- `tests/test_trajectory_import.py`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Missing RatInABox | `pip install -r requirements.txt` inside `.hippo` |
| No figures / empty PDF | Ensure outputs exist; `python run_visualizations.py --experiment ... --all --compile-pdf` |
| Timestamp mismatch | Behavior is 20 Hz (`behavior_dt=0.05`); check `summary.json` |
| Want plots without re-decoding | Use `run_visualizations.py` only |
| Re-run only comparison for debugging | `run_decoder_comparison.py` |
| README equations look like raw LaTeX | Root README uses Unicode/code blocks on purpose |

## Previous models

Adapted from:

- `previous_models/CA1.m` — Gaussian place fields, ensemble drift, Poisson spikes
- `previous_models/hw2simulationmethodinneuroscience.ipynb` — rate equation notation, drift parameters

## Documentation map

| Doc | Contents |
|-----|----------|
| [neural_model.md](neural_model.md) | Rate equations, populations, feedforward weights |
| [neuropixels_model.md](neuropixels_model.md) | Recording / sorting parameters |
| [anatomy_and_trajectory.md](anatomy_and_trajectory.md) | Trajectories, NTE, capture allowlist |
| [decoding_methods.md](decoding_methods.md) | Decoder zoo, targets, temporal W×L |
| [manifolds.md](manifolds.md) | Isomap details, distillation, diagnostics |
| [dynamic_latents.md](dynamic_latents.md) | LDS / GPFA commands and outputs |
| [realtime_deployment.md](realtime_deployment.md) | Gating, registry, latency |
| [visualizations.md](visualizations.md) | Figure catalog |
| [output_schema.md](output_schema.md) | Artifact tree |
| [cli_reference.md](cli_reference.md) | Flags and profiles |
