# Engineering audit report — README ↔ public workflow

Date: 2026-07-26  
Scope: Harden the consolidated public workflow (`run_simulation.py` → `run_decoder.py` → `run_visualizations.py`) so code matches the README contract. No new scientific features.

## 1. README contracts — status

| # | Contract | Status |
|---|----------|--------|
| 1 | Three-script happy path works as documented | **Satisfied** |
| 2 | `run_decoder.py` orchestrates comparison → registry → realtime replay | **Satisfied** |
| 3 | Deployable selection is sorted-spikes only | **Satisfied** (fixed) |
| 4 | Ground-truth is oracle / non-deployable | **Satisfied** |
| 5 | Causal half-open windows `[t−W, t)` | **Satisfied** |
| 6 | Decoded rows include window metadata | **Satisfied** (fixed) |
| 7 | Target-specific causal `W` in registry | **Satisfied** |
| 8 | Uniform-`W` warning points at `all_sorted_window_scores.csv` | **Satisfied** |
| 9 | Classic Isomap cannot enter deployable registry | **Satisfied** (hardened) |
| 10 | Distilled Isomap latency/distortion gated | **Satisfied** |
| 11 | Latency profiling under `latency_profiling/` | **Satisfied** (canonical summary added) |
| 12 | Visualization is read-only (no retrain / no mutation of model artifacts) | **Satisfied** (hardened) |
| 13 | Output layout matches README | **Satisfied** (README + validators) |
| 14 | Public smoke test exists and passes | **Satisfied** |

## 2. Code fixes made in this pass

1. **Sorted-only deployable selection** (`realtime/deployment_selection.py`)
   - Removed the empty-filter fallback that could relabel ground-truth rows as sorted.
   - `write_deployment_selection_artifacts` now raises if only `ground_truth` rows are present.

2. **Classic Isomap remapping** (`realtime/deployment_selection.py`)
   - Under `best_accuracy`, offline Isomap winners are remapped to the realtime-recommended counts/PCA artifact.
   - Model path and manifold transform path are cleared/replaced (no dangling Isomap `.joblib` in the registry).
   - Registry rows record `remapped_from_offline_isomap` and `realtime_compatible`.

3. **Causal window metadata on realtime + example outputs** (`realtime/realtime_decoder.py`, `realtime/decoder_comparison.py`)
   - Every realtime decoded row now includes:
     `decode_time`, `window_start`, `window_end`, `decode_window_s`, `update_dt_s`,
     `n_spikes_in_window`, `n_active_units_in_window`.
   - Best decoded-example CSVs include the same window fields.

4. **Latency stage reporting** (`realtime/latency_profiler.py`, `realtime/realtime_decoder.py`, `realtime/latency_benchmark.py`)
   - Stages now include `manifold_transform` in addition to spike binning / feature transform / per-target decode / closed-loop / total.
   - Writes `latency_profiling/latency_summary.csv` and `latency_summary.json`.
   - Annotates `models/best_realtime_decoders.json` with budget compatibility; RF classifier heads over the 50 ms / 20 Hz budget are flagged and marked not realtime-compatible.

5. **Output contract validators** (`realtime/output_contracts.py`)
   - Machine-checkable required artifacts for simulation / decode / visualization.

6. **Visualization read-only hardening** (`visualization/publication_isomap_plots.py`)
   - Replaced sklearn `PCA.fit_transform` with an in-memory SVD used only for figure display (never written under `models/` or `decoder_comparison/`).

7. **README alignment**
   - Output layout now lists `models/`, `deployment_decoder_selection/`, `latency_profiling/`, and required simulation files.
   - Developer utilities section no longer claims a non-existent `run_full_decoder_workflow.py` script or a public `run_realtime_decoding.py`.

8. **Smoke test** (`scripts/smoke_test_public_workflow.sh`)
   - Runs short simulate → `run_decoder.py --profile quick` → `run_visualizations.py --all --compile-pdf`.
   - Asserts contracts and verifies visualization does not mutate registry/metrics mtimes.

## 3. Remaining TODOs (non-blocking)

| Item | Notes |
|------|-------|
| Closed-loop replay uses one target’s `W` for all heads | Registry stores per-target `W`; replay loads the `closed_loop_target` window for the shared feature front-end. Multi-head per-target `W` at replay time is a future enhancement. |
| Display SVD for counts latents | Geometry figures still rebuild spike matrices + frozen transforms for plotting (no decoder retrain). Could later cache embedding coords during comparison. |
| `run_BCI.py` staging | Remains a developer utility; not part of the happy path. |

## 4. Sorted-only deployable selection?

**Yes.** `models/best_realtime_decoders.json` and `deployment_decoder_selection/` are built from sorted rows only. Ground-truth comparison dirs receive `ORACLE_NON_DEPLOYABLE.json`. Realtime replay rejects `spike_source != "sorted"`.

## 5. Target-specific windows?

**Yes in the registry.** Each target stores `selected_causal_window_s`. Uniform-window warning points users to `deployment_decoder_selection/all_sorted_window_scores.csv`. Closed-loop replay currently applies the selected closed-loop target’s `W` to the shared update loop (documented TODO above).

## 6. Classic Isomap prevented from deployment?

**Yes.** Offline Isomap feature modes are remapped away from the deployable registry, including model artifact paths under `best_accuracy`.

## 7. Distilled Isomap latency-gated?

**Yes.** Distilled transforms that fail `realtime_compatible` (latency / held-out distortion gates from distillation) are remapped to counts. Latency benchmark reports teacher vs distilled costs and RT flags.

## 8. Visualization read-only?

**Yes for the public contract.** `run_visualizations.py` does not call comparison, realtime replay, or decoder fitting. Smoke test confirms registry/metrics files are not rewritten. Display-only SVD may run in memory for figure axes.

## 9. Public smoke test?

**Passes.**

```bash
bash scripts/smoke_test_public_workflow.sh
```

Observed on 2026-07-26:

- simulation contract OK  
- decode/deployment contract OK (sorted-only, no classic Isomap)  
- RF classifier over-budget warnings emitted into registry  
- visualization PDF written; no model/comparison mutation  
- `smoke_test_public_workflow PASSED`

## How to re-verify

```bash
python -m pytest tests/test_deployment_selection.py tests/test_output_contracts.py tests/test_causal_windows.py -q
bash scripts/smoke_test_public_workflow.sh
```
