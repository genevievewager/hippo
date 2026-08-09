# Realtime deployment selection and closed-loop replay

The project does **not** simply choose the highest offline score. Exact gates depend on which selection artifact you mean.

## Public registry (`models/best_realtime_decoders.json`)

Used by `run_decoder.py` → `realtime/workflow.py` → `write_deployment_selection_artifacts` and subsequent closed-loop replay.

```text
candidate F × E × D × W configurations (sorted spikes)
        ↓
held-out sorted-spike performance
        ↓
shortest-near-optimal window selection
        ↓
prefer realtime-compatible E (remap offline Isomap; GPFA not closed-loop)
        ↓
best_realtime_decoders.json
        ↓
causal realtime replay + closed-loop policy C
```

Mandatory for this path:

1. Sorted spikes only (`deployment_spike_source: sorted`)
2. Held-out metric selection with `shortest_near_optimal` (default) or `best_accuracy`
3. Prefer `realtime_compatible` embeddings when recommending closed-loop models (classic `global_isomap` remapped; `gpfa` offline-only)

Latency microbenchmarks and a session latency profile are recorded, but the public registry is **not** required to pass a total-update ≤ 50 ms hard gate before export.

## Parallel lab-deployable selection

`select_best_lab_deployable_decoders` (called at the end of a comparison) writes `best_lab_deployable_decoders.csv` and per-target profiles. When columns are available, candidates must pass:

1. Sorted spikes
2. `passes_realtime_gate` (feature + embedding + predict microbench vs `max_compute_ms`, and `W` ≤ `max_effective_history_s`)
3. Categorical calibration (`is_well_calibrated`)
4. Sorting-robustness label ≠ `large_loss` (informative mainly when GT is compared)
5. Beats negative controls when control rows exist
6. Cross-run stats when a multi-run summary is merged

This path is **implemented** but is not the same object as `models/best_realtime_decoders.json`.

## Additional validation (optional / conditional)

| Analysis | When | Role |
|----------|------|------|
| Closed-loop trigger score table `C` | Default comparison (`enable_trigger_search`) | Downstream policy evaluation on decoded predictions |
| Sorted-vs-GT information loss | GT diagnostics compared | Degradation robustness summary |
| Negative controls | `--include-controls` (comparison CLI) | Sanity |
| Population / region / layer ablation | Opt-in flags | Interpretability |
| Cross-run generalization | `run_decoder_comparison.py --inputs …` | Multi-run selection stats |
| Full causal latency profile | Public workflow after replay | Engineering budget reporting |

## Closed-loop rule `C`

`C` is **not** searched as a fifth axis inside the same Cartesian decoder grid as `F × E × D × W`. Default trigger rules are scored after decoding (`closed_loop_trigger_comparison.csv`) and applied again during registry replay (`--closed-loop-target`, default `position`).

## Selection policy: `shortest_near_optimal`

Default for the public registry. Per target, choose the shortest causal window `W` whose primary metric is within **5%** of the best:

- lower-is-better: metric ≤ 1.05 × best
- higher-is-better: metric ≥ 0.95 × best

Closed-loop replay loads saved artifacts (transforms + `.joblib`) and does **not** retrain when comparison artifacts exist.

## Causal window grids (public profiles)

- Public default update rate: **20 Hz / 50 ms** (`update_dt=0.050`). Overridable with `--update-dt` (e.g. 0.025 supported).
- `--profile manifolds` / `quick`: `W ∈ [0.050, 0.250, 0.500, 1.000]`
- `--profile standard`: `W ∈ [0.050, 0.100, 0.250, 0.500, 1.000]`
- `--profile full` / `feature_robustness`: can include 25 ms

## Comparison-only developer grids

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/decoder_comparison \
    --compare-sources \
    --decode-windows 0.025 0.050 0.100 0.250 0.500 1.000

python run_decoder_comparison.py \
    --inputs outputs/run_001 outputs/run_002 outputs/run_003 \
    --output outputs/decoder_comparison_multi_run \
    --spike-source sorted \
    --feature-types counts rates sqrt_counts \
    --embedding-types identity global_pca region_pca \
    --decode-windows 0.050 0.100 0.250 0.500 \
    --max-models quick \
    --include-controls \
    --population-ablation
```
