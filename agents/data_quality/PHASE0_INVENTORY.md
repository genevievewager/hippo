# Phase 0 inventory — STOP for user review

Date: 2026-10-05. Worktree: `/home/gmwager/projects/hippo-dq` @ `24cc602`
(`data-quality`). Live checkout untouched (porcelain empty; HEAD unchanged).

## A. Live-run fingerprint

See `LIVE_RUN.md`. Summary: tmux `m2`, PID **1177494**,
`python -m analysis.real_quadrant.run_m2`, cwd live checkout, venv `.hippo`,
`HIPPO_DATA_ROOT=/data/ecephys/Shai_Daniel_12032026`, peak RSS ~2.8 GiB,
191 threads, outputs under `outputs/real_quadrant/{m2,cache,manifest,m2_run.log}`.

## B. Worktree

```text
git worktree add /home/gmwager/projects/hippo-dq -b data-quality 24cc602
```

Import check (must run with cwd = worktree):

```bash
cd /home/gmwager/projects/hippo-dq
PYTHONPATH=/home/gmwager/projects/hippo-dq \
  /home/gmwager/projects/hippo/.hippo/bin/python -c \
  'import analysis.real_quadrant.adapter as a; print(a.__file__)'
# → .../hippo-dq/analysis/real_quadrant/adapter.py
```

**Verified IMPORT_OK.** With worktree imports, adapter defaults become:

- cache → `/home/gmwager/projects/hippo-dq/outputs/real_quadrant/cache`
- manifest → `/home/gmwager/projects/hippo-dq/outputs/real_quadrant/manifest.jsonl`

Those are **not** under `outputs/data_quality/`. Phase 1+ must wrap
`build_segment_bundle` / `_append_manifest` (new code only) to force
`outputs/data_quality/cache` and never touch the live tree. See deferred note.

**Cursor workspace warning:** this agent session still has the *live* checkout
open. Format-on-save / relative writes are dangerous. Re-open Cursor on
`/home/gmwager/projects/hippo-dq` before Phase 1.

## C. Lab inputs (via existing adapter contracts; no adapter edits)

Sources: PLAN §3.0 / §3.0a, `SURVEY.local.md`, direct aggregate reads of
2-room sessions (n=41). Spike parsing via imported `_parse_spike_times` only
(no `build_segment_bundle`, so no cache/manifest writes).

| Item | Finding |
| ---- | ------- |
| Spike-time units | **Seconds.** Median of per-session median ISI ≈ **0.112 s**. If misread as 30 kHz samples → ≈ 3.7 µs (absurd). Seconds guard: 0 sessions with median ISI < 1e-4; max fraction of sampled units with rate > 500 Hz = **0**. |
| Unit region labels | `MEC`, `SUB`, `NC`, `TEa`, `VI`, `PARA` (totals over 2rooms). **No `LEC` rows** in the 2-room aggregate (SURVEY listed LEC historically — flag). |
| Native position rate | **20 Hz** (median Δt = 0.05 s) on `postions_dataset.csv`; same on `dataset.csv` grid. |
| Room segments | From `config.yaml` `map_rooms`; primary room `A`. Retained length after 60 s / 10 s trim: med ≈ **1120 s** (min ~1073, max ~1713) → floor(/300) ≈ **3** chunks. |
| Valid mask | `valid` on positions; PLAN: masks targets only. |
| Exclusions | ≥30 units after NON-SOMA; ≥0.7 valid fraction on retained. Median units/session ≈ **231** (46–860). |
| Phy / Kilosort | **None** under `HIPPO_DATA_ROOT` (depth≤5): no `cluster_info.tsv`, `amplitudes.npy`, waveforms, `params.py`. Sorting-realism Phy metrics are lab-N/A; use refractory / sim degradation equivalents. |
| Adapter caches | **Yes.** `DEFAULT_CACHE_ROOT = <repo>/outputs/real_quadrant/cache` writes `.npz`. Manifest appends to `outputs/real_quadrant/manifest.jsonl`. `_append_manifest` is **not** path-parameterized on exclusion call sites. Must wrap / redirect in new IO layer. Does **not** write beside lab data. |

### Region-matched unit counts (ambiguous — needs decision)

Sim `unit_inclusion.regions` (7 probe bands): entorhinal_transition,
hippocampal_formation_transition, entorhinal_cortex_layer{6a,5},
medial_entorhinal_layer{3,2,1}. Lab labels are coarse (`MEC`, …).

| Lab region set | median units | min | n sessions &lt; 10 | n &lt; 30 |
| -------------- | -----------: | --: | -----------------: | -------: |
| MEC only | 184 | 0 | **7** | 11 |
| MEC+LEC | 184 | 0 | **7** | 11 |
| MEC+LEC+PARA | 184 | 0 | **6** | 10 |
| MEC+LEC+PARA+SUB | 217 | 32 | **0** | 0 |

Spec §3 / §10: if region-matched sessions have &lt; 10 units → **stop and ask**.
MEC-only fails that for 7 sessions. Including SUB clears the floor.

## D. Sim inputs

| Item | Finding |
| ---- | ------- |
| `OUTPUT_ROOT` | Hardcoded `outputs/quadrant_n5` in `realtime/quadrant_n5_run.py`. `generate_seed_dataset(..., output_dir=)` accepts override. |
| **Chosen approach** | **Read Report 1 existing sim outputs read-only** at `/home/gmwager/projects/hippo/outputs/quadrant_n5/seed_{0..4}/sim/`. Avoids regenerating under memory pressure; byte-identical to live commit’s generators. Copy or stream into analysis; never write back into that tree. |
| Pre-degradation GT | **Yes.** `spikes_ground_truth.csv` + `spikes_sorted.csv` + `units.csv` + `behavior.csv` + `rates.npy`. Three-way GT → degraded/sorted → lab is feasible. |
| Sim ISI (agg) | sorted median ISI ≈ 0.144 s; GT ≈ 0.077 s; frac &gt;500 Hz = 0. Behavior Δt = 0.05 s. |
| Sim included regions (seed 0) | 7 bands; n_units ≈ 86; cell types MEC_grid/hd/speed, Sub_bvc, INT_CA1. |

## E. Reuse (Report 3 §9 groundwork)

**Report 3 hparam-sweep SPEC is not in this repo** (no `agents/hparam*`, no
§9 document under `docs/` or `agents/`). Treated as **all metrics need writing**
in `agents/data_quality/metrics/`, importing only low-level primitives where they
already exist:

| Need | Status |
| ---- | ------ |
| Fano, ISI CV/CV2, burst, refractory, drift | **Write** (no dedicated functions found) |
| Place fields, spatial info, sparsity, coverage, split-half | **Write** (viz helpers only, not metrics API) |
| Theta modulation from spike ACG | **Write** (`hippo_sim.features.theta_modulation` is rate×phase, not spike ACG) |
| Noise correlations, synchrony, participation ratio, PCA spectrum | **Write** |
| LDS eigenvalues | Can **import** `realtime.dynamic_latents.lds` fit path behind a wrapper; no ready “eigenvalue metric” helper |
| Behaviour (speed, occupancy entropy, wall distance, turning, dwell) | Partial columns exist on sim `behavior.csv`; lab must derive from positions — **write** shared functions |
| Decoding error/floor vs N, W | **Reuse** `analyze_source` / prepared-bundle seam + frozen configs via wrapper; do not edit `quadrant_n5_run.py` |
| Sorting robustness labels | `realtime/sorting_robustness.py` is decoder-drop summary, not Phy metrics — limited reuse |
| sklearn `HistGradientBoosting` | **Installed** (sklearn 1.9.1) — no pip needed |

## F. Resources and proposed caps

| Resource | Value |
| -------- | ----- |
| `nproc` | 128 |
| MemTotal | ~755 GiB |
| MemAvailable (Phase 0) | ~650–683 GiB |
| Live peak RSS | ~2.8 GiB |
| Free-RAM floor (spec) | max(0.25×755, 2×2.8) ≈ **189 GiB** available |
| Headroom now | large (~400+ GiB free) |

**Proposed caps for Report 4 jobs:**

| Knob | Cap |
| ---- | --- |
| `nice` | 19 |
| `ionice` | `-c3` if available |
| `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `NUMEXPR_NUM_THREADS` | **4** |
| joblib / metrics `n_jobs` | **2** |
| Watchdog abort if MemAvailable &lt; **189 GiB** | yes |
| Soft RSS budget for our job | **32 GiB** (well under floor) |
| Pilot wall-time ask threshold | &gt; 6 h projected → stop and ask |

## Ambiguous / ask before Phase 1

1. **Region matching map** lab `MEC`/`PARA`/`SUB`/… ↔ sim’s 7 depth bands.
   MEC-only leaves 7 sessions with &lt;10 units. Propose primary =
   `{MEC, PARA, SUB}` (and LEC if present), secondary = all non-NON-SOMA.
2. **Report 3 §9 source text** missing from repo — confirm metric list is
   exactly the Cursor brief §4, or provide Report 3 SPEC path.
3. **Sim input mode** — confirm read-only reuse of
   `hippo/outputs/quadrant_n5/` (recommended) vs regenerate into worktree.
4. **Cursor workspace** — please re-open `/home/gmwager/projects/hippo-dq`
   before authorizing Phase 1 edits.
5. Adapter manifest always appends to `<repo>/outputs/real_quadrant/manifest.jsonl`
   even under worktree imports — wrapper required (new file); not an edit to
   `adapter.py`.

## Safety check at Phase 0 end

- Live PID 1177494 still running.
- Live HEAD `24cc602`, dirty count 0.
- No Report 4 writes under live tree (only worktree `agents/data_quality/` +
  `outputs/data_quality/_phase0/`).
- Shared env not modified.
