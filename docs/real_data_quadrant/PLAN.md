# Real-data quadrant extension — design plan

Status: design locked; seam and M1 not started. Branch: `offline-decoding`
(rebased onto `master` after merging `quadrant-n5`).

Companion to the simulated `quadrant_n5` experiment (`agents/quadrant_n5/`,
`realtime/quadrant_n5*.py`, `configs/quadrant_n5.yaml`).

**Repo is public.** No real-data paths, session names, animal IDs, cell counts,
figures, or derived results belong in git. Dataset-specific survey numbers live
only in gitignored `docs/real_data_quadrant/SURVEY.local.md`. Runtime outputs
go under an ignored directory (see §3.5).

**Code placement:** new analysis under `analysis/real_quadrant/`. Approved
exception: a small seam on `analyze_source` so sim and real share one pipeline
(see §3.2). Do not modify `hippo/`, `ui/`, or `agents/quadrant_n5/`. Real data
root = env var `HIPPO_DATA_ROOT` only.

---

## 0. Setup done on this branch

1. Branch `offline-decoding` created; `pytest.ini` has `pythonpath = .`.
2. `quadrant-n5` merged into `master`; `offline-decoding` rebased onto that tip.
3. Dataset survey written to gitignored `SURVEY.local.md` (not committed).

---

## 1. Map of existing `quadrant_n5` (simulated)

### 1.1 Pipeline stages and input contracts

| Stage | What it does | Input contract |
| ----- | ------------ | -------------- |
| **Simulate** (`generate_seed_dataset`) | Hippocampal sim + Neuropixels degradation + simulated sorting | `session_s=600`, square arena `100 cm`, `behavior_dt=0.050 s`, lab probe track YAML; seed streams for trajectory / neural / recording_noise / sorting_errors / methods |
| **Load** (`load_simulation_data`) | Spikes + behavior + units for one spike source | Sim dir with `behavior.csv`, `spikes_sorted.csv` / ground-truth spikes, `units`; region allow-list from config |
| **Decode times** | Causal grid | `W=0.250 s`, `update_dt=0.050 s`, half-open `[t−W, t)`, label at **right edge** |
| **Spike matrix** (`build_causal_spike_matrix`) | Integer counts per unit × time | Shape `(n_times, n_units)`; times in seconds |
| **Align behavior** | Position at decode times | `y` shape `(n_times, 2)` in **cm** (`x`,`y`) |
| **Split** | Contiguous train/test + purge | Train first 80%, test last 20%, `gap_s=1.0` (= `max(W+max_history, 1.0)` stepped); identical index set for every method |
| **Features** | `sqrt(counts)` then z-score | Scaler **fit on train only**; same `X` fed to every representation (audit A2) |
| **Representations** | Fit on train (or per inner-CV fold); transform full session | Latent / feature matrix `Z` `(n_times, d)` or raw dim; LDS filters from session start |
| **Decoders** | Ridge (primary), kNN (sensitivity) | Target `position_xy_cm`; alphas / k from inner CV; decoder scalers train-only |
| **Metrics + A13** | Error, floor, circular time-shift null | Eval = intersection of valid test indices; chance = train-mean position |
| **Replay / audit / report** | Causal step path, A1–A14, PDFs + figure set | Results under `outputs/quadrant_n5/` (gitignored) |

**Binning summary (frozen for real data too):** window **250 ms**, step **50 ms**,
labels at window right edge, spatial units **cm**. Window sweep is a later
supplement, not M1–M4.

### 1.2 Methods and outputs

| Key | Role | Fit / transform | Output |
| --- | ---- | --------------- | ------ |
| `raw` | Control | Identity on z-scored counts | `Z = X` (n_units dims) |
| `raw_lag` | History control | Stack current + previous 5 steps (extra 250 ms) | Wider feature vec; loses first 5 samples |
| `pca` | Linear–static quadrant | Train PCA; nested fit at d=20, slice | `(n, d)`, d∈{2,3,5,10,20} |
| `dm` | Nonlinear–static | Diffusion maps + Nyström OOS; landmarks if n_fit>12k | `(n, d)` |
| `lds` | Linear–dynamic | EM on train; **Kalman filter only** (no RTS on eval); continuous from session start | `(n, d)`; refit per d |
| `isomap` | Baseline | Global Isomap (+ optional pre-PCA); OOS transform | `(n, d)` |
| `gpfa` | Baseline (expect offline) | GPFA; smoother-style inference | `(n, d)`; refit per d; A9 typically offline-only |
| nonlinear–dynamic | Empty cell | Not implemented | Drawn empty in figures |

Primary `d` per method: lowest inner-CV median Ridge error; ties → smaller d.
Each method also produces Ridge/kNN test errors, A13 null stats, optional
d-sweep rows, and (when complete) predictions for trajectory panels.

### 1.3 What “quadrant” means here

Not spatial arena quadrants. The **representation quadrant** is the 2×2:

|            | Static | Dynamic        |
| ---------- | ------ | -------------- |
| Linear     | PCA    | LDS (filter)   |
| Nonlinear  | DM     | *(empty)*      |

**Arena geometry (sim):** fixed square, `arena_size_cm=100`, origin-aligned
occupancy bins for coverage / spatial error maps / centre-pull (distance from
arena centre). Position labels are continuous `(x,y)` cm, not discrete
quadrant classes.

### 1.4 Metrics

Per (seed, spike source, method, d, decoder):

- Euclidean error: mean, median, p90 (+ full distribution where saved)
- R² per axis; centre-pull slope; spatial error maps
- Ridge − kNN gap; inner-CV vs test; learning curves (sorted, Ridge)
- Floor = predict training-mean position; A13 = circular label shifts vs floor
- Replay: per-step transform/decoder latency vs 50 ms budget; A9/A11

Across seeds (N=5): per-seed values, mean, SD, **paired contrasts** with sign
counts (no p-values): `dm−pca`, `lds−pca`, `raw_lag−raw`, `lds−raw_lag`,
each method − `raw`, kNN−Ridge, `sorted−gt`.

### 1.5 Seeds and aggregation

- `n_seeds=5`, `master_seed` fixed; `SeedSequence(master).spawn(5)[i]` then
  component streams: trajectory, neural, recording_noise, sorting_errors, methods
- Same five data seeds for every method (paired design)
- Aggregate PDF + figure set: mean/SD and sign counts; story page interprets
  direction of effects (must be re-read if signs flip)

### 1.6 Report / figure inventory (`quadrant_n5`)

**Phase-5 PDFs** (`realtime/quadrant_n5_report.py`): per-seed provenance, audit
table, population composition + occupancy, primary-d errors (sorted + GT),
A9/A11 replay; aggregate quadrant medians + planned contrasts + A9/A11.

**Publication figure set** (`agents/quadrant_n5/figures/`):

| Figure | Content |
| ------ | ------- |
| Story | Narrative answer at N=5 (numbers from tidy CSVs only) |
| Fig1 design | Quadrant diagram, pipeline, population, trajectories |
| Fig2 validity | Audits, A13 null, CV vs test, offline vs replay |
| Fig3 quadrant answer | Per-seed errors + planned contrasts (Ridge / kNN) |
| Fig4 mechanism | kNN vs Ridge; sorted vs GT; LDS beyond history |
| Fig5 deployability | Latency, LDS vs GPFA, selected d, learning curves, coverage |
| Fig6 answer | Summary quadrant cells + contrast arrows |
| Fig7 trajectories | Example decoded paths (PCA/DM/LDS) |
| FigS1 | Latent-d sweeps |
| FigS2 | Full A13 strip plots |
| FigS3 | All-seed trajectory grids |
| FigS4 | Spatial error, centre-pull, CDFs, jump rate |
| FigS5 | kNN pressure / Phase-8 criteria |

### 1.7 Places that assume simulated data

| Assumption | Where |
| ---------- | ----- |
| Synthetic-only guardrail | `SPEC.md` §1 |
| Simulator generation + seed streams | `generate_seed_dataset`, `derive_seed_streams`, `sim_*_hash` |
| Spike sources `sorted` vs `ground_truth` | Config `spike_sources`; dual analysis; Fig4/S sorted vs GT |
| Known latents / cell-type composition from sim | Population panels; Phase-8 cell-type subsets |
| Fixed 100 cm square arena | Occupancy, centre-pull, Fig1 trajectories |
| Probe-track region allow-list (sim anatomy labels) | `unit_inclusion.regions` |
| Session length 600 s | Split timeline art; A13 shift fracs × `session_s` |
| Deployability = sorted only; GT non-deployable | A7, legends, story |
| Seed isolation / methods-stream audits (A10) | Sim-specific |
| UI frozen-config audit (A12) | Streamlit path |
| Realtime replay from sim spike CSVs | `quadrant_n5_replay.py` |
| LDS / `raw_lag` history from **session** start | Real data resets at **segment** start (§3.0b) |

---

## 2. Real-data layout (generic only)

Per-session directory under `HIPPO_DATA_ROOT/<session>/` (path from env only):

| File | Role |
| ---- | ---- |
| `dataset.csv` | Binned spike counts (`Cell_*`) plus aligned behavior columns |
| `postions_dataset.csv` | Tracking (`timestamp`, `X`, `Y`, `valid`, `room`, …) |
| `config.yaml` | Room time ranges (`map_rooms`), boundary polygons, project meta |
| `clusters_dataset.csv` | Per-cell metadata (`Region`, `BClabel`, …); load with `usecols` only |

`Cell_*` is a centre-labeled 250 ms sliding count on the 50 ms grid (see §3.0a);
rebuild causal windows for decoding. Room labels are time segments from
`map_rooms` (e.g. `A`, `B`, `a`, `b` as present). Arena geometry comes from
`preprocessing.boundary`, not a fixed square. Dataset-specific counts and
ranges: see `SURVEY.local.md`.

---

## 3. Design

### 3.0 Locked decisions (was §3.7)

1. **Shared pipeline:** `quadrant-n5` is on `master`. Next implementation commit
   (after this plan): add a seam to `analyze_source` that accepts a prepared
   data bundle (`X_counts`, `y`, `decode_times`, unit meta, arena) instead of
   only a sim dir. Sim path must stay behavior-unchanged (existing sim tests
   byte-identical). Approved exception to the earlier no-modify-`realtime/` rule.
2. **`HIPPO_DATA_ROOT`:** always from the environment; never hardcode.
3. **No room transfer.** Each room segment (`A`, `B`, `a`, `b`, … as present) is
   its own independent analysis: blocked CV with purge **inside that segment
   only**, room-local coordinates, fits never cross segment boundaries. `A` and
   `a` are separate segments even when they share a physical box. **Primary
   result: room `A`** (present in every session, first exposure). `B` and `a`
   are replications in the same tables.
4. **M3 cohort:** 2-room sessions only. 3-room and morph held out.
5. **Units:** all regions; exclude `BClabel == NON-SOMA`. Region subsets are
   secondary analyses.
6. **Coordinates:** room-local, origin at the room centre from the boundary
   polygon.
7. **`valid==False`:** masks **targets only**; report dropped fraction. Spike
   counts in those bins stay in `X` and in causal windows (§3.0c).
8. **M4:** offline report only; no realtime replay arm.
9. **`W=250 ms` locked** (5 × 50 ms bins). Window sweep is a later supplement.

### 3.0a Bin-edge check (before M1) — conclusion

Checked on 3 sessions × 5 moderate-firing units by re-binning
`clusters_dataset` spike times onto the `dataset.csv` timestamp grid.

**Findings (aggregates only):**

- Spike times (`clusters_dataset.timestamp` comma-lists) are in **seconds**,
  on the **same clock** as `dataset.csv` timestamps (**constant offset = 0**).
- `Cell_*` is **not** a raw 50 ms bin count. Each spike appears in **five**
  consecutive rows (`sum(Cell_*) / n_spikes ≈ 5`), i.e. a **250 ms sliding
  window** stepped at 50 ms.
- That window is **centre-labeled**: `Cell_*(t)` = spike count in
  `[t − 0.125, t + 0.125)`. Exact bin match fraction **1.0** on every
  checked session under this rule; start- and end-aligned 250 ms windows
  do not match. Re-binning as if `Cell_*` were 50 ms start/centre/end
  counts does not match (and cannot, given the ×5 multiplicity).

**Causal rule for the adapter (matches sim contract):** for a label at time
`t`, use spike counts in the half-open window **`[t − 0.250, t)`** so the
window contains **only spikes strictly before `t`**. Do **not** feed
`Cell_*(t)` in as features — it includes **125 ms of future** spikes. Rebuild
causal 250 ms counts from per-unit spike times (or an equivalent causal
construction on the 50 ms grid). Position/valid/room still align to the
dataset timestamp `t` (label = right edge of the causal window).

### 3.0b Segment isolation

All evaluation is within a **single room segment**. No split, fit, or
normalization spans two segments. Exclusion rules (min units, min valid
fraction) apply **per segment**.

- Exclude the **first 60 s** of each room segment from all fitting and
  evaluation (settling after the room change), and the **last 10 s** (removal
  from the room). Record these trim values in the config hash.
- LDS filtering and `raw_lag` history still start at **segment start** (not
  session start), so the excluded minute is warm-up and the first evaluated
  point has a settled filter state.
- Compute the valid-fraction exclusion rule (§3.0f) on the **retained** part
  only.

### 3.0c Invalid frames

`valid==False` masks targets (`y`) only. Spike counts in those bins remain in
`X` and contribute to any window that covers them. Report the fraction of
eval labels dropped for invalid targets.

### 3.0d Aggregation

Per room segment, contrasts across animals with sessions nested within animals.
Report room `A` as primary; `B` and `a` in the same tables as replications.

### 3.0e Sim-vs-real metric

Sim-vs-real comparisons use **normalized error**
`(error / chance-floor error)`, not raw cm.

### 3.0f Pre-registered exclusions

Fixed before any results:

- Minimum **30 units** after the NON-SOMA filter.
- Minimum **0.7** valid-target fraction within the **retained** part of the
  room segment (after the 60 s / 10 s trims in §3.0b).

Log excluded sessions with reasons in the **local manifest only** (ignored
outputs; never commit).

### 3.1 Mapping table: `quadrant_n5` → real data

| Concept | Real-data equivalent | Notes |
| ------- | -------------------- | ----- |
| Independent seeds (N=5) | Sessions nested in animals; per room segment | Paired method contrasts within segment; pool across animals |
| Simulator | **Not applicable** | Adapter → prepared bundle for `analyze_source` seam |
| `sorted` spikes | Real sorted binned counts (`Cell_*`) | Only spike source |
| `ground_truth` spikes | **Not applicable** | Drop sorted-vs-GT panels |
| Known sim latents / cell types | `clusters_dataset.Region` (+ `BClabel`) | Region subsets secondary; no whole-session place-cell selection |
| Region allow-list (sim anatomy) | All regions; drop NON-SOMA | Record rule in config hash |
| Fixed 100 cm arena | Per-room boundary; **room-local** coords | Centre from boundary polygon |
| `behavior_dt` / bin | Native **50 ms** timestamp grid | `Cell_*` is centre 250 ms (see §3.0a); rebuild causal `[t−W,t)` |
| Seed streams / A10 | Session + segment identity; analysis RNG for methods | No trajectory/neural/noise streams |
| Train/test + purge | Time-blocked **inside one segment** | Never cross segment boundaries |
| Inner CV + purge | Keep, within segment | All fits train-fold only |
| Floor / A13 time-shift | Keep, within segment | Chance baseline for every metric; sim-vs-real uses error/floor |
| LDS / lag history from session start | From **segment** start | 60 s / 10 s edge trims for fit/eval; warm-up uses full segment prefix |
| Realtime replay arm | **Not in scope** for M1–M4 | |
| UI config audit (A12) | **Not applicable** | |
| Outputs | Ignored real-data output root (§3.5) | Never commit |

### 3.2 Architecture

```text
HIPPO_DATA_ROOT/<session>/
        │
        ▼
analysis/real_quadrant/adapter.py
        │  one room segment → prepared bundle:
        │  X_counts, y (room-local cm), decode_times,
        │  unit meta, arena, valid-target mask, dropped fraction
        ▼
realtime/quadrant_n5_run.analyze_source  (seam: accept bundle OR sim dir)
        │  same splits, CV, methods, metrics as sim
        ▼
analysis/real_quadrant/outputs/…         # gitignored
```

**Seam (next commit after this plan, with tests — not in this commit):**

- `analyze_source` accepts either the existing sim-dir path or a prepared
  bundle with `X_counts`, `y`, `decode_times`, unit meta, and arena.
- Loading a sim dir and building the bundle internally must leave sim results
  **byte-identical** (existing `tests/test_quadrant_n5*.py` / golden checks).
- One pipeline for sim and real keeps the comparison valid.

**Package layout**

- `analysis/real_quadrant/` — adapter (per segment), CLI/runner wrappers,
  report builder, synthetic unit tests (no real data in CI).
- Import transformers / LDS / GPFA / purge helpers from `realtime/`.
- Do not import `agents/quadrant_n5` into core; report pages may live under
  `analysis/real_quadrant/report/` or call figure code with tidy tables only.

**Adapter duties (per segment)**

1. Build causal `X_counts` for `W=250 ms` as `[t−0.250, t)` per §3.0a (not
   raw `Cell_*(t)`).
2. Slice to the segment range from `map_rooms`; keep neural history from segment
   start; mark fit/eval indices after excluding the first 60 s and last 10 s
   (trim values in config hash).
3. `y` in room-local cm; mask targets where `valid==False` (counts stay in `X`).
4. Units: all regions except NON-SOMA; attach `Region` via clusters `usecols`.
5. Arena centre/extents from boundary polygon for that room.
6. Enforce exclusion thresholds (§3.0f) on the retained part, or skip with a
   manifest reason.

### 3.3 Evaluation and leakage rules

1. **Time-blocked CV** with purge ≥ `W + max_history` (and ≥ 1 s), outer and
   inner, **entirely inside one room segment**.
2. **No cross-segment** split, fit, or normalization. `A` and `a` never share a
   model.
3. **All fitting on training folds only:** scalers, PCA/DM/Isomap/LDS/GPFA,
   Ridge α, kNN k, landmarks, bandwidths.
4. **No cell selection using whole-session place-cell flags.** Primary filter:
   exclude NON-SOMA only; region subsets secondary and a priori.
5. **Chance baseline for every metric** (train-mean position floor; keep
   circular time-shift null). Sim-vs-real uses error / floor (§3.0e).
6. Intersection eval set across methods within the segment.
7. Pre-registered exclusions (§3.0f) fixed before results; exclusions only in
   the local manifest.

### 3.4 Compute plan

Real segments are longer and often higher-dimensional than the 600 s sim; LDS
and especially GPFA dominate wall time (see local survey for scale notes).

| Method | Proposal |
| ------ | -------- |
| `raw`, `raw_lag`, `pca` | All eligible 2-room segments early |
| `dm`, `isomap` | After M2 on the pilot segment; then cohort |
| `lds` | Subset first (one segment, then a few animals), then cohort; consider trimmed d grid on pilot |
| `gpfa` | Subset only until timed on one real segment; optional for full M3 |

Pilot rule: time one full method grid on one segment; if wall time explodes,
trim GPFA/LDS d-sweep or keep heavy methods on a designated subset.

### 3.5 Report plan

Mirror the sim figure set for side-by-side reading. Write only under an
**ignored** dir (`analysis/real_quadrant/outputs/` or `outputs/real_quadrant/`).
**Never commit** PDFs, NPZs, or tidy CSVs from real data.

| Sim section | Real-data report |
| ----------- | ---------------- |
| Fig1 design | Same class; real pipeline (no GT branch); population by `Region`; room-`A` trajectory example at render time only |
| Fig2 validity | Audits that apply within a segment; drop A7 GT, A10 sim seeds, A12 UI, A9/A11 replay |
| Fig3 quadrant answer | Same method contrasts; **room A primary**, B and a as replication columns/facets |
| Fig4 mechanism | Drop sorted-vs-GT; keep kNN vs Ridge and LDS vs `raw_lag`; **sim-vs-real** via normalized error (error/floor) from published sim aggregates |
| Fig5 deployability | Drop latency/replay; keep d-selection + learning curves; coverage in room-local arena |
| Fig6 answer | Same summary cells; N = animals/sessions language |
| Fig7 / S3 trajectories | Same on real segments (outputs only) |
| FigS1 d-sweep | Same |
| FigS2 A13 | Same (within segment) |
| FigS4 failure modes | Same with per-room centre |
| FigS5 Phase-8 cell-type | **Replace** with a priori **region subsets** |
| sorted vs GT | **Dropped** |
| Replay / seed-isolation story | **Dropped** |

**Real-data-only sections**

1. Per-animal variability (sessions nested within animals), by room segment.
2. Region-subset decoding (secondary).
3. Valid-target drop fraction and exclusion manifest summary (counts/reasons
   only in local outputs).

### 3.6 Milestones

| ID | Deliverable | Done when |
| -- | ----------- | --------- |
| **Bin-edge** | Done (§3.0a): `Cell_*(t)` = centre 250 ms; causal rebuild `[t−0.250,t)` | Conclusion in PLAN; adapter must not use `Cell_*(t)` as features |
| **Seam** | `analyze_source` accepts prepared bundle; sim tests byte-identical | Separate commit with tests (after this plan commit) |
| **M1** | Adapter + `raw` on one 2-room **room-A** segment; within-segment split+purge; chance floor; target-only valid mask | Synthetic unit tests; local metrics JSON; error vs floor; `HIPPO_DATA_ROOT` only |
| **M2** | All methods on that segment (GPFA optional if too slow) | Per-method JSON + d-selection; time-shift null; timings |
| **M3** | Eligible 2-room cohort; room A primary, B/a replications; heavy methods per §3.4 | Local manifest (incl. exclusions); per-animal aggregates; nothing real in git |
| **M4** | Offline side-by-side report | Fig1–6 parallels + real-only sections; ignored outputs only |

---

## 4. Stop

Plan and branch integration committed. Next after review: bin-edge check,
then the `analyze_source` seam commit, then M1. No seam or adapter in this
commit.
