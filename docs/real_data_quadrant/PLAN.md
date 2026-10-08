# Real-data quadrant extension — design plan

Status: **Report 2 COMPLETE** (2026-10-08). Seam, M1–M4 done. Branch:
`offline-decoding`; tag `report2-v1`. Sim-vs-real comparison lives in the
**Report 1 revision** (default + matched sim), not M4.

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
| `lds_smooth` | Smoothing control on LDS | Causal EMA on LDS Ridge predictions; τ from train blocked CV | Same as LDS + EMA |
| `isomap` | Baseline | Global Isomap (+ optional pre-PCA); OOS transform | `(n, d)` |
| `gpfa` | Offline reference (non-causal smoother) | GPFA; RTS smoother-style inference | `(n, d)`; not a causal method |
| `gpfa_causal` | Linear–dynamic (verified causal) | Same GPFA params (train-only); **Kalman filter only**, forward from segment start | `(n, d)`; leakage-tested |
| `raw_smooth` / `pca_smooth` / `dm_smooth` | Smoothing controls | Causal EMA on Ridge predictions; τ∈{0.1,0.25,0.5,1,2}s by train CV | No representation refit |
| nonlinear–dynamic | Empty cell | Not implemented | Drawn empty in figures |

Primary `d` per method: lowest inner-CV median Ridge error; ties → smaller d.
Each method also produces Ridge/kNN test errors, A13 null stats, optional
d-sweep rows, and (when complete) predictions for trajectory panels.

**Latent-d grid (sim + real; shared extension rule):** default
`latent_dims: [2, 3, 5, 10, 20]` and `nested_fit_d: 20` in
`configs/quadrant_n5.yaml`. Rule (recorded before extending):
if the selected `d` equals the grid max on most sessions/seeds, extend the
grid by doubling until the mode selected `d` is strictly below the max or
`d` reaches `⌊n_units / 2⌋`. Nested methods set `nested_fit_d = max(grid)`.
Sim N=5 selections sat at 20 for most methods (pca/dm/isomap/gpfa 9/10,
lds 10/10); sim extends to `{2,3,5,10,20,40}` (`⌊86/2⌋=43`). Real cohort
extends to `{2,3,5,10,20,40,80}` (sessions often ≫ 100 units). Prior rows
kept alongside labeled `grid20`.

### 1.3 What “quadrant” means here

Not spatial arena quadrants. The **representation quadrant** is the 2×2:

|            | Static | Dynamic        |
| ---------- | ------ | -------------- |
| Linear     | PCA    | LDS / gpfa_causal (filters) |
| Nonlinear  | DM     | *(empty)*      |

Smoothing controls (`*_smooth`) are post-hoc causal EMA on Ridge predictions
(same τ rule for every base, including `lds_smooth`). They are not a fifth
quadrant cell; they equalize post-processing across contrast arms.

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

**Effective lag (causal methods):** reports use **train-based lag** (sync /
train XC under the behind-truth convention) plus an **eval plateau range**
(lags within 0.01 of peak correlation). Eval **point estimates are dropped**.
Positive lag means the prediction is behind the truth.

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
6. **Coordinates:** room-local **cm**, origin at the room centre from the
   boundary polygon. Targets come from ``postions_dataset.csv`` ``X,Y`` only —
   never ``dataset.csv`` ``X,Y`` (those are normalized). Load-time check:
   retained path extent must be 50–110% of the boundary width and height.
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
window contains **only spikes strictly before `t`**. Rebuild causal 250 ms
counts from per-unit spike times. Position/valid/room still align to the
dataset timestamp `t` (label = right edge of the causal window).

**Hard rule — never decoder features:** Do **not** use `dataset.csv` `Cell_*`
columns, `spike_rate_dataset.csv`, or `dataset_polar.csv` as decoder / latent
inputs unless a separate check has proven that series is strictly causal
(no spikes at times `≥ t` in the feature for label `t`). `Cell_*(t)` is
centre-aligned and leaks **125 ms** of future spikes; the other two files are
treated as non-causal until proven otherwise. They may be used only for the
load-time integrity check (rebuild centre window vs `Cell_*`) or for
non-feature metadata.

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

**M3 planned contrasts (locked before any M3 results):**

Primary (normalized error; sign counts across animals, N≤6):

- `lds − raw_smooth`
- `lds_smooth − raw_smooth`
- `gpfa_causal − raw_smooth` (included: leakage + train-only fit verified)
- `pca_smooth − raw_smooth`
- `dm_smooth − pca_smooth`
- `raw_smooth − raw`

Secondary (original unsmoothed sim-report contrasts):

- `dm − pca`, `lds − pca`, `raw_lag − raw`, `lds − raw_lag`
- each base method − `raw` (`pca`, `dm`, `lds`, `isomap`, `gpfa_causal`)

Both arms of every primary contrast have the same post-processing option
(`lds_smooth` exists so LDS can be compared under EMA when needed).

**M3 cohort rule (recorded in local manifest before run):** 2 sessions per
animal closest to that animal's median eligible unit count among sessions
passing all exclusions (incl. rate-map stability); ties → lexicographic. If an
animal has fewer than 2 eligible, use what exists and record the shortfall.
Room A only. Parallel: one process per session, max 6, BLAS threads = 8.
Resume keyed on config hash + git SHA; refuse dirty tree.

**Latent-d grid extension (pre-registered before extended-grid results):**
If selected `d` equals the grid max on most cohort sessions (real) or
seed×source cells (sim), extend by doubling (`…, 20 → 40 → 80 → …`) until
the mode selected `d` is below the max or `d = ⌊n_units/2⌋`. Apply to
pca/lds (and their smooth controls); archive prior `grid20` rows alongside.
Sim cap: d=40 (`⌊86/2⌋`). Real cohort: d=80.

**Plateau stop (recorded before reading final contrasts):** after the first
extension to d=80, compare median normalized error at d=40 vs d=80 across
real cohort sessions, per method. If the 40→80 improvement is **&lt; 0.01**
normalized error, stop extending that method and report
“selection at grid max; error at plateau”. Otherwise run **one** more
doubling to `d = min(160, ⌊n_units/2⌋)` for that method only and treat that
as the final grid. Decision (real): pca improvement ≈ 0.005 → **stop /
plateau**; lds improvement ≈ −0.008 (d=80 not better) → **stop / plateau**.
Final real grid remains `{2,3,5,10,20,40,80}`.

**Frozen (post plateau decision):** no new methods, controls, or analysis
reruns after this decision. Report 2 uses the frozen final grid (+ `grid20`
rows alongside). Further sim-vs-real work belongs to the Report 1 revision.

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

### 3.5 Report plan (M4 figure specification)

M4 builds an offline **real-data-only** figure report that reads alongside
`agents/quadrant_n5/figures` for style, not for a sim-vs-real panel.
Number real-data figures **RD1–RD7 and RD9** (RD8 removed — see below).
Same visual style as `agents/quadrant_n5/figures`: reuse its style constants
and plotting helpers by import where those helpers accept arrays (do not
import sim report code into core analysis; report pages only).

Write only under an **ignored** dir (`analysis/real_quadrant/outputs/` or
`outputs/real_quadrant/`). **Never commit** PDFs, NPZs, tidy CSVs, figures, or
tables from real data.

#### Figure story (RD1–RD7, RD9)

| Figure | Content | Parallel |
| ------ | ------- | -------- |
| **RD1** The recording | Arena outline with the animal's path per room segment; occupancy maps; valid vs dropped frames; units by region and depth | Fig1 design |
| **RD2** From spikes to features | Raster of example units aligned with position over ~1 min of behavior; methods panel comparing the source file's centred 250 ms window with our causal `[t−250 ms, t)` window, with the 125 ms lookahead marked | *(real-data only; no direct sim twin)* |
| **RD3** What the neurons encode | Rate maps for example units computed on **training data only**; speed and head-direction context | *(real-data only)* |
| **RD4** Representations | PCA, DM and LDS latent trajectories colored by position | Fig1 / Fig7 latent panels |
| **RD5** Predictions | Decoded vs true path on the arena; x(t) and y(t) traces over a test block; error over time | Fig7 trajectories |
| **RD6** Is it real? | Error distributions vs chance floor; time-shift null; split timeline showing train, test, purge gaps and the trimmed segment edges | Fig2 validity |
| **RD7** The quadrant answer | Method contrasts across animals; room A primary, B and a as replications | Fig3 / Fig6 |
| **RD8** *(removed from M4)* | Real vs simulated | **Moved to the Report 1 revision** (default + matched sim). Not part of Report 2 / M4. |
| **RD9** Region subsets | A priori region-subset decoding (supplementary) | Replaces FigS5 |

Dropped from the sim set for real data (no RD twin): sorted-vs-GT, realtime
replay / A9–A11 latency, A10 seed-isolation, A12 UI audit, deployability
latency panels. Optional later supplements (not M4): d-sweeps (FigS1), full
A13 strips (FigS2), spatial-error / centre-pull grids (FigS4).

#### Figure data contract

Each milestone must save enough under the ignored outputs directory that **M4
can draw every figure without rerunning analyses**.

**Per session and room segment**, save at least:

| Artifact | Used by | Notes |
| -------- | ------- | ----- |
| `decode_times` | RD1–RD6 | Causal grid timestamps |
| `y` (room-local cm) | RD1, RD3–RD6 | Position labels; origin at room centre |
| `valid` mask | RD1, RD6 | Target validity; dropped-frame panels |
| Trim boundaries | RD1, RD6 | First 60 s / last 10 s edges (§3.0b) |
| Train / test / purge masks | RD5, RD6 | Split timeline; eval intersection |
| Predictions per method and decoder | RD5–RD7 | Test-block traces and aggregate error |
| Latents per method (or a documented subsample) | RD4 | If subsampled, document rule and seed |
| Chance-floor predictions | RD6 | Train-mean position floor |
| Per-fold metrics | RD6–RD7 | Errors, R², null stats as computed |
| Example-unit spike times in the plotted window | RD2 | Only the units/window chosen by the fixed selection rule |

**Session-level JSON** (alongside segment artifacts): provenance with git
commit, config hash, and integrity-check result (centre-window rebuild vs
`Cell_*`).

**Milestone ownership:** M1 must start saving this contract for the `raw`
method (and shared masks / times / `y` / floor). Later milestones extend the
same layout with additional methods, latents, nulls, and cohort aggregates.
M4 / Report 2 is render-only over these real-data artifacts. Sim-vs-real
(former RD8) is owned by the Report 1 revision.

#### Report rules

1. Real-data figures and tables are written **only** to ignored outputs and
   are **never committed**.
2. Captions are filled in **at render time** (numbers from saved tidy
   artifacts / CSVs only — same discipline as the sim story page).
3. Example sessions and units are chosen by a **fixed rule** (e.g. median by a
   stated criterion), not hand-picked for how good they look; the rule is
   recorded in the report config / provenance JSON.
4. Room A is primary in RD7; B and a appear as replications. Sim-vs-real
   lives in the Report 1 revision (normalized error, §3.0e), not in M4.

### 3.6 Milestones

| ID | Deliverable | Done when |
| -- | ----------- | --------- |
| **Bin-edge** | Done (§3.0a) | Done |
| **Seam** | `analyze_source` accepts prepared bundle; sim tests byte-identical | Done |
| **M1** | Adapter + `raw` on one room-A segment; figure data contract started | Done |
| **M2** | All methods on that segment; smooth / lag / gpfa_causal | Done |
| **M3** | Cohort (2/animal); contrasts; latent-d extension + plateau freeze | Done (grid frozen) |
| **M4 / Report 2** | Offline real-data RD1–RD7 + RD9 (RD8 out of scope) | **COMPLETE** 2026-10-08 (tag `report2-v1`) |

---

## 4. Stop — Report 2 COMPLETE (2026-10-08)

Report 2 (real-data-only RD1–RD7/RD9) is frozen and closed.

| Field | Value |
| -- | -- |
| Date | 2026-10-08 |
| Tag | `report2-v1` |
| Git sha | `35b20f1831792731e7c36f98e19b834ffa2d16f7` |
| Config sha256 | `0192b55b4ad30367f3a2b35ef272ae9da40ee3a352ed459141561612921676b5` |
| Named PDF sha256 | `e2e3a0bf81a173e5ab5951d691492e5d6076a0d43c91787569ab7c3acc13d16f` |
| Anon PDF sha256 | `1be0a7be8b29ca9850369c9a83bda1fdee09e624512cf4c61e2566b045415d0b` |
| Named figure-set sha256 | `081e5d03a4b85f29943005c6a08c8313c77611880047f162b6e6b29d678efa7b` |
| Anon figure-set sha256 | `e35682278158e531b95e182f02dd1eb4b271fe2afb39fc183d095a3712d96d96` |

**Answer (aggregate, N = 6 animals; mean over animals of per-animal session medians; normalized error = median Euclidean / chance floor).** (1) Temporal integration helps: raw+EMA − raw = −0.088 (6/6). (2) Static compression does not beat the full population: PCA+EMA − raw+EMA = +0.050 (1/6). (3) DM does not beat PCA at matched grid20: dm+EMA − pca+EMA = +0.021 (1/6). (4) LDS+EMA − raw+EMA = −0.020 (3/6); causal GPFA − raw+EMA = +0.042 (0/6). (5) Readout: at the pre-registered Ridge-selected d, kNN − Ridge is slightly negative for every method (general gain, not LDS-specific as in Report 1); at each method’s kNN-best d (descriptive, chosen on test) the gain is larger (6/6 where defined). Primary contrasts stay at Ridge-selected d. EMA is Ridge-only; causal GPFA has no kNN run.

Next: Report 1 revision (sim-vs-real normalized error and the readout gap). Sim extend-d outputs remain for that revision. Local artifact paths: gitignored `outputs/real_quadrant/HANDOFF.local.md`.
