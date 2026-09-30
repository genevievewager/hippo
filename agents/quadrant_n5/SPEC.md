# Quadrant N=5 experiment agent — spec for Cursor

Repo: `hippo` (public). Runs on `bionet-neuroai` via Cursor Remote-SSH.
Companion to `agents/pressure_test/` (built 2026-09-20 @ `8c01b734`; pre-rewrite: `01b86a3`).
Suggested location: `agents/quadrant_n5/SPEC.md`. Point Cursor at this file and
tell it to start at Phase 0.

---

## 0. Purpose

Produce a first empirical answer, at N = 5 independent simulation seeds, to:

> What aspects of hippocampal population structure need to be preserved to
> create an accurate, stable, and realtime-deployable neural representation
> for behavioral decoding?

The comparison is a 2 × 2 of representations applied to spike counts
(250 ms causal window, 50 ms step):

|               | Static                         | Dynamic                         |
| ------------- | ------------------------------ | ------------------------------- |
| **Linear**    | PCA                            | LDS (Kalman *filter*)           |
| **Nonlinear** | Diffusion Maps + Nyström       | **Empty — not implemented.**    |

Baselines outside the quadrant: Isomap, GPFA.
Controls: raw counts, raw counts + lagged history.

**This agent does not look for a winner.** It measures each representation
under identical conditions, audits that the comparison is fair, and reports
consistent effects, variance, failure modes, and open questions. A result
where PCA or raw counts win is a finding to explain, not a bug to tune away.
Never tune a method after seeing test results.

---

## 1. Guardrails (data, IP, repo) — non-negotiable

1. **Synthetic data only.** Inputs are the hippocampal simulator and its
   Neuropixels degradation stage. Refuse any input path that is not produced
   by the simulator inside this run's output directory. Never read, list, or
   open real lab recordings, even if they exist on the server.
2. **Everything executes on the server.** No uploads, no external APIs, no
   telemetry, no cloud storage, no new dependencies that make network calls.
   Adding any dependency requires asking the user first.
3. **Keep data out of the chat context.** Do not `cat`/print data files,
   arrays, or large CSVs into the conversation. Inspect them with scripts that
   print shapes, dtypes, summary statistics, and hashes.
4. **Outputs go to `outputs/quadrant_n5/`.** Before writing anything, add it to
   `.gitignore` and verify with `git check-ignore -v outputs/quadrant_n5/x`.
   (The `.hippo/` entry was added after those files were tracked and does
   nothing; don't repeat that.) Never commit generated data, models, or PDFs
   to `master`. Commit PDFs to `pressure-test-reports` only if the user asks.
5. **Work on a branch `quadrant-n5`.** Do not push, merge, force-push, or
   rewrite history without explicit user approval.
6. **The frozen config is frozen.** Any change to `configs/quadrant_n5.yaml`
   after Phase 3 starts requires stopping and asking the user, and invalidates
   all runs made under the old config hash.

---

## 2. Phase 0 — Inventory (read-only; then STOP)

Locate and report, as a table of `concept → file:symbol → notes`:

- simulation entry point; how trajectory, neural noise, and seeds are set
- Neuropixels degradation stage; simulated sorting stage; how ground-truth
  spikes and sorted spikes are each accessed
- spike-count feature construction (`spike_binner`, `count_spikes_in_window`)
  and whether labels are timestamped at the window's right edge
- each representation: PCA, Diffusion Maps + Nyström, LDS, Isomap, GPFA —
  fit, batch transform, per-step transform (`transform_one` or equivalent),
  save/load
- Ridge decoder; any existing kNN decoder
- realtime replay path (`ReplaySpikeStream`, `CausalSpikeBuffer`,
  `ObservationTransformer`)
- where the Streamlit UI builds the analysis config
- current 7 pre-existing test failures (confirm still the same 7)

Flag anything missing, duplicated, or ambiguous. **Stop and show the table to
the user before writing code.**

---

## 3. Phase 1 — Correctness gate

The realtime-replay arm is untrustworthy until these are fixed. Fix each with
a regression test that fails before and passes after.

1. `CausalSpikeBuffer.counts_at` undercounts on out-of-order arrival (linear
   scan breaks on first `ts >= t`). Fix: maintain sorted order on insert
   (bisect insort) or count without early exit. Test: shuffled arrival order
   gives identical counts to sorted arrival.
2. NaN timestamp counted into every window and disables `_prune`. Fix: reject
   non-finite timestamps at ingest (drop + count them in a diagnostics
   counter; never silently). Test: one NaN leaves counts and buffer size
   identical to the clean stream.
3. `count_spikes_in_window` on unsorted input. Fix: enforce the precondition
   (check monotonicity; sort or raise — prefer raise in library code, sort at
   ingest boundaries). Test: unsorted input either raises or matches brute force.
4. + 5. `ReplaySpikeStream.connect` cannot load Phy column naming. Fix: rename
   columns **before** unit-id extraction; replace the three divergent alias
   tables (`spike_binner`, `spike_stream`, `spike_buffer`) with one shared
   table. Test: `cluster_id`/`clusters`/`times`/`spike_time_s` variants all load.
6. Time units: every spike-time and position-time array carries seconds. Add
   an assertion at every ingest point that rejects values consistent with
   sample indices (e.g. max time > session length × 2 or integer dtype).
7. Set the `--latency-budget-ms` default to **50** (= step size = update_dt).

Gate (record the result verbatim in every PDF):
- existing suite: no failures beyond the known 7
- `agents/pressure_test` re-run: **0 critical**
- if the gate is not green, the replay arm is disabled and the PDFs say so.
  Offline analyses may still run, but only with sorted-input enforcement from fix 3.

Do not fix the non-critical pressure-test findings here unless they block the
experiment; list them in the report instead.

---

## 4. Phase 2 — Frozen experimental config

Create `configs/quadrant_n5.yaml`. Record its SHA-256 in every output.

### Session and data
- `session_s: 600` (fallback 300 if the Phase 3 pilot says so — one value)
- arena geometry and size: fixed, identical across seeds
- `n_seeds: 5`, `master_seed`: fixed integer
- per data seed, derive independent streams with
  `np.random.SeedSequence(master_seed).spawn(5)`, then spawn per component:
  `trajectory`, `neural`, `recording_noise`, `sorting_errors`, `methods`.
  Trajectory, neural activity, and noise all vary across seeds.
- **spike sources** (each run on the same seed's data):
  - `sorted` — degraded + simulated sorting. **Primary. Only source allowed
    in any "deployable" or realtime claim.**
  - `gt` — simulator ground-truth spikes. **Reference only, labelled
    non-deployable everywhere it appears.** Used to separate representation
    effects from recording/sorting effects (Q6).

### Features
- window `W = 0.250 s`, step `0.050 s`, half-open causal `[t − W, t)`
- label = position at `t` (window right edge). Never window center.
- transform: identical for all methods (use the pipeline's existing one; if
  none, `sqrt(counts)`), then z-score with **training-block statistics only**.

### Split
- contiguous: train = first 80%, test = last 20%
- purge gap between them: `W + max_history_used_by_any_method`, rounded up to
  whole steps (≥ 1.0 s is safe with lags below)
- inner cross-validation (all hyperparameter selection): 5 contiguous blocks
  inside the training segment, same purge gap on each side of the held-out block
- evaluation set: the **intersection** of valid test indices across all
  methods (methods needing history lose leading samples; all are scored on
  the same samples)
- training samples: identical index set for every method

### Representations
| key          | quadrant / role          | notes |
| ------------ | ------------------------ | ----- |
| `raw`        | control                  | z-scored counts, no reduction |
| `raw_lag`    | history control          | `raw` stacked with its previous 5 steps (250 ms extra history). Not a quadrant member. Separates "has a dynamics model" from "can see the past." |
| `pca`        | linear-static            | fit on train |
| `dm`         | nonlinear-static         | kernel bandwidth by a train-only rule (median kNN distance), recorded; fit on all training samples if ≤ 12 000, else a fixed landmark count drawn from the `methods` stream; Nyström for test and replay |
| `lds`        | linear-dynamic           | EM on train; inference by **Kalman filter only**. RTS smoothing is forbidden anywhere in the evaluation path. |
| `isomap`     | baseline                 | out-of-sample via its own transform if causal (see audit A9) |
| `gpfa`       | baseline                 | typically a smoother → expect offline-only label |
| nonlinear-dynamic | —                   | empty cell; shown as empty in figures |

Latent dimensionality sweep: `d ∈ {2, 3, 5, 10, 20}` for every reducing method.
Where embeddings are nested (PCA, DM, Isomap), fit once at d = 20 and slice.
LDS and GPFA are refit per d.
**Primary d** per method per seed: choose by the same rule for all — lowest
inner-CV median error with the Ridge readout; ties → smaller d.

### Decoders
- `ridge` — **primary**. Alpha from `logspace(-3, 3, 13)` via inner CV.
- `knn` — **sensitivity check**, applied to every representation.
  `KNeighborsRegressor`, k from `{5, 10, 20, 40}` via inner CV, uniform
  weights, inputs standardized with train statistics.
- Target: 2-D position (x, y), in the simulator's spatial units, recorded.

---

## 5. Phase 3 — Runs

Grid: 5 seeds × 2 spike sources × 7 representation keys × d sweep × 2 decoders,
plus realtime replay for `sorted` at primary d with both decoders (gate
permitting).

Additional analyses:
- **Learning curve (Q10, partial):** for `sorted`, primary d, Ridge — train on
  the last 25%, 50%, 100% of the training segment, same test set.
- **Floor and ceiling references:** predict the training-mean position
  (floor; center-bias reference), and a circular time-shift control: position
  shifted by ≥ half the session relative to spikes must decode at floor level.
  If it does not, stop — that is leakage.

Execution rules:
- **Pilot first:** run seed 0 fully, time each stage, project total wall time.
  If the projection exceeds 12 h, stop and give the user options (5-min
  sessions, running seeds in parallel, trimming LDS/GPFA's d sweep) with
  estimated runtimes for each.
- resumable: each (seed, source, rep, d, decoder) writes one result JSON and
  is skipped if its JSON exists with a matching config hash
- every result JSON records: config hash, git SHA, dirty-tree flag, all seed
  values, sample-index hashes, hyperparameters chosen, per-stage timings
- saved fitted objects go under `outputs/quadrant_n5/models/`

---

## 6. Phase 4 — Audit (pressure test for the experiment)

Each check produces PASS / FAIL / N/A per seed and appears as a table in every
PDF. A FAIL means the affected comparisons are marked unreliable, not hidden.

- **A1 Split identity** — train/test/eval index hashes identical across all
  methods within a seed.
- **A2 Preprocessing identity** — hash of the feature matrix fed to each
  representation is identical within a (seed, source).
- **A3 Sample counts** — identical training and evaluation sample counts.
- **A4 Window identity** — every method's features built with W = 0.250 s,
  step 0.050 s; labels at window right edge.
- **A5 Future perturbation** — for each representation's per-step path:
  perturb all spikes after time t; outputs at ≤ t must be bit-identical
  (or within 1e-9). LDS additionally: assert no smoother is called.
- **A6 Hyperparameter hygiene** — instrument selection code; assert no test
  index ever reaches it. Record chosen values.
- **A7 Deployability labelling** — any figure or table containing `gt`
  results is labelled non-deployable; any "realtime" claim uses `sorted`
  only.
- **A8 Save/reload** — serialize each fitted representation + decoder,
  reload in a fresh process, re-transform test: max abs diff ≤ 1e-9
  (float64) or 1e-5 (float32).
- **A9 Realtime compatibility, determined empirically** — run the per-step
  causal transform over the test segment and compare with the batch
  transform. Match → `realtime-compatible`. Mismatch or no per-step path →
  `offline-only`. The code's existing labels are ignored; disagreements with
  them are reported.
- **A10 Seed isolation** — re-running a seed reproduces results; changing
  only the `methods` stream leaves the generated data bit-identical; changing
  the data seed changes trajectory, neural, and noise streams.
- **A11 Offline vs replay** — per-sample difference between offline and
  replay predictions on the same test segment. Any nonzero difference is
  traced to a cause and reported, not averaged away.
- **A12 UI config** — build the analysis config through the Streamlit code
  path headlessly (`streamlit.testing.v1.AppTest`) with defaults, diff it
  against the frozen YAML. Any silent difference = FAIL.
- **A13 Time-shift control** — see Phase 3. Must be at floor.
- **A14 LDS readout inference** — Ridge and kNN trained on LDS latents
  must see Kalman-**filtered** training states, never RTS-smoothed ones.
  Train and test latents come from the same inference (`causal=True`).
  A FAIL marks that seed's LDS decoder results unreliable.

**Suspicious-result tracing.** Automatically flag and write a short trace
note (which stage, which evidence) when: test error < inner-CV error by a
large margin; replay beats offline; any method beats `raw` on `gt` but loses
on `sorted` (or vice versa) by more than the seed SD; a method's error is
below the time-shift control by less than the floor-to-best gap would
suggest; or one seed drives a mean difference (sign flips in ≥ 2 seeds).
Trace; do not "fix" by retuning.

---

## 7. Metrics

Per (seed, source, rep, d, decoder):
- Euclidean error: mean, median, p90; full distribution saved
- R² per axis
- center-pull: slope of decoded vs. true distance from arena center
  (< 1 = shrinkage toward center); also per-axis regression slopes
- spatial error map (binned by true position)
- Ridge − kNN error gap (readout-limited geometry indicator, Q9)
- information retained: variance explained (linear methods); for all,
  inner-CV Ridge error vs `raw` as a reference

Realtime (replay arm):
- representation transform time, decoder time, total per step: p50, p99, max
- fraction of steps exceeding the **50 ms** budget
- dropped/late steps

Across seeds (N = 5):
- every seed's value, mean, SD
- paired per-seed differences for the planned contrasts:
  - linear vs nonlinear: `dm − pca`
  - static vs dynamic: `lds − pca`
  - history alone: `raw_lag − raw`
  - dynamics beyond history: `lds − raw_lag`
  - reduction vs none: each rep − `raw`
  - readout limitation: (`knn` − `ridge`) per rep
  - recording effect: `sorted − gt` per rep
- sign consistency (k/5 seeds) and standardized paired effect
  (mean paired diff / SD of paired diffs), labelled descriptive
- **No p-values.** State once in the report: with N = 5, a two-sided sign
  test or Wilcoxon signed-rank cannot go below p = 0.0625, so these numbers
  estimate effect size and consistency for planning the next experiment.
- The linear×nonlinear × static×dynamic interaction is **not estimable**
  (empty cell); say so rather than inferring it.

---

## 8. Phase 5 — PDFs

Use matplotlib `PdfPages` (or reportlab if already a dependency). No web
fonts, no network. Write to `outputs/quadrant_n5/reports/`.

**Per-seed PDF** (`seed_{i}.pdf`):
1. Provenance: git SHA, dirty flag, config hash, seed values, gate status,
   date, host
2. Audit table A1–A14 for this seed
3. Trajectory and occupancy map (arena coverage in train and test)
4. Decoded vs true trajectories (test segment) per representation, primary d,
   Ridge and kNN
5. Error distributions and spatial error maps
6. d-sweep curves per method (Ridge and kNN)
7. Replay latency panel (or "replay disabled: gate not green")
8. Flagged anomalies and trace notes

**Aggregate PDF** (`quadrant_n5_summary.pdf`):
1. Provenance + gate + audit summary across seeds
2. The quadrant figure: one panel per cell showing all 5 seed values with
   paired lines across methods; empty cell drawn as empty
3. Paired-contrast plot for every planned contrast (per-seed points, mean,
   sign count)
4. d-sweep curves, mean ± SD across seeds
5. `sorted` vs `gt` per representation (recording/sorting sensitivity)
6. Ridge vs kNN per representation
7. Center-pull and failure-mode summary per representation
8. Learning curves
9. Offline vs replay; latency distributions vs the 50 ms budget
10. Findings: **measured facts only** — each statement cites the figure and
    the sign count behind it. No causal language beyond what a contrast
    supports.
11. Open questions for the next experiment: anomalies, contrasts with 3/5 or
    worse consistency, and each of Q1–Q10 marked as `answered at N=5`,
    `suggestive`, or `not addressed`.

---

## 9. Tests for the agent itself

Add `tests/test_quadrant_n5.py` covering: split/purge logic (no index within
the gap), label timing, A5 future-perturbation harness catches an injected
smoother, A9 flags an injected non-causal transform, A13 catches injected
leakage, seed-stream isolation, and result-JSON resume logic. Run on tiny
synthetic inputs (< 30 s total).

---

## 10. Stop and ask the user when

- Phase 0 inventory is done
- the Phase 1 gate is not green after the fixes
- the pilot projects more than 12 h
- a representation or decoder is missing or behaves differently than this
  spec assumes
- anything would require real data, network access, a new dependency, a
  config change, or touching `master`

## 11. Final message to the user

Short: gate status, audit failures (if any), where the two PDFs are, the
3–5 most consistent effects with sign counts, and the flagged anomalies. No
recap of steps.
