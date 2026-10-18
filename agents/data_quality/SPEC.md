# Report 4: Simulated vs lab data quality. Spec for Cursor

**Checkout:** detached worktree at `24cc602`, no branch
(`/home/gmwager/projects/hippo-dq`). Work stays uncommitted. Do not create a
`data-quality` branch.

Repo: `hippo`. Work happens in this **separate git worktree**, never in the
checkout the live Report 2 run uses. Runs on `bionet-neuroai` via Cursor
Remote-SSH. Location: `agents/data_quality/SPEC.md`.

Companions:
- Report 1: `quadrant_n5` (sim, N = 5 seeds). `agents/quadrant_n5/SPEC.md`.
- Report 2: real-data quadrant. `docs/real_data_quadrant/PLAN.md`. **Running live in tmux.**
- Report 3: hparam sweep. Its §9 "Report 4 groundwork" list is implemented here.

Start at Phase 0. **Read §1 first.**

---

## 0. Purpose

> Where does the simulator's output differ from the lab's recordings, and by
> how much? Check units, sorting, spatial coding, population structure,
> behaviour and decoding.

A difference counts as divergence only if it is larger than the lab's own
session-to-session variability. The main output is a ranked list of
divergences. Each entry has a direction (sim too high or too low), a size
relative to lab variability, and an uncertainty interval. That list becomes
the simulator's to-do list.

---

## 1. Live-process safety. Non-negotiable; overrides everything else

A Report 2 run is live in tmux on this server. It reads lab data through the
current checkout. Changing any file, module, config, environment or output
directory it depends on can corrupt it, possibly silently.

Editing is dangerous even though the run has already started:
- worker processes (joblib/loky, `multiprocessing` spawn) re-import modules
  from disk;
- lazily imported modules load later;
- configs may be re-read per segment;
- an editable install points the venv at the checkout.

Rules:

1. **Never touch the live checkout's working tree.** No edits, no new files,
   no `git checkout / switch / stash / pull / reset / rebase / merge / clean`
   in it. All work goes in the detached worktree at the live run's base commit.
   Creating the worktree writes only `.git` metadata, not the working files.
2. **Open only the worktree in Cursor.** Do not open the live checkout as a
   workspace: format-on-save, agent edits or refactors could touch it.
3. **Never interact with the tmux session.** No `send-keys`. Do not attach,
   because a stray keystroke such as Ctrl-C kills the run. Read-only
   inspection only: `tmux capture-pane -p -t <target>`, `ps`, `/proc/<pid>/*`.
   Do not attach `py-spy`, `gdb` or `strace`; ptrace pauses the process.
4. **Do not modify the shared Python environment.** No pip/conda
   install/uninstall/upgrade. No `pip install -e` of the worktree, which would
   repoint the package for the live run's future workers. Run worktree code
   with `PYTHONPATH=<worktree>`, and verify that
   `python -c "import <pkg>; print(<pkg>.__file__)"` resolves into the
   worktree. If a dependency is missing, stop and ask.
5. **New code only, in new files.** Import existing modules; never edit them,
   even to fix a bug found during this work. Log such bugs in
   `agents/data_quality/DEFERRED_FIXES.md` for after the live run finishes. If
   you need different behaviour, write a wrapper in the new package and note
   it.
6. **Separate outputs.** Write everything under
   `<worktree>/outputs/data_quality/`. Never write into Report 2's output dirs.
   Never use Report 2's in-progress outputs as inputs. Before the first write,
   confirm that path resolution does not fall back to an absolute path or env
   var shared with the live run.
7. **Lab data is read-only.** Check whether the existing real-data adapter
   writes caches (`.npz`, `.parquet`, pickles) next to the data or into a
   shared cache dir. If it does, redirect the cache into the worktree's
   outputs, or do not use that code path.
8. **Resource ceiling.**
   - Run under `nice -n 19`, plus `ionice -c3` if available.
   - Cap threads with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
     `MKL_NUM_THREADS` and `n_jobs`, using the caps set in Phase 0.
   - Keep free RAM ≥ max(25 % of total, 2 × the live run's peak RSS).
   - A watchdog aborts our job if free memory drops below that threshold.
     The OOM killer may otherwise pick the live process.
9. **Tests.** Run only inside the worktree. Run `tests/test_data_quality.py`
   and targeted tests. Run the full suite only after confirming it writes
   nothing outside the worktree.
10. No push, merge, or touching `master` without explicit approval.
11. **No commits in this worktree during Report 4.** Work stays uncommitted.

**Before any command that writes, installs or runs heavy compute, ask:** does
this touch the live checkout, its environment, its outputs, its data cache,
or compete with it for memory? If unsure, stop and ask.

---

## 2. Phase 0: Inventory (read-only; then STOP)

**A. Live-run fingerprint.** Write `agents/data_quality/LIVE_RUN.md` in the
worktree. It is the do-not-touch list and must include:
- tmux session/pane, PID and child PIDs, full command line, python executable;
- `readlink /proc/<pid>/cwd`;
- checkout path, branch and commit (`git -C <path> rev-parse HEAD`, read-only),
  and the dirty-file list;
- files open under the repo and under `HIPPO_DATA_ROOT` (`ls -l /proc/<pid>/fd`);
- output dir(s), config path(s), cache dir(s);
- peak RSS, cores in use, and whether it spawns workers.

**B. Worktree.**
- Base the worktree on the live run's commit, so Report 4 uses byte-identical
  loaders and adapter.
- Create the worktree and verify import resolution (§1.4).

**C. Lab inputs, through the existing adapter (imported, not edited).**
- Spike-time units: seconds or samples? Give an aggregate check only, e.g.
  median ISI interpreted both ways.
- Unit region labels.
- Native position rate.
- Room segments, valid mask, and the pre-registered exclusions.
- Whether Kilosort/Phy quality data exist (`cluster_info.tsv`,
  `amplitudes.npy`, waveforms).
- Whether the adapter writes caches (§1.7).

**D. Sim inputs.**
- `OUTPUT_ROOT` in `quadrant_n5_run.py` is hardcoded. Choose one approach and
  say which:
  - call the generator functions directly with our own output dir; or
  - read Report 1's existing sim outputs read-only.
- Does the simulator expose pre-degradation ground truth? It is useful for
  three-way comparisons: ground truth → degraded sim → lab.

**E. Reuse.** Which Report 3 §9 metrics already exist as functions (import
them), and which need writing.

**F. Resources.** Report `nproc`, total and free memory, and the live run's
usage. Propose thread, job and memory caps.

Flag anything ambiguous. **Stop and show.**

---

## 3. Phase 1: Matching the two datasets

The comparison must not report a divergence that comes only from comparing
unlike things. Use the defaults below. Anything that cannot be matched is
reported as a confound.

### 3.1 Units / region sets

- **Primary:** `{MEC, SUB}` (mapped from the simulator's 7 probe bands).
- **Secondary:** `{MEC, PARA, SUB}` and all lab units.
- **Tertiary:** MEC-only; drop sessions with < 10 matched units and report
  the drop count.
- If primary region-matched lab sessions have < 10 units, stop and ask.

### 3.2 Lab preprocessing = PLAN §3.0

- Analyse within room segments.
- Trim 60 s from the start and 10 s from the end.
- Apply the valid mask.
- Apply the pre-registered exclusions (30 units, 0.7 valid fraction).
- Apply the same exclusion rules to sim.
- **Matched-arm output dir:**
  `<worktree>/outputs/data_quality/matched/`
  (per-session dirs `*_A/`, fleet status, calibration artifacts). Not
  `sim_matched/`.

### 3.3 Duration / space / time / N

- Compute metrics on matched-length chunks, default 300 s, so estimators
  (Fano factor, correlations, place fields) see equal sample sizes.
- A sim session (600 s) gives 2 chunks. A lab segment gives
  floor(length / 300) chunks.
- Aggregate chunks within a session.
- Spatial metrics on two grids: (i) fixed 3 cm bins; (ii) a fixed
  20 × 20 grid over each arena's bounding box.
- Same occupancy threshold, speed filter (> 2 cm/s) and smoothing
  for both datasets.
- Count-based metrics use the shared W grid
  {0.05, 0.1, 0.25, 0.5, 1.0} s.
- Metrics that depend on N (dimensionality, synchrony, decoding) are
  computed on matched-N subsamples, N ∈ {10, 20, 40, max common}, with
  20 random subsamples each.
- **Seconds guard.** Fail loudly if either dataset has median ISI < 1e-4 s, or
  more than 1 % of units fire above 500 Hz. Sample-index vs seconds is a
  30000× error.

### 3.4 Sim inputs for Report 4

- Copy `outputs/quadrant_n5/seed_*/sim/` from the live checkout into
  `<worktree>/outputs/data_quality/sim_inputs/` with a SHA-256 manifest;
  read only the copy.
- Live run also writes `outputs/quadrant_n5/models/seed_*/real/` (hardcoded
  `OUTPUT_ROOT`); that tree is do-not-touch. Do not copy models.

### 3.5 Phy / sorting realism

- Phy files are absent in this dataset: drop amplitude cutoff; keep
  presence ratio (spike-time based).

---

## 4. Phase 2: Metric module

Create a new package `agents/data_quality/metrics/`. Use pure functions.

- **Input:** spike times per unit, unit metadata, positions, segment metadata.
- **Output:** tidy rows
  `(dataset, animal|seed, session, chunk, region_set, metric, W, N, value)`.
- **One code path.** There is one entry point, and `dataset` enters only as a
  label. Audit Q1 enforces this.

Metric families. These cover Report 3 §9 plus additions.

**Units**
- firing rate
- ISI CV and CV2
- burst fraction (ISI < 10 ms)
- refractory violations (< 1.5 ms)
- Fano factor at each W
- within-session rate drift (slope across chunk thirds / mean)

**Sorting realism**
- contamination estimate from refractory violations
- presence ratio (spike-time based); amplitude cutoff dropped (no Phy)
- the simulator's degradation-model equivalents
- sim pre-degradation vs post-degradation vs lab

**Spatial coding**
- fraction of units with ≥ 1 place field; field count, size and peak rate
- spatial information (bits/spike)
- sparsity
- coverage
- split-half stability
- all computed from train-only rate maps

**Rhythm (spike-based)**
- theta modulation index from the spike autocorrelogram (6–10 Hz)
- theta frequency
- If the simulator has no theta, that is a finding. Do not suppress it.

**Population**
- noise correlations at each W, with the position-tuned expectation removed
- synchrony: fraction of bins with ≥ k % of units active
- participation ratio and PCA spectrum at each W
- LDS dynamics eigenvalues, reusing the Report 3 code if it exists

**Behaviour**
- speed and acceleration
- occupancy entropy
- wall-distance distribution
- turning angles
- dwell times

**Decoding (within each dataset)**
- error/floor vs N units (matched-N), using the frozen Report 1/2 config
- error/floor vs W

Train-on-sim / test-on-lab is **not** possible with unit-level decoders
because units do not correspond across datasets. State this once and do not
attempt it.

---

## 5. Phase 3: Divergence scoring

### 5.1 Divergence ratio

For each metric, at each W, N and region set:

- **Lab-to-lab distance**, d_lab: the median over lab session pairs of the
  Wasserstein-1 distance between unit-level distributions. Prefer pairs from
  different animals.
- **Sim-to-lab distance**, d_sim: the median over (sim seed, lab session) pairs
  of the same distance.
- **Divergence ratio:** R = d_sim / d_lab.
  - R ≤ 1: sim is within lab variability.
  - R > 1: sim diverges.
- **Direction:** the sign of the median difference.
- **Location:** the percentile of the sim median among lab session medians.
- **Scale:** log-transform rates. Otherwise standardise by the pooled lab
  robust scale (MAD) before computing W1.
- **Uncertainty:** bootstrap over lab animals and sim seeds to get a 90 %
  interval for R. Flag metrics whose interval excludes 1.

### 5.2 Variability check

Also compute the sim-to-sim distance, d_simsim, between seeds. If
d_simsim ≪ d_lab, the simulator is too regular. Report that as a finding in
its own right.

### 5.3 Discriminator ("what gives the simulation away")

- **Features:** unit-count-invariant population summaries per 10 s window, at
  matched N. Examples: population rate mean and variance, synchrony, mean
  pairwise correlation, participation ratio, median ISI CV, burst fraction,
  speed.
- **Models:** logistic regression and sklearn `HistGradientBoosting`. Check
  that it is already installed; no installs (§1.4).
- **CV:** grouped by session or seed. Never split a session across folds.
- **Report:**
  - AUC with CI;
  - permutation importance;
  - an ablation curve: drop the top feature, re-score, repeat until AUC < 0.6.
    The drop order is the ranked list of giveaways.
- **Null:** shuffle labels at the group level. AUC must sit at ≈ 0.5.

---

## 6. Audits

- **Q1 One code path:** the same function objects (hash the source) produce
  sim and lab rows. The dataset label affects nothing but the output column.
- **Q2 Units:** the seconds guard (§3) passes for both datasets.
- **Q3 Report 2 consistency:** lab unit counts, segment counts and valid
  fractions equal what the Report 2 adapter yields. Compare aggregates only.
- **Q4 Estimator bias:** split one sim seed into halves. R(half vs half) must
  be ≈ ≤ 1 at lab chunk lengths and at each matched N.
- **Q5 Discriminator leakage:** CV is grouped, and the group-shuffle null is
  at chance.
- **Q6 Live-run isolation:** check at the start, after Phase 2, and at the
  end:
  - the live process is still running, or ended on its own;
  - the live checkout's `git status` and HEAD match `LIVE_RUN.md`;
  - no file in its output or cache dirs was created or modified by us
    (check mtime against our run window);
  - the shared env import path is unchanged.

A FAIL marks the affected comparisons as unreliable. They are shown, not
hidden.

---

## 7. Phase 4: Report 4 PDF

Write the PDF to `outputs/data_quality/reports/`. Reuse the
`agents/quadrant_n5/figures` style constants. In every distribution plot,
draw lab sessions as thin lines and sim seeds as thick lines.

| fig | content |
| --- | --- |
| **Q1 Design** | Matching protocol, chunking, matched-N, the definition of R, audit summary |
| **Q2 Divergence overview** | All metrics ranked by R with 90 % intervals, coloured by family, R = 1 line. **The headline figure.** |
| **Q3 Units** | Rate, ISI, burst, refractory, Fano-vs-W, drift |
| **Q4 Sorting realism** | Lab quality metrics vs sim degradation; sim pre- vs post-degradation vs lab |
| **Q5 Spatial coding** | Field stats, spatial info, stability; both binning schemes |
| **Q6 Rhythm** | Theta modulation and frequency |
| **Q7 Population** | Noise correlations, synchrony, PCA spectra and participation ratio vs W and N |
| **Q8 Behaviour** | Speed, occupancy, wall distance, turning, dwell |
| **Q9 Decoding** | Error/floor vs N and vs W, sim vs lab |
| **Q10 Discriminator** | AUC, importances, ablation curve, null |
| **Q11 Variability** | d_simsim vs d_lab per family |
| **Q12 Confounds** | The Report 3 §9 mismatch table; region-matched vs all-units R |
| **Q13 Validity** | Audits Q1–Q6 |

- **Findings page:** measured facts only. Each statement cites a figure and R
  with its interval.
- **Simulator to-do page:** the top divergences, ranked. For each, give the
  direction and the simulator setting most likely responsible, labelled as a
  hypothesis.

---

## 8. Relationship to Report 3

Report 3 §9 asks for the same metrics during its sweep. This package is that
code. When Report 3 runs, it imports from here instead of duplicating it.

---

## 9. Integration after the live run

After the live run finishes, copy the new files into the main checkout on
`offline-decoding`, commit there, then `git worktree remove`. Triage
`DEFERRED_FIXES.md` before or with that commit. Do not merge a `data-quality`
branch (none exists).

---

## 10. Tests: `tests/test_data_quality.py`

Synthetic inputs only; runs in < 60 s. Cover:

- Poisson units give Fano ≈ 1 and ISI CV ≈ 1, and injected refractory
  violations are recovered;
- sample-index input is caught by the seconds guard;
- two draws from the same generator give R ≈ 1, and shifted rates give R ≫ 1;
- grouped CV never splits a group, and shuffled labels give AUC ≈ 0.5;
- matched-N subsampling is correct;
- one code path is used for both dataset labels (Q1);
- the output-path guard refuses to write outside `outputs/data_quality/` and
  refuses every path listed in `LIVE_RUN.md`.

---

## 11. Stop and ask the user when

- Phase 0 is done;
- a change to an existing module, the shared environment, or anything in
  `LIVE_RUN.md` would be needed;
- free memory falls toward the threshold;
- the live run ends or errors (never restart it yourself);
- region-matched lab sessions have < 10 units;
- the pilot projects > 6 h at the capped resources.

## 12. Final message

Short. Include:
- audit FAILs, especially Q6;
- the top 5 divergences, each with R, its interval and direction;
- discriminator AUC and the top giveaways;
- whether the sim is less variable than the lab;
- where the PDF is.

No recap of steps.
