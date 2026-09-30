# Cursor prompt: generate the missing data, then the missing panels

Paste the block below into Cursor (Agent mode) on branch `quadrant-n5`, after the Phase 6 figure module is in `agents/quadrant_n5/figures/`. It works in stages and stops at each gate.

```
Goal: produce the data behind four missing publication panels (decoded
trajectories, error over time, spatial error maps, center-pull) plus a
latent-d sweep, then plot them with the existing figure module. The
experiment is frozen: config 4da17a5d…, seeds code dcc78ed7… (pre-rewrite: 775fa1c3). Nothing
numerical in the saved results may change. Everything runs on the
server; no network, no new dependencies without asking me; outputs stay
under outputs/quadrant_n5/ (gitignored). Don't print data arrays into
chat — shapes, hashes and summary numbers only.

STAGE 0 — plan (report, then STOP)
- Show how you will obtain primary-d test predictions for all
  5 seeds x 2 sources x 7 methods x 2 decoders (140 cells), reusing the
  runner's OWN functions (feature construction, split, _sqrt_zscore_train,
  representation transform, decoder fit). Do not re-implement any of them.
- Preferred route: load the saved representation models under
  outputs/quadrant_n5/models/ (A8 already verified save/reload), transform
  features, then refit ONLY the decoders on the training rows with the
  hyperparameters recorded in each method JSON (ridge_alpha, knn_k).
  Fallback, only where a model can't be loaded: deterministic refit with
  the recorded hyperparameters and the recorded methods seed stream.
  List which cells need the fallback and the estimated time.
- Confirm the eval rows are the saved eval set (index hash aecdcfd66f49eebe)
  and that LDS uses the Kalman filter, never the smoother. GPFA uses its
  offline smoother exactly as in Phase 3 and stays labelled offline-only.
Wait for my OK.

STAGE 1 — reproduction gate (must pass before anything is written)
- For every one of the 140 cells, recompute the median test error from
  the regenerated predictions and compare it with the saved median in
  the method JSON / source_summary. Tolerance 1e-6 cm.
- Print a 140-row pass/fail table (seed, source, method, decoder,
  saved, recomputed, |diff|). If ANY cell fails, write nothing, STOP and
  report which cells failed and your diagnosis. Don't adjust anything
  to make it pass.

STAGE 2 — write predictions (only if Stage 1 is 140/140)
- New script: agents/quadrant_n5/export_predictions.py
  (CLI: --results outputs/quadrant_n5 --seeds 0-4).
- Per seed and source, write
  outputs/quadrant_n5/seed_{s}/{source}/predictions.npz with:
    t_s        (n_eval,)    window right-edge time, seconds
    y_true     (n_eval, 2)  x, y in cm
    pred_{method}_{ridge|knn}   (n_eval, 2) for all 7 methods
    eval_index_hash, config_sha256, seeds_code_sha, export_code_sha
- Add a test with a tiny synthetic run: export -> reload -> medians
  equal the stored medians.

STAGE 2b — make the runner save predictions from now on
- The spec (§8 item 4) required decoded-vs-true trajectories per seed,
  but the runner never saved predictions and the reports silently
  dropped that page. Fix this at the source:
  - In realtime/quadrant_n5_run.py, after each method's final test
    scoring, write the same predictions.npz schema as Stage 2 (reuse the
    export function; don't duplicate it). Add nothing else, and change
    no computation: the saved medians and every existing JSON field must
    be byte-identical before and after this change.
  - Add audit check A15 "Required outputs present": every seed/source
    has predictions.npz with all 7 methods x 2 decoders, and its
    recomputed medians match the method JSONs to 1e-6 cm. FAIL if
    missing. Add it to the audit table in the reports.
  - Test: run the runner on the tiny synthetic fixture with and without
    the change; the JSON results must be identical, and the npz must
    exist and reproduce them.
  - This doesn't touch the frozen config. Confirm the config hash is
    unchanged.

STAGE 3 — latent-d sweep (plan cost first)
- PCA, DM, Isomap are nested: slice the SAVED d=20 embedding to
  d = 2, 3, 5, 10, 20 (no representation refit). For each d, refit the
  decoders on training rows using the same inner-CV rule as Phase 3 to
  choose alpha / k, then score the test set. Write
  seed_{s}/{source}/d_sweep.json: list of {method, d, ridge_median,
  knn_median, ridge_alpha, knn_k, refit_representation: false}.
- Check: the d that the sweep's own inner CV picks must equal the saved
  primary_d for every cell. Report any mismatch and stop.
- LDS and GPFA need a refit per d. Estimate the wall time from the Phase 3
  timings and ASK me before running them. If I approve, run them in tmux
  with logging, refit_representation: true.

STAGE 4 — plots (extend the existing module; keep its style)
- In agents/quadrant_n5/figures/extract_json.py: if predictions.npz
  exists, derive (never in the runner):
    data_predictions_window.csv  one 60 s window per seed: the FIRST 60 s
        of the test segment (fixed rule, no picking), true + PCA, DM, LDS,
        raw for both decoders
    data_error_maps.csv          median Euclidean error per 10x10 spatial
        bin (true position), per seed/source/method/decoder, plus the
        occupancy count per bin
    data_center_pull.csv         per seed/source/method/decoder: slope of
        decoded vs true distance from arena centre (OLS), and binned means
        in 5 cm distance bins
    data_error_cdf.csv           per method/decoder/source: pooled error
        quantiles 0..1 in 0.01 steps
  If d_sweep.json exists: data_d_sweep.csv.
  If a file is absent, skip its panels. Never fabricate.
- In figures.py add fig7(D, out) "Decoded trajectories", the final main
  figure: what each representation actually decodes. 180 mm wide, same
  rcParams, colours, markers and head_fig() convention as Fig 1–6.
  Sorted spikes throughout. The seed shown is the one whose LDS − PCA
  Ridge difference is closest to the across-seed median; the window is
  the first 60 s of the test segment. Both rules are stated in the legend.
    a  Arena view, 2 rows x 3 columns (rows = Ridge, kNN; columns = PCA,
       DM, LDS). Each square 100 cm arena shows the true path (black,
       0.8 pt) and the decoded path (quadrant colour, 0.7 pt) over the
       60 s window, start marked with an open circle. Each panel is
       annotated with that window's median error.
    b  x(t) and y(t) over the same window, stacked: true (black 1.0 pt)
       with PCA, DM, LDS overlaid (0.8 pt), Ridge. A small inset or a
       second row shows kNN.
    c  Euclidean error over the window (2 s rolling median), same
       colours, chance floor as a dashed grey line.
    d  Per-seed summary strip: the window's median error for PCA, DM and
       LDS in all 5 seeds (dots, seed shown in a highlighted), so the
       reader can see the example is typical, not the best case.
- Add figS3(D, out) "Decoded trajectories, all seeds": the Fig 7a arena
  view repeated for every seed (5 rows x 3 methods, kNN, plus a Ridge
  version as a second page). This is the no-cherry-picking backup.
- Add figS4(D, out) "Failure modes":
    a  spatial error maps: rows = Ridge, kNN; columns = PCA, DM, LDS;
       median across seeds of per-bin medians; one shared sequential
       single-hue scale (white -> #3a3936), colourbar in cm; bins with <10
       samples masked grey-hatched.
    b  centre-pull: slope per method per seed (dots, mean bar), reference
       line at slope 1 ("no shrinkage"); both decoders, sorted.
    c  pooled error CDFs per method, Ridge solid / kNN dashed.
- Add figS1(D, out) "Latent dimensionality": median test error vs d
  (log-x ticks at 2, 3, 5, 10, 20), mean ± s.e.m. across seeds, one
  panel per decoder x source (2x2), selected d marked with an open ring.
  LDS/GPFA drawn only if their refits exist, otherwise noted "not refit".
- Add figS2(D, out) "Full time-shift null": for each method, the 20
  per-shift (shifted − floor) values per seed from data_a13_shifts.csv,
  strip plot, −2 cm line, sorted and GT side by side.
- In build_story.py, add Fig 7 to the figure set as the last main figure
  (after Fig 6), and add S1–S4 after it under a "Supplementary" divider
  page. Add legends for all of them. Every number is
  computed from the data files (same pattern as the existing legends),
  and the story page's "What this does not show" paragraph drops the
  sentence about missing trajectories/error maps/center-pull once those
  files exist.
- Render everything, open each PNG, and fix any overlapping text before
  reporting. Width stays 180 mm; fonts stay embedded (fonttype 42).

STAGE 5 — wrap up
- make_figures.py runs the whole chain; tests pass; commit on
  quadrant-n5 (code only, nothing under outputs/). No push.
- Report: Stage 1 table summary (140/140?), that the runner change
  left every JSON byte-identical and A15 passes, d-sweep check, which
  panels were produced, and any anomalies, especially centre-pull slopes, which
  the hypothesis doc lists as a key failure mode.
```

## What to check when it reports back

- **Stage 1 must be 140/140.** Anything less means the regenerated predictions aren't the ones behind your results, and none of the new panels should be trusted. Bring me the failing rows.
- **Stage 3 must pick the same d as the saved primary d.** If it doesn't, the sweep isn't using the same selection rule.
- **Stage 2b must leave every existing result JSON byte-identical.** The runner change only adds a file. If anything else moved, the change touched computation and must be reverted.
- **Fig 7 picks its seed and window by fixed rules.** Panel d and Fig S3 show every seed, so nobody can say the example trajectory was cherry-picked. Keep those rules in the legend.
- **Fig 7 is the last main figure.** The failure-mode panels (error maps, center-pull, error CDFs) moved to Fig S4, so Fig 7 stays focused on the trajectories.
