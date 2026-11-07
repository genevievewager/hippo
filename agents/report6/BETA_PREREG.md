# Report 6 — beta-animal pre-registration

**Committed before any beta-animal data is read.** Data-free. Binding for the
first beta run. Nothing from the animal may appear in chat or in tracked files
(aliases only if a run is later reported).

The animal is not on the server yet. Do not run the pipeline on beta data until
this file is on the branch and the fixture tests below have passed.

---

## Binding choices

| Item | Choice |
| --- | --- |
| Targets | **PCA defines targets** (LDS may detect; it does not redefine the target set) |
| Target space | **d = 3** (first three PCA dimensions) |
| Stable-dimension count | **Reported**, not used to select d or the target set |
| Detection window | **W = 250 ms** |
| Reproducibility (hard-split) window | **W = 1 s** (secondary; 250 ms remains primary for detection) |
| Recurrence rule | **Filter only**: occupancy ≥ 5 % of training time and ≥ 10 visits (visit ends after ≥ 1 s outside). Show target occupancy next to **1/M** (uniform share if M equal tiles). Recurrence alone is not evidence. |
| Surrogate (null b) | **20** PCA surrogates; pass if at most 20 % succeed (≤ 4 of 20) |
| Split-population | Unchanged pass rule (Cohen kappa ≥ 0.2 and above shuffle) |
| Hard-split | **S1** (alternating 60 s blocks) and **S2** (first vs second half); pre-registered conjunction unchanged |
| Fixture gate | `tests/test_report6_location_pca.py` (and detector/visits fixtures) **must pass before the run** |

---

## Pre-registered pass rules (unchanged from the pilot / revision)

1. **Recurs** — ≥ 5 % training occupancy and ≥ 10 visits.
2. **Distinguishable** — held-out delta P ≥ 0.10 and Brier ≤ 0.25.
3. **Survives the surrogate** — ≤ 20 % of surrogates yield a qualifying, separable target (PCA: ≤ 4 of 20).
4. **Split-population** — kappa ≥ 0.2 and above shuffle.
5. **Hard-split conjunction (S1, u out)** —
   - same subspace: `n_stable ≥ 2` and `n_stable` above the N1 p95 floor;
   - same tiles: ARI above the N1 p95 floor;
   - resolvable: median r/σ_noise ≥ 2 and frac(shift < radius) ≥ 0.5.
6. **S2** — reported beside S1 (first vs second half); same metrics; not a second gate for pass/fail of the conjunction unless stated at analysis time in a dated amendment.

Coordinate error (rigid) remains a reported number with no pass rule.

---

## What is not allowed on this run

- Reading beta recordings, Kilosort exports, or caches into chat or notebooks
  before this commit exists on the branch.
- Writing real session or animal identifiers into tracked files, captions, or
  chat (use a reserved alias, e.g. `S_beta` / animal `G`, if results are later
  summarised).
- Changing d for target selection from the stable-dimension count.
- Changing the recurrence thresholds, surrogate fraction, split-pop kappa, or
  hard-split conjunction without a dated amendment committed first.

---

## Before launch checklist

1. This file is committed on `report6`.
2. Fixture tests pass:
   `pytest tests/test_report6_location_pca.py tests/test_report6_detector.py tests/test_report6_visits.py`.
3. Outputs only under `outputs/report6/` (gitignored); alias map gitignored.
4. No beta paths, spike files, or real IDs staged for commit.
