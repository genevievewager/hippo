# Report 6 pilot plan (pre-registration)

**Committed before any target computation.** Binding decisions from
`notes/report6-pilot-plan-2026-11-03.md` (2026-11-03). Lab meeting 2026-11-04.

Worktree: `hippo-r6` @ `9730510` (report2-v2-phaseB). Outputs only under
`outputs/report6/` (gitignored). Anonymised labels only: sessions S01–S10,
animals A–F. No real session or animal IDs in tracked files, captions, or chat.

---

## 0. Data (read-only sources)

| Arm | Source (absolute, read-only) | n |
| --- | --- | --- |
| Real cohort | `/home/gmwager/projects/hippo-r2v2/outputs/real_quadrant/m3_v2/room_A/*/latent_cache` + `figure_contract` | 10 |
| Matched sim, k = 0 | `/home/gmwager/projects/hippo-r1/outputs/report1_revision/matched_sim/*/subsample_00/figure_contract` | 10 |
| Spike counts (real) | `/home/gmwager/projects/hippo-r2v2/outputs/real_quadrant/cache/<hash>.npz` (`X_counts`), matched by `(T, n_hf)` | — |
| Spike counts (sim) | `arrays.npz` key `X_counts` | — |
| Optional beta | `outputs/report6/beta_animal/` Kilosort export if present | 0–1 |

Alias map (real IDs → S01–S10 / A–F) lives only under `outputs/report6/` (gitignored).

Splits and purge: reuse Report 2 `train_mask`, `test_mask`, `train_ok`,
`retained`, `valid` from `figure_contract/arrays.npz`. No re-splitting.

---

## 1. M0 — Manifold (no refit for primary latents)

- **Primary:** LDS (causal; A5/A9 by construction — filter-only
  `transform(..., causal=True)`, `causal_default=True` in
  `realtime/quadrant_n5_run.py`; A13 PASS in cohort `a13.json`).
- **Beside:** PCA (`global_pca`).
- **d:** Report 2 v2 `primary_d` per session from `latent_cache/final/{lds,pca}.npz`.
- **Latents:** load `Z` from cache / `figure_contract/latents.npz` (`Z_lds`, `Z_pca`).
  Slice to `[:, :primary_d]` when the stored array is wider than selected d.
- Target definition and detection use the **first 3 latent dims** (or all if
  `primary_d < 3`).

Refits (null b, split-population, bootstrap, plant-and-recover) call existing
factories only — never reimplement:

- `realtime/quadrant_n5_run.py` `make_model` / `fit_transform_representation`
  (PCA ~L347, LDS ~L366, fit ~L407)
- Underlying: `realtime/manifolds/pca.py`, `realtime/dynamic_latents/lds.py`

---

## 2. M1 — Target rule (training block only)

**Candidates:** k-means on training latents in the first `min(3, d)` dims.
- k chosen by **elbow** on inertia for k ∈ {2, 3, 4, 5, 6}; pick smallest k
  where relative inertia drop `(I_{k-1} − I_k) / I_2 < 0.10` (else k = 4).
- Seed: `20261103`. Standardise dims on train only before clustering.

**Qualify** a cluster as a target if, on the training block (`train_ok`):

| Parameter | Value |
| --- | --- |
| Occupancy | ≥ 5 % of training samples assigned to the cluster |
| Visits | ≥ 10 separate visits |
| Visit definition | entry after ≥ 1.0 s outside (contiguous out-of-cluster) |
| Max targets / session | 3 (rank by occupancy descending among qualifiers) |

Report all candidates (occupancy, visits, mean/median dwell) even if they fail
qualification. If fewer than 1 qualifier, mark session `no_qualifying_target`
and skip M2 for that session/method (still count in pass/fail table).

---

## 3. M2 — Detector (thresholds on train; eval on held-out only)

1. Target centroid μ and covariance Σ from training samples in the target
   cluster (shrinkage: `sklearn.covariance.LedoitWolf` on train target samples;
   if < 3d + 5 samples, use diagonal variance + 1e-3 floor).
2. Mahalanobis distance → logistic calibration on train:
   `P(in) = σ(a − b · D²)` with `(a, b)` fit by logistic regression on train
   (label = in-target cluster membership). Clip P to [1e-6, 1 − 1e-6].
3. **Hysteresis** (set on train so train dwell matches cluster visits order of
   magnitude; fixed defaults if search fails):
   - enter if `P ≥ 0.70` and not in; exit if `P ≤ 0.40` and in;
   - minimum dwell 250 ms (5 samples at 50 ms update).
4. **Held-out evaluation** on `test_mask` ∩ `retained` ∩ `valid`:
   - separability: mean P on target vs mean P on each neighbouring cluster
     (same k-means labels mapped by nearest train centroid);
   - calibration: reliability curve (5 bins) on held-out;
   - binary in/out time series after hysteresis.

**Pass (distinguishable):** held-out mean P(target) − mean P(best neighbour) ≥ 0.10
and Brier score on held-out ≤ 0.25.

---

## 4. Nulls

### (a) Circular time shift (no refit)

- 20 shifts of the latent time series relative to the target definition
  timeline: shift Z by `⌊i · T / 21⌋` samples, i = 1…20 (never 0).
- Re-run M1–M2 on shifted Z with **same** train/test masks.
- **Pass null:** fraction of shifts that produce a qualifying target **and**
  pass the distinguishable criterion ≤ 0.20 (≤ 4 / 20).

### (b) Correlation-breaking surrogate (refit)

- Independent circular shift per unit of `X_counts` (same shift grid size N).
- Refit manifold at selected d via `fit_transform_representation`; re-run M1–M2.
- **N_surrogate = 5** (runtime cut; registered here).
- **PCA:** all 10 real + all 10 sim sessions.
- **LDS:** 3 real sessions with smallest `n_units` only (runtime cut).
- **Pass null:** ≤ 1 / 5 surrogates pass qualify + distinguishable.

---

## 5. Split-population κ

- One random partition of units into two disjoint halves (seed `20261103 + 1`).
- Refit each half at selected d (PCA all sessions; LDS on the same 3 as null b).
- Align neither half to the full-pop target; instead define target on half A
  train, detect on half A and half B held-out after Procrustes of half-B Z to
  half-A train cloud; compute Cohen's κ of binary in-target series on the
  intersection of valid times.
- Chance: κ under 20 label shuffles of half-B.
- **Pass:** κ ≥ 0.2 and κ > 95th percentile of chance.

---

## 6. Refit stability (bootstrap + Procrustes)

- Bootstrap: resample contiguous train blocks (10 blocks of equal length;
  sample blocks with replacement) — **B = 10** (PCA all); **B = 5** (LDS, 3 sessions).
- Refit; Procrustes-align bootstrap Z to original train Z; map original target
  through alignment; overlap = Jaccard of in-target times on held-out.
- **Pass:** median Jaccard ≥ 0.3 across bootstraps.

---

## 7. Plant-and-recover (sim only)

- One matched-sim session (S01 of the sim alias list).
- Plant: add a fixed population pattern `p` (unit vector in count space, norm
  scaled to 2 × median train count L2) at 40 known windows of 500 ms, spaced
  through the session (≥ 5 s apart), half in train and half in test by time.
- Refit PCA (and LDS if time) on planted X; run M1–M2.
- **Hit rate:** fraction of planted test windows overlapping a detected visit.
- **False alarms:** detected visits with no planted window within 1 s /
  session-minutes of test.
- **Pass:** hit rate ≥ 0.5 and false-alarm rate ≤ 2 / min.

---

## 8. M4 — Post-hoc description only

Where `y` (position) exists: for each qualifying target, report mean speed
proxy (`‖Δy‖/Δt`) and positional occupancy on a coarse grid during in-target
vs out. Never used to define or select targets.

---

## 9. Metrics summarised for the deck

Per session × method (LDS primary, PCA beside):

1. n_candidates, n_qualifying
2. occupancy, visits, dwell stats for top target
3. held-out ΔP vs best neighbour; Brier; pass/fail distinguishable
4. null (a) pass/fail; null (b) pass/fail (where run)
5. split κ pass/fail (where run)
6. bootstrap Jaccard pass/fail (where run)

**Three numbers to say aloud** (computed after all runs; placeholders until then):

1. Fraction of real sessions with ≥ 1 qualifying LDS target.
2. Fraction of those that pass held-out distinguishable **and** null (a).
3. Plant-and-recover hit rate on the sim session.

---

## 10. Figures and packaging

- F6.1–F6.5 required; F6.6 if beta KS export present.
- Report 1 figure style (mm widths, white face, lettered panels, full captions,
  takeaway titles, n stated).
- One-page summary PDF + SHA256SUMS + caption number audit + `pdftotext` ID grep.
- Figures/tables/PDFs never committed.

---

## 11. Runtime cuts (explicit)

Registered because full LDS nested-CV history is ~15 min/session but a **single**
final fit is ~47 s; still:

| Step | PCA | LDS |
| --- | --- | --- |
| M0–M2 + null (a) | 10 real + 10 sim | 10 real + 10 sim |
| Null (b) N=5 | all 10+10 | 3 real (smallest n_units) |
| Split κ | all | same 3 |
| Bootstrap | B=10 all | B=5 on same 3 |
| Plant-and-recover | 1 sim | 1 sim if wall allows else PCA-only |

Workers ≤ 6, `nice -n 19`, tmux session `r6pilot` for jobs > 5 min.

---

## 12. Pass/fail criteria table (deck F6.1)

| Criterion | Test | Pass rule |
| --- | --- | --- |
| Recurs | occupancy + visits | ≥ 5 % and ≥ 10 visits |
| Distinguishable | held-out ΔP, Brier | ΔP ≥ 0.10, Brier ≤ 0.25 |
| Population state | split κ | κ ≥ 0.2 and > chance 95 % |
| Survives refitting | bootstrap Jaccard | median ≥ 0.3 |
| Fails null (a) | 20 time shifts | ≤ 4/20 false passes |
| Fails null (b) | 5 surrogates | ≤ 1/5 false passes |
| Plant-and-recover | sim planted windows | hit ≥ 0.5, FA ≤ 2/min |

---

## Changelog

- 2026-11-03: initial pre-registration (Phase 0 complete; no targets computed yet).
