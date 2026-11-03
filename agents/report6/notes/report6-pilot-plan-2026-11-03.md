# Report 6 pilot plan (2026-11-03; lab meeting 2026-11-04)

Decision: focus on Report 6 (latent-target clamp) because the beta-BCI animal
has neural data only (camera to be added later). Pilot = M0–M2 + nulls +
plant-and-recover, on data whose causal latents already exist, deliverable by
the lab meeting.

## Decisions taken 2026-11-03 (user may override)
1. Manifold: LDS (causal; A5/A9 pass) primary, PCA beside it. d = Report 2 v2's
   selected d per session. Latents from the existing caches (Report 2 v2
   latent_cache / figure_contract; Report 1 matched-arm figure_contract),
   read-only; no refits for M0.
2. Target rule (M1): density peaks in training-block latent space (KDE on
   the first 3 latent dims, or k-means with k by the elbow rule, stated);
   qualify if occupancy ≥ 5 % of training time in ≥ 10 separate visits
   (visit = entry after ≥ 1 s outside); up to 3 targets per session.
3. Detector (M2): Mahalanobis distance to target centroid/covariance from
   training data → P(in target) via logistic calibration on training; enter /
   exit thresholds with hysteresis, minimum dwell 250 ms; evaluated on the
   held-out block only.
4. Nulls: (a) circular time shift of the latent relative to itself (20
   shifts); (b) correlation-breaking surrogate: independent per-unit circular
   shifts of spike counts, then refit the manifold (PCA always; LDS where
   time allows), rerun M1–M2.
5. Split-population test: two random disjoint halves of units; refit; agreement
   of in-target time series (Cohen's κ vs chance).
6. Refit stability: bootstrap blocks; Procrustes alignment; same region recovers
   the same activity patterns (overlap of in-target times).
7. Plant-and-recover: on a matched-sim session, add a recurring planted
   population pattern (known times); hit rate and false alarms.
8. Within-session only. No behaviour in target definition; M4 post-hoc
   description only (position / speed where behaviour exists).
9. Data: Report 2 v2 cohort (10 sessions, HF units) and the matched sim (10
   sessions, k = 0 subsample). Optional: beta animal's own 60-min KS4 output if
   copied to the server (spike times, clusters, labels; neural-only).

## Deliverables for the deck (by 2026-11-04 morning)
- F6.1 Idea + validation table (schematic, data-free).
- F6.2 Targets per session: occupancy, visits, dwell distributions (cohort,
  anonymised; sim beside).
- F6.3 Held-out detection vs neighbours; calibration; nulls (a) and (b).
- F6.4 Split-population agreement and refit stability.
- F6.5 Plant-and-recover on sim.
- F6.6 (if data copied) beta animal pilot: targets + detection, neural-only.
- One-page summary with counts and pass/fail per criterion; anonymised PDF;
  SHA256SUMS; nothing committed but code.

## Rules
Worktree hippo-r6 from report2-v2-phaseB @ 9730510 (latent cache present).
Read-only on all other worktrees and hippo/. Outputs under hippo-r6/outputs/
report6/ (gitignored). No new dependencies. tmux for anything > 5 min. No real
IDs anywhere outside gitignored local files. Figures via Report 1's package.
