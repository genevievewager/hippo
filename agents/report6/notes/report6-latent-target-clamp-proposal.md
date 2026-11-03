# Report 6 (proposed): latent-space target clamp

Status: **PROPOSED, not started.** Written 2026-10-13. Follows Report 5 (see `updates/2026-10-06-status-and-scope.md`). Can start on a generic latent before Report 5 finishes; swaps in Report 5's manifold when it exists.

## The idea

1. Fit a **manifold**: a low-dimensional latent space that the population activity lives on.
2. Pick a **target**: a region of that manifold. It is defined by neural activity alone and is **not tied to any behavioural variable**.
3. Account for **coordinate transformations**: latent coordinates are not fixed. They can rotate, flip or shift between fits, time blocks, sessions and days, and as units drift. The target has to stay the same *place on the manifold*, not the same coordinates.
4. **Detect** the target causally at every time point: is the neural state in the target region now?
5. Eventually, **hold** the state there (closed loop): the clamp.

This fits the 2026-10-05 direction. What was ruled out there was a separate detector per *behavioural* zone. Here the target is a region of neural state space, which is the project's stated goal: hold onto a target state in neural state space.

## Why now

- Report 2: the leading latent dimensions of every generic method are largely non-spatial.
- Report 4a: measured behaviour explains little of real neural activity, and few real units are tuned.

So most of the structure in real activity is not about the measured behaviour. A purely neural target is the natural way to work with that structure. It is also the case where behaviour-based decoding cannot help.

Report 4a also found that real place maps drift within a session while simulated ones do not. Alignment is therefore a core problem even within one session.

## Question

> Can a region of the neural manifold be defined so that it is a real, recurring population state; detected causally at every time point; and tracked as the same region across coordinate changes (within session first, then across days)?

## The central problem: validation without a behavioural ground truth

The existence of a region on a fitted manifold does not make it real. A target counts as real only if it passes all of these:

| criterion | test |
| --- | --- |
| **Recurs** | the state enters the region repeatedly across the session (number of separate visits, spread over time), not once |
| **Distinguishable** | on held-out data, the region is separable from its neighbours beyond noise (detection score vs neighbouring regions; calibration) |
| **Population state** | split the units into two disjoint random halves; each half's detector agrees on when the state is in the target, beyond chance |
| **Survives refitting** | refit the manifold (bootstrap, different time block); after alignment, the same region captures the same activity patterns |
| **Fails the nulls** | (a) circular time-shifted data; (b) surrogates that keep each unit's rate and autocorrelation but break correlations between units (independent per-unit circular shifts). Neither should yield a target that passes the criteria above |
| **Plant and recover** | in synthetic data, and in the simulator, plant a known population state and confirm the method finds it at the right times |

The simulator is a weak testbed for anything beyond plant-and-recover: its activity is almost entirely behaviour-driven and its maps are unrealistically stable (Report 4a).

## Design

**Data.** The Report 2 v2 cohort (hippocampal-formation units), same splits, purge, trims and causal [t − W, t) counts from spike times. The live mouse once recorded. Sim and synthetic data for plant-and-recover only.

**M0. Manifold.**
- Start with a generic causal latent from Reports 1–2 (e.g. PCA or LDS on causal counts; `lds` is causal and passes A5/A9). Swap in Report 5's manifold when it exists.
- Fitted on training blocks only; causal at every step (A5 future-perturbation and A9 per-step == batch must pass).
- Dimensionality chosen by a fixed rule (held-out reconstruction or Report 2's doubling rule with plateau check). Not tuned on target results.

**M1. Target definition by a pre-registered rule.**
- Candidate regions come from training data only, for example density peaks or clusters in the latent space.
- A candidate qualifies if it is occupied ≥ X % of training time in ≥ K separate visits. X and K are fixed before results.
- Evaluate **several** qualifying targets per session, not one, so the result cannot rest on picking the best-looking region.

**M2. Detection at every time point.**
- A causal score: distance to the region or membership probability, from the current latent state.
- Threshold and hysteresis (separate enter and exit thresholds, minimum dwell) set on training data.
- Output: a time series of P(in target) and binary in/out, on held-out data.

**M3. Coordinate transformations.**
- **Within session (first):** the units are the same, so the latent can be tracked or refit across blocks. Compare detection consistency between early and late blocks with and without alignment.
- **Across days (second):** units change, so alignment must come from the structure of the activity (e.g. Procrustes or optimal transport between latent point clouds, density-structure matching). Behaviour may be used as an alignment *tool* if the user approves, but it never defines the target.
- **Across animals:** deferred.

**M4. Post-hoc description (never used to define targets).**
- What the animal is doing while in each target state: position, speed, movement state, head direction. Descriptive only.

**M5. Closed-loop readiness.**
- Latency per step, false alarms per minute, dwell fidelity, entry rate, on replay.
- Requires the pressure-test buffer fixes first.
- Design of the clamp experiment itself (feedback modality, yoked controls with the same event count but not contingent on the neural state, necessity vs sufficiency tiers from the 2026-10-05 direction) is planned with the user before anything runs live.

## Metrics

- Target occupancy fraction, number of visits, dwell-time distribution.
- Held-out detection: separability from neighbours, calibration of P(in target).
- Split-population agreement (above chance).
- Refit stability after alignment.
- Null results (time-shift; correlation-breaking surrogate).
- Plant-and-recover hit rate and false alarms.
- Within-session and across-day detection consistency, with and without alignment.
- Operating metrics for closed loop: latency, false alarms per minute, time-to-detect.

## Scope guards

- Report 6 does not build or compare representations; it uses Report 5's manifold, or a generic Reports 1–2 latent until then.
- Targets are defined on training data only; held-out data is never used to choose targets or thresholds.
- No behavioural variable defines a target.
- No new dependencies without asking.
- Hyperparameter tuning for deployment belongs to Report 3.

## Decisions for the user before starting

1. The starting manifold (PCA vs LDS on causal counts) until Report 5 delivers.
2. Target rule parameters: occupancy threshold X, minimum visits K, how many targets per session.
3. Whether behaviour may be used as an alignment tool across days (never to define targets).
4. The order of transformations: within session, then across days (proposed).
5. What "holding" means for the eventual experiment (feedback type), to be decided later; it does not block M0–M4.

## Milestones (proposed)

- M0 manifold + causality checks.
- M1 target rule + candidate targets per session.
- M2 causal detector + held-out validation (all criteria in the table above).
- M3 within-session alignment; then across days.
- M4 post-hoc behavioural description.
- M5 closed-loop readiness on replay; then on the live mouse, after the deployment track's prerequisites.
- Report to the figure standard (`claude/report-figure-standard.md`).
