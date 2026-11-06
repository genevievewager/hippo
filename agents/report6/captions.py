"""Accessible figure captions for the Report 6 draft (non-expert readable).

Revision 2026-11-06 section B: insert as written; US spelling.
"""

from __future__ import annotations

from agents.report6.prose import FIG61_CAPTION, SUPPLEMENT_BOOT, SUPPLEMENT_NULL_A, SUPPLEMENT_PANELS_NOTE

CAPTIONS: dict[str, dict[str, str]] = {
    "F6_0_board": {
        "title": "Figure 6.0  Results at a glance",
        "takeaway": (
            "Every session has a recurring, separable neural target; the open "
            "questions are the surrogate, split-population agreement, and hard-split "
            "reproducibility."
        ),
        "caption": (
            "Each column is one recording session (S01-S10). Each row is a "
            "pre-registered test. Green = PASS, red = FAIL, dash = not run yet. "
            "Recurs: the target is visited often enough in training. "
            "Distinguishable: on held-out time the detector scores the target higher "
            "than neighboring regions. "
            "Survives the surrogate (PCA): after independently time-shifting each unit "
            "and refitting, success should be rare (at most 20 %). LDS has its own "
            "surrogate row (4 of 10); PCA defines targets. "
            "Split-population: two random halves of the units agree on when the state "
            "is in the target. "
            "Same subspace / same tiles / resolvable: the pre-registered hard-split "
            "conjunction (n_stable, ARI, r/sigma and frac<r). "
            "Counts are per run (session by method). "
            "Bottom boxes: headline rates for the lab meeting, plus plant-and-recover."
        ),
    },
    "F6_numbers": {
        "title": "Figure 6.0b  Session-by-session numbers",
        "takeaway": (
            "Hard-split columns are numbers at d = 3 and at Report 2's d, not "
            "PASS/FAIL cells."
        ),
        "caption": (
            "Per real session: occupancy %, visits, delta P; surrogate pass counts "
            "(PCA / LDS); split-population kappa; n_stable at d = 3 and at Report 2's "
            "d (N1 p95 in parentheses); median rigid coordinate error at d = 3; "
            "r/sigma at 250 ms and 1 s; ARI S1 and S2 at d = 3 (N1 p95 in parentheses "
            "for S1). Occupancy above 50 % is most of the cloud. "
            "Null (a) and bootstrap are retired (supplement)."
        ),
    },
    "F6_1": {
        "title": "Figure 6.1  What we are testing",
        "takeaway": (
            "A neural target is defined without behavior, then proven with held-out "
            "detection, population tests, and hard-split reproducibility."
        ),
        "caption": FIG61_CAPTION,
    },
    "F6_2": {
        "title": "Figure 6.2  Do qualifying targets exist?",
        "takeaway": (
            "Yes: every real and matched-sim session has at least one target that "
            "meets the occupancy and visit rules."
        ),
        "caption": (
            "Blue bars: fraction of training time spent in the top qualifying target "
            "(left axis). Orange points: number of separate visits (right axis; a new "
            "visit requires at least 1 s outside first). "
            "Dashed line: 5 % occupancy minimum; dotted line: 10-visit minimum. "
            "Gray markers: 1/M (uniform share if M equal tiles). Surrogate targets "
            "match the real targets' occupancy, so the recurrence rule filters "
            "candidates and is not evidence on its own; compare occupancy to 1/M. "
            "Panels a-b are real hippocampal-formation recordings (LDS and PCA). "
            "Panels c-d are matched simulations with the same unit count (subsample "
            "k = 0). Shared y-limits across panels. Any top target above 50 % "
            "occupancy is most of the cloud, not a specific state."
        ),
    },
    "F6_3": {
        "title": "Figure 6.3  Detection quality and the surrogate",
        "takeaway": (
            "Held-out detection is strong; the correlation-breaking surrogate is the "
            "test of whether coordination between units is required."
        ),
        "caption": (
            "(a) Separability on held-out time: green bars exceed the 0.10 delta-P "
            "threshold. "
            "(b) Calibration: each line is one session; the diagonal is perfect "
            "calibration. "
            "(c) Surrogate results for PCA at 20 surrogates per session (pass if at "
            "most 4 of 20 succeed). S03 is a pass at the boundary (4 of 20) and is "
            "also the session whose top target covers most of the cloud; S09 fails. "
            "(d) Surrogate results for LDS at 5 surrogates per session (pass if at "
            "most 1 of 5 succeed; 4 of 10 PASS). The LDS density rule is the weaker "
            "definition; PCA defines targets. "
            "The whole-series time-shift null is retired (supplement)."
        ),
    },
    "F6_4": {
        "title": "Figure 6.4  Population agreement and hard-split reproducibility",
        "takeaway": (
            "Pre-registered conjunction at Report 2's decoding d: 1 of 10; "
            "stable-dimension count is the headline; r/sigma vs W tests structure."
        ),
        "caption": (
            "(a) Split-population (per run). (b) n_stable against d with the N1 p95 "
            "floor (dotted) and the pre-registered n_stable ≥ 2 line (dashed). "
            "(c) Fraction of sessions passing each criterion against d; Report 2 "
            "primary d marked. (d) Per-axis rigid error for one example with the S0 "
            "ceiling. (e) ARI by session for S1, S2, S0 at d = 3 with N1 p95. "
            "(f) Hard-split conjunction pass counts: real 1/10 at Report 2 d and "
            "2/10 at d = 3; sim 0/10 at Report 2 d. "
            "(g) median r/sigma against W at d = 3: median 1 s / 250 ms = 1.56, "
            "with 6 of 10 at or above 1.5 (structure direction, not pure noise). "
            "(h) Example session: first two aligned axes, A-tiles colored, aligned "
            "B centroids overlaid, matched pairs joined. PCA targets are primary. "
            "Bootstrap panel retired (supplement)."
        ),
    },
    "F6_appendix": {
        "title": "Appendix  Drift, d-grid numbers, LDS split-pop, full test table",
        "takeaway": (
            "Numbers that do not fit the main figures: null-(a) drift check, "
            "hard-split across d, LDS split-population, and the full pass-rule table."
        ),
        "caption": (
            "(a) First vs second half training occupancy of the top LDS target for "
            "the null-(a) outlier session. (b) n_stable across the d grid and ARI at "
            "d = 3 for each real session (S1, u out). (c) LDS split-population kappa "
            "on its three sessions. (d) Full question / pass-if / fail table at "
            "readable type (≥ 7 pt)."
        ),
    },
    "F6_5": {
        "title": "Figure 6.5  Plant-and-recover (simulation sanity check)",
        "takeaway": (
            "When we insert a known population pattern, both PCA and LDS find it at "
            "the right times with no false alarms."
        ),
        "caption": (
            "On one matched-sim session we replaced spike counts in known time "
            "windows with a fixed population pattern (strong enough to meet the "
            "5 % occupancy rule), then refit and ran the same target+detector "
            "pipeline. "
            "(a) Hit rate: fraction of planted test windows that overlap a detected "
            "visit (PASS at least 0.5). "
            "(b) False alarms per minute of held-out time (PASS at most 2). "
            "This shows the pipeline can recover a genuine recurring state when one "
            "is present."
        ),
    },
    "F6_S_retired": {
        "title": "Supplement  Retired tests (pilot)",
        "takeaway": (
            "Two pilot tests are retired: the whole-series time shift cannot fail for "
            "a density-defined target, and the bootstrap refit reuses most of its data."
        ),
        "caption": (
            f"{SUPPLEMENT_NULL_A} "
            f"{SUPPLEMENT_BOOT} "
            f"{SUPPLEMENT_PANELS_NOTE}"
        ),
    },
}
