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
            "Every session has a recurring, separable target; hard-split columns use "
            "the pre-registered conjunction at Report 2's decoding d."
        ),
        "caption": (
            "For each real session we report training occupancy and visit count of the "
            "top target, held-out separability (delta P), and PASS/FAIL for the "
            "PCA and LDS surrogates, the split-population test, and the hard-split "
            "rows (same subspace; same tiles; resolvable). "
            "Occupancy is typically about 25-50 % of training time; any top target "
            "above 50 % is called most of the cloud. "
            "Null (a) and bootstrap refit are retired (see supplement)."
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
            "(a) Split-population (per run). (b) n_stable against d (S1, u out). "
            "(c) Fraction of sessions passing each criterion against d; Report 2 "
            "primary d marked. (d) Per-axis rigid error for one example session "
            "with the S0 ceiling. (e) ARI by session for S1, S2, S0 at d = 3 with "
            "N1 p95. (f) Sim condensed and real pass counts. (g) median r/sigma "
            "against W at d = 3 with the line at 2; if tiles capture state, the "
            "1 s value should be about twice the 250 ms value. (h) Tile-match "
            "reminder. PCA targets are primary. Bootstrap panel retired (supplement)."
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
