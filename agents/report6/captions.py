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
            "Survives the surrogate: after independently time-shifting each unit and "
            "refitting, success should be rare (at most 20 %). "
            "Split-population: two random halves of the units agree on when the state "
            "is in the target. "
            "Same subspace / same tiles: independent fits on alternating halves of the "
            "session recover the same latent space and the same regions (hard-split "
            "test; filled when the amendment finishes). "
            "Counts are per run (session by method). Survives-surrogate, split-population and "
            "reproducibility cells use PCA until the LDS runs are complete. "
            "Bottom boxes: headline rates for the lab meeting, plus plant-and-recover."
        ),
    },
    "F6_numbers": {
        "title": "Figure 6.0b  Session-by-session numbers",
        "takeaway": (
            "Every session has a recurring, separable target; reproducibility columns "
            "are filled from the hard-split amendment."
        ),
        "caption": (
            "For each real session we report training occupancy and visit count of the "
            "top target, held-out separability (delta P), and PASS/FAIL for the "
            "surrogate, the split-population test, and the hard-split reproducibility "
            "rows (stable dimensions; coordinate error; ARI on S1 and S2). "
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
            "most 1 of 5 succeed). "
            "The whole-series time-shift null is retired (supplement)."
        ),
    },
    "F6_4": {
        "title": "Figure 6.4  Population agreement and hard-split reproducibility",
        "takeaway": (
            "Neuron-halves should agree on the target; independent halves of the "
            "session should recover the same subspace, coordinates, and tiles."
        ),
        "caption": (
            "(a) Split-population test (per run, session by method): two disjoint "
            "unit halves, each with its own latent fit; Cohen's kappa of in-target "
            "time series. Dashed line: kappa = 0.2. "
            "(b)-(e) Hard-split reproducibility (amendment): principal angles, "
            "coordinate error, ARI, matched-tile shifts, with an example session. "
            "(f) Matched sim condensed. "
            "The bootstrap refit panel is retired (supplement)."
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
