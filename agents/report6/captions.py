"""Accessible figure captions for the Report 6 draft (non-expert readable)."""

from __future__ import annotations

# Each entry: takeaway (bold lead) + caption body. n / PASS rates filled at build time.
CAPTIONS: dict[str, dict[str, str]] = {
    "F6_0_board": {
        "title": "Figure 6.0  Results at a glance",
        "takeaway": "Density-based neural targets always appear and can be detected, but a simple time scramble rarely removes them.",
        "caption": (
            "Each column is one recording session (S01–S10). Each row is a pre-registered "
            "test of whether a purely neural 'target' region in latent space is real. "
            "Green = PASS, red = FAIL. "
            "Recurs: the target is visited often enough in training (≥5% of time, ≥10 separate visits). "
            "Distinguishable: on held-out time, the detector scores the target higher than neighbouring regions. "
            "Fails null (a): after circularly time-shifting the latent trajectory, the same rule should "
            "rarely still 'succeed' — if it still succeeds, the target may just reflect static density, not a timed state. "
            "Fails null (b): after breaking correlations between neurons and refitting, success should be rare. "
            "Split-pop κ: two random halves of the neurons should agree on when the animal is in the target. "
            "Refit Jaccard: after refitting the latent model on bootstrap blocks, the same times should still be marked in-target. "
            "Bottom boxes: headline rates for the lab meeting, plus plant-and-recover hit rate on simulated data."
        ),
    },
    "F6_numbers": {
        "title": "Figure 6.0b  Session-by-session numbers",
        "takeaway": "Every session has a recurring, separable target; the main failure mode is the time-shift null.",
        "caption": (
            "For each real session we report training occupancy and visit count of the top LDS target, "
            "held-out separability (ΔP = mean detector probability on the target minus the best neighbouring region), "
            "and PASS/FAIL for the two nulls, the split-population test, and bootstrap refit stability. "
            "Occupancy is typically ~25–50% (S03 is an outlier at ~86%). "
            "ΔP is comfortably above the 0.10 pass line in every session. "
            "Null (a) passes in only one session; null (b) and bootstrap usually pass."
        ),
    },
    "F6_1": {
        "title": "Figure 6.1  What we are testing",
        "takeaway": "The goal is a neural target defined without behaviour, then proven with held-out detection and nulls.",
        "caption": (
            "Left: analysis steps. We start from already-fitted causal latents (linear dynamical system, LDS, "
            "and PCA as a companion). On training time only we find dense regions (candidate targets), "
            "then train a detector and evaluate it only on held-out time. "
            "Right: a target is accepted only if it passes every listed rule — recurrence, held-out separation, "
            "agreement across neuron halves, stability under refits, failure of structured nulls, "
            "and recovery of a planted pattern in simulation. Behaviour is never used to define the target."
        ),
    },
    "F6_2": {
        "title": "Figure 6.2  Do qualifying targets exist?",
        "takeaway": "Yes — every real and matched-sim session has at least one target that meets the occupancy and visit rules.",
        "caption": (
            "Blue bars: fraction of training time spent in the top qualifying target (left axis). "
            "Orange points: number of separate visits (right axis; a new visit requires ≥1 s outside first). "
            "Dashed line: 5% occupancy minimum; dotted line: 10-visit minimum. "
            "Panels a–b are real hippocampal-formation recordings (LDS and PCA). "
            "Panels c–d are matched simulations with the same unit count (subsample k = 0). "
            "Real sessions show more variable visit counts than sim; S03 has unusually high occupancy."
        ),
    },
    "F6_3": {
        "title": "Figure 6.3  Detection quality and null tests",
        "takeaway": "Held-out detection is strong, but circular time shifts usually still look like 'real' targets — correlation-breaking is the tougher null.",
        "caption": (
            "(a) Separability on held-out time for LDS: green bars exceed the 0.10 ΔP threshold in all sessions. "
            "(b) Calibration: each line is one session; the diagonal is perfect calibration. "
            "Curves generally rise with predicted probability but sit somewhat below the diagonal (over-confident scores). "
            "(c) Null (a): number of time-shifted copies (out of 20) that still pass qualify+detect. "
            "PASS requires ≤4; only S04 clears this bar — so for most sessions the density rule is not killed by scrambling time. "
            "(d) Null (b): after independently time-shifting each neuron's spike counts and refitting, "
            "PASS requires ≤1 of 5 surrogates to succeed. Most PCA runs pass; a few sessions (notably S06, S09) fail."
        ),
    },
    "F6_4": {
        "title": "Figure 6.4  Is the target a population state that survives refits?",
        "takeaway": "Neuron-halves usually agree, and bootstrap refits recover the same in-target times.",
        "caption": (
            "(a) Split-population test: units are split into two disjoint halves, each half gets its own latent fit, "
            "and we ask whether their binary in-target time series agree (Cohen's κ). "
            "Dashed line: κ = 0.2 minimum; green = PASS (also above a shuffle chance level). "
            "(b) Bootstrap stability: refit on resampled training blocks, align with Procrustes, "
            "and measure overlap (Jaccard) of held-out in-target times with the original. "
            "Dashed line: median Jaccard = 0.3; all runs shown pass. "
            "LDS refits were limited to the three smallest sessions for runtime; PCA covers all ten."
        ),
    },
    "F6_5": {
        "title": "Figure 6.5  Plant-and-recover (simulation sanity check)",
        "takeaway": "When we insert a known population pattern, both PCA and LDS find it at the right times with no false alarms.",
        "caption": (
            "On one matched-sim session we replaced spike counts in known time windows with a fixed population pattern "
            "(strong enough to meet the ≥5% occupancy rule), then refit and ran the same target+detector pipeline. "
            "(a) Hit rate: fraction of planted test windows that overlap a detected visit (PASS ≥ 0.5). "
            "Both methods reach 1.0. "
            "(b) False alarms per minute of held-out time (PASS ≤ 2). Both are 0. "
            "This shows the pipeline can recover a genuine recurring state when one is present."
        ),
    },
}
