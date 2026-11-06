"""Report 6 revision prose (2026-11-06 section B). Data-free; insert as written."""

from __future__ import annotations

# US spelling throughout (revision D3).

PAGE1_QUESTION = (
    "Question. Can we pick a region of the neural population state (a \"target\") "
    "using only spike activity (not position or speed) and still show that the "
    "region is a real, recurring state that can be detected on held-out time?"
)

PAGE1_WHAT_WE_DID = (
    "What we did. On 10 real hippocampal-formation sessions we took the causal "
    "LDS and PCA latents already fitted for Report 2. On training time we "
    "clustered the first three latent dimensions, kept the clusters the state "
    "occupies for at least 5 % of the time in at least 10 separate visits, trained "
    "a detector on each, and scored it only on held-out time. We then asked "
    "whether each target is a property of the population (two tests), whether it "
    "survives a fully independent refit on the other half of the session (three "
    "measures), and whether the pipeline finds a planted state in simulation."
)

PAGE1_TESTS_HEADER = "The tests, one line each."

PAGE1_TESTS = [
    (
        "Recurs",
        "the state keeps coming back to the region (at least 5 % of training "
        "time, at least 10 visits).",
    ),
    (
        "Distinguishable",
        "on held-out time, a detector tells the region from its neighbors.",
    ),
    (
        "Survives the surrogate",
        "when each unit's activity is shifted in time on its own, which keeps "
        "every unit's rate but destroys coordination between units, the region "
        "should disappear. If it does not, single units could have made it.",
    ),
    (
        "Split-population",
        "two random halves of the units, fitted separately, agree about when "
        "the state is in the region.",
    ),
    (
        "Reproducible",
        "fitted again on the other half of the session and aligned, the two "
        "halves span the same subspace, put each moment at the same "
        "coordinates, and find the same tiles.",
    ),
    (
        "Plant-and-recover",
        "a known state inserted into simulated data is found at the right "
        "times with no false alarms.",
    ),
]

PAGE1_SAY_ALOUD = [
    "1. 10 of 10 real sessions have at least one recurring neural target that is "
    "separable on held-out time.",
    "2. 9 of 10 real sessions keep that target when coordination between units is "
    "destroyed (PCA surrogate at 20: S03 passes at the boundary, 4 of 20, and is "
    "also the session whose top target covers most of the cloud; S09 fails the "
    "surrogate test); 9 of 13 session-by-method runs pass the split-population test.",
    "3. Plant-and-recover in simulation: every planted visit found, no false "
    "alarms.",
    "4. Reproducibility across independent halves: [filled in from the amendment's "
    "results table; the sign count over animals].",
)

PAGE1_READING_GUIDE = (
    "Figure 6.0 is the pass/fail board: start there. Figure 6.0b lists the raw "
    "numbers. Figures 6.1-6.5 walk through the pipeline, targets, detection and "
    "surrogate, population agreement and hard-split reproducibility, and "
    "plant-and-recover. Retired tests (whole-series time shift; bootstrap refit) "
    "are on the supplement page. Each figure page has a bold takeaway and a "
    "caption written for a reader who has not followed the project day to day."
)

PAGE1_BOARD_NOTE = (
    "PASS = meets the pre-registered rule for that test. Dash = not run yet. "
    "Counts are per run (session by method). Survives-surrogate, split-population "
    "and reproducibility cells use PCA until the LDS runs are complete."
)

PAGE1_TABLE_COLS = [
    "Session",
    "Recurs",
    "Distinguishable",
    "Survives surrogate",
    "Split-pop",
    "Same subsp.",
    "Same tiles",
]

# Fig 6.1b columns: test, question, how, pass if, a fail means
FIG61B_ROWS: list[tuple[str, str, str, str, str]] = [
    (
        "Recurs",
        "Does the state keep coming back here?",
        "Cluster training-time latent points (k-means on the first three dimensions); "
        "count occupancy and separate visits (a visit ends after at least 1 s outside)",
        "at least 5 % of training time and at least 10 visits",
        "the region is visited too rarely to hold",
    ),
    (
        "Distinguishable",
        "On new data, can a detector tell this region from its surroundings?",
        "Mahalanobis detector fitted on training time, calibrated to a probability, "
        "with hysteresis; on held-out time compare the mean probability on the target "
        "with the best neighboring region (delta P) and check calibration (Brier)",
        "delta P at least 0.10 and Brier at most 0.25",
        "on new data the region blends into its neighbors",
    ),
    (
        "Survives the surrogate (null b)",
        "Could uncoordinated units produce this region on their own?",
        "Shift each unit's count series by its own random offset, refit the latent, "
        "rerun the two tests above; 20 surrogates (PCA), 5 (LDS)",
        "at most 20 % of surrogates yield a qualifying, separable target",
        "the target is no better than what single units do alone; probably carried "
        "by one or a few units",
    ),
    (
        "Split-population",
        "Is it a population state or one loud unit?",
        "Two random disjoint halves of the units, each with its own latent, target "
        "and detector; agreement of their in/out time series (Cohen's kappa)",
        "kappa at least 0.2 and above shuffle",
        "the halves disagree about when the state is in the target",
    ),
    (
        "Reproducible: same subspace",
        "Do two independent halves of the session find the same latent space?",
        "Alternating 60 s blocks, purged at the edges, to halves A and B; PCA fitted "
        "on each; principal angles between the two subspaces",
        "at least 2 stable dimensions (angle under 30 degrees), more than the "
        "surrogate gives",
        "the latent space itself does not reproduce",
    ),
    (
        "Reproducible: same coordinates",
        "Does each moment land in the same place on both maps?",
        "Rotate map B onto map A from the loadings alone; distance between a "
        "moment's two coordinates, scaled by the cloud size, shown against the "
        "count-noise floor and the interleaved-bin ceiling",
        "reported as a number, no pass rule",
        "large: the coordinate system is unstable within the session",
    ),
    (
        "Reproducible: same tiles",
        "Is \"here\" the same place?",
        "Tiles fitted on each half and matched by centroid; centroid shift, "
        "membership agreement (Jaccard, adjusted Rand)",
        "agreement above the surrogate's 95th percentile; at least half the tiles "
        "move less than their own radius; tiles larger than twice the noise floor",
        "the regions do not survive an independent refit",
    ),
    (
        "Plant-and-recover",
        "If a real recurring state exists, does the pipeline find it?",
        "Overwrite known windows in a simulated session with a fixed pattern, "
        "rerun everything, compare detected visits with the planted windows",
        "hit rate at least 0.5, at most 2 false alarms per minute",
        "the pipeline misses even an obvious state",
    ),
]

FIG61_CAPTION = (
    "Left: the analysis steps. We start from the causal latents fitted for "
    "Report 2 (a linear dynamical system, LDS, with PCA beside it). On training "
    "time only we find dense regions, the candidate targets, train a detector on "
    "each, and evaluate it only on held-out time. Right: every test the target has "
    "to pass, the question it answers, and what a failure would mean. The first "
    "two tests say the region exists and can be found on new data. The next two "
    "say it belongs to the population rather than to a few units. The three "
    "reproducibility rows say the same place is found again when everything is "
    "refitted on the other half of the session, which is the error bar on \"here\". "
    "The last row is a sanity check in simulation. Behavior is never used to "
    "define a target. Two tests from the pilot were retired (supplement): the "
    "whole-series time shift, which cannot fail for a region defined by density, "
    "and the bootstrap refit, which reuses most of the data it is tested against."
)

SUPPLEMENT_NULL_A = (
    "Null (a), whole-series time shift (retired). Shifted the whole latent "
    "trajectory by one circular offset, 20 offsets per session, and reran qualify "
    "and detect; passed if at most 4 of 20 shifted copies still passed. Because the "
    "offset moves every point together, the cloud, its dense regions, occupancy, "
    "visits and dwell times are unchanged, so the shifted copies pass by "
    "construction. The pilot's 1 of 10 therefore says nothing about the data. The "
    "one passing session is the one whose target occupancy differs most between "
    "the first and second half of the session (numbers in the appendix), which is "
    "within-session drift, now measured by the first-half versus second-half split."
)

SUPPLEMENT_BOOT = (
    "Bootstrap refit stability (retired). Refit the latent on resampled "
    "training blocks, aligned by Procrustes, and measured the overlap (Jaccard) of "
    "held-out in-target times with the original; passed at median Jaccard of at "
    "least 0.3. Each refit shares most of its training data with the original and "
    "is scored on the same held-out block, so overlap is largely guaranteed; all "
    "13 runs passed with Jaccard 0.75 to 0.97. Replaced by the hard-split test."
)

SUPPLEMENT_PANELS_NOTE = (
    "The original panels (old Fig 6.3c and 6.4b) are kept here unchanged for the "
    "record."
)

# Board rows (Fig 6.0a / page-1 table), revision B5
BOARD_CRITERIA = [
    ("qualifies", "Recurs"),
    ("distinguishable", "Distinguishable"),
    ("null_b", "Survives surrogate"),
    ("split", "Split-population"),
    ("same_subspace", "Same subspace (A vs B)"),
    ("same_tiles", "Same tiles (A vs B)"),
]
