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

# Four one-line statements for the deck.
PAGE1_SAY_ALOUD = [
    "1. 10 of 10 real sessions have at least one recurring neural target that is separable on held-out time.",
    "2. 9 of 10 keep that target under the PCA surrogate; split-population passes in 9 of 13 session-by-method runs; LDS surrogate 4 of 10.",
    "3. Plant-and-recover: every planted visit found, no false alarms.",
    "4. Hard-split at Report 2's d: 1 of 10; at d = 3 (post hoc): 2 of 10; stable-dimension count is the headline.",
]

PAGE1_FINDINGS = (
    "Every real session has a recurring, held-out-separable target "
    "(Fig 6.0, 6.2, 6.3). PCA defines targets (9 of 10 survive the surrogate; "
    "S03 is a boundary pass at 4 of 20 and occupies most of the cloud; S09 fails "
    "the population tests); LDS is the weaker density rule (4 of 10; Fig 6.3d). "
    "Split-population: 9 of 13 runs (Fig 6.4a). Pre-registered hard-split "
    "conjunction at Report 2's decoding d: 1 of 10; post hoc at d = 3: 2 of 10; "
    "n_stable < 2 in 4 of 10 at d = 3 (Fig 6.4b-c). S2 tile agreement is a little "
    "below S1 but above the N1 floor in 9 of 10 (Fig 6.0b, 6.4e). Window length: "
    "median r/sigma at 1 s is 1.56 times the 250 ms value, with 6 of 10 at or "
    "above 1.5 (Fig 6.4g)."
)

PAGE1_IMPLICATIONS = (
    "Recurring neural targets exist without behavior in the definition, and the "
    "population tests (surrogate, split-pop) carry most of the \"real state\" "
    "claim. Independent halves reproduce a low-dimensional subspace more "
    "reliably than a full Report 2 decoding space: count noise grows with d "
    "while structure sits in the leading dimensions. Longer windows raise tile "
    "resolution in the direction expected for structure, not pure noise."
)

PAGE1_CAVEATS = (
    "Within-session only; no behavioral description of the targets; beta animal "
    "not run; LDS hard-split level unrun; 250 ms resolution; S03 most of the "
    "cloud; S09 fails the population tests."
)

PAGE1_HOW_TO_READ = (
    "Start with Fig 6.0a (pass/fail) and 6.0b (numbers); then 6.1-6.5 for "
    "pipeline, targets, detection/surrogate, hard-split, and plant. Retired "
    "tests and the appendix table are after Fig 6.5."
)

# Board footnote lives only under Fig 6.0a (not on page 1).
PAGE1_BOARD_NOTE = (
    "Counts are per run (session by method). Surrogate, split-population and "
    "reproducibility cells use PCA; LDS appears only in its own surrogate row."
)

PAGE1_TABLE_COLS: list[str] = []  # unused; page-1 board removed

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

FIG61_ONE_LINE_RULES = [
    (name, pass_if) for name, _q, _how, pass_if, _fail in FIG61B_ROWS
]

FIG61_CAPTION = (
    "Left: the analysis steps. We start from the causal latents fitted for "
    "Report 2 (a linear dynamical system, LDS, with PCA beside it). On training "
    "time only we find dense regions, the candidate targets, train a detector on "
    "each, and evaluate it only on held-out time. Right: one-line pass rules; the "
    "full question/how/fail table is on the appendix page. PCA defines targets; "
    "the LDS density rule is the weaker definition. Behavior is never used to "
    "define a target. Two tests from the pilot were retired (supplement)."
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

SUPPLEMENT_ALIGNMENT_FIX = (
    "Hard-split alignment bug (fixed). An early pass applied the Procrustes "
    "rotation twice when scoring rigid coordinate error; the S0 interleaved-bin "
    "check caught it (e_med near 1 while e_aff near 0). Fixture tests are in "
    "tests/test_report6_location_pca.py."
)

SUPPLEMENT_PANELS_NOTE = (
    "The original panels (old Fig 6.3c and 6.4b) are kept here unchanged for the "
    "record. "
    + SUPPLEMENT_ALIGNMENT_FIX
)

BOARD_CRITERIA = [
    ("qualifies", "Recurs"),
    ("distinguishable", "Distinguishable"),
    ("null_b_pca", "Survives surrogate (PCA)"),
    ("null_b_lds", "Survives surrogate (LDS)"),
    ("split", "Split-population"),
    ("same_subspace", "Same subspace"),
    ("same_tiles", "Same tiles"),
    ("resolvable", "Resolvable"),
]

HARD_SPLIT_OUTCOME = (
    "At Report 2's decoding d the pre-registered conjunction passes in 1 of 10 "
    "sessions: count noise adds about one uncertain unit per dimension while the "
    "reproducible structure sits in the leading ones. In the targets' own space "
    "(d = 3, labeled post hoc) 2 of 10 pass; the stable-dimension count is the "
    "headline of this test."
)

WINDOW_EXPECTATION = (
    "Window length. The pilot bins are 250 ms. If the tiles capture a real state, "
    "median r/sigma at W = 1 s should be about twice the 250 ms value; if they are "
    "mostly count noise it stays near 1. Observed median ratio 1.56, with 6 of 10 "
    "at or above 1.5. The purge g scales with the new tau_ac and is recorded."
)

TILE_LOW_D_NOTE = (
    "k-means at the elbow M in the primary-d space is unstable even between "
    "interleaved halves in some sessions (S0 ARI 0.39 for one real session at "
    "d = 40): tiles belong in low d or in the stable subspace in the next round. "
    "At 250 ms a single bin places the state only to within about one tile."
)
