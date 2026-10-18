# Leave-one-animal-out + shared generation (revised with calibration_space v2)

## Animals (6 folds)

{1120, 1130, 1140, 1150, 1170, 7394}

## Shared generation

Each θ is simulated **once** under matched conditions for every animal
(≥3 replicate seeds where needed). Fold *k* scores θ only on sims matched to
its calibration animals vs those animals' lab data. Held-out animal data are
touched only at validation after `calibration_log` is locked (C1). Validation
seeds disjoint. **Only GP is per fold.**

## Budget (see `calibration_space.yaml`)

| Stage | Evals | wall @ n_jobs=12 (per_θ≈112.8 s) |
| --- | ---: | ---: |
| Morris screening r=4, k=75 | 4×76 = **304** | **~0.79 h** |
| Shared LHS ≈10×k_kept (assume 20) | **200** | **~0.52 h** |
| GP 15 θ × 6 folds | **90** | **~0.24 h** |
| Validation 6×3 seeds | **18** | **~0.05 h** |
| **Total (approx)** | **612** | **~1.6 h** |

Rescale LHS when k_kept is known. Categorical dynamics arms are discrete strata
on top of this continuous budget.

## Bootstrap grouping

`recording_day = animal_DDMMYYYY`. Behaviour metrics: if two sessions share
identical `postions_dataset.csv` SHA-256, assign a common `recording_id` and
count behaviour once; otherwise keep separate sessions under the same day group.
