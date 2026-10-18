# Calibration plan (cheap design — draft, not approved)

See `calibration_space.yaml` v3 (`design: cheap_v1`). **Do not run** until approved.

## Knobs (k=6; dynamics fixed)

| # | Knob | Bounds | Source |
| - | ---- | ------ | ------ |
| 1 | `rate_scale_MEC` | [5, 40] Hz | guessed |
| 2 | `rate_scale_SUB` | [3, 30] Hz | guessed |
| 3 | `sorting_miss_rate` | [0.02, 0.40] | guessed |
| 4 | `sorting_fp_contam` (FP+contam coupled) | FP [0.001,0.05]; contam [0.01,0.25] | guessed |
| 5 | `sorting_jitter_ms` | [0.05, 1.5] | guessed |
| 6 | `recording_noise_std_uv` | [5, 40] | guessed |

Everything else fixed (incl. hippocampal dynamics / feedforward switches at default).

## Proxy

- 300 s, ~50 units in each animal's MEC/SUB proportions
- 1 session × 6 animals = 6 sims per θ, shared across LOAO folds

## Budget

- Morris r=4 → 28 θ
- LHS ≈ 10×k_kept → ~60
- GP 15 θ × 6 folds → 90
- Validation: ≤3 eligible full-size sessions per held-out animal

Hard cap: projected wall > 3 h at current `n_jobs` → STOP AND SHOW.

## Proxy fidelity gate

Default θ scored on proxy vs full matched; R must agree within bootstrap CI.
