# Report 4a — Results (v8)

**Tag:** `report4a-v8` · **PDF:** `reports/report4/report4a_full.pdf` (+ anon) · **Build:** `hippo-dq/outputs/data_quality/reports/report4a_v8/`

## v8 figure map

| Page | Content |
|------|---------|
| 1 | Prose story only (version line: Report 4a v8, tag report4a-v8) |
| 2 | **Fig 1** — Behavioural information (ex-Fig 3): decode info at N=20, position vs N ladder, region splits, VE share, error/floor vs N, paired N=20, reweighted sensitivity |
| 3 | **Fig 2** — Tuning and drift: rate maps, place-tuned fraction / bits, adjacent-block reliability, within-session drift |
| 4 | **Fig 3** — Spikes and behaviour (ex-Fig 1): ISI / refractory, speed, turning, still, HD–movement residual |
| 5 | **Fig 4** — What matches: paired strips for unit / population metrics with \(R \le 1\) |
| 6 | **Fig 5** — Divergence \(R\) + simulator to-do |
| 7 | **Fig S1** — Extra distributions / sensitivity |
| 8 | **Appendix** — Gates / eligibility |

Numbers frozen at v5 tidy + `n_curve_ladder.csv`. Audits (prose, number, anon leak, overlap, whitespace, duplicate) PASS for v8.

## N-ladder (lab position information)

Session-median net decode information (position), ladder \(N \in \{10, 20, 40, 80, 160, \max\}\).

| \(N\) | Lab | Matched |
|------:|----:|--------:|
| 10 | −0.113 | 0.081 |
| 20 | −0.080 | 0.178 |
| 40 | −0.007 | 0.373 |
| 80 | 0.081 | 0.527 |
| 160 | 0.169 | 0.681 |
| max | 0.227 (≈184) | 0.756 (≈415) |

Lab zero-crossing (log-\(N\) interp between 40 and 80): \(N \approx 42\).

## The question

Before using the simulator as a stand-in for real data, which statistics of spikes, spatial coding, population structure and behaviour already match the lab, and which diverge? Every metric uses one code path; dataset is a label only. Divergence is summarised by \(R = d_\mathrm{sim}/d_\mathrm{lab}\) (lab-MAD-scaled distances of session medians, 90% bootstrap by recording day). \(R \le 1\) means the sim sits within lab variability.

Cohort: 6 animals, 47 room-A lab sessions; matched arm n = 47; default seeds = 5; thigmotaxis = 0.8 (not fitted).

## Three takeaways

1. **Refractory structure is missing.** Lab units violate a 1.5 ms refractory window in 0.09% of ISIs (session median); matched sim sorted spikes do so in 0.87%. After lab-equivalent 0.3 ms duplicate removal the gap remains. Ground-truth sim spikes also show sub-ms ISIs (inhomogeneous Poisson, no public refractory / min-ISI). Matched \(R \approx 18.3\).

2. **Behaviour is too regular and too fast.** Median speed 4.26 cm/s lab vs 7.13 cm/s matched (\(R = 3.20\)). Turning-angle SD (speed > 2 cm/s) 0.32 vs 0.11 rad (\(R = 4.15\)). Thigmotaxis held at 0.8; speed / rotational knobs not public (DF6).

3. **Place coding is weaker, less reliable, and drifts in the lab.** Place-tuned fraction 0.19 lab vs 0.37 matched (chance 0.05); held-out bits/spike among place-tuned units 0.07 vs 0.43. Linear position decoding at \(N = 20\) is near chance in the lab (net \(R^2 = -0.06\)) vs clear in the sim (0.19). Lab place maps are less reliable (adjacent-block \(r \approx 0.48\) vs 0.89) and drift slowly over a session (slope \(-0.024\)/block vs \(\sim 0\) in the sim).

## What the simulation gets right

Matched-arm \(R \le 1\) (within lab variability) includes firing rate (\(R = 0.83\)), Fano factor (\(0.81\)), ISI CV (\(0.78\)), within-session rate drift (\(0.72\)), field size among place-tuned (\(0.94\)), Skaggs sparsity (\(0.94\)), peak rate (\(0.93\)), and spatial information (\(0.68\)). Grid-cell fractions are near chance in both arms and are not counted as a match.

## Structural requests to the other lab (1..7)

1. Refractory period (public min-ISI).
2. Speed / turning knobs or external trajectories (DF6).
3. Independent head direction (~45° lab dissociation).
4. Tuned-cell prevalence and per-spike information (border/HD/place; speed-cell excess near floor, \(R \approx 1.26\)).
5. Within-session drift (adjacent-block \(r \approx 0.48\) vs 0.89; slope \(\approx -0.024\)/block).
6. Expose the capture cap (DF7).
7. Honour `SimConfig.seed` (DF5).

## Keep-mask spike fix + v5 consistency

`_spike_positions` assigns each spike to its parent frame interval; occupancy and spikes share `keep`. Lab peak rate 13.7 → 7.5 Hz. **v5:** every figure, caption, story and findings page is built from the post keep-mask tidy (`data_fig3_divergence.csv`); page-number audit + anon leak check both PASS. Fig7 dropped (trajectories in Fig5). Findings page is prose from that tidy (story is authority).

## Deferred items

- Calibration (Q13–Q15) → Report **4b** (one line on the story; no empty page).
- Report 2 v2 behavioural speed (250 ms) recheck when Report 2 v2 is final.
- Occupancy-matched joint bins across lab polygon vs sim square remain unused (not meaningful).
- SUB sparse in most lab sessions; room A only.

## Artefacts

- PDF: `hippo/reports/report4/report4a_full.pdf` (also `hippo-dq/outputs/data_quality/reports/report4a_v8/`)
- Anon PDF: `…/report4a_full_anon.pdf`
- Delivery: `hippo/tmp_r4_review/v8/` (full + anon + audit JSONs + page PNGs)
- Tag: `report4a-v8` (prior: `report4a-draft` = v4 package on `offline-decoding`)
