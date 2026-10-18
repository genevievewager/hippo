# Report 4a — Results (draft v4)

## The question

Before using the simulator as a stand-in for real data, which statistics of spikes, spatial coding, population structure and behaviour already match the lab, and which diverge? Every metric uses one code path; dataset is a label only. Divergence is summarised by \(R = d_\mathrm{sim}/d_\mathrm{lab}\) (lab-MAD-scaled distances of session medians, 90% bootstrap by recording day). \(R \le 1\) means the sim sits within lab variability.

Cohort: 6 animals, 47 room-A lab sessions; matched arm n = 47; default seeds = 5; thigmotaxis = 0.8 (not fitted).

## Three takeaways

1. **Refractory structure is missing.** Lab units violate a 1.5 ms refractory window in 0.09% of ISIs (session median); matched sim sorted spikes do so in 0.87%. After lab-equivalent 0.3 ms duplicate removal the gap remains. Ground-truth sim spikes also show sub-ms ISIs (inhomogeneous Poisson, no public refractory / min-ISI). Matched \(R \approx 18.3\).

2. **Behaviour is too regular and too fast.** Median speed 4.26 cm/s lab vs 7.13 cm/s matched (\(R = 3.20\)). Turning-angle SD (speed > 2 cm/s) 0.32 vs 0.11 rad (\(R = 4.15\)). Thigmotaxis held at 0.8; speed / rotational knobs not public (DF6).

3. **Place coding is weaker, less reliable, and drifts in the lab.** Place-tuned fraction 0.19 lab vs 0.37 matched (chance 0.05); held-out bits/spike among place-tuned units 0.07 vs 0.43. Linear position decoding at \(N = 20\) is near chance in the lab (net \(R^2 = -0.06\)) vs clear in the sim (0.19). Lab place maps are less reliable (adjacent-block \(r \approx 0.48\) vs 0.89) and drift slowly over a session (slope \(-0.024\)/block vs \(\sim 0\) in the sim).

## What the simulation gets right

Matched-arm \(R \le 1\) (within lab variability) includes firing rate (\(R = 0.83\)), Fano factor (\(0.81\)), ISI CV (\(0.78\)), within-session rate drift (\(0.72\)), field size among place-tuned (\(0.94\)), Skaggs sparsity (\(0.94\)), peak rate (\(0.93\)), and spatial information (\(0.68\)). Grid-cell fractions are near chance in both arms and are not counted as a match.

## Structural requests to the other lab

- Add a public refractory / min-ISI parameter (generator is inhomogeneous Poisson).
- Expose speed and rotational knobs (DF6); thigmotaxis was held at 0.8 and not fitted.
- Reduce excess **border** and **HD** cells and the over-strong behavioural encoding of position, movement direction and speed. Speed-cell excess is near the reliability floor after the keep-mask fix (\(R \approx 1.26\)) and is **not** a primary cell-type request.
- Lab head direction differs from movement direction by a median ~45°; the sim has no such dissociation (sim HD = movement heading). Add an independent HD drive.
- Within-session drift: lab place maps are less reliable (adjacent-block \(r \approx 0.48\) vs 0.89) and drift slowly (slope \(\approx -0.024\)/block vs ~0). Add a public nonstationarity / representational-drift mechanism.
- Accept external trajectories (DF6) so lab behaviour can be replayed exactly when knobs are insufficient.

## Keep-mask spike fix (v4)

`_spike_positions` now assigns each spike to its parent frame interval and drops spikes whose frame is excluded (speed filter, valid mask, split arm). Occupancy and spikes share the same `keep` mask. Sanity: 20 lab + 20 sim eligible units — occupancy-weighted map mean equals mean rate over kept frames within 5% (all 40 passed, max rel err = 0). Lab peak rate fell 13.7 → 7.5 Hz and field size rose 0.43 → 0.67; matched barely moved. See `OLD_VS_NEW_spike_keep.csv` in the report4a_v4 outputs.

## Deferred items

- Calibration pages Q13–Q15 → Report **4b** (proxy-fidelity gate failed).
- Report 2 v2 behavioural speed (250 ms) recheck when Report 2 v2 is final.
- Occupancy-matched joint bins across lab polygon vs sim square remain unused (not meaningful).
- SUB sparse in most lab sessions; room A only.

## Artefacts

- PDF: `hippo-dq/outputs/data_quality/reports/report4a_v4/report4a_full.pdf`
- Anon PDF: `…/report4a_full_anon.pdf`
- OLD vs NEW: `…/OLD_VS_NEW_spike_keep.csv`
- Tag: `report4a-draft` (this package on `offline-decoding`)
