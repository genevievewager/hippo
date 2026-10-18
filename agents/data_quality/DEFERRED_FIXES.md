# Deferred fixes (do not edit live checkout modules for these)

Live M2 finished 2026-10-05; still do not write into Report 2 output trees.

| ID | Location | Issue | Workaround / request |
| -- | -------- | ----- | -------------------- |
| DF1 | `analysis/real_quadrant/adapter.py` `_append_manifest` | Exclusion paths always use `DEFAULT_MANIFEST_PATH` under `outputs/real_quadrant/`. | Wrapper redirects to `outputs/data_quality/`. |
| DF2 | `adapter.DEFAULT_CACHE_ROOT` | Defaults to `outputs/real_quadrant/cache`. | Always pass `cache_root=outputs/data_quality/cache`. |
| DF3 | `realtime/quadrant_n5_run.py` `OUTPUT_ROOT` | Live real run writes `outputs/quadrant_n5/models/seed_{0..5}/real/` (M2 used seed_0; **M3 uses seeds 0–5** via `seed_index`). Hardcoded module `OUTPUT_ROOT`; Report 1 `seed_*/sim/` models untouched so far. | Isolation exclude `models/seed_*/real/`; Report 4 `analyze_wrap.patch_output_root` redirects module attribute under `outputs/data_quality/`. Permanent fix: accept env/arg for OUTPUT_ROOT. |
| DF4 | Live M2 thread env | No `OMP_*` caps (191 threads). | Our jobs set caps. |
| DF5 | `hippo_sim/behavior.py` | `config.seed` builds unused RNG; Agent loop unseeded. **"seed unused in Agent loop"** | Request to other lab: wire seed into Agent / document public seeding. No monkeypatch. |
| DF6 | `hippo_sim/behavior.py` | No public route for `speed_*`, `rotational_velocity_*`, `wall_repel_*` without reimplementing `simulate_behavior`. | Request to other lab: expose knobs + **accept external trajectories**. Report 4 fits **thigmotaxis only**. |
| DF7 | `hippo/anatomy/cell_capture.py` `apply_cell_capture_to_ratinabox_params` | Lab-format path **caps** each `n_*` at baseline RatInABox size, so raising `unit_density_per_mm` cannot grow yield past ~100 MEC units. | Report 4 boosts `SimConfig.ratinabox_params` after `apply_trajectory_to_config` (wrapper). Request other lab: allow `n_*` to scale above baseline with density. |
