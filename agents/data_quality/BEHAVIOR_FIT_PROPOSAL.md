# Behaviour fit setup (show only — no fit until calibration split exists)

## Reachable knobs (public, no monkeypatch / no reimplementation)

| Knob | Route | Fit? |
| ---- | ----- | ---- |
| `thigmotaxis` | `SimConfig.thigmotaxis` → `Agent(params={dt, thigmotaxis})` | **Yes** (only) |
| `behavior_dt`, `arena_size_cm`, `session_duration_s` | `SimConfig` | Matching, not free fit |

## Unreachable without reimplementing `simulate_behavior` (DF6 → other lab)

`speed_mean`, `speed_std`, `speed_coherence_time`, `rotational_velocity_*`,
`wall_repel_*`, `head_direction_smoothing_timescale`, plus **external
trajectories**. Do **not** build a wrapper.

## Targets

| Statistic | Role | Notes |
| --------- | ---- | ----- |
| Wall-distance CDF, distance / half-width | **Fit target** for `thigmotaxis` | Lab half-width = 0.5×boundary width (or mean half-extent); sim = 0.5×`arena_size_cm`. Addresses aspect 1.57 vs square. |
| Speed CDF (>2 cm/s) | **Reported divergence** (knobs unreachable) | Not a fit target until other lab exposes speed_* |
| Occupancy entropy | **Check only** | Not optimized |

## Protocol (when calibration animals exist)

1. Fit `thigmotaxis` on calibration-group animals only (grid or 1-D search).
2. Freeze; evaluate wall-distance + occupancy checks on validation animals.
3. Until split exists: this document is the setup; **no fitting**.
