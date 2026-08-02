"""Summary CSV tables for the visualization report.

The former multi-panel ``simulation_report_summary`` figure was removed —
its panels already appear in behavior, neural, sorting, and trajectory figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from visualization.load_outputs import SimulationOutputs


def build_summary_tables(data: SimulationOutputs, output_dir: Path) -> None:
    """Save CSV summary tables for the visualization report."""
    cell_rows = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        rates = data.unit_mean_rates_gt.reindex(uids).dropna()
        gt_n = data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0]
        sorted_n = data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0]
        cell_rows.append({
            "cell_type": ct,
            "n_units": int(len(uids)),
            "total_ground_truth_spikes": int(gt_n),
            "total_sorted_spikes": int(sorted_n),
            "mean_rate_hz": float(rates.mean()) if len(rates) else 0.0,
            "median_rate_hz": float(rates.median()) if len(rates) else 0.0,
            "std_rate_hz": float(rates.std()) if len(rates) else 0.0,
        })
    pd.DataFrame(cell_rows).to_csv(output_dir / "figure_summary_cell_classes.csv", index=False)

    beh = data.behavior
    x_min, x_max, y_min, y_max = data.bounds
    behavior_row = {
        "session_duration_s": data.session_duration_s,
        "mean_speed_cm_s": float(beh["speed"].mean()),
        "median_speed_cm_s": float(beh["speed"].median()),
        "max_speed_cm_s": float(beh["speed"].max()),
        "mean_distance_to_wall_cm": float(beh["distance_to_wall"].mean()),
        "arena_x_min": x_min,
        "arena_x_max": x_max,
        "arena_y_min": y_min,
        "arena_y_max": y_max,
    }
    pd.DataFrame([behavior_row]).to_csv(output_dir / "figure_summary_behavior.csv", index=False)

    sorting_rows = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        gt_n = data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0]
        sorted_n = data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0]
        loss = 1.0 - sorted_n / gt_n if gt_n > 0 else np.nan
        sorting_rows.append({
            "cell_type": ct,
            "ground_truth_spike_count": int(gt_n),
            "sorted_spike_count": int(sorted_n),
            "apparent_sorting_loss": float(loss),
        })
    pd.DataFrame(sorting_rows).to_csv(output_dir / "figure_summary_sorting.csv", index=False)


def generate_report_figures(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> None:
    """Write summary CSVs; remove the obsolete composite report PNG if present."""
    del rate_bin_size  # retained for call-site compatibility
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_summary_tables(data, output_dir)
    (output_dir / "simulation_report_summary.png").unlink(missing_ok=True)
