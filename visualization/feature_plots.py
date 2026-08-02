"""Behavioral and neural driver feature visualization.

Latent navigation variables (position, speed, heading) drive RatInABox
neural populations; these plots show how those features evolve over time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0

from hippo_sim.config import RATE_PARAMS
from visualization.constants import CELL_CLASS_ORDER, FIGURE_DPI, MAX_LINE_POINTS, cell_class_colors
from visualization.load_outputs import SimulationOutputs, downsample_series


def plot_behavior_features_over_time(data: SimulationOutputs, output_dir: Path) -> None:
    beh = data.behavior
    t = beh["time"].to_numpy()

    panels = [
        ("x", "x position (cm)"),
        ("y", "y position (cm)"),
        ("speed", "Speed (cm/s)"),
        ("acceleration", "Acceleration (cm/s²)"),
        ("head_direction", "Head direction (rad)"),
        ("distance_to_wall", "Distance to wall (cm)"),
        ("theta_phase", "Theta phase (rad)"),
        ("ripple", "Ripple envelope"),
    ]
    available = [(col, label) for col, label in panels if col in beh.columns]
    if not available:
        raise ValueError("No behavioral features available to plot.")

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 2 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available):
        y = beh[col].to_numpy()
        t_ds, y_ds = downsample_series(t, y, MAX_LINE_POINTS)
        ax.plot(t_ds, y_ds, linewidth=0.6)
        ax.set_ylabel(label)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Behavioral features over time", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_features_over_time.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_behavior_feature_distributions(data: SimulationOutputs, output_dir: Path) -> None:
    beh = data.behavior
    features = [
        ("speed", "Speed (cm/s)"),
        ("acceleration", "Acceleration (cm/s²)"),
        ("head_direction", "Head direction (rad)"),
        ("distance_to_wall", "Distance to wall (cm)"),
    ]
    available = [(col, label) for col, label in features if col in beh.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available):
        ax.hist(beh[col], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)

    fig.suptitle("Behavioral feature distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_feature_distributions.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _compute_class_driver_averages(data: SimulationOutputs) -> dict[str, dict[str, np.ndarray]]:
    """Compute population-mean driver features per cell class over time."""
    beh = data.behavior
    t = beh["time"].to_numpy()
    pos = np.column_stack([beh["x"], beh["y"]])
    hd = beh["head_direction"].to_numpy()
    speed = beh["speed"].to_numpy()
    acc = beh["acceleration"].to_numpy()
    dist_wall = beh["distance_to_wall"].to_numpy()
    theta_ph = beh["theta_phase"].to_numpy()
    ripple = beh["ripple"].to_numpy()

    class_drivers: dict[str, dict[str, list[np.ndarray]]] = {
        ct: {k: [] for k in ["place", "hd", "speed", "accel", "boundary", "theta", "ripple"]}
        for ct in CELL_CLASS_ORDER
    }

    for _, unit in data.units.iterrows():
        ct = unit["cell_type"]
        if ct not in class_drivers:
            continue
        p = RATE_PARAMS.get(ct, RATE_PARAMS["CA1_pyr"])
        center = np.array([
            unit.get("place_x", 50.0),
            unit.get("place_y", 50.0),
        ])
        hd_pref = unit.get("hd_pref", 0.0)

        dist_sq = np.sum((pos - center) ** 2, axis=1)
        f_place = np.exp(-dist_sq / (2 * p["sigma_place_cm"] ** 2))

        kappa = p["kappa_hd"]
        f_hd = np.exp(kappa * np.cos(hd - hd_pref)) / (np.exp(kappa) / i0(kappa))

        denom = max(1.0, 30.0 - p["speed_thresh_cm_s"])
        f_speed = np.clip(speed - p["speed_thresh_cm_s"], 0, None) / denom
        f_acc = np.clip(np.abs(acc) / 50.0, 0, 1)
        f_bnd = np.exp(-dist_wall ** 2 / (2 * 15.0 ** 2))
        f_theta = 1.0 + p["w_theta"] * np.cos(theta_ph)

        class_drivers[ct]["place"].append(f_place)
        class_drivers[ct]["hd"].append(f_hd * p["w_hd"])
        class_drivers[ct]["speed"].append(f_speed * p["w_speed"])
        class_drivers[ct]["accel"].append(f_acc * 0.2)
        class_drivers[ct]["boundary"].append(f_bnd * p["w_boundary"])
        class_drivers[ct]["theta"].append((f_theta - 1.0))
        class_drivers[ct]["ripple"].append(ripple * p.get("w_ripple", 0))

    result: dict[str, dict[str, np.ndarray]] = {}
    for ct, drivers in class_drivers.items():
        result[ct] = {}
        for key, arrs in drivers.items():
            result[ct][key] = np.mean(arrs, axis=0) if arrs else np.zeros_like(t)
    return result


def plot_neural_driver_features_over_time(data: SimulationOutputs, output_dir: Path) -> None:
    class_drivers = _compute_class_driver_averages(data)
    t = data.behavior["time"].to_numpy()
    driver_names = ["place", "hd", "speed", "accel", "boundary", "theta", "ripple"]
    labels = ["Place drive", "HD drive", "Speed drive", "Accel drive",
              "Boundary drive", "Theta drive", "Ripple drive"]

    fig, axes = plt.subplots(len(driver_names), 1, figsize=(10, 2.2 * len(driver_names)), sharex=True)
    colors = cell_class_colors(CELL_CLASS_ORDER)

    for ax, driver, label in zip(axes, driver_names, labels):
        for ct in CELL_CLASS_ORDER:
            if ct not in class_drivers:
                continue
            y = class_drivers[ct][driver]
            t_ds, y_ds = downsample_series(t, y, MAX_LINE_POINTS)
            ax.plot(t_ds, y_ds, linewidth=0.7, label=ct, color=colors[ct])
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Neural driver features over time (population mean by cell class)", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "neural_driver_features_over_time.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_neural_driver_features_by_cell_class(data: SimulationOutputs, output_dir: Path) -> None:
    class_drivers = _compute_class_driver_averages(data)
    t = data.behavior["time"].to_numpy()
    driver_names = ["place", "hd", "speed", "boundary", "theta", "ripple"]
    labels = ["Place", "HD", "Speed", "Boundary", "Theta", "Ripple"]

    n_classes = len([c for c in CELL_CLASS_ORDER if c in class_drivers])
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 2.5 * n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]

    idx = 0
    for ct in CELL_CLASS_ORDER:
        if ct not in class_drivers:
            continue
        ax = axes[idx]
        for driver, label in zip(driver_names, labels):
            y = class_drivers[ct][driver]
            if ct == "DG_granule" and driver == "ripple" and np.allclose(y, 0):
                continue
            if ct != "CA3_pyr" and driver == "ripple" and np.allclose(y, 0):
                pass
            t_ds, y_ds = downsample_series(t, y, MAX_LINE_POINTS)
            ax.plot(t_ds, y_ds, linewidth=0.7, label=label)
        ax.set_ylabel(ct)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        idx += 1

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Neural driver features by cell class", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "neural_driver_features_by_cell_class.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_feature_plots(data: SimulationOutputs, output_dir: Path) -> None:
    """Generate neural-driver figures (covariates live in behavior/)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from visualization.publication_behavior_plots import (
        plot_fig_neural_drivers,
        _cleanup_legacy_pngs,
    )

    (output_dir / "fig_behavior_features.png").unlink(missing_ok=True)
    plot_fig_neural_drivers(data, output_dir)
    _cleanup_legacy_pngs(output_dir)

