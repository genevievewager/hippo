"""Load and normalize simulation output files for visualization.

RatInABox generates the locomotor trajectory; ground-truth Poisson spike trains
are the known neural activity before Neuropixels recording degradation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hippo_sim.config import RATE_PARAMS, RIPPLE_PARAMS
from visualization.constants import CELL_CLASS_ORDER, REGION_ORDER


def _pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No {label} column found. Expected one of {candidates}; "
        f"got: {list(df.columns)}"
    )


def resolve_behavior_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        "time": _pick_column(df, ["time", "time_s", "t", "timestamp"], "time"),
        "x": _pick_column(df, ["x", "pos_x", "position_x", "x_cm"], "x"),
        "y": _pick_column(df, ["y", "pos_y", "position_y", "y_cm"], "y"),
        "speed": _pick_column(df, ["speed", "velocity", "v", "speed_cm_s"], "speed"),
        "head_direction": _pick_column(
            df, ["head_direction", "head_direction_rad", "hd", "heading", "theta"],
            "head_direction",
        ),
    }


def resolve_spike_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        "time": _pick_column(df, ["time", "spike_time", "spike_time_s", "timestamp", "times"], "time"),
        "unit_id": _pick_column(df, ["unit_id", "unit", "cluster_id"], "unit_id"),
    }


def resolve_unit_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping = {
        "unit_id": _pick_column(df, ["unit_id", "unit", "cluster_id"], "unit_id"),
        "cell_type": _pick_column(df, ["cell_type", "cell_class", "type"], "cell_type"),
        "region": _pick_column(df, ["region", "brain_region"], "region"),
    }
    for optional, cands in [
        ("rate_equation", ["rate_equation", "rate_model", "equation_class"]),
        ("depth_um", ["depth_um", "depth", "probe_depth_um"]),
        ("channel", ["channel", "channel_id", "electrode"]),
        ("place_x", ["place_x_cm", "place_x", "field_x"]),
        ("place_y", ["place_y_cm", "place_y", "field_y"]),
        ("hd_pref", ["hd_pref_rad", "hd_pref", "preferred_hd"]),
    ]:
        for col in cands:
            if col in df.columns:
                mapping[optional] = col
                break
    return mapping


def normalize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    cols = resolve_behavior_columns(df)
    out = pd.DataFrame({
        "time": df[cols["time"]].astype(float),
        "x": df[cols["x"]].astype(float),
        "y": df[cols["y"]].astype(float),
        "speed": df[cols["speed"]].astype(float),
        "head_direction": df[cols["head_direction"]].astype(float),
    })
    if "acceleration" in df.columns or "accel" in df.columns or "a" in df.columns:
        acc_col = next(c for c in ["acceleration", "accel", "a", "acceleration_cm_s2"] if c in df.columns)
        out["acceleration"] = df[acc_col].astype(float)
    else:
        dt = float(np.median(np.diff(out["time"].to_numpy())))
        if dt <= 0:
            dt = 0.05
        out["acceleration"] = np.gradient(out["speed"].to_numpy(), dt)

    if "distance_to_wall" in df.columns or "distance_to_wall_cm" in df.columns:
        dw_col = "distance_to_wall_cm" if "distance_to_wall_cm" in df.columns else "distance_to_wall"
        out["distance_to_wall"] = df[dw_col].astype(float)
    return out


def normalize_spikes(df: pd.DataFrame) -> pd.DataFrame:
    cols = resolve_spike_columns(df)
    return pd.DataFrame({
        "time": df[cols["time"]].astype(float),
        "unit_id": df[cols["unit_id"]].astype(int),
    })


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    cols = resolve_unit_columns(df)
    out = pd.DataFrame({
        "unit_id": df[cols["unit_id"]].astype(int),
        "cell_type": df[cols["cell_type"]].astype(str),
        "region": df[cols["region"]].astype(str),
    })
    if "rate_equation" in cols:
        out["rate_equation"] = df[cols["rate_equation"]].astype(str)
    else:
        out["rate_equation"] = out["cell_type"]
    if "depth_um" in cols:
        out["depth_um"] = df[cols["depth_um"]].astype(float)
    elif "channel" in cols:
        out["depth_um"] = (df[cols["channel"]].astype(float) - 1) * 20.0
    if "place_x" in cols:
        out["place_x"] = df[cols["place_x"]].astype(float)
    if "place_y" in cols:
        out["place_y"] = df[cols["place_y"]].astype(float)
    if "hd_pref" in cols:
        out["hd_pref"] = df[cols["hd_pref"]].astype(float)
    return out


def infer_arena_bounds(behavior: pd.DataFrame, summary: dict) -> tuple[float, float, float, float]:
    arena = summary.get("arena_size_cm")
    if arena is not None:
        size = float(arena)
        return 0.0, size, 0.0, size
    return (
        float(behavior["x"].min()),
        float(behavior["x"].max()),
        float(behavior["y"].min()),
        float(behavior["y"].max()),
    )


def add_distance_to_wall(behavior: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    if "distance_to_wall" in behavior.columns:
        return behavior
    x_min, x_max, y_min, y_max = bounds
    x = behavior["x"].to_numpy()
    y = behavior["y"].to_numpy()
    dist = np.minimum.reduce([x - x_min, x_max - x, y - y_min, y_max - y])
    behavior = behavior.copy()
    behavior["distance_to_wall"] = dist
    return behavior


def add_theta_and_ripple(behavior: pd.DataFrame) -> pd.DataFrame:
    """Add theta phase and ripple envelope using simulator formulas."""
    from hippo_sim.features import _generate_ripple_envelope

    behavior = behavior.copy()
    t = behavior["time"].to_numpy()
    theta_freq = RATE_PARAMS["CA1_pyr"]["theta_freq_hz"]
    behavior["theta_phase"] = (2 * np.pi * theta_freq * t) % (2 * np.pi)
    behavior["theta_modulation"] = 1.0 + 0.25 * np.cos(behavior["theta_phase"])
    behavior["ripple"] = _generate_ripple_envelope(t, RIPPLE_PARAMS)
    return behavior


def downsample_series(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx], y[idx]


@dataclass
class SimulationOutputs:
    input_dir: Path
    behavior: pd.DataFrame
    units: pd.DataFrame
    spikes_gt: pd.DataFrame
    spikes_sorted: pd.DataFrame
    summary: dict
    anatomy: pd.DataFrame | None = None
    bounds: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)
    behavior_dt: float = 0.05
    session_duration_s: float = 600.0
    unit_mean_rates_gt: pd.Series = field(default_factory=pd.Series)
    unit_mean_rates_sorted: pd.Series = field(default_factory=pd.Series)

    @property
    def cell_class_order(self) -> list[str]:
        return [c for c in CELL_CLASS_ORDER if c in self.units["cell_type"].unique()]

    @property
    def region_order(self) -> list[str]:
        return [r for r in REGION_ORDER if r in self.units["region"].unique()]


def load_simulation_outputs(input_dir: Path) -> SimulationOutputs:
    input_dir = Path(input_dir)
    required = {
        "behavior.csv": "behavior",
        "units.csv": "units",
        "spikes_ground_truth.csv": "spikes_gt",
        "spikes_sorted.csv": "spikes_sorted",
        "summary.json": "summary",
    }
    paths = {}
    for fname, key in required.items():
        path = input_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        paths[key] = path

    behavior = normalize_behavior(pd.read_csv(paths["behavior"]))
    units = normalize_units(pd.read_csv(paths["units"]))
    spikes_gt = normalize_spikes(pd.read_csv(paths["spikes_gt"]))
    spikes_sorted = normalize_spikes(pd.read_csv(paths["spikes_sorted"]))
    with open(paths["summary"]) as f:
        summary = json.load(f)

    anatomy_path = input_dir / "anatomy_regions.csv"
    anatomy = pd.read_csv(anatomy_path) if anatomy_path.exists() else None

    bounds = infer_arena_bounds(behavior, summary)
    behavior = add_distance_to_wall(behavior, bounds)
    behavior = add_theta_and_ripple(behavior)

    session_duration = float(summary.get("session_duration_s", behavior["time"].max()))
    times = behavior["time"].to_numpy()
    behavior_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05

    duration = max(session_duration, 1e-6)
    gt_counts = spikes_gt.groupby("unit_id").size()
    unit_mean_rates_gt = gt_counts / duration
    sorted_counts = spikes_sorted.groupby("unit_id").size()
    unit_mean_rates_sorted = sorted_counts / duration

    return SimulationOutputs(
        input_dir=input_dir,
        behavior=behavior,
        units=units,
        spikes_gt=spikes_gt,
        spikes_sorted=spikes_sorted,
        summary=summary,
        anatomy=anatomy,
        bounds=bounds,
        behavior_dt=behavior_dt,
        session_duration_s=session_duration,
        unit_mean_rates_gt=unit_mean_rates_gt,
        unit_mean_rates_sorted=unit_mean_rates_sorted,
    )


def sort_units_by_class_and_rate(
    units: pd.DataFrame,
    mean_rates: pd.Series,
    cell_class_order: list[str] | None = None,
) -> pd.DataFrame:
    """Sort units by cell class then descending mean firing rate."""
    cell_class_order = cell_class_order or CELL_CLASS_ORDER
    units = units.copy()
    units["mean_rate_hz"] = units["unit_id"].map(mean_rates).fillna(0.0)
    class_rank = {c: i for i, c in enumerate(cell_class_order)}
    units["class_rank"] = units["cell_type"].map(class_rank).fillna(999)
    return units.sort_values(["class_rank", "mean_rate_hz"], ascending=[True, False])


def sort_units_by_rate_equation(
    units: pd.DataFrame,
    mean_rates: pd.Series,
) -> pd.DataFrame:
    units = units.copy()
    units["mean_rate_hz"] = units["unit_id"].map(mean_rates).fillna(0.0)
    return units.sort_values(["rate_equation", "mean_rate_hz"], ascending=[True, False])
