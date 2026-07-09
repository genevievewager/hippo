"""Offline decoder training with causal time-based train/test split.

Behavioral variables are latent from the Neuropixels recording perspective.
Training uses only spike counts from causal windows; labels come from true
behavior (available offline for evaluation, not to the online decoder).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MOVEMENT_THRESHOLDS = {"still_max": 2.0, "slow_max": 10.0}
WALL_MARGIN_FRAC = 0.15
CORNER_MARGIN_FRAC = 0.25


@dataclass
class TrainedDecoders:
    position: Pipeline
    spatial_context: Pipeline
    movement_state: Pipeline
    speed: Pipeline
    spatial_context_classes: list[str]
    movement_state_classes: list[str]


def _resolve_behavior_columns(behavior_df: pd.DataFrame) -> dict[str, str]:
    time_candidates = ["time", "time_s", "t"]
    x_candidates = ["x", "pos_x", "position_x", "x_cm"]
    y_candidates = ["y", "pos_y", "position_y", "y_cm"]
    speed_candidates = ["speed", "velocity", "v", "speed_cm_s"]
    hd_candidates = ["head_direction", "hd", "heading", "theta", "head_direction_rad"]

    def pick(cands: list[str]) -> str | None:
        return next((c for c in cands if c in behavior_df.columns), None)

    mapping = {
        "time": pick(time_candidates),
        "x": pick(x_candidates),
        "y": pick(y_candidates),
        "speed": pick(speed_candidates),
        "head_direction": pick(hd_candidates),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing behavior columns for {missing}. "
            f"Available columns: {list(behavior_df.columns)}"
        )
    return mapping  # type: ignore[return-value]


def infer_arena_bounds(
    behavior_df: pd.DataFrame,
    summary: dict | None = None,
) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) for spatial-context zoning."""
    if summary is not None:
        arena = summary.get("arena_size_cm")
        if arena is not None:
            size = float(arena)
            return 0.0, size, 0.0, size

    cols = _resolve_behavior_columns(behavior_df)
    x = behavior_df[cols["x"]].to_numpy()
    y = behavior_df[cols["y"]].to_numpy()
    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def classify_spatial_context(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Assign center / wall / corner from position relative to arena bounds."""
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    wall_x = WALL_MARGIN_FRAC * width
    wall_y = WALL_MARGIN_FRAC * height
    corner_x = CORNER_MARGIN_FRAC * width
    corner_y = CORNER_MARGIN_FRAC * height

    near_left = x <= x_min + wall_x
    near_right = x >= x_max - wall_x
    near_bottom = y <= y_min + wall_y
    near_top = y >= y_max - wall_y

    near_corner_x = (x <= x_min + corner_x) | (x >= x_max - corner_x)
    near_corner_y = (y <= y_min + corner_y) | (y >= y_max - corner_y)
    is_corner = near_corner_x & near_corner_y

    is_wall = (near_left | near_right | near_bottom | near_top) & ~is_corner

    labels = np.full(len(x), "center", dtype=object)
    labels[is_wall] = "wall"
    labels[is_corner] = "corner"
    return labels


def classify_movement_state(speed: np.ndarray) -> np.ndarray:
    """Assign still / slow / fast from speed (cm/s)."""
    labels = np.full(len(speed), "slow", dtype=object)
    labels[speed < MOVEMENT_THRESHOLDS["still_max"]] = "still"
    labels[speed >= MOVEMENT_THRESHOLDS["slow_max"]] = "fast"
    return labels


def align_behavior_to_decoder_times(
    behavior_df: pd.DataFrame,
    decode_times: np.ndarray,
    summary: dict | None = None,
) -> pd.DataFrame:
    """
    Align each decoder time to the nearest behavior frame.

    Returns dataframe with time, x, y, speed, head_direction,
    spatial_context, and movement_state.
    """
    cols = _resolve_behavior_columns(behavior_df)
    beh_times = behavior_df[cols["time"]].to_numpy()

    nearest_idx = np.searchsorted(beh_times, decode_times, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(beh_times) - 1)
    prev_idx = np.clip(nearest_idx - 1, 0, len(beh_times) - 1)
    choose_prev = np.abs(beh_times[prev_idx] - decode_times) < np.abs(
        beh_times[nearest_idx] - decode_times
    )
    idx = np.where(choose_prev, prev_idx, nearest_idx)

    x = behavior_df[cols["x"]].to_numpy()[idx]
    y = behavior_df[cols["y"]].to_numpy()[idx]
    speed = behavior_df[cols["speed"]].to_numpy()[idx]
    hd = behavior_df[cols["head_direction"]].to_numpy()[idx]

    x_min, x_max, y_min, y_max = infer_arena_bounds(behavior_df, summary)
    spatial_context = classify_spatial_context(x, y, x_min, x_max, y_min, y_max)
    movement_state = classify_movement_state(speed)

    return pd.DataFrame({
        "time": decode_times,
        "x": x,
        "y": y,
        "speed": speed,
        "head_direction": hd,
        "spatial_context": spatial_context,
        "movement_state": movement_state,
    })


def causal_train_test_split(
    decode_times: np.ndarray,
    train_frac: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    """Split decode times by session chronology (not random shuffle)."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    split_time = decode_times[0] + train_frac * (decode_times[-1] - decode_times[0])
    train_mask = decode_times < split_time
    test_mask = ~train_mask
    return train_mask, test_mask


def train_decoders(
    X_train: np.ndarray,
    behavior_train: pd.DataFrame,
    models_dir: Path | None = None,
) -> TrainedDecoders:
    """Train position, context, movement, and speed decoders on causal spike counts."""
    y_pos = behavior_train[["x", "y"]].to_numpy()
    y_context = behavior_train["spatial_context"].to_numpy()
    y_movement = behavior_train["movement_state"].to_numpy()
    y_speed = behavior_train["speed"].to_numpy()

    position = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    position.fit(X_train, y_pos)

    spatial_context = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        )),
    ])
    spatial_context.fit(X_train, y_context)

    movement_state = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        )),
    ])
    movement_state.fit(X_train, y_movement)

    speed = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    speed.fit(X_train, y_speed.reshape(-1, 1))

    decoders = TrainedDecoders(
        position=position,
        spatial_context=spatial_context,
        movement_state=movement_state,
        speed=speed,
        spatial_context_classes=list(spatial_context.named_steps["clf"].classes_),
        movement_state_classes=list(movement_state.named_steps["clf"].classes_),
    )

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(decoders.position, models_dir / "position_decoder.joblib")
        joblib.dump(decoders.spatial_context, models_dir / "spatial_context_decoder.joblib")
        joblib.dump(decoders.movement_state, models_dir / "movement_state_decoder.joblib")
        joblib.dump(decoders.speed, models_dir / "speed_decoder.joblib")
        joblib.dump(decoders.spatial_context_classes, models_dir / "spatial_context_classes.joblib")
        joblib.dump(decoders.movement_state_classes, models_dir / "movement_state_classes.joblib")

    return decoders


def evaluate_offline_training(
    decoders: TrainedDecoders,
    X_test: np.ndarray,
    behavior_test: pd.DataFrame,
) -> dict:
    """Compute offline test metrics for trained decoders."""
    y_pos = behavior_test[["x", "y"]].to_numpy()
    pred_pos = decoders.position.predict(X_test)
    pos_err = np.linalg.norm(pred_pos - y_pos, axis=1)

    y_context = behavior_test["spatial_context"].to_numpy()
    pred_context = decoders.spatial_context.predict(X_test)

    y_movement = behavior_test["movement_state"].to_numpy()
    pred_movement = decoders.movement_state.predict(X_test)

    y_speed = behavior_test["speed"].to_numpy()
    pred_speed = decoders.speed.predict(X_test).ravel()

    return {
        "mean_position_error_cm": float(np.mean(pos_err)),
        "median_position_error_cm": float(np.median(pos_err)),
        "spatial_context_accuracy": float(accuracy_score(y_context, pred_context)),
        "spatial_context_confusion_matrix": confusion_matrix(
            y_context, pred_context, labels=["center", "wall", "corner"]
        ).tolist(),
        "spatial_context_labels": ["center", "wall", "corner"],
        "movement_state_accuracy": float(accuracy_score(y_movement, pred_movement)),
        "movement_state_confusion_matrix": confusion_matrix(
            y_movement, pred_movement, labels=["still", "slow", "fast"]
        ).tolist(),
        "movement_state_labels": ["still", "slow", "fast"],
        "speed_r2": float(r2_score(y_speed, pred_speed)),
        "speed_correlation": float(np.corrcoef(y_speed, pred_speed)[0, 1]),
    }
