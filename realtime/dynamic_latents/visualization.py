"""Standardized figures for dynamic latent representations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_behavior_col(behavior: pd.DataFrame, *names: str) -> np.ndarray | None:
    for n in names:
        if n in behavior.columns:
            return behavior[n].to_numpy()
    return None


def figure_latent_trajectory_time(
    Z: np.ndarray,
    output_path: Path,
    *,
    title: str = "Latent trajectory (colored by time)",
) -> Path:
    """Figure A — 2D/3D latent trajectory colored by time."""
    Z = np.asarray(Z, dtype=float)
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    t = np.arange(Z.shape[0])
    fig = plt.figure(figsize=(7, 5))
    if Z.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=t, cmap="viridis", s=6)
        ax.set_zlabel("z3")
    else:
        ax = fig.add_subplot(111)
        dims = min(2, Z.shape[1])
        if dims == 1:
            sc = ax.scatter(t, Z[:, 0], c=t, cmap="viridis", s=6)
            ax.set_xlabel("time index")
            ax.set_ylabel("z1")
        else:
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="viridis", s=6)
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")
    fig.colorbar(sc, ax=ax, label="time index", fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def figure_latent_trajectory_colored(
    Z: np.ndarray,
    color: np.ndarray,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> Path:
    """Figures B/C/D — latent trajectory colored by a behavioral scalar."""
    Z = np.asarray(Z, dtype=float)
    color = np.asarray(color, dtype=float)
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if Z.shape[1] == 1:
        sc = ax.scatter(np.arange(len(Z)), Z[:, 0], c=color, cmap=cmap, s=8)
        ax.set_xlabel("time index")
        ax.set_ylabel("z1")
    else:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap=cmap, s=8)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    fig.colorbar(sc, ax=ax, label=colorbar_label, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def figure_latent_timeseries(
    Z: np.ndarray,
    behavior: pd.DataFrame,
    output_path: Path,
    *,
    max_dims: int = 5,
    title: str = "Latent-state time series",
) -> Path:
    """Figure E — z_i(t) aligned with selected behavioral variables."""
    Z = np.asarray(Z, dtype=float)
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    k = min(max_dims, Z.shape[1])
    beh_keys = [c for c in ("speed", "distance_to_wall", "x") if c in behavior.columns]
    n_rows = k + len(beh_keys)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    t = np.arange(Z.shape[0])
    for i in range(k):
        axes[i].plot(t, Z[:, i], lw=0.8, color="C0")
        axes[i].set_ylabel(f"z{i+1}")
    for j, key in enumerate(beh_keys):
        ax = axes[k + j]
        ax.plot(t, behavior[key].to_numpy()[: len(t)], lw=0.8, color="C3")
        ax.set_ylabel(key)
    axes[-1].set_xlabel("time index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def figure_static_vs_dynamic_prediction(
    *,
    true_xy: np.ndarray,
    static_xy: np.ndarray | None,
    dynamic_xy: np.ndarray | None,
    output_path: Path,
    title: str = "Static vs dynamic position decoding",
) -> Path:
    """Figure F — true / static / dynamic decoded trajectories."""
    true_xy = np.asarray(true_xy, dtype=float)
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(true_xy[:, 0], true_xy[:, 1], lw=0.8, color="0.3", label="true")
    if static_xy is not None:
        axes[0].plot(static_xy[:, 0], static_xy[:, 1], lw=0.8, color="C0", alpha=0.8, label="static")
    if dynamic_xy is not None:
        axes[0].plot(dynamic_xy[:, 0], dynamic_xy[:, 1], lw=0.8, color="C1", alpha=0.8, label="dynamic")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Trajectories")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    t = np.arange(len(true_xy))
    axes[1].plot(t, true_xy[:, 0], color="0.3", lw=0.8, label="true x")
    if static_xy is not None:
        axes[1].plot(t, static_xy[:, 0], color="C0", lw=0.8, alpha=0.8, label="static x")
    if dynamic_xy is not None:
        axes[1].plot(t, dynamic_xy[:, 0], color="C1", lw=0.8, alpha=0.8, label="dynamic x")
    axes[1].legend(fontsize=8)
    axes[1].set_title("x(t)")
    axes[1].set_xlabel("time index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_dynamic_latent_figures(
    *,
    Z: np.ndarray,
    behavior: pd.DataFrame,
    output_dir: Path,
    representation: str = "global_lds",
    causal_status: str = "causal_filtered",
    true_xy: np.ndarray | None = None,
    static_pred_xy: np.ndarray | None = None,
    dynamic_pred_xy: np.ndarray | None = None,
) -> dict[str, str]:
    """Write Figures A–F under ``output_dir``; return stem→path map."""
    out = _ensure_dir(Path(output_dir))
    prefix = f"{representation}_{causal_status}"
    paths: dict[str, str] = {}

    p = out / f"figA_latent_trajectory_time__{prefix}.png"
    figure_latent_trajectory_time(Z, p, title=f"{representation}: latent trajectory (time)")
    paths["fig_A_time"] = str(p)

    pos_color = None
    if "x" in behavior.columns and "y" in behavior.columns:
        pos_color = np.hypot(
            behavior["x"].to_numpy(dtype=float)[: len(Z)],
            behavior["y"].to_numpy(dtype=float)[: len(Z)],
        )
    elif "x" in behavior.columns:
        pos_color = behavior["x"].to_numpy(dtype=float)[: len(Z)]
    if pos_color is not None:
        p = out / f"figB_latent_trajectory_position__{prefix}.png"
        figure_latent_trajectory_colored(
            Z, pos_color, p,
            title=f"{representation}: latent colored by position",
            colorbar_label="position scalar",
            cmap="plasma",
        )
        paths["fig_B_position"] = str(p)

    reward = _get_behavior_col(behavior, "reward_state", "in_reward_zone", "reward")
    if reward is not None:
        p = out / f"figC_latent_trajectory_reward__{prefix}.png"
        figure_latent_trajectory_colored(
            Z, np.asarray(reward, dtype=float)[: len(Z)], p,
            title=f"{representation}: latent colored by reward state",
            colorbar_label="reward",
            cmap="coolwarm",
        )
        paths["fig_C_reward"] = str(p)

    wall = _get_behavior_col(behavior, "distance_to_wall")
    if wall is not None:
        p = out / f"figD_latent_trajectory_wall__{prefix}.png"
        figure_latent_trajectory_colored(
            Z, np.asarray(wall, dtype=float)[: len(Z)], p,
            title=f"{representation}: latent colored by distance to wall",
            colorbar_label="distance_to_wall",
            cmap="magma",
        )
        paths["fig_D_wall"] = str(p)

    p = out / f"figE_latent_timeseries__{prefix}.png"
    figure_latent_timeseries(Z, behavior.iloc[: len(Z)], p, title=f"{representation}: z(t)")
    paths["fig_E_timeseries"] = str(p)

    if true_xy is not None and (static_pred_xy is not None or dynamic_pred_xy is not None):
        p = out / f"figF_static_vs_dynamic_prediction__{prefix}.png"
        figure_static_vs_dynamic_prediction(
            true_xy=true_xy,
            static_xy=static_pred_xy,
            dynamic_xy=dynamic_pred_xy,
            output_path=p,
        )
        paths["fig_F_comparison"] = str(p)

    return paths
