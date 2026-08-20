"""Two-panel winner embeddings for Manifold Explorer (counts vs best manifold).

Writes::

    figures/manifolds/fig_winner_counts_<target>.png
    figures/manifolds/fig_winner_manifold_<target>.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from realtime.decoder_comparison import ALL_TARGETS
from visualization.constants import FIGURE_DPI
from visualization.publication_isomap_plots import (
    MAX_EMBED_POINTS,
    _color_for_feature,
    _scatter_latent,
    _scatter_position_2d,
    to_display_latent_2d,
)
from visualization.publication_style import apply_publication_theme, save_pub_figure

apply_publication_theme()

FIGURE_SUBDIR_MANIFOLDS = "manifolds"

TARGET_TITLES: dict[str, str] = {
    "position": "Position",
    "speed": "Speed",
    "acceleration": "Acceleration",
    "head_direction": "Head direction",
    "distance_to_wall": "Distance to wall",
    "spatial_context": "Spatial context",
    "movement_state": "Movement state",
    "wall_distance_bin": "Wall-distance bin",
}


def winner_png_paths(figures_dir: Path, target: str) -> tuple[Path, Path]:
    out = Path(figures_dir) / FIGURE_SUBDIR_MANIFOLDS
    return (
        out / f"fig_winner_counts_{target}.png",
        out / f"fig_winner_manifold_{target}.png",
    )


def _subtitle(winner) -> str:
    w_ms = int(round(float(winner.decode_window) * 1000))
    bits = [str(winner.embedding_type), f"W={w_ms} ms", f"k={int(winner.n_components)}"]
    if winner.n_neighbors:
        bits.append(f"nn={int(winner.n_neighbors)}")
    if winner.decoder_name:
        bits.append(str(winner.decoder_name))
    if winner.metric_value is not None:
        bits.append(f"{winner.metric_name}={winner.metric_value:.3g}")
    if not winner.from_metrics:
        bits.append("fallback")
    return " · ".join(bits)


def _load_winner_embedding(experiment_dir: Path, winner, *, spike_source: str = "sorted"):
    from ui.services.manifolds import compute_manifold_diagnostics

    return compute_manifold_diagnostics(
        experiment_dir,
        winner.embedding_type,
        feature_set=winner.feature_set,
        spike_source=spike_source,
        decode_window=float(winner.decode_window),
        n_components=int(winner.n_components),
        n_neighbors=winner.n_neighbors,
        max_samples=MAX_EMBED_POINTS,
        persist=True,
    )


def _write_single_winner_png(
    out_path: Path,
    *,
    Z: np.ndarray,
    behavior: pd.DataFrame,
    target: str,
    mode: str,
    title: str,
    subtitle: str,
) -> Path | None:
    Z2 = to_display_latent_2d(np.asarray(Z, dtype=float), mode)
    if Z2.ndim != 2 or Z2.shape[0] == 0 or Z2.shape[1] < 2:
        return None
    beh = behavior.reset_index(drop=True)
    if len(beh) != len(Z2):
        n = min(len(beh), len(Z2))
        beh = beh.iloc[:n].reset_index(drop=True)
        Z2 = Z2[:n]
    if Z2.shape[0] > MAX_EMBED_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(Z2.shape[0], size=MAX_EMBED_POINTS, replace=False)
        Z2 = Z2[idx]
        beh = beh.iloc[idx].reset_index(drop=True)

    color = _color_for_feature(beh, target)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    if color is None:
        ax.scatter(Z2[:, 0], Z2[:, 1], s=4, alpha=0.65, c="0.4")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
    elif isinstance(color, str) and color == "position_xy":
        _scatter_position_2d(ax, Z2, beh)
    else:
        _scatter_latent(ax, Z2, color, label=TARGET_TITLES.get(target, target))

    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pub_figure(fig, out_path, dpi=FIGURE_DPI)
    return out_path


def plot_fig_winner_embeddings(
    experiment_dir: Path,
    figures_dir: Path | None = None,
    *,
    spike_source: str = "sorted",
    progress_callback=None,
) -> list[Path]:
    from ui.services.manifolds import best_counts_and_manifold_winners
    from visualization.publication_decoding_plots import load_comparison_metrics

    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_MANIFOLDS
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir, prefer=spike_source)
    if not metrics.empty and "spike_source" in metrics.columns:
        sub = metrics[metrics["spike_source"].astype(str) == str(spike_source)]
        if not sub.empty:
            metrics = sub

    written: list[Path] = []
    diag_cache: dict[tuple, object] = {}
    n_targets = len(ALL_TARGETS)

    for ti, target in enumerate(ALL_TARGETS, start=1):
        if progress_callback is not None:
            progress_callback(
                f"{TARGET_TITLES.get(target, target)} ({ti}/{n_targets})",
                ti,
                n_targets,
            )
        counts_w, man_w = best_counts_and_manifold_winners(
            metrics if not metrics.empty else None,
            target,
            spike_source=spike_source,
        )
        for kind, winner in (("counts", counts_w), ("manifold", man_w)):
            key = (
                winner.embedding_type,
                winner.feature_set,
                float(winner.decode_window),
                int(winner.n_components),
                int(winner.n_neighbors or 0),
            )
            try:
                if key not in diag_cache:
                    diag_cache[key] = _load_winner_embedding(
                        experiment_dir, winner, spike_source=spike_source,
                    )
                diag = diag_cache[key]
            except Exception as exc:
                print(f"  warning: winner {kind}/{target} embed failed ({exc})")
                continue

            mode = "counts" if winner.is_counts else winner.embedding_type
            title = (
                f"Raw counts · {TARGET_TITLES.get(target, target)}"
                if kind == "counts"
                else f"Best manifold · {TARGET_TITLES.get(target, target)}"
            )
            out_path = out_dir / f"fig_winner_{kind}_{target}.png"
            try:
                path = _write_single_winner_png(
                    out_path,
                    Z=diag.latent,
                    behavior=diag.behavior,
                    target=target,
                    mode=mode,
                    title=title,
                    subtitle=_subtitle(winner),
                )
            except Exception as exc:
                print(f"  warning: winner {kind}/{target} plot failed ({exc})")
                continue
            if path is not None:
                written.append(path)
                print(f"  wrote {path.relative_to(figures_dir)}")

    return written


def generate_publication_winner_figures(
    experiment_dir: Path,
    figures_dir: Path | None = None,
    *,
    spike_source: str = "sorted",
    progress_callback=None,
) -> list[Path]:
    return plot_fig_winner_embeddings(
        experiment_dir,
        figures_dir,
        spike_source=spike_source,
        progress_callback=progress_callback,
    )
