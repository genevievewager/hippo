"""Diagnostics for fitted Diffusion Maps + Nyström embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from realtime.manifolds.diffusion_nystrom import DiffusionNystrom


def summarize_ood(diag: dict[str, np.ndarray]) -> dict[str, Any]:
    """Aggregate batch OOD diagnostics into scalar session stats."""
    flag = np.asarray(diag.get("ood_flag"), dtype=bool)
    n = int(flag.size) if flag.size else 0
    out: dict[str, Any] = {
        "n_samples": n,
        "n_ood": int(np.sum(flag)) if n else 0,
        "ood_fraction": float(np.mean(flag)) if n else 0.0,
    }
    for key in (
        "nearest_landmark_distance",
        "sigma_x",
        "max_kernel_weight",
        "kernel_entropy",
        "effective_n_landmarks",
    ):
        vals = np.asarray(diag.get(key), dtype=float)
        if vals.size == 0:
            continue
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        out[f"{key}_mean"] = float(np.mean(finite))
        out[f"{key}_median"] = float(np.median(finite))
        out[f"{key}_p95"] = float(np.percentile(finite, 95))
        out[f"{key}_max"] = float(np.max(finite))
    return out


def write_diffusion_diagnostics(
    encoder: DiffusionNystrom,
    output_dir: Path,
    *,
    X_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    behavior_train: Any | None = None,
    behavior_test: Any | None = None,
    Z_train: np.ndarray | None = None,
    Z_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write eigenvalue / coordinate / coverage / Nyström / OOD artifacts.

    Plots are stored under ``output_dir/diagnostics/``. Missing optional
    arrays skip the corresponding figure rather than failing the run.
    """
    output_dir = Path(output_dir)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "nystrom_consistency": encoder.nystrom_landmark_consistency(),
        "eigenvalues": (
            np.asarray(encoder.eigenvalues_).tolist()
            if encoder.eigenvalues_ is not None else None
        ),
        "eigenvalues_all_leading": (
            np.asarray(encoder.eigenvalues_all_[: min(32, encoder.eigenvalues_all_.size)]).tolist()
            if encoder.eigenvalues_all_ is not None else None
        ),
        "n_landmarks": encoder.n_landmarks_fitted_,
        "n_components": encoder.actual_n_components_,
        "n_components_dropped": encoder.n_components_dropped_,
        "memory_bytes": encoder.memory_bytes_,
    }

    if X_train is not None:
        Z_train = encoder.transform(X_train) if Z_train is None else Z_train
        ood_train = encoder.query_diagnostics_batch(X_train)
        summary["ood_train"] = summarize_ood(ood_train)
        np.savez_compressed(diag_dir / "ood_train.npz", **ood_train)
    if X_test is not None:
        Z_test = encoder.transform(X_test) if Z_test is None else Z_test
        ood_test = encoder.query_diagnostics_batch(X_test)
        summary["ood_test"] = summarize_ood(ood_test)
        np.savez_compressed(diag_dir / "ood_test.npz", **ood_test)

    try:
        _plot_eigenvalue_spectrum(encoder, diag_dir / "eigenvalue_spectrum.png")
    except Exception as exc:  # noqa: BLE001
        summary["eigenvalue_plot_error"] = str(exc)
    try:
        _plot_diffusion_coordinates(
            Z_train, Z_test, behavior_train, behavior_test,
            diag_dir / "diffusion_coordinates.png",
        )
    except Exception as exc:  # noqa: BLE001
        summary["coordinate_plot_error"] = str(exc)
    try:
        if X_train is not None:
            _plot_landmark_coverage(
                encoder, X_train, diag_dir / "landmark_coverage.png",
            )
    except Exception as exc:  # noqa: BLE001
        summary["coverage_plot_error"] = str(exc)

    with open(diag_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _plot_eigenvalue_spectrum(encoder: DiffusionNystrom, path: Path) -> None:
    import matplotlib.pyplot as plt

    all_e = encoder.eigenvalues_all_
    kept = encoder.eigenvalues_
    if all_e is None and kept is None:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    if all_e is not None:
        idx = np.arange(min(40, all_e.size))
        ax.plot(idx, np.abs(all_e[: idx.size]), "o-", color="0.55", ms=4, label="leading |λ|")
        ax.axvline(0, color="0.75", ls=":", lw=1)
    if kept is not None:
        start = 1
        ax.plot(
            np.arange(start, start + kept.size),
            np.abs(kept),
            "o",
            color="C0",
            ms=7,
            label="retained nontrivial",
        )
    ax.set_xlabel("eigenmode index (0 = trivial)")
    ax.set_ylabel("|λ|")
    ax.set_title("Diffusion operator spectrum")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _behavior_color(behavior: Any, n: int) -> tuple[np.ndarray | None, str]:
    if behavior is None:
        return None, "index"
    try:
        import pandas as pd

        if isinstance(behavior, pd.DataFrame):
            for col, label in (
                ("speed", "speed"),
                ("spatial_context", "spatial context"),
                ("movement_state", "movement state"),
                ("x", "x position"),
            ):
                if col in behavior.columns and len(behavior) >= n:
                    vals = behavior[col].to_numpy()[:n]
                    if np.issubdtype(np.asarray(vals).dtype, np.number):
                        return np.asarray(vals, dtype=float), label
                    # Categorical: factorize
                    codes, _ = pd.factorize(vals)
                    return codes.astype(float), label
            if "x" in behavior.columns and "y" in behavior.columns:
                xy = behavior[["x", "y"]].to_numpy(dtype=float)[:n]
                return np.hypot(xy[:, 0], xy[:, 1]), "radial position"
    except Exception:
        pass
    return None, "index"


def _plot_diffusion_coordinates(
    Z_train: np.ndarray | None,
    Z_test: np.ndarray | None,
    behavior_train: Any,
    behavior_test: Any,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    packs = []
    if Z_train is not None:
        packs.append(("train", np.asarray(Z_train), behavior_train))
    if Z_test is not None:
        packs.append(("test", np.asarray(Z_test), behavior_test))
    if not packs:
        return
    n_cols = min(3, max(z.shape[1] for _, z, _ in packs if z.shape[1] >= 2), 3)
    if n_cols < 2:
        n_cols = 2
    fig, axes = plt.subplots(len(packs), 1, figsize=(5.2, 3.6 * len(packs)), squeeze=False)
    for row, (split, Z, beh) in enumerate(packs):
        ax = axes[row][0]
        color, clabel = _behavior_color(beh, len(Z))
        if color is None:
            color = np.arange(len(Z), dtype=float)
            clabel = "time index"
        x = Z[:, 0]
        y = Z[:, 1] if Z.shape[1] > 1 else np.zeros(len(Z))
        sc = ax.scatter(x, y, c=color, s=6, cmap="viridis", linewidths=0, alpha=0.75)
        ax.set_xlabel("diffusion 1")
        ax.set_ylabel("diffusion 2" if Z.shape[1] > 1 else "—")
        ax.set_title(f"{split} diffusion coordinates")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=clabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_landmark_coverage(
    encoder: DiffusionNystrom,
    X_train: np.ndarray,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if encoder.landmarks_ is None:
        return
    Xp = encoder._transform_preprocess(np.asarray(X_train))
    L = np.asarray(encoder.landmarks_, dtype=np.float64)
    n_show = min(4000, len(Xp))
    if n_show < len(Xp):
        idx = np.linspace(0, len(Xp) - 1, n_show).astype(int)
        Xs = Xp[idx]
    else:
        Xs = Xp
    pca = PCA(n_components=2, random_state=0)
    Xs2 = pca.fit_transform(Xs)
    L2 = pca.transform(L)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(Xs2[:, 0], Xs2[:, 1], s=5, c="0.75", linewidths=0, label="train")
    ax.scatter(L2[:, 0], L2[:, 1], s=18, c="C3", linewidths=0, label="landmarks")
    ax.set_xlabel("train PCA 1")
    ax.set_ylabel("train PCA 2")
    ax.set_title(f"Landmark coverage (M={len(L)})")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
