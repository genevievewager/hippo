"""On-demand feature analysis runs (multi feature-set × window), mirroring manifolds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ui.services.features import compute_feature_diagnostics, feature_set_families, list_feature_sets
from ui.services.registry import new_run_id, try_git_commit
from visualization.artifact_manifest import register_artifact

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class FeatureAnalysisRequest:
    input_dir: Path
    feature_sets: tuple[str, ...] = ("counts",)
    decode_windows: tuple[float, ...] = (0.250,)
    spike_source: str = "sorted"
    regenerate_simulation_figures: bool = True


def _analysis_dir(input_dir: Path, run_id: str) -> Path:
    return Path(input_dir) / "features" / run_id


def run_feature_analysis(
    req: FeatureAnalysisRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Extract requested feature sets × windows; save figures under features/<run_id>/."""
    unknown = [fs for fs in req.feature_sets if fs not in list_feature_sets()]
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")
    if not req.feature_sets:
        raise ValueError("Select at least one feature set")
    if not req.decode_windows:
        raise ValueError("Select at least one decode window")

    run_id = new_run_id(prefix="feature")
    out_dir = _analysis_dir(req.input_dir, run_id)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (fs, float(w))
        for fs in req.feature_sets
        for w in req.decode_windows
    ]
    results: list[dict[str, Any]] = []
    n = max(len(jobs), 1)

    for i, (fs, window) in enumerate(jobs, start=1):
        if progress_callback:
            progress_callback(f"Feature `{fs}` @ {window}s", i, n + 1)
        diag = compute_feature_diagnostics(
            req.input_dir,
            fs,
            spike_source=req.spike_source,
            decode_window=window,
        )
        stem = f"{fs}_w{int(window * 1000):04d}ms"
        saved = _save_feature_figures(diag, fig_dir, stem=stem)
        for path in saved:
            register_artifact(
                path,
                category="Features",
                title=f"{fs} · {int(window * 1000)} ms",
                feature_set=fs,
                decode_window=window,
                run_id=run_id,
            )
        results.append({
            "feature_set": fs,
            "families": list(feature_set_families(fs)),
            "decode_window_s": window,
            "n_features": diag.n_features,
            "n_samples": diag.n_samples,
            "figures": [str(p) for p in saved],
        })

    # Copy into discoverable figures/features/
    exp_fig = Path(req.input_dir) / "figures" / "features"
    exp_fig.mkdir(parents=True, exist_ok=True)
    for r in results:
        for fp in r["figures"]:
            src = Path(fp)
            if src.exists() and src.suffix == ".png":
                dest = exp_fig / src.name
                dest.write_bytes(src.read_bytes())
                register_artifact(
                    dest,
                    category="Features",
                    title=src.stem.replace("_", " "),
                    feature_set=r["feature_set"],
                    decode_window=r["decode_window_s"],
                    run_id=run_id,
                )

    sim_figs: dict[str, Any] | None = None
    if req.regenerate_simulation_figures:
        if progress_callback:
            progress_callback("Regenerating simulation / feature gallery figures…", n + 1, n + 1)
        try:
            from ui.services.visualizations import VisualizationRequest, generate_visualizations

            sim_figs = generate_visualizations(
                VisualizationRequest(
                    experiment_dir=Path(req.input_dir),
                    include_simulation=True,
                    include_comparison=False,
                    include_realtime=False,
                ),
            )
        except Exception as exc:  # noqa: BLE001 — gallery regen is best-effort
            sim_figs = {"error": str(exc)}

    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": try_git_commit(),
        "request": {
            "input_dir": str(req.input_dir),
            "feature_sets": list(req.feature_sets),
            "decode_windows": list(req.decode_windows),
            "spike_source": req.spike_source,
            "regenerate_simulation_figures": req.regenerate_simulation_figures,
        },
        "n_jobs_requested": len(jobs),
        "n_jobs_run": len(results),
        "results": results,
        "simulation_figures": sim_figs,
        "output_dir": str(out_dir),
    }
    (out_dir / "analysis_config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (out_dir / "analysis_summary.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


def _save_feature_figures(diag, fig_dir: Path, *, stem: str) -> list[Path]:
    saved: list[Path] = []

    # Variance profile
    if diag.variance is not None and len(diag.variance):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(np.arange(len(diag.variance)), diag.variance, lw=1.2)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Variance")
        ax.set_title(f"{diag.feature_set}: feature variance")
        path = fig_dir / f"{stem}_variance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # Top traces
    traces = diag.example_traces
    value_cols = [c for c in traces.columns if c != "time_s"]
    if value_cols and "time_s" in traces.columns:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        t = traces["time_s"].to_numpy()
        for col in value_cols:
            ax.plot(t, traces[col].to_numpy(), lw=1.0, label=col)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Feature value")
        ax.set_title(f"{diag.feature_set}: top-variance traces")
        if len(value_cols) <= 8:
            ax.legend(fontsize=7, loc="upper right")
        path = fig_dir / f"{stem}_traces.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # Correlation subset
    corr = diag.correlation_subset
    if corr is not None and not corr.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(corr.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{diag.feature_set}: feature correlation (subset)")
        ax.set_xticks([])
        ax.set_yticks([])
        path = fig_dir / f"{stem}_correlation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved
