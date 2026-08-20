"""On-demand feature analysis runs with Manifold-Explorer-style panneled pages.

Display contract:
  - First generation (no decoder metrics): all feature-set panels use 250 ms.
  - After decoder comparison exists: each feature set uses its own best window
    from metrics (per-feature ranking), even if that set did not win overall.

Per-window diagnostic PNGs can still be written for exploration; the primary
Feature Explorer gallery is the three panneled overview pages under
``figures/features/fig_feature_panel_{variance,traces,correlation}.png``.
"""

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

from ui.services.features import (
    checkpoint_feature_transform,
    compute_feature_diagnostics,
    feature_set_families,
    list_feature_sets,
)
from ui.services.registry import new_run_id, try_git_commit
from visualization.artifact_manifest import register_artifact
from visualization.publication_feature_plots import (
    DEFAULT_FEATURE_PANEL_WINDOW_S,
    plot_feature_panel_pages,
    resolve_feature_panel_windows,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class FeatureAnalysisRequest:
    input_dir: Path
    feature_sets: tuple[str, ...] = ("counts",)
    decode_windows: tuple[float, ...] = (DEFAULT_FEATURE_PANEL_WINDOW_S,)
    spike_source: str = "sorted"
    regenerate_simulation_figures: bool = True
    # When True (default), write the panneled overview pages used by Feature Explorer.
    write_panel_pages: bool = True
    # When True, also write per-window variance/traces/correlation PNGs.
    write_per_window_diagnostics: bool = False
    # When False (default), skip when the selection is already on disk.
    force_recompute: bool = False


def _analysis_dir(input_dir: Path, run_id: str) -> Path:
    return Path(input_dir) / "features" / run_id


def run_feature_analysis(
    req: FeatureAnalysisRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Extract feature diagnostics and write panneled Feature Explorer pages."""
    unknown = [fs for fs in req.feature_sets if fs not in list_feature_sets()]
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")
    if not req.feature_sets:
        raise ValueError("Select at least one feature set")

    windows = tuple(float(w) for w in req.decode_windows) or (DEFAULT_FEATURE_PANEL_WINDOW_S,)
    if not req.force_recompute:
        coverage = selection_already_covered(
            req.input_dir,
            feature_sets=req.feature_sets,
            decode_windows=windows,
            spike_source=req.spike_source,
        )
        panels_ok = not req.write_panel_pages or all(
            (Path(req.input_dir) / "figures" / "features" / f"fig_feature_panel_{kind}.png").exists()
            for kind in ("variance", "traces", "correlation")
        )
        matrix_ok = not req.write_per_window_diagnostics or coverage["fully_covered"]
        if coverage["fully_covered"] and panels_ok and matrix_ok:
            return {
                "run_id": None,
                "skipped": True,
                "reason": "feature_transforms_on_disk",
                "n_skipped": coverage["n_wanted"],
                "matching_run_ids": coverage["matching_run_ids"],
                "output_dir": str(Path(req.input_dir) / "features"),
            }

    run_id = new_run_id(prefix="feature")
    out_dir = _analysis_dir(req.input_dir, run_id)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolve display windows: 250 ms default, else each set's best decoder window.
    resolved = resolve_feature_panel_windows(
        req.input_dir,
        req.feature_sets,
        fallback_s=DEFAULT_FEATURE_PANEL_WINDOW_S,
        spike_source=req.spike_source,
    )

    results: list[dict[str, Any]] = []
    panel_meta: dict[str, Any] | None = None

    # Optional dense per-window diagnostics (advanced / legacy gallery).
    per_window_jobs: list[tuple[str, float]] = []
    if req.write_per_window_diagnostics:
        per_window_jobs = [(fs, float(w)) for fs in req.feature_sets for w in windows]

    total_steps = (
        (1 if req.write_panel_pages else 0)
        + len(per_window_jobs)
        + (1 if req.regenerate_simulation_figures else 0)
    )
    total_steps = max(total_steps, 1)
    step = 0

    if req.write_panel_pages:
        step += 1
        if progress_callback:
            progress_callback("Writing panneled feature overview pages…", step, total_steps)

        def _cb(msg: str, s: int, n: int) -> None:
            if progress_callback:
                # Nest panel progress inside the outer step slot.
                progress_callback(msg, step, total_steps)

        panel_meta = plot_feature_panel_pages(
            req.input_dir,
            req.feature_sets,
            spike_source=req.spike_source,
            figures_dir=Path(req.input_dir) / "figures",
            fallback_window_s=DEFAULT_FEATURE_PANEL_WINDOW_S,
            run_id=run_id,
            progress_callback=_cb,
        )
        # Also copy panel pages into this run folder.
        for kind, path_s in (panel_meta.get("figures") or {}).items():
            src = Path(path_s)
            if src.exists():
                dest = fig_dir / src.name
                dest.write_bytes(src.read_bytes())
        results.append({
            "kind": "panel_pages",
            "windows": panel_meta.get("windows"),
            "figures": list((panel_meta.get("figures") or {}).values()),
        })

    # Shared F checkpoints for Decoder Benchmark (selected windows × sets).
    checkpoint_jobs = [
        (str(fs), float(w))
        for fs in req.feature_sets
        for w in windows
    ]
    # Also include resolved panel windows if they differ from the selection.
    for fs, info in (resolved or {}).items():
        try:
            if isinstance(info, dict):
                wf = float(info.get("window_s"))
            else:
                wf = float(info)
        except (TypeError, ValueError):
            continue
        if fs in req.feature_sets and (str(fs), wf) not in {
            (a, b) for a, b in checkpoint_jobs
        }:
            checkpoint_jobs.append((str(fs), wf))

    n_ckpt = len(checkpoint_jobs)
    total_steps = max(total_steps + n_ckpt, 1)
    n_feature_checkpoints = 0
    checkpoint_errors: list[dict[str, Any]] = []
    for fs, window in checkpoint_jobs:
        step += 1
        if progress_callback:
            progress_callback(
                f"Checkpoint F `{fs}` @ {window}s", step, total_steps,
            )
        try:
            ck = checkpoint_feature_transform(
                req.input_dir,
                fs,
                spike_source=req.spike_source,
                decode_window=window,
            )
            if ck.get("persisted") or ck.get("from_cache"):
                n_feature_checkpoints += 1
            results.append({
                "kind": "feature_transform_checkpoint",
                **ck,
            })
        except Exception as exc:  # noqa: BLE001
            err_row = {
                "kind": "feature_transform_checkpoint",
                "feature_set": fs,
                "decode_window_s": window,
                "error": str(exc),
            }
            checkpoint_errors.append(err_row)
            results.append(err_row)

    for fs, window in per_window_jobs:
        step += 1
        if progress_callback:
            progress_callback(f"Feature `{fs}` @ {window}s", step, total_steps)
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

    exp_fig = Path(req.input_dir) / "figures" / "features"
    exp_fig.mkdir(parents=True, exist_ok=True)
    for r in results:
        for fp in r.get("figures") or []:
            src = Path(fp)
            if src.exists() and src.suffix == ".png":
                dest = exp_fig / src.name
                dest.write_bytes(src.read_bytes())
                register_artifact(
                    dest,
                    category="Features",
                    title=src.stem.replace("_", " "),
                    feature_set=r.get("feature_set"),
                    decode_window=r.get("decode_window_s"),
                    run_id=run_id,
                )

    sim_figs: dict[str, Any] | None = None
    if req.regenerate_simulation_figures:
        step += 1
        if progress_callback:
            progress_callback("Regenerating simulation / feature gallery figures…", step, total_steps)
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
            "decode_windows": list(windows),
            "spike_source": req.spike_source,
            "regenerate_simulation_figures": req.regenerate_simulation_figures,
            "write_panel_pages": req.write_panel_pages,
            "write_per_window_diagnostics": req.write_per_window_diagnostics,
            "force_recompute": req.force_recompute,
        },
        "resolved_panel_windows": resolved,
        "panel_pages": panel_meta,
        "n_jobs_requested": len(per_window_jobs) + (1 if req.write_panel_pages else 0) + n_ckpt,
        "n_feature_checkpoints": n_feature_checkpoints,
        "n_checkpoint_errors": len(checkpoint_errors),
        "checkpoint_errors": checkpoint_errors,
        "n_jobs_run": len(results),
        "results": results,
        "simulation_figures": sim_figs,
        "output_dir": str(out_dir),
        "ok": not checkpoint_errors,
    }
    (out_dir / "analysis_config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (out_dir / "analysis_summary.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    if checkpoint_errors:
        details = "; ".join(
            f"{row['feature_set']} @ {float(row['decode_window_s']):g}s"
            f" ({row.get('error') or 'unknown error'})"
            for row in checkpoint_errors
        )
        raise RuntimeError(
            "Feature Construction did not write F caches for: "
            f"{details}. Downstream pages will not treat these as generated."
        )
    return meta


def refresh_feature_panels_from_metrics(
    experiment_dir: Path,
    *,
    feature_sets: tuple[str, ...] | list[str] | None = None,
    spike_source: str = "sorted",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any] | None:
    """Update panneled Feature Explorer pages after a decoder run."""
    from visualization.publication_feature_plots import update_feature_panels_after_decoding

    return update_feature_panels_after_decoding(
        Path(experiment_dir),
        feature_sets=feature_sets,
        spike_source=spike_source,
        progress_callback=progress_callback,
    )


def list_feature_analysis_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    """Discover prior ``features/<run_id>/`` analyses (newest first)."""
    root = Path(experiment_dir) / "features"
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        summary_path = path / "analysis_summary.json"
        meta: dict[str, Any] = {}
        if summary_path.exists():
            try:
                meta = json.loads(summary_path.read_text())
            except Exception:  # noqa: BLE001
                meta = {}
        req = meta.get("request") or {}
        feature_sets = list(req.get("feature_sets") or [])
        decode_windows = [float(w) for w in (req.get("decode_windows") or [])]
        # Fall back to result rows when older summaries omit request windows.
        if not feature_sets or not decode_windows:
            for row in meta.get("results") or []:
                fs = row.get("feature_set")
                if fs and fs not in feature_sets:
                    feature_sets.append(str(fs))
                w = row.get("decode_window_s")
                if w is not None:
                    wf = float(w)
                    if wf not in decode_windows:
                        decode_windows.append(wf)
        fig_dir = path / "figures"
        n_figs = (
            len(list(fig_dir.glob("*.png"))) if fig_dir.is_dir() else 0
        )
        runs.append({
            "run_id": meta.get("run_id") or path.name,
            "created_at": meta.get("created_at"),
            "output_dir": str(path),
            "feature_sets": feature_sets,
            "decode_windows": decode_windows,
            "spike_source": req.get("spike_source"),
            "write_panel_pages": bool(req.get("write_panel_pages", False)),
            "write_per_window_diagnostics": bool(
                req.get("write_per_window_diagnostics", True)
            ),
            "n_figures": n_figs,
            "n_jobs_run": meta.get("n_jobs_run"),
            "has_panels": any(
                (fig_dir / f"fig_feature_panel_{k}.png").exists()
                for k in ("variance", "traces", "correlation")
            ) if fig_dir.is_dir() else False,
        })
    runs.sort(key=lambda r: str(r.get("created_at") or r.get("run_id") or ""), reverse=True)
    return runs


def selection_already_covered(
    input_dir: Path,
    *,
    feature_sets: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...],
    spike_source: str | None = None,
) -> dict[str, Any]:
    """Whether selected feature×window F caches exist on disk.

    Uses the same ``feature_transforms`` inventory Decoder Benchmark and
    Latent Representations load. Prior run JSON is not a success signal.
    """
    from realtime.transform_cache import inventory_feature_construction_cache

    return inventory_feature_construction_cache(
        Path(input_dir),
        feature_sets=feature_sets,
        decode_windows=decode_windows,
        spike_source=str(spike_source or "sorted"),
    )


def _save_feature_figures(diag, fig_dir: Path, *, stem: str) -> list[Path]:
    saved: list[Path] = []

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
