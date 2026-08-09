"""On-demand manifold analysis runs for the active dataset.

Writes publication-format ``fig_latent_geometry_<feature>.png`` pages
(same layout as the PDF suite under ``figures/decoder_comparison/``):
one page per behavioral color variable, panels = selected manifolds.

Window selection:
  - If ``decoder_comparison`` metrics exist for counts, regenerate via the
    publication plotter (optimal W per mode, exact PDF match).
  - Otherwise fit at ~250 ms (or closest selected window); when metrics exist
    for a mode later, that optimal W is used for UI fits.

When multiple neural feature sets are requested, ``counts`` keeps the
canonical PDF filenames; other sets are saved as
``fig_latent_geometry_<feature>__<feature_set>.png``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from realtime.neural_features import embedding_compatible_with_feature_set
from realtime.search_space import resolve_manifold_alias
from ui.services.manifolds import compute_manifold_diagnostics
from ui.services.registry import new_run_id, try_git_commit
from visualization.artifact_manifest import register_artifact
from visualization.constants import FIGURE_SUBDIR_DECODER
from visualization.experiment_viz import has_decoder_comparison
from visualization.publication_isomap_plots import (
    COLOR_FEATURES,
    LatentGeometryPanel,
    plot_fig_latent_geometry,
    plot_fig_latent_geometry_for_feature,
    plot_latent_geometry_page_from_embeddings,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class ManifoldAnalysisRequest:
    input_dir: Path
    feature_sets: tuple[str, ...] | None = None
    feature_set: str | None = None  # legacy single-set alias
    manifolds: tuple[str, ...] = ("global_pca",)
    decode_windows: tuple[float, ...] = (0.250,)
    n_components: int = 3
    spike_source: str = "sorted"
    color_by: str = "time"  # unused for PDF suite; kept for API compat
    behavioral_colors: tuple[str, ...] | None = None  # None → full COLOR_FEATURES

    def resolved_feature_sets(self) -> tuple[str, ...]:
        if self.feature_sets:
            return tuple(self.feature_sets)
        if self.feature_set:
            return (self.feature_set,)
        return ("counts",)

    def resolved_behavioral_colors(self) -> tuple[tuple[str, str], ...]:
        if self.behavioral_colors is None:
            return COLOR_FEATURES
        wanted = set(self.behavioral_colors)
        return tuple((k, t) for k, t in COLOR_FEATURES if k in wanted)


def _analysis_dir(input_dir: Path, run_id: str) -> Path:
    return Path(input_dir) / "manifolds" / run_id


def _preferred_window(windows: tuple[float, ...] | list[float]) -> float:
    """Prefer 250 ms when present; else closest to 250 ms."""
    wins = [float(w) for w in windows]
    if not wins:
        return 0.250
    if 0.250 in wins:
        return 0.250
    return min(wins, key=lambda w: abs(w - 0.250))


def _geometry_stem(feature: str, feature_set: str) -> str:
    if feature_set in {"counts", "identity"}:
        return f"fig_latent_geometry_{feature}"
    safe = feature_set.replace("/", "_")
    return f"fig_latent_geometry_{feature}__{safe}"


def _optimal_window_for_mode(
    experiment_dir: Path,
    mode: str,
    *,
    fallback: float,
) -> float:
    """Best decode window from comparison metrics when available."""
    try:
        from visualization.publication_decoding_plots import load_comparison_metrics
        from visualization.publication_isomap_plots import _best_row_for_mode_target
    except Exception:
        return fallback
    metrics = load_comparison_metrics(experiment_dir, prefer="sorted")
    if metrics is None or metrics.empty:
        return fallback
    if "spike_source" in metrics.columns:
        metrics = metrics[metrics["spike_source"].astype(str) == "sorted"].copy()
    emb = "counts" if resolve_manifold_alias(mode) == "identity" else resolve_manifold_alias(mode)
    row = _best_row_for_mode_target(metrics, feature_mode=emb, target="position")
    if row is None:
        return fallback
    w = row.get("decode_window_s")
    if w is None or (isinstance(w, float) and pd.isna(w)):
        return fallback
    return float(w)


def run_manifold_analysis(
    req: ManifoldAnalysisRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Fit manifolds and write PDF-format latent-geometry pages."""
    run_id = new_run_id(prefix="manifold")
    out_dir = _analysis_dir(req.input_dir, run_id)
    run_fig_dir = out_dir / "figures"
    run_fig_dir.mkdir(parents=True, exist_ok=True)
    pub_fig_dir = Path(req.input_dir) / "figures" / FIGURE_SUBDIR_DECODER
    pub_fig_dir.mkdir(parents=True, exist_ok=True)

    color_pages = req.resolved_behavioral_colors()
    windows = tuple(float(w) for w in req.decode_windows) or (0.250,)
    preferred = _preferred_window(windows)
    feature_sets = req.resolved_feature_sets()
    all_figures: list[str] = []
    results: list[dict[str, Any]] = []

    # --- counts + existing decoder comparison → exact PDF plotter ----------
    used_publication = False
    if "counts" in feature_sets and has_decoder_comparison(req.input_dir):
        if progress_callback:
            progress_callback("Regenerating publication latent-geometry suite…", 1, 2)
        try:
            if req.behavioral_colors is None:
                path = plot_fig_latent_geometry(
                    req.input_dir,
                    figures_dir=Path(req.input_dir) / "figures",
                    compact=False,
                )
                if path is not None:
                    used_publication = True
                for color_key, color_title in COLOR_FEATURES:
                    p = pub_fig_dir / f"fig_latent_geometry_{color_key}.png"
                    if p.exists():
                        all_figures.append(str(p))
                        register_artifact(
                            p,
                            category="Manifolds",
                            title=f'Latent geometry colored by "{color_title}"',
                            feature_set="counts",
                            target=color_key,
                            run_id=run_id,
                        )
            else:
                for color_key, color_title in color_pages:
                    path = plot_fig_latent_geometry_for_feature(
                        req.input_dir, color_key, color_title,
                        figures_dir=Path(req.input_dir) / "figures",
                    )
                    if path is not None:
                        used_publication = True
                        all_figures.append(str(path))
                        register_artifact(
                            path,
                            category="Manifolds",
                            title=f'Latent geometry colored by "{color_title}"',
                            feature_set="counts",
                            target=color_key,
                            run_id=run_id,
                        )
            results.append({
                "feature_set": "counts",
                "source": "publication_plot_fig_latent_geometry",
                "figures": list(all_figures),
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "feature_set": "counts",
                "source": "publication_plot_fig_latent_geometry",
                "error": str(exc),
            })
            used_publication = False

    # --- remaining feature sets (or counts without metrics) via UI fits ----
    fit_sets = [fs for fs in feature_sets if not (fs == "counts" and used_publication)]
    fit_jobs: list[tuple[str, str, float]] = []
    for fs in fit_sets:
        for m in req.manifolds:
            emb = resolve_manifold_alias(m)
            if not embedding_compatible_with_feature_set(emb, fs):
                continue
            label = "counts" if emb == "identity" else emb
            w = (
                _optimal_window_for_mode(req.input_dir, label, fallback=preferred)
                if fs == "counts"
                else preferred
            )
            fit_jobs.append((fs, m, float(w)))

    panels_by_fs: dict[str, list[LatentGeometryPanel]] = {fs: [] for fs in fit_sets}
    n_fits = max(len(fit_jobs), 1)
    n_pages = len(fit_sets) * len(color_pages)
    total_steps = max(n_fits + n_pages + (1 if used_publication else 0), 1)
    step = 1 if used_publication else 0

    for fs, manifold, window in fit_jobs:
        step += 1
        if progress_callback:
            progress_callback(
                f"Fit `{manifold}` on `{fs}` @ {window}s", step, total_steps,
            )
        try:
            diag = compute_manifold_diagnostics(
                req.input_dir,
                manifold,
                feature_set=fs,
                spike_source=req.spike_source,
                decode_window=window,
                n_components=req.n_components,
                comparison_root=None,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({
                "feature_set": fs,
                "manifold": manifold,
                "decode_window_s": window,
                "error": str(exc),
            })
            continue

        mode = resolve_manifold_alias(manifold)
        if mode == "identity":
            mode = "counts"
        nn = None
        meta_ex = (diag.extras or {}).get("metadata") or {}
        if isinstance(meta_ex, dict) and meta_ex.get("n_neighbors") is not None:
            try:
                nn = int(meta_ex["n_neighbors"])
            except (TypeError, ValueError):
                nn = None
        panels_by_fs.setdefault(fs, []).append(
            LatentGeometryPanel(
                mode=mode,
                Z=diag.latent,
                behavior=diag.behavior,
                decode_window_s=float(window),
                n_components=int(diag.n_components),
                n_neighbors=nn,
            ),
        )
        results.append({
            "feature_set": fs,
            "manifold": manifold,
            "embedding_type": diag.embedding_type,
            "decode_window_s": window,
            "n_components": diag.n_components,
            "latent_shape": list(diag.latent.shape),
            "n_neural_features": (diag.extras or {}).get("n_neural_features"),
            "from_cache": diag.from_cache,
        })

    for fs, panels in panels_by_fs.items():
        if not panels:
            continue
        order: list[LatentGeometryPanel] = []
        for m in req.manifolds:
            emb = resolve_manifold_alias(m)
            label = "counts" if emb == "identity" else emb
            for p in panels:
                if p.mode == label and p not in order:
                    order.append(p)
        ordered = order or panels

        for color_key, color_title in color_pages:
            step += 1
            if progress_callback:
                progress_callback(
                    f"Page `{color_key}` · `{fs}`", step, total_steps,
                )
            stem = _geometry_stem(color_key, fs)
            pub_path = pub_fig_dir / f"{stem}.png"
            path = plot_latent_geometry_page_from_embeddings(
                ordered, color_key, color_title, pub_path,
            )
            if path is None:
                continue
            run_copy = run_fig_dir / path.name
            run_copy.write_bytes(path.read_bytes())
            all_figures.append(str(path))
            register_artifact(
                path,
                category="Manifolds",
                title=f'Latent geometry colored by "{color_title}"',
                feature_set=fs,
                target=color_key,
                decode_window=ordered[0].decode_window_s if ordered else preferred,
                run_id=run_id,
            )

    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": try_git_commit(),
        "request": {
            "input_dir": str(req.input_dir),
            "feature_sets": list(feature_sets),
            "manifolds": list(req.manifolds),
            "decode_windows": list(windows),
            "preferred_window_s": preferred,
            "n_components": req.n_components,
            "spike_source": req.spike_source,
            "behavioral_colors": [k for k, _ in color_pages],
            "used_publication_latent_geometry": used_publication,
        },
        "n_manifold_fits": len([r for r in results if "error" not in r and "manifold" in r]),
        "n_geometry_pages": len(all_figures),
        "n_jobs_run": len([r for r in results if "error" not in r]),
        "results": results,
        "figures": all_figures,
        "output_dir": str(out_dir),
        "figures_dir": str(pub_fig_dir),
    }
    (out_dir / "analysis_config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n",
    )
    return meta
