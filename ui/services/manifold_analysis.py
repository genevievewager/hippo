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
    # When False (default), skip fits / pages already present for this dataset.
    force_recompute: bool = False

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


def _norm_manifold_label(name: str) -> str:
    emb = resolve_manifold_alias(str(name))
    return "counts" if emb == "identity" else emb


def _window_key(w: float) -> float:
    """Round windows for stable set membership (ms precision)."""
    return round(float(w), 3)


def _fit_key(feature_set: str, manifold: str, decode_window_s: float | None = None) -> tuple:
    base = (str(feature_set), _norm_manifold_label(manifold))
    if decode_window_s is None:
        return base
    return (*base, _window_key(decode_window_s))


def _geometry_pages_exist(
    input_dir: Path,
    feature_set: str,
    color_pages: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> bool:
    """True when published latent-geometry PNGs for this feature set already exist."""
    pub_fig_dir = Path(input_dir) / "figures" / FIGURE_SUBDIR_DECODER
    if not pub_fig_dir.is_dir():
        return False
    for color_key, _ in color_pages:
        stem = _geometry_stem(color_key, feature_set)
        if not (pub_fig_dir / f"{stem}.png").exists():
            return False
    return bool(color_pages)


def planned_manifold_fit_jobs(
    input_dir: Path | str,
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...] | None = None,
    *,
    skip_counts_publication: bool = False,
) -> list[tuple[str, str, float]]:
    """Checkpoint jobs: full Cartesian of compatible (F, E) × selected windows.

    Each selected window is fitted and written into the shared
    ``manifold_transforms`` cache so Decoder Benchmark can reuse them.
    Geometry pages still use :func:`geometry_manifold_fit_jobs` (one W per pair).
    """
    del input_dir  # kept for call-site compatibility
    windows = tuple(float(w) for w in (decode_windows or [])) or (0.250,)
    fit_sets = list(feature_sets)
    if skip_counts_publication:
        fit_sets = [fs for fs in fit_sets if fs != "counts"]
    jobs: list[tuple[str, str, float]] = []
    for fs in fit_sets:
        for m in manifolds:
            emb = resolve_manifold_alias(m)
            if not embedding_compatible_with_feature_set(emb, fs):
                continue
            for w in windows:
                jobs.append((str(fs), str(m), float(w)))
    return jobs


def geometry_manifold_fit_jobs(
    input_dir: Path | str,
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...] | None = None,
    *,
    skip_counts_publication: bool = False,
) -> list[tuple[str, str, float]]:
    """One preferred/optimal window per compatible (F, E) for geometry pages."""
    input_dir = Path(input_dir)
    windows = tuple(float(w) for w in (decode_windows or [])) or (0.250,)
    preferred = _preferred_window(windows)
    fit_sets = list(feature_sets)
    if skip_counts_publication:
        fit_sets = [fs for fs in fit_sets if fs != "counts"]
    jobs: list[tuple[str, str, float]] = []
    for fs in fit_sets:
        for m in manifolds:
            emb = resolve_manifold_alias(m)
            if not embedding_compatible_with_feature_set(emb, fs):
                continue
            label = _norm_manifold_label(m)
            w = (
                _optimal_window_for_mode(input_dir, label, fallback=preferred)
                if fs == "counts"
                else preferred
            )
            # Prefer an explicit selected window when optimal W is outside selection.
            if _window_key(w) not in {_window_key(x) for x in windows}:
                w = preferred
            jobs.append((str(fs), str(m), float(w)))
    return jobs


def _prior_completed_fit_keys(
    runs: list[dict[str, Any]],
    *,
    spike_source: str | None = None,
    n_components: int | None = None,
) -> tuple[set[tuple], set[tuple], list[str]]:
    """Return (keys_with_window, keys_fs_manifold, matching_run_ids)."""
    with_w: set[tuple] = set()
    fs_m: set[tuple] = set()
    matching: list[str] = []
    for run in runs:
        if spike_source and run.get("spike_source") and run["spike_source"] != spike_source:
            continue
        if (
            n_components is not None
            and run.get("n_components") is not None
            and int(run["n_components"]) != int(n_components)
        ):
            continue
        hit_any = False
        for cell in run.get("completed_fits") or []:
            fs = cell.get("feature_set")
            m = cell.get("manifold") or cell.get("embedding_type")
            if not fs or not m:
                continue
            fs_m.add(_fit_key(str(fs), str(m)))
            w = cell.get("decode_window_s")
            if w is not None:
                with_w.add(_fit_key(str(fs), str(m), float(w)))
            hit_any = True
        # Legacy summaries without per-fit rows: cartesian of request lists.
        if not (run.get("completed_fits") or []):
            for fs in run.get("feature_sets") or []:
                for m in run.get("manifolds") or []:
                    fs_m.add(_fit_key(str(fs), str(m)))
                    for w in run.get("decode_windows") or []:
                        with_w.add(_fit_key(str(fs), str(m), float(w)))
                        hit_any = True
                    if not (run.get("decode_windows") or []):
                        hit_any = True
        if hit_any:
            matching.append(str(run["run_id"]))
    return with_w, fs_m, matching


def job_is_covered(
    feature_set: str,
    manifold: str,
    decode_window_s: float,
    *,
    covered_with_window: set[tuple],
    covered_fs_m: set[tuple] | None = None,
    require_exact_window: bool = True,
) -> bool:
    """Match exact (F, E, W). Optional legacy (F, E) fallback when exact W unknown."""
    exact = _fit_key(feature_set, manifold, decode_window_s)
    if exact in covered_with_window:
        return True
    if require_exact_window:
        return False
    if covered_fs_m is None:
        return False
    return _fit_key(feature_set, manifold) in covered_fs_m


def _shared_transform_exists(
    input_dir: Path,
    feature_set: str,
    manifold: str,
    decode_window_s: float,
    *,
    n_components: int,
    spike_source: str,
) -> bool:
    """True when a reloadable manifold transform is already on disk."""
    from realtime.transform_cache import (
        discover_comparison_roots,
        find_manifold_transform_in_roots,
        preferred_comparison_root,
    )

    roots: list[Path] = []
    preferred = preferred_comparison_root(Path(input_dir), spike_source=spike_source)
    if preferred is not None:
        roots.append(preferred)
    roots.extend(discover_comparison_roots(Path(input_dir)))
    if not roots:
        return False
    return (
        find_manifold_transform_in_roots(
            roots,
            feature_set=feature_set,
            embedding_type=resolve_manifold_alias(manifold),
            decode_window=float(decode_window_s),
            n_components=int(n_components),
        )
        is not None
    )


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
    """Fit manifolds and write PDF-format latent-geometry pages.

    By default, skips feature×manifold fits (and geometry pages) already
    present for this dataset. Pass ``force_recompute=True`` to regenerate.
    """
    run_id = new_run_id(prefix="manifold")
    out_dir = _analysis_dir(req.input_dir, run_id)
    run_fig_dir = out_dir / "figures"
    run_fig_dir.mkdir(parents=True, exist_ok=True)
    pub_fig_dir = Path(req.input_dir) / "figures" / FIGURE_SUBDIR_DECODER
    pub_fig_dir.mkdir(parents=True, exist_ok=True)

    color_pages = req.resolved_behavioral_colors()
    windows = tuple(float(w) for w in req.decode_windows) or (0.250,)
    from realtime.transform_cache import assert_cached_decode_windows

    assert_cached_decode_windows(
        req.input_dir,
        windows,
        spike_source=req.spike_source,
    )
    preferred = _preferred_window(windows)
    feature_sets = req.resolved_feature_sets()
    all_figures: list[str] = []
    results: list[dict[str, Any]] = []
    n_skipped = 0
    n_computed = 0

    force = bool(req.force_recompute)

    # --- counts + existing decoder comparison → exact PDF plotter ----------
    used_publication = False
    if "counts" in feature_sets and has_decoder_comparison(req.input_dir):
        pages_ready = _geometry_pages_exist(req.input_dir, "counts", color_pages)
        if not force and pages_ready:
            used_publication = True
            n_skipped += 1
            if progress_callback:
                progress_callback("Reusing existing publication latent-geometry suite…", 1, 2)
            for color_key, color_title in (
                COLOR_FEATURES if req.behavioral_colors is None else color_pages
            ):
                stem = _geometry_stem(color_key, "counts")
                p = pub_fig_dir / f"{stem}.png"
                if p.exists():
                    all_figures.append(str(p))
                    run_copy = run_fig_dir / p.name
                    if not run_copy.exists():
                        run_copy.write_bytes(p.read_bytes())
                    register_artifact(
                        p,
                        category="Manifolds",
                        title=f'Latent geometry colored by "{color_title}"',
                        feature_set="counts",
                        target=color_key,
                        run_id=run_id,
                    )
            results.append({
                "feature_set": "counts",
                "source": "publication_plot_fig_latent_geometry",
                "skipped": True,
                "figures": list(all_figures),
            })
        else:
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
                n_computed += 1
                results.append({
                    "feature_set": "counts",
                    "source": "publication_plot_fig_latent_geometry",
                    "skipped": False,
                    "figures": list(all_figures),
                })
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "feature_set": "counts",
                    "source": "publication_plot_fig_latent_geometry",
                    "error": str(exc),
                })
                used_publication = False

    # --- checkpoint every F×E×W; geometry pages use one W per pair ---------
    # Always include counts in the checkpoint grid even when publication
    # figures already cover the visual suite.
    fit_sets = [fs for fs in feature_sets if not (fs == "counts" and used_publication)]
    checkpoint_jobs = planned_manifold_fit_jobs(
        req.input_dir,
        feature_sets,
        req.manifolds,
        windows,
        skip_counts_publication=False,
    )
    geometry_jobs = geometry_manifold_fit_jobs(
        req.input_dir,
        fit_sets,
        req.manifolds,
        windows,
        skip_counts_publication=False,
    )
    geometry_key_set = {
        (fs, m, _window_key(w)) for fs, m, w in geometry_jobs
    }

    jobs_to_run: list[tuple[str, str, float]] = []
    jobs_skipped: list[tuple[str, str, float]] = []
    for fs, m, w in checkpoint_jobs:
        on_disk = _shared_transform_exists(
            req.input_dir,
            fs,
            m,
            w,
            n_components=req.n_components,
            spike_source=req.spike_source,
        )
        # Skip only when a reloadable transform already exists. Prior figure-only
        # analysis runs do not count — they may lack shared checkpoints.
        if not force and on_disk:
            jobs_skipped.append((fs, m, w))
            n_skipped += 1
            results.append({
                "feature_set": fs,
                "manifold": m,
                "decode_window_s": w,
                "skipped": True,
                "from_cache": True,
                "transform_on_disk": True,
            })
        else:
            jobs_to_run.append((fs, m, w))

    dirty_fs: set[str] = set()
    for fs in fit_sets:
        if force or not _geometry_pages_exist(req.input_dir, fs, color_pages):
            dirty_fs.add(fs)

    # Reload geometry panels from skipped transforms when pages need rewrite.
    load_for_page: list[tuple[str, str, float]] = []
    skipped_set = set(jobs_skipped)
    seen_page: set[tuple] = set()
    for fs, m, w in list(jobs_to_run) + list(jobs_skipped):
        gkey = (fs, m, _window_key(w))
        if gkey not in geometry_key_set:
            continue
        if fs not in dirty_fs and (fs, m, w) in skipped_set:
            continue
        if gkey in seen_page:
            continue
        seen_page.add(gkey)
        load_for_page.append((fs, m, w))

    panels_by_fs: dict[str, list[LatentGeometryPanel]] = {fs: [] for fs in fit_sets}
    n_fits = max(len(jobs_to_run) + len(load_for_page), 1)
    n_pages = len(dirty_fs) * len(color_pages)
    total_steps = max(
        len(jobs_to_run) + len(load_for_page) + n_pages + (
            1 if ("counts" in feature_sets and has_decoder_comparison(req.input_dir)) else 0
        ),
        1,
    )
    step = 1 if ("counts" in feature_sets and has_decoder_comparison(req.input_dir)) else 0

    # Persist / refresh shared transform checkpoints for the full window grid.
    fitted_diags: dict[tuple, Any] = {}
    for fs, manifold, window in jobs_to_run:
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
                persist=True,
                force_refit=force,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({
                "feature_set": fs,
                "manifold": manifold,
                "decode_window_s": window,
                "error": str(exc),
            })
            continue
        fitted_diags[(fs, manifold, _window_key(window))] = diag
        n_computed += 1
        results.append({
            "feature_set": fs,
            "manifold": manifold,
            "embedding_type": diag.embedding_type,
            "decode_window_s": window,
            "n_components": diag.n_components,
            "latent_shape": list(diag.latent.shape),
            "n_neural_features": (diag.extras or {}).get("n_neural_features"),
            "from_cache": diag.from_cache,
            "saved_path": (diag.extras or {}).get("saved_path"),
            "skipped": False,
        })

    for fs, manifold, window in load_for_page:
        step += 1
        gkey = (fs, manifold, _window_key(window))
        diag = fitted_diags.get(gkey)
        is_skip_reload = diag is None
        if progress_callback:
            verb = "Reload" if is_skip_reload else "Panel"
            progress_callback(
                f"{verb} `{manifold}` on `{fs}` @ {window}s", step, total_steps,
            )
        if diag is None:
            try:
                diag = compute_manifold_diagnostics(
                    req.input_dir,
                    manifold,
                    feature_set=fs,
                    spike_source=req.spike_source,
                    decode_window=window,
                    n_components=req.n_components,
                    comparison_root=None,
                    persist=True,
                    force_refit=False,
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

    for fs in fit_sets:
        if fs not in dirty_fs:
            for color_key, color_title in color_pages:
                stem = _geometry_stem(color_key, fs)
                pub_path = pub_fig_dir / f"{stem}.png"
                if not pub_path.exists():
                    continue
                run_copy = run_fig_dir / pub_path.name
                if not run_copy.exists():
                    run_copy.write_bytes(pub_path.read_bytes())
                all_figures.append(str(pub_path))
                register_artifact(
                    pub_path,
                    category="Manifolds",
                    title=f'Latent geometry colored by "{color_title}"',
                    feature_set=fs,
                    target=color_key,
                    run_id=run_id,
                )
            continue

        panels = panels_by_fs.get(fs) or []
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
            n_computed += 1
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
            "force_recompute": force,
        },
        "n_manifold_fits": len(
            [r for r in results if "error" not in r and r.get("manifold") and not r.get("skipped")]
        ),
        "n_skipped": n_skipped,
        "n_computed": n_computed,
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



_MANIFOLD_FIT_SEC = 18.0
_MANIFOLD_PAGE_SEC = 2.5
_MANIFOLD_OVERHEAD_S = 15.0
_MANIFOLD_PUB_SUITE_SEC = 25.0
_REFERENCE_SESSION_S = 600.0


def list_manifold_analysis_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    """Discover prior ``manifolds/<run_id>/`` analyses (newest first)."""
    root = Path(experiment_dir) / "manifolds"
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
        manifolds = list(req.get("manifolds") or [])
        decode_windows = [float(w) for w in (req.get("decode_windows") or [])]
        if not feature_sets or not manifolds:
            for row in meta.get("results") or []:
                fs = row.get("feature_set")
                if fs and fs not in feature_sets:
                    feature_sets.append(str(fs))
                m = row.get("manifold") or row.get("embedding_type")
                if m and str(m) not in manifolds:
                    manifolds.append(str(m))
                w = row.get("decode_window_s")
                if w is not None:
                    wf = float(w)
                    if wf not in decode_windows:
                        decode_windows.append(wf)
        fig_dir = path / "figures"
        n_figs = len(list(fig_dir.glob("*.png"))) if fig_dir.is_dir() else 0
        pub_n = len(meta.get("figures") or [])
        completed_fits: list[dict[str, Any]] = []
        for row in meta.get("results") or []:
            if row.get("error"):
                continue
            # Publication suite covers counts modes without per-manifold rows.
            if row.get("source") == "publication_plot_fig_latent_geometry":
                for m in (req.get("manifolds") or manifolds or ["counts"]):
                    completed_fits.append({
                        "feature_set": "counts",
                        "manifold": m,
                        "decode_window_s": req.get("preferred_window_s"),
                        "from_publication": True,
                    })
                continue
            m = row.get("manifold") or row.get("embedding_type")
            fs = row.get("feature_set")
            if not fs or not m:
                continue
            completed_fits.append({
                "feature_set": str(fs),
                "manifold": str(m),
                "decode_window_s": (
                    float(row["decode_window_s"])
                    if row.get("decode_window_s") is not None
                    else None
                ),
                "skipped": bool(row.get("skipped")),
            })
        runs.append({
            "run_id": meta.get("run_id") or path.name,
            "created_at": meta.get("created_at"),
            "output_dir": str(path),
            "feature_sets": feature_sets,
            "manifolds": manifolds,
            "decode_windows": decode_windows,
            "n_components": req.get("n_components"),
            "spike_source": req.get("spike_source"),
            "n_figures": max(n_figs, pub_n),
            "n_jobs_run": meta.get("n_jobs_run"),
            "n_geometry_pages": meta.get("n_geometry_pages"),
            "n_skipped": meta.get("n_skipped"),
            "completed_fits": completed_fits,
            "used_publication": bool(
                (req.get("used_publication_latent_geometry"))
            ),
        })
    runs.sort(
        key=lambda r: str(r.get("created_at") or r.get("run_id") or ""),
        reverse=True,
    )
    return runs


def selection_already_covered(
    runs: list[dict[str, Any]],
    *,
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...] | None = None,
    spike_source: str | None = None,
    n_components: int | None = None,
    input_dir: Path | str | None = None,
    require_geometry_pages: bool = True,
) -> dict[str, Any]:
    """Summarize coverage of planned F×E×W checkpoint fits for this dataset.

    A job is covered only when a shared ``manifold_transforms`` checkpoint exists
    (and geometry pages when ``require_geometry_pages``). Prior figure-only
    analysis runs are listed in ``matching_run_ids`` but do not alone cover a job.
    """
    windows = [float(w) for w in (decode_windows or [])] or [0.250]
    spike = spike_source or "sorted"
    k = int(n_components) if n_components is not None else 3

    if input_dir is not None:
        jobs = planned_manifold_fit_jobs(
            input_dir,
            feature_sets,
            manifolds,
            windows,
            skip_counts_publication=False,
        )
    else:
        jobs = []
        for fs in feature_sets:
            for m in manifolds:
                emb = resolve_manifold_alias(m)
                if not embedding_compatible_with_feature_set(emb, fs):
                    continue
                for w in windows:
                    jobs.append((str(fs), str(m), float(w)))

    _, _, matching_run_ids = _prior_completed_fit_keys(
        runs, spike_source=spike_source, n_components=n_components,
    )

    if not jobs:
        return {
            "n_wanted": 0,
            "n_covered": 0,
            "n_to_compute": 0,
            "covered": set(),
            "missing": set(),
            "matching_run_ids": matching_run_ids,
            "fully_covered": False,
            "planned_jobs": [],
        }

    covered: set[tuple] = set()
    missing: set[tuple] = set()
    for fs, m, w in jobs:
        key = _fit_key(fs, m, w)
        on_disk = False
        if input_dir is not None:
            on_disk = _shared_transform_exists(
                Path(input_dir),
                fs,
                m,
                w,
                n_components=k,
                spike_source=spike,
            )
        pages_ok = True
        if require_geometry_pages and input_dir is not None:
            pages_ok = _geometry_pages_exist(Path(input_dir), fs, COLOR_FEATURES)
        if on_disk and pages_ok:
            covered.add(key)
        elif on_disk and not require_geometry_pages:
            covered.add(key)
        else:
            missing.add(key)

    n_wanted = len(jobs)
    n_covered = len(covered)
    return {
        "n_wanted": n_wanted,
        "n_covered": n_covered,
        "n_to_compute": max(n_wanted - n_covered, 0),
        "covered": covered,
        "missing": missing,
        "matching_run_ids": matching_run_ids,
        "fully_covered": not missing and n_wanted > 0,
        "planned_jobs": jobs,
    }


def estimate_manifold_workload(
    *,
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...] | None = None,
    n_behavioral_colors: int | None = None,
    session_duration_s: float | None = None,
    has_decoder_comparison: bool = False,
) -> dict[str, Any]:
    """Heuristic runtime for a manifold analysis selection (display only)."""
    from ui.services.comparison import format_duration, valid_feature_manifold_pairs

    pairs = valid_feature_manifold_pairs(feature_sets, manifolds)
    n_pairs = len(pairs)
    n_fs = max(len(feature_sets), 1) if feature_sets else 0
    n_windows = max(len(decode_windows or []), 1)
    n_colors = int(n_behavioral_colors) if n_behavioral_colors is not None else len(COLOR_FEATURES)
    # Full F×E×W checkpoint grid, plus one geometry page per color × feature set.
    n_fits = n_pairs * n_windows
    n_pages = n_fs * max(n_colors, 1)
    planned = max(n_fits, 0) + max(n_pages, 0)
    duration_scale = 1.0
    if session_duration_s is not None and session_duration_s > 0:
        duration_scale = max(float(session_duration_s) / _REFERENCE_SESSION_S, 0.25)

    use_pub = bool(has_decoder_comparison and any(fs == "counts" for fs in feature_sets))
    # Publication regenerates counts geometry, but checkpoints still fit the
    # full selected window grid for decode reuse.
    fit_jobs = n_fits

    estimated_s = (
        _MANIFOLD_OVERHEAD_S
        + (_MANIFOLD_PUB_SUITE_SEC if use_pub else 0.0)
        + float(fit_jobs) * _MANIFOLD_FIT_SEC * duration_scale
        + float(n_pages) * _MANIFOLD_PAGE_SEC
    )
    low_s = max(estimated_s * 0.7, _MANIFOLD_OVERHEAD_S)
    high_s = estimated_s * 1.6
    return {
        "planned_configurations": int(planned),
        "n_feature_sets": len(feature_sets),
        "n_manifolds": len(manifolds),
        "n_windows": n_windows,
        "n_valid_feature_manifold_pairs": n_pairs,
        "n_geometry_pages": n_pages,
        "detail_label": (
            f"{n_fits} feature×manifold×window checkpoint(s) · "
            f"{n_pages} geometry page(s)"
            + (" · + publication suite" if use_pub else "")
        ),
        "estimated_runtime_s": float(estimated_s),
        "estimated_runtime_low_s": float(low_s),
        "estimated_runtime_high_s": float(high_s),
        "estimated_runtime_label": format_duration(estimated_s),
        "estimated_runtime_range_label": (
            f"{format_duration(low_s)} – {format_duration(high_s)}"
        ),
    }


def count_published_geometry_pages(experiment_dir: Path) -> int:
    """Count published ``fig_latent_geometry_*.png`` under figures/decoder_comparison."""
    pub = Path(experiment_dir) / "figures" / FIGURE_SUBDIR_DECODER
    if not pub.is_dir():
        return 0
    return len(list(pub.glob("fig_latent_geometry_*.png")))
