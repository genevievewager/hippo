"""Ridge-only quadrant diagnostics: center bias, calibration, regularization sweeps."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.search_space import is_dynamic_embedding, resolve_manifold_alias
from realtime.spike_features import build_causal_spike_matrix
from realtime.timing import extract_behavior_times, resolve_update_dt_s
from realtime.train_decoder import (
    align_behavior_to_decoder_times,
    causal_train_test_split,
    infer_arena_bounds,
)

RIDGE_ALPHA_GRID: tuple[float, ...] = (
    1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0,
)
DIM_SWEEP_GRID: tuple[int, ...] = (2, 5, 10, 20, 30, 50)
DEFAULT_RIDGE_ALPHA = 1.0
_GALLERY_FEATURE_SET = "counts"
_EPS = 1e-6

# Quadrant-framed ridge comparison (dynamic_nonlinear reserved for future methods).
RIDGE_QUADRANT_METHODS: tuple[dict[str, str], ...] = (
    {"id": "counts", "quadrant": "baseline", "label": "Counts", "embedding": "counts"},
    {"id": "global_pca", "quadrant": "static_linear", "label": "Global PCA", "embedding": "global_pca"},
    {"id": "region_pca", "quadrant": "static_linear", "label": "Region PCA", "embedding": "region_pca"},
    {
        "id": "diffusion_nystrom",
        "quadrant": "static_nonlinear",
        "label": "Diffusion + Nyström",
        "embedding": "diffusion_nystrom",
    },
    {"id": "global_lds", "quadrant": "dynamic_linear", "label": "LDS", "embedding": "global_lds"},
    {
        "id": "dynamic_nonlinear",
        "quadrant": "dynamic_nonlinear",
        "label": "Dynamic nonlinear (TODO)",
        "embedding": "",
    },
)


@dataclass
class CalibrationMetrics:
    slope_x: float
    intercept_x: float
    r2_x: float
    slope_y: float
    intercept_y: float
    r2_y: float
    slope_r: float
    intercept_r: float
    r2_r: float
    mean_radial_bias: float
    contraction_ratio: float
    mean_position_error_cm: float

    def to_dict(self) -> dict[str, float]:
        return {
            "slope_x": self.slope_x,
            "intercept_x": self.intercept_x,
            "r2_x": self.r2_x,
            "slope_y": self.slope_y,
            "intercept_y": self.intercept_y,
            "r2_y": self.r2_y,
            "slope_r": self.slope_r,
            "intercept_r": self.intercept_r,
            "r2_r": self.r2_r,
            "mean_radial_bias": self.mean_radial_bias,
            "contraction_ratio": self.contraction_ratio,
            "mean_position_error_cm": self.mean_position_error_cm,
        }


@dataclass
class MethodDiagnostics:
    method_id: str
    label: str
    quadrant: str
    embedding: str
    available: bool
    skip_reason: str | None = None
    decoded: pd.DataFrame | None = None
    metrics: CalibrationMetrics | None = None
    alpha_sweep: pd.DataFrame | None = None
    dim_sweep: pd.DataFrame | None = None
    coef_norm_vs_alpha: pd.DataFrame | None = None
    occupancy_weighted: CalibrationMetrics | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _fit_axis_calibration(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    lr = LinearRegression().fit(true.reshape(-1, 1), pred)
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)
    r2 = float(r2_score(true, pred))
    return slope, intercept, r2


def _arena_center(behavior_train: pd.DataFrame, summary: dict) -> tuple[float, float]:
    x_min, x_max, y_min, y_max = infer_arena_bounds(behavior_train, summary)
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def compute_calibration_metrics(
    true_xy: np.ndarray,
    pred_xy: np.ndarray,
    center: tuple[float, float],
) -> CalibrationMetrics:
    true_xy = np.asarray(true_xy, dtype=float)
    pred_xy = np.asarray(pred_xy, dtype=float)
    sx, bx, r2x = _fit_axis_calibration(true_xy[:, 0], pred_xy[:, 0])
    sy, by, r2y = _fit_axis_calibration(true_xy[:, 1], pred_xy[:, 1])
    cx, cy = center
    r_true = np.linalg.norm(true_xy - np.array([cx, cy]), axis=1)
    r_pred = np.linalg.norm(pred_xy - np.array([cx, cy]), axis=1)
    sr, br, r2r = _fit_axis_calibration(r_true, r_pred)
    err = np.linalg.norm(pred_xy - true_xy, axis=1)
    denom = np.maximum(r_true, _EPS)
    return CalibrationMetrics(
        slope_x=sx,
        intercept_x=bx,
        r2_x=r2x,
        slope_y=sy,
        intercept_y=by,
        r2_y=r2y,
        slope_r=sr,
        intercept_r=br,
        r2_r=r2r,
        mean_radial_bias=float(np.mean(r_pred - r_true)),
        contraction_ratio=float(np.mean(r_pred / denom)),
        mean_position_error_cm=float(np.mean(err)),
    )


def fit_ridge_position(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    ridge = Ridge(alpha=float(alpha))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", ridge),
    ])
    if sample_weight is not None:
        pipe.fit(X_train, y_train, ridge__sample_weight=sample_weight)
    else:
        pipe.fit(X_train, y_train)
    return pipe


def _ridge_coef_norm(pipe: Pipeline) -> float:
    ridge = pipe.named_steps["ridge"]
    coef = getattr(ridge, "coef_", None)
    if coef is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(coef)))


def _comparison_roots(experiment_dir: Path, spike_source: str) -> list[Path]:
    from realtime.transform_cache import discover_comparison_roots, preferred_comparison_root

    roots: list[Path] = []
    seen: set[Path] = set()
    preferred = preferred_comparison_root(experiment_dir, spike_source=spike_source)
    for root in (*([preferred] if preferred is not None else []), *discover_comparison_roots(experiment_dir)):
        if root is None:
            continue
        key = root.resolve() if root.exists() else root
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _find_cached_transform(
    experiment_dir: Path,
    spike_source: str,
    embedding_type: str,
    decode_window: float,
    n_components: int,
) -> Path | None:
    from realtime.transform_cache import find_manifold_transform_in_roots

    roots = _comparison_roots(experiment_dir, spike_source)
    if not roots:
        return None
    return find_manifold_transform_in_roots(
        roots,
        feature_set=_GALLERY_FEATURE_SET,
        embedding_type=resolve_manifold_alias(embedding_type),
        decode_window=float(decode_window),
        n_components=int(n_components),
    )


def _load_or_fit_transformer(
    *,
    experiment_dir: Path,
    spike_source: str,
    embedding_type: str,
    decode_window: float,
    n_components: int,
    X_counts_train: np.ndarray,
    data: dict,
    update_dt: float,
):
    emb = resolve_manifold_alias(embedding_type)
    hit = _find_cached_transform(
        experiment_dir, spike_source, emb, decode_window, n_components,
    )
    if hit is not None:
        from realtime.transform_cache import try_load_manifold

        loaded = try_load_manifold(Path(hit))
        if loaded is not None:
            return loaded, str(hit), "cached"

    from realtime.manifold_features import make_feature_transformer

    transformer = make_feature_transformer(
        emb,
        decode_window=float(decode_window),
        n_components=int(n_components),
        units_df=data["units_df"],
        unit_ids=data["unit_ids"],
        update_dt=float(update_dt),
        spike_source=spike_source,
    )
    if transformer is None:
        raise ValueError(f"Could not build transformer for {embedding_type!r}")
    transformer.fit(X_counts_train)
    return transformer, None, "fit_on_train"


def transform_features(
    transformer,
    embedding_type: str,
    X_counts: np.ndarray,
    *,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    emb = resolve_manifold_alias(embedding_type)
    if is_dynamic_embedding(emb):
        X_train = transformer.transform(X_counts[train_mask], causal=True, reset=True)
        X_test = transformer.transform(X_counts[test_mask], causal=True, reset=False)
    else:
        X_train = transformer.transform(X_counts[train_mask])
        X_test = transformer.transform(X_counts[test_mask])
    return np.asarray(X_train), np.asarray(X_test)


def build_representation_matrix(
    *,
    experiment_dir: Path,
    spike_source: str,
    embedding_type: str,
    decode_window: float,
    n_components: int,
    X_counts: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    data: dict,
    update_dt: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    if resolve_manifold_alias(embedding_type) == "counts":
        return X_counts[train_mask], X_counts[test_mask], "identity"
    transformer, _, source = _load_or_fit_transformer(
        experiment_dir=experiment_dir,
        spike_source=spike_source,
        embedding_type=embedding_type,
        decode_window=decode_window,
        n_components=n_components,
        X_counts_train=X_counts[train_mask],
        data=data,
        update_dt=update_dt,
    )
    X_train, X_test = transform_features(
        transformer, embedding_type, X_counts,
        train_mask=train_mask, test_mask=test_mask,
    )
    return X_train, X_test, source


def _decoded_dataframe(
    beh_test: pd.DataFrame,
    pred_xy: np.ndarray,
    *,
    decode_window: float,
    method_id: str,
    alpha: float,
) -> pd.DataFrame:
    pred_xy = np.asarray(pred_xy, dtype=float)
    true_xy = beh_test[["x", "y"]].to_numpy(dtype=float)
    err = np.linalg.norm(pred_xy - true_xy, axis=1)
    return pd.DataFrame({
        "time": beh_test["time"].to_numpy(),
        "true_x": true_xy[:, 0],
        "true_y": true_xy[:, 1],
        "decoded_x": pred_xy[:, 0],
        "decoded_y": pred_xy[:, 1],
        "position_error_cm": err,
        "decode_window_s": float(decode_window),
        "method_id": method_id,
        "ridge_alpha": float(alpha),
    })


def compute_occupancy_weights(
    xy: np.ndarray,
    arena_bounds: tuple[float, float, float, float],
    *,
    n_bins: int = 20,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = arena_bounds
    xy = np.asarray(xy, dtype=float)
    hist, x_edges, y_edges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=n_bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    ix = np.clip(np.digitize(xy[:, 0], x_edges) - 1, 0, n_bins - 1)
    iy = np.clip(np.digitize(xy[:, 1], y_edges) - 1, 0, n_bins - 1)
    occ = hist[ix, iy].astype(float)
    weights = 1.0 / (occ + _EPS)
    weights *= len(weights) / np.sum(weights)
    return weights


def compute_occupancy_map(
    xy: np.ndarray,
    arena_bounds: tuple[float, float, float, float],
    *,
    n_bins: int = 30,
) -> dict[str, Any]:
    x_min, x_max, y_min, y_max = arena_bounds
    hist, x_edges, y_edges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=n_bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    return {
        "hist": hist.tolist(),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "n_bins": int(n_bins),
    }



def _run_alpha_sweep_on_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    true_test: np.ndarray,
    center: tuple[float, float],
    *,
    method_id: str,
    alphas: tuple[float, ...] = RIDGE_ALPHA_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        pipe = fit_ridge_position(X_train, y_train, alpha=float(alpha))
        pred = pipe.predict(X_test)
        m = compute_calibration_metrics(true_test, pred, center)
        sweep_rows.append({
            "method_id": method_id,
            "alpha": float(alpha),
            **{k: v for k, v in m.to_dict().items() if k != "mean_position_error_cm"},
            "mean_position_error_cm": m.mean_position_error_cm,
        })
        norm_rows.append({
            "method_id": method_id,
            "alpha": float(alpha),
            "coef_l2_norm": _ridge_coef_norm(pipe),
        })
    return pd.DataFrame(sweep_rows), pd.DataFrame(norm_rows)


def _available_k_values(
    experiment_dir: Path,
    spike_source: str,
    embedding_type: str,
    decode_window: float,
    grid: tuple[int, ...] = DIM_SWEEP_GRID,
) -> list[int]:
    if resolve_manifold_alias(embedding_type) == "counts":
        return []
    out: list[int] = []
    for k in grid:
        if _find_cached_transform(experiment_dir, spike_source, embedding_type, decode_window, k):
            out.append(int(k))
    return out


def _run_dim_sweep(
    *,
    experiment_dir: Path,
    spike_source: str,
    embedding_type: str,
    decode_window: float,
    X_counts: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    beh_train: pd.DataFrame,
    beh_test: pd.DataFrame,
    data: dict,
    update_dt: float,
    center: tuple[float, float],
    method_id: str,
    k_values: list[int],
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_train = beh_train[["x", "y"]].to_numpy()
    true_test = beh_test[["x", "y"]].to_numpy()
    for k in k_values:
        try:
            X_train, X_test, _ = build_representation_matrix(
                experiment_dir=experiment_dir,
                spike_source=spike_source,
                embedding_type=embedding_type,
                decode_window=decode_window,
                n_components=int(k),
                X_counts=X_counts,
                train_mask=train_mask,
                test_mask=test_mask,
                data=data,
                update_dt=update_dt,
            )
        except (FileNotFoundError, ValueError):
            continue
        pipe = fit_ridge_position(X_train, y_train, alpha=alpha)
        pred = pipe.predict(X_test)
        m = compute_calibration_metrics(true_test, pred, center)
        rows.append({
            "method_id": method_id,
            "n_components": int(k),
            **m.to_dict(),
        })
    return pd.DataFrame(rows)


def _evaluate_method(
    *,
    experiment_dir: Path,
    spike_source: str,
    method: Mapping[str, str],
    decode_window: float,
    n_components: int,
    X_counts: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    beh_train: pd.DataFrame,
    beh_test: pd.DataFrame,
    data: dict,
    update_dt: float,
    center: tuple[float, float],
    arena_bounds: tuple[float, float, float, float],
    run_weighted: bool,
) -> MethodDiagnostics:
    mid = str(method["id"])
    label = str(method["label"])
    quadrant = str(method["quadrant"])
    embedding = str(method.get("embedding") or "")

    if quadrant == "dynamic_nonlinear" or not embedding:
        return MethodDiagnostics(
            method_id=mid,
            label=label,
            quadrant=quadrant,
            embedding=embedding,
            available=False,
            skip_reason="dynamic_nonlinear not implemented",
        )

    try:
        X_train, X_test, source = build_representation_matrix(
            experiment_dir=experiment_dir,
            spike_source=spike_source,
            embedding_type=embedding,
            decode_window=decode_window,
            n_components=n_components,
            X_counts=X_counts,
            train_mask=train_mask,
            test_mask=test_mask,
            data=data,
            update_dt=update_dt,
        )
    except (FileNotFoundError, ValueError) as exc:
        return MethodDiagnostics(
            method_id=mid,
            label=label,
            quadrant=quadrant,
            embedding=embedding,
            available=False,
            skip_reason=str(exc),
        )

    y_train = beh_train[["x", "y"]].to_numpy()
    true_test = beh_test[["x", "y"]].to_numpy()
    pipe = fit_ridge_position(X_train, y_train, alpha=DEFAULT_RIDGE_ALPHA)
    pred = pipe.predict(X_test)
    metrics = compute_calibration_metrics(true_test, pred, center)
    decoded = _decoded_dataframe(
        beh_test, pred,
        decode_window=decode_window,
        method_id=mid,
        alpha=DEFAULT_RIDGE_ALPHA,
    )
    alpha_sweep, coef_norm = _run_alpha_sweep_on_test(
        X_train, y_train, X_test, true_test, center, method_id=mid,
    )
    k_values = _available_k_values(
        experiment_dir, spike_source, embedding, decode_window,
    )
    if not k_values and resolve_manifold_alias(embedding) != "counts":
        k_values = [int(n_components)] if _find_cached_transform(
            experiment_dir, spike_source, embedding, decode_window, n_components,
        ) else []
    dim_sweep = _run_dim_sweep(
        experiment_dir=experiment_dir,
        spike_source=spike_source,
        embedding_type=embedding,
        decode_window=decode_window,
        X_counts=X_counts,
        train_mask=train_mask,
        test_mask=test_mask,
        beh_train=beh_train,
        beh_test=beh_test,
        data=data,
        update_dt=update_dt,
        center=center,
        method_id=mid,
        k_values=k_values,
    ) if k_values else pd.DataFrame()

    weighted_metrics = None
    if run_weighted:
        w = compute_occupancy_weights(y_train, arena_bounds)
        w_pipe = fit_ridge_position(
            X_train, y_train, alpha=DEFAULT_RIDGE_ALPHA, sample_weight=w,
        )
        w_pred = w_pipe.predict(X_test)
        weighted_metrics = compute_calibration_metrics(true_test, w_pred, center)

    return MethodDiagnostics(
        method_id=mid,
        label=label,
        quadrant=quadrant,
        embedding=embedding,
        available=True,
        decoded=decoded,
        metrics=metrics,
        alpha_sweep=alpha_sweep,
        dim_sweep=dim_sweep,
        coef_norm_vs_alpha=coef_norm,
        occupancy_weighted=weighted_metrics,
        extra={"feature_source": source},
    )


def generate_interpretation_summary(results: Mapping[str, MethodDiagnostics]) -> str:
    available = [r for r in results.values() if r.available and r.metrics is not None]
    if not available:
        return "No ridge quadrant methods were available for this run."

    best_err = min(available, key=lambda r: r.metrics.mean_position_error_cm)  # type: ignore[union-attr]
    worst_bias = max(available, key=lambda r: abs(r.metrics.mean_radial_bias))  # type: ignore[union-attr]
    strongest_shrink = min(available, key=lambda r: r.metrics.slope_r)  # type: ignore[union-attr]

    lines = [
        f"Lowest mean position error: **{best_err.label}** "
        f"({best_err.metrics.mean_position_error_cm:.2f} cm).",  # type: ignore[union-attr]
        f"Strongest center bias (|mean radial bias|): **{worst_bias.label}** "
        f"({worst_bias.metrics.mean_radial_bias:+.2f} cm).",  # type: ignore[union-attr]
        f"Largest radial contraction (lowest radial slope): **{strongest_shrink.label}** "
        f"(slope={strongest_shrink.metrics.slope_r:.3f}).",  # type: ignore[union-attr]
    ]

    pca = results.get("global_pca")
    rpca = results.get("region_pca")
    if (
        pca and rpca
        and pca.available and rpca.available
        and pca.metrics and rpca.metrics
    ):
        if rpca.metrics.slope_r > pca.metrics.slope_r:
            lines.append(
                "Region PCA preserves radial extent better than global PCA "
                f"(radial slopes {rpca.metrics.slope_r:.3f} vs {pca.metrics.slope_r:.3f})."
            )
        else:
            lines.append(
                "Global PCA shows less radial shrinkage than region PCA on this session."
            )

    lds = results.get("global_lds")
    if lds and lds.available and lds.metrics and pca and pca.available and pca.metrics:
        if lds.metrics.mean_radial_bias < pca.metrics.mean_radial_bias:
            lines.append("LDS reduces center bias relative to global PCA on this split.")
        else:
            lines.append("LDS does not reduce center bias versus global PCA here.")

    alpha_rows = []
    for r in available:
        if r.alpha_sweep is not None and not r.alpha_sweep.empty:
            sub = r.alpha_sweep.sort_values("alpha")
            if len(sub) >= 2:
                d_slope = float(sub.iloc[-1]["slope_r"] - sub.iloc[0]["slope_r"])
                alpha_rows.append((r.label, d_slope))
    if alpha_rows:
        worsened = max(alpha_rows, key=lambda t: abs(t[1]))
        lines.append(
            f"Increasing ridge α most changes radial slope for **{worsened[0]}** "
            f"(Δ slope_r ≈ {worsened[1]:+.3f} across the sweep)."
        )

    return " ".join(lines)


def run_quadrant_ridge_diagnostics(
    *,
    input_dir: Path,
    spike_source: str = "sorted",
    decode_window: float = 0.250,
    update_dt: float = 0.050,
    n_components: int = 3,
    train_frac: float = 0.70,
    seed: int = 42,
    run_weighted: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    """Run ridge-only quadrant diagnostics and write artifacts under quadrant_ridge/."""
    input_dir = Path(input_dir)
    out_root = input_dir / "realtime_decoding" / "quadrant_ridge"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = input_dir / "figures" / "realtime_decoding" / "quadrant_ridge"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_simulation_data(input_dir, spike_source)
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"],
        derive_from_behavior=True,
        update_dt_s=update_dt,
        behavior_times=behavior_times,
    )
    decode_times = make_decode_times(
        data["session_duration"],
        float(decode_window),
        update_dt,
        behavior_times=behavior_times,
    )
    aligned = align_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"],
    )
    train_mask, test_mask = causal_train_test_split(decode_times, train_frac)
    beh_train = aligned.loc[train_mask].reset_index(drop=True)
    beh_test = aligned.loc[test_mask].reset_index(drop=True)
    arena_bounds = infer_arena_bounds(data["behavior_df"], data["summary"])
    center = _arena_center(beh_train, data["summary"])

    X_counts = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, float(decode_window),
    )

    methods = [m for m in RIDGE_QUADRANT_METHODS if m["quadrant"] != "dynamic_nonlinear"]
    results: dict[str, MethodDiagnostics] = {}
    total = len(methods)
    for i, method in enumerate(methods, start=1):
        if progress_callback:
            progress_callback(
                f"Ridge diagnostics · {method['label']}",
                i,
                total,
            )
        diag = _evaluate_method(
            experiment_dir=input_dir,
            spike_source=spike_source,
            method=method,
            decode_window=float(decode_window),
            n_components=int(n_components),
            X_counts=X_counts,
            train_mask=train_mask,
            test_mask=test_mask,
            beh_train=beh_train,
            beh_test=beh_test,
            data=data,
            update_dt=float(update_dt),
            center=center,
            arena_bounds=arena_bounds,
            run_weighted=run_weighted and method["id"] in ("counts", "global_pca"),
        )
        results[str(method["id"])] = diag
        if diag.available and diag.decoded is not None:
            diag.decoded.to_csv(out_root / f"decoded_{method['id']}.csv", index=False)

    true_decoded = _decoded_dataframe(
        beh_test,
        beh_test[["x", "y"]].to_numpy(),
        decode_window=float(decode_window),
        method_id="true",
        alpha=float("nan"),
    )
    true_decoded.to_csv(out_root / "decoded_true.csv", index=False)

    metrics_rows = []
    for mid, diag in results.items():
        if not diag.available or diag.metrics is None:
            metrics_rows.append({
                "method_id": mid,
                "label": diag.label,
                "quadrant": diag.quadrant,
                "available": False,
                "skip_reason": diag.skip_reason,
            })
            continue
        metrics_rows.append({
            "method_id": mid,
            "label": diag.label,
            "quadrant": diag.quadrant,
            "embedding": diag.embedding,
            "available": True,
            **diag.metrics.to_dict(),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_root / "ridge_calibration_metrics.csv", index=False)

    alpha_parts = [d.alpha_sweep for d in results.values() if d.alpha_sweep is not None and not d.alpha_sweep.empty]
    if alpha_parts:
        pd.concat(alpha_parts, ignore_index=True).to_csv(
            out_root / "ridge_alpha_sweep.csv", index=False,
        )
    dim_parts = [d.dim_sweep for d in results.values() if d.dim_sweep is not None and not d.dim_sweep.empty]
    if dim_parts:
        pd.concat(dim_parts, ignore_index=True).to_csv(
            out_root / "ridge_dim_sweep.csv", index=False,
        )
    coef_parts = [
        d.coef_norm_vs_alpha for d in results.values()
        if d.coef_norm_vs_alpha is not None and not d.coef_norm_vs_alpha.empty
    ]
    if coef_parts:
        pd.concat(coef_parts, ignore_index=True).to_csv(
            out_root / "ridge_coef_norm.csv", index=False,
        )

    occ = compute_occupancy_map(
        beh_test[["x", "y"]].to_numpy(), arena_bounds,
    )
    (out_root / "occupancy_map.json").write_text(json.dumps(occ, indent=2) + "\n")

    summary_text = generate_interpretation_summary(results)

    from visualization.quadrant_ridge_figures import (
        plot_fig_quadrant_ridge_shrinkage,
        plot_fig_quadrant_ridge_trajectory,
    )

    fig1 = plot_fig_quadrant_ridge_trajectory(
        true_decoded,
        results,
        metrics_df,
        center=center,
        arena_bounds=arena_bounds,
        output_dir=fig_dir,
    )
    fig2 = plot_fig_quadrant_ridge_shrinkage(
        results,
        occ,
        arena_bounds=arena_bounds,
        output_dir=fig_dir,
    )

    sidecar = {
        "decoder": "ridge",
        "decoder_class": "linear",
        "target": "position",
        "decode_window_s": float(decode_window),
        "update_dt_s": float(update_dt),
        "n_components": int(n_components),
        "spike_source": spike_source,
        "seed": int(seed),
        "methods": {
            mid: {
                "label": d.label,
                "quadrant": d.quadrant,
                "embedding": d.embedding,
                "available": d.available,
                "skip_reason": d.skip_reason,
                "metrics": d.metrics.to_dict() if d.metrics else None,
                "feature_source": d.extra.get("feature_source"),
            }
            for mid, d in results.items()
        },
        "interpretation": summary_text,
        "figures": {
            "trajectory": str(fig1) if fig1 else None,
            "shrinkage": str(fig2) if fig2 else None,
        },
        "tables": {
            "calibration": str(out_root / "ridge_calibration_metrics.csv"),
            "alpha_sweep": str(out_root / "ridge_alpha_sweep.csv"),
            "dim_sweep": str(out_root / "ridge_dim_sweep.csv"),
        },
        "output_dir": str(out_root),
    }
    sidecar_path = out_root / "quadrant_ridge_summary.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    sidecar["sidecar_path"] = str(sidecar_path)
    return sidecar
