"""Manifold vs counts summaries, ablation helpers, and markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realtime.decoder_comparison import ALL_TARGETS, PRIMARY_METRIC, _is_better, _near_optimal
from realtime.manifold_features import MANIFOLD_FEATURE_MODES, is_manifold_feature_mode


def _metric_direction(target: str) -> tuple[str, str]:
    return PRIMARY_METRIC[target]


def interpret_manifold_vs_counts(
    manifold_value: float,
    counts_value: float,
    direction: str,
    tol: float = 0.05,
) -> str:
    """Compare manifold vs counts with ±5% relative tolerance."""
    if direction == "lower":
        # lower is better: relative improvement if manifold < counts
        if counts_value == 0:
            rel = 0.0
        else:
            rel = (counts_value - manifold_value) / abs(counts_value)
        if rel > tol:
            return "manifold improves decoding"
        if rel < -tol:
            return "manifold reduces decoding"
        return "manifold comparable to raw counts"
    # higher is better
    if counts_value == 0:
        rel = 0.0 if manifold_value == 0 else (1.0 if manifold_value > 0 else -1.0)
    else:
        rel = (manifold_value - counts_value) / abs(counts_value)
    if rel > tol:
        return "manifold improves decoding"
    if rel < -tol:
        return "manifold reduces decoding"
    return "manifold comparable to raw counts"


def build_best_manifold_decoder_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Best manifold-informed setup per target (excludes raw counts/rates)."""
    rows = []
    man = metrics_df[metrics_df["feature_type"].isin(MANIFOLD_FEATURE_MODES)].copy()
    if man.empty:
        return pd.DataFrame()
    for target in ALL_TARGETS:
        target_df = man[man["target_name"] == target]
        if target_df.empty:
            continue
        metric_key, direction = _metric_direction(target)
        if metric_key not in target_df.columns:
            continue
        idx = target_df[metric_key].idxmin() if direction == "lower" else target_df[metric_key].idxmax()
        best = target_df.loc[idx]
        rows.append({
            "target_name": target,
            "primary_metric": metric_key,
            "best_feature_type": best.get("feature_type"),
            "best_manifold_type": best.get("manifold_type"),
            "best_manifold_grouping": best.get("manifold_grouping"),
            "best_manifold_n_components": best.get("manifold_n_components"),
            "best_decode_window_s": best.get("decode_window_s"),
            "best_decoder_name": best.get("decoder_name"),
            "best_metric_value": float(best[metric_key]),
            "spike_source": best.get("spike_source"),
            "actual_n_features": best.get("actual_n_features"),
        })
    return pd.DataFrame(rows)


def build_manifold_vs_counts_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-target comparison of best counts vs best manifold decoder."""
    rows = []
    for spike_source in sorted(metrics_df["spike_source"].unique()):
        src = metrics_df[metrics_df["spike_source"] == spike_source]
        for target in ALL_TARGETS:
            target_df = src[src["target_name"] == target]
            if target_df.empty:
                continue
            metric_key, direction = _metric_direction(target)
            if metric_key not in target_df.columns:
                continue
            counts_df = target_df[target_df["feature_type"].isin(("counts", "rates"))]
            man_df = target_df[target_df["feature_type"].isin(MANIFOLD_FEATURE_MODES)]
            if counts_df.empty or man_df.empty:
                continue
            c_idx = counts_df[metric_key].idxmin() if direction == "lower" else counts_df[metric_key].idxmax()
            m_idx = man_df[metric_key].idxmin() if direction == "lower" else man_df[metric_key].idxmax()
            c_row = counts_df.loc[c_idx]
            m_row = man_df.loc[m_idx]
            c_val = float(c_row[metric_key])
            m_val = float(m_row[metric_key])
            if direction == "lower":
                diff = c_val - m_val  # positive => manifold better
            else:
                diff = m_val - c_val
            rows.append({
                "spike_source": spike_source,
                "target_name": target,
                "primary_metric": metric_key,
                "best_counts_feature_type": c_row["feature_type"],
                "best_counts_decoder": c_row["decoder_name"],
                "best_counts_window_s": c_row["decode_window_s"],
                "best_counts_metric_value": c_val,
                "best_manifold_feature_type": m_row["feature_type"],
                "best_manifold_decoder": m_row["decoder_name"],
                "best_manifold_window_s": m_row["decode_window_s"],
                "best_manifold_n_components": m_row.get("manifold_n_components"),
                "best_manifold_metric_value": m_val,
                "performance_difference": diff,
                "interpretation": interpret_manifold_vs_counts(m_val, c_val, direction),
            })
    return pd.DataFrame(rows)


def enrich_best_decoder_table(best_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add manifold columns to best_decoder_by_target from the winning metric row."""
    if best_df.empty or metrics_df.empty:
        return best_df
    out = best_df.copy()
    for col in (
        "best_manifold_type",
        "best_manifold_grouping",
        "best_manifold_n_components",
        "manifold_transform_path",
    ):
        if col not in out.columns:
            out[col] = None
    for i, row in out.iterrows():
        target = row["target_name"]
        metric_key = row["primary_metric"]
        target_df = metrics_df[metrics_df["target_name"] == target]
        if target_df.empty or metric_key not in target_df.columns:
            continue
        direction = PRIMARY_METRIC[target][1]
        idx = target_df[metric_key].idxmin() if direction == "lower" else target_df[metric_key].idxmax()
        win = target_df.loc[idx]
        out.at[i, "best_feature_type"] = win.get("feature_type", row.get("best_feature_type"))
        out.at[i, "best_manifold_type"] = win.get("manifold_type")
        out.at[i, "best_manifold_grouping"] = win.get("manifold_grouping")
        out.at[i, "best_manifold_n_components"] = win.get("manifold_n_components")
        out.at[i, "manifold_transform_path"] = win.get("manifold_transform_path")
    return out


def write_manifold_decoder_report(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    vs_counts: pd.DataFrame,
    best_manifold: pd.DataFrame,
    explained_df: pd.DataFrame | None = None,
) -> Path:
    """Generate manifold_decoder_report.md from metrics (no hard-coded conclusions)."""
    output_dir = Path(output_dir)
    lines = [
        "# Manifold-informed decoder report",
        "",
        "Region-specific PCA tests whether each hippocampal subregion contains a "
        "distinct low-dimensional code for behavioral variables. If region-specific "
        "manifolds outperform global PCA, this suggests that preserving anatomical "
        "structure helps decoding.",
        "",
        "## 1. Manifold vs raw counts",
        "",
    ]
    if vs_counts.empty:
        lines.append("No manifold-vs-counts comparison available.")
    else:
        improved = vs_counts[vs_counts["interpretation"] == "manifold improves decoding"]
        comparable = vs_counts[vs_counts["interpretation"] == "manifold comparable to raw counts"]
        worse = vs_counts[vs_counts["interpretation"] == "manifold reduces decoding"]
        lines.append(
            f"- Targets where manifold improves decoding (>5%): "
            f"{', '.join(improved['target_name'].astype(str)) or 'none'}"
        )
        lines.append(
            f"- Targets where manifold is comparable (±5%): "
            f"{', '.join(comparable['target_name'].astype(str)) or 'none'}"
        )
        lines.append(
            f"- Targets where manifold reduces decoding (>5%): "
            f"{', '.join(worse['target_name'].astype(str)) or 'none'}"
        )
        lines.append("")
        lines.append("| target | spike_source | interpretation | counts | manifold | diff |")
        lines.append("|---|---|---|---:|---:|---:|")
        for _, r in vs_counts.iterrows():
            lines.append(
                f"| {r['target_name']} | {r['spike_source']} | {r['interpretation']} | "
                f"{r['best_counts_metric_value']:.4f} | {r['best_manifold_metric_value']:.4f} | "
                f"{r['performance_difference']:.4f} |"
            )

    lines.extend(["", "## 2. Targets benefiting most from manifold features", ""])
    if not vs_counts.empty:
        ranked = vs_counts.sort_values("performance_difference", ascending=False)
        for _, r in ranked.head(5).iterrows():
            lines.append(
                f"- **{r['target_name']}** ({r['spike_source']}): "
                f"{r['interpretation']} using `{r['best_manifold_feature_type']}` "
                f"(k={r['best_manifold_n_components']})"
            )

    lines.extend(["", "## 3. Region / layer manifold strength", ""])
    if explained_df is not None and not explained_df.empty:
        for grouping in ("region", "layer"):
            g = explained_df[explained_df.get("grouping_name", explained_df.get("feature_type", "")) == grouping] \
                if "grouping_name" in explained_df.columns else pd.DataFrame()
            # Use feature_type filter
            g = explained_df[explained_df["feature_type"] == f"{grouping}_pca"] if "feature_type" in explained_df.columns else pd.DataFrame()
            if g.empty:
                continue
            summary = g.groupby("group_name")["explained_variance_sum"].mean().sort_values(ascending=False)
            lines.append(f"### {grouping}")
            for name, val in summary.items():
                lines.append(f"- {name}: mean explained variance (selected components) = {val:.3f}")
            lines.append("")
    else:
        lines.append("No explained-variance table available.")

    if not best_manifold.empty and "best_feature_type" in best_manifold.columns:
        lines.extend(["", "## 4. Components sufficient for best manifold setups", ""])
        for _, r in best_manifold.iterrows():
            lines.append(
                f"- {r['target_name']}: `{r['best_feature_type']}` with "
                f"k={r.get('best_manifold_n_components')} "
                f"(window={r.get('best_decode_window_s')}s, "
                f"metric={r.get('best_metric_value'):.4f})"
            )

    lines.extend(["", "## 5. Sorted vs ground-truth manifold decoding", ""])
    if "spike_source" in metrics_df.columns and metrics_df["spike_source"].nunique() > 1:
        man = metrics_df[metrics_df["feature_type"].isin(MANIFOLD_FEATURE_MODES)]
        for target in ALL_TARGETS:
            metric_key, direction = PRIMARY_METRIC[target]
            if metric_key not in man.columns:
                continue
            parts = []
            for src in ("ground_truth", "sorted"):
                sub = man[(man["target_name"] == target) & (man["spike_source"] == src)]
                if sub.empty:
                    continue
                val = sub[metric_key].min() if direction == "lower" else sub[metric_key].max()
                parts.append(f"{src}={val:.4f}")
            if parts:
                lines.append(f"- {target}: " + ", ".join(parts))
    else:
        lines.append("Single spike source only; cross-source comparison unavailable.")

    lines.extend(["", "## 6. Realtime suitability", ""])
    if "mean_inference_latency_s" in metrics_df.columns:
        lat = metrics_df["mean_inference_latency_s"].dropna()
        if len(lat):
            ok = float((lat <= 0.050).mean())
            lines.append(
                f"- Fraction of configs with mean inference ≤ 50 ms: {ok:.1%}"
            )
            lines.append(
                f"- Median mean inference latency: {float(np.median(lat))*1000:.2f} ms"
            )
        else:
            lines.append("Latency not recorded for these runs.")
    else:
        lines.append(
            "PCA manifold transforms are linear and typically realtime-compatible; "
            "deploy using the selected transform fitted on training data only."
        )

    path = output_dir / "manifold_decoder_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
