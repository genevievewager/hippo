"""Union metrics across partial decoder comparison runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from realtime.comparison_metrics_union import (
    load_or_collect_union,
    merge_comparison_metrics,
    persist_merged_comparison_metrics,
    union_csv_path,
)


def _row(
    *,
    decoder: str,
    metric: float,
    target: str = "position",
    feature_mode: str = "global_pca",
    window: float = 0.250,
    k: int = 3,
) -> dict:
    return {
        "spike_source": "sorted",
        "target_name": target,
        "decoder_name": decoder,
        "feature_set": "counts",
        "feature_mode": feature_mode,
        "embedding_type": feature_mode,
        "decode_window_s": window,
        "manifold_n_components": k,
        "primary_metric": "mean_position_error_cm",
        "mean_position_error_cm": metric,
    }


def test_merge_keeps_older_decoder_and_replaces_same_key():
    ridge = pd.DataFrame([_row(decoder="ridge", metric=12.0)])
    rf_a = pd.DataFrame([_row(decoder="random_forest_regressor", metric=11.0)])
    rf_b = pd.DataFrame([_row(decoder="random_forest_regressor", metric=9.0)])
    merged = merge_comparison_metrics(ridge, rf_a)
    assert set(merged["decoder_name"]) == {"ridge", "random_forest_regressor"}
    replaced = merge_comparison_metrics(merged, rf_b)
    rf = replaced[replaced["decoder_name"] == "random_forest_regressor"]
    assert len(rf) == 1
    assert float(rf["mean_position_error_cm"].iloc[0]) == 9.0
    assert "ridge" in set(replaced["decoder_name"])


def test_persist_rf_only_run_does_not_wipe_ridge(tmp_path: Path):
    exp = tmp_path / "exp"
    out = exp / "decoder_comparison" / "sorted"
    out.mkdir(parents=True)
    prior = pd.DataFrame([_row(decoder="ridge", metric=12.0)])
    prior.to_csv(out / "decoder_comparison_metrics.csv", index=False)
    incoming = pd.DataFrame([_row(decoder="random_forest_regressor", metric=8.0)])
    scoped = persist_merged_comparison_metrics(
        incoming,
        output_dir=out,
        experiment_dir=exp,
        spike_source="sorted",
    )
    assert set(scoped["decoder_name"]) == {"ridge", "random_forest_regressor"}
    union = pd.read_csv(union_csv_path(exp))
    assert set(union["decoder_name"]) == {"ridge", "random_forest_regressor"}
    rerun = pd.read_csv(out / "decoder_comparison_metrics.csv")
    assert set(rerun["decoder_name"]) == {"ridge", "random_forest_regressor"}
    assert "config_id" in scoped.columns
    assert scoped["config_id"].nunique() == 2


def test_load_or_collect_union_globs_when_union_missing(tmp_path: Path):
    exp = tmp_path / "exp"
    nested = exp / "decoder_comparison" / "sorted"
    nested.mkdir(parents=True)
    pd.DataFrame([_row(decoder="ridge", metric=12.0)]).to_csv(
        nested / "decoder_comparison_metrics.csv", index=False,
    )
    loaded = load_or_collect_union(exp)
    assert list(loaded["decoder_name"]) == ["ridge"]
