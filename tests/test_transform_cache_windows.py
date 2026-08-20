"""Tests for Feature Construction window inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime.transform_cache import (
    assert_cached_decode_windows,
    inventory_feature_construction_cache,
    list_cached_decode_windows,
    window_ms,
)


def _write_f_cache(root: Path, *, feature_set: str, f_eff: str, decode_window: float) -> None:
    name = f"{feature_set}__{f_eff}_w{window_ms(decode_window):04d}ms"
    d = root / "models" / "feature_transforms" / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "feature_set": feature_set,
        "feature_type_eff": f_eff,
        "decode_window_s": float(decode_window),
        "source": "feature_explorer",
    }))


def test_list_cached_decode_windows_parses_meta(tmp_path: Path):
    exp = tmp_path / "exp"
    sorted_root = exp / "decoder_comparison" / "sorted"
    _write_f_cache(sorted_root, feature_set="counts", f_eff="counts", decode_window=0.25)
    _write_f_cache(sorted_root, feature_set="counts", f_eff="counts", decode_window=0.1)
    # Wrong spike source must be ignored
    gt_root = exp / "decoder_comparison" / "ground_truth"
    _write_f_cache(gt_root, feature_set="counts", f_eff="counts", decode_window=0.5)

    got = list_cached_decode_windows(exp, spike_source="sorted")
    assert got == [0.1, 0.25]


def test_list_cached_decode_windows_parses_dirname_without_meta_field(tmp_path: Path):
    exp = tmp_path / "exp"
    d = (
        exp / "decoder_comparison" / "sorted" / "models" / "feature_transforms"
        / "counts__counts_w0500ms"
    )
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{}")
    got = list_cached_decode_windows(exp, spike_source="sorted")
    assert got == [0.5]


def test_list_cached_decode_windows_includes_one_second(tmp_path: Path):
    exp = tmp_path / "exp"
    sorted_root = exp / "decoder_comparison" / "sorted"
    _write_f_cache(sorted_root, feature_set="counts", f_eff="counts", decode_window=1.0)
    got = list_cached_decode_windows(exp, spike_source="sorted")
    assert got == [1.0]


def test_list_cached_decode_windows_counts_does_not_unlock_other_set(tmp_path: Path):
    exp = tmp_path / "exp"
    sorted_root = exp / "decoder_comparison" / "sorted"
    _write_f_cache(sorted_root, feature_set="counts", f_eff="counts", decode_window=0.25)
    _write_f_cache(
        sorted_root, feature_set="counts_dynamics", f_eff="counts", decode_window=0.5,
    )
    unfiltered = list_cached_decode_windows(exp, spike_source="sorted")
    assert unfiltered == [0.25, 0.5]
    assert list_cached_decode_windows(
        exp, spike_source="sorted", feature_sets=("counts_dynamics",),
    ) == [0.5]
    assert list_cached_decode_windows(
        exp, spike_source="sorted", feature_sets=("counts",),
    ) == [0.25]
    assert list_cached_decode_windows(
        exp, spike_source="sorted", feature_sets=("counts", "counts_dynamics"),
    ) == []


def test_inventory_uses_f_cache_not_run_json(tmp_path: Path):
    exp = tmp_path / "exp"
    features = exp / "features" / "feature_old"
    features.mkdir(parents=True)
    (features / "analysis_summary.json").write_text(json.dumps({
        "run_id": "feature_old",
        "request": {
            "feature_sets": ["counts"],
            "decode_windows": [0.25, 1.0],
            "spike_source": "sorted",
        },
    }))
    sorted_root = exp / "decoder_comparison" / "sorted"
    _write_f_cache(sorted_root, feature_set="counts", f_eff="counts", decode_window=0.25)

    cov = inventory_feature_construction_cache(
        exp,
        feature_sets=["counts"],
        decode_windows=[0.25, 1.0],
        spike_source="sorted",
    )
    assert cov["n_wanted"] == 2
    assert cov["n_covered"] == 1
    assert cov["n_to_compute"] == 1
    assert cov["fully_covered"] is False
    assert ("counts", 1.0) in cov["missing"]


def test_inventory_one_second_is_covered_when_w1000ms_exists(tmp_path: Path):
    exp = tmp_path / "exp"
    sorted_root = exp / "decoder_comparison" / "sorted"
    _write_f_cache(sorted_root, feature_set="counts", f_eff="counts", decode_window=1.0)
    cov = inventory_feature_construction_cache(
        exp,
        feature_sets=["counts"],
        decode_windows=[1.0],
        spike_source="sorted",
    )
    assert cov["fully_covered"] is True
    assert cov["n_to_compute"] == 0


def test_assert_cached_decode_windows_rejects_missing(tmp_path: Path):
    exp = tmp_path / "exp"
    _write_f_cache(
        exp / "decoder_comparison" / "sorted",
        feature_set="counts",
        f_eff="counts",
        decode_window=0.25,
    )
    assert_cached_decode_windows(exp, (0.25,), spike_source="sorted")
    with pytest.raises(ValueError, match="Feature Construction"):
        assert_cached_decode_windows(exp, (0.25, 0.1), spike_source="sorted")


def test_checkpoint_reuses_existing_cache_without_simulation(tmp_path: Path):
    from ui.services.features import checkpoint_feature_transform

    exp = tmp_path / "exp"
    _write_f_cache(
        exp / "decoder_comparison" / "sorted",
        feature_set="counts",
        f_eff="counts",
        decode_window=0.25,
    )
    out = checkpoint_feature_transform(exp, "counts", decode_window=0.25)
    assert out["from_cache"] is True
    assert "0.25" in out["saved_path"] or "0250" in out["saved_path"]
