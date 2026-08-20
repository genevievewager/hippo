"""Display-only ETA scaling by already-generated work."""

from __future__ import annotations

import pytest

from ui.services.comparison import (
    DECODER_TRANSFORM_TIME_SHARE,
    apply_remaining_work_to_workload,
    estimate_workload,
)
from ui.services.feature_analysis import (
    FeatureAnalysisRequest,
    run_feature_analysis,
    selection_already_covered,
)


def _base_workload(*, planned: int = 42, seconds: float = 519.0) -> dict:
    return {
        "planned_configurations": planned,
        "detail_label": "7 feature set(s) × 6 window(s) · + panel pages",
        "estimated_runtime_s": seconds,
        "estimated_runtime_low_s": seconds * 0.7,
        "estimated_runtime_high_s": seconds * 1.5,
        "estimated_runtime_label": "placeholder",
        "estimated_runtime_range_label": "placeholder",
    }


def test_fully_covered_hides_estimate():
    out = apply_remaining_work_to_workload(
        _base_workload(),
        n_wanted=42,
        n_covered=42,
        n_to_compute=0,
        fully_covered=True,
    )
    assert out["skip_estimate"] is True
    assert out["planned_configurations"] == 0
    assert out["estimated_runtime_s"] == 0.0


def test_partial_coverage_scales_runtime_and_count():
    out = apply_remaining_work_to_workload(
        _base_workload(planned=42, seconds=420.0),
        n_wanted=42,
        n_covered=30,
        n_to_compute=12,
        fully_covered=False,
    )
    assert out.get("skip_estimate") is not True
    assert out["planned_configurations"] == 12
    assert out["estimated_runtime_s"] == 120.0
    assert "12 new · 30 reusable" in out["detail_label"]


def test_no_prior_work_leaves_workload_unchanged():
    base = _base_workload()
    out = apply_remaining_work_to_workload(
        base,
        n_wanted=0,
        n_covered=0,
        fully_covered=False,
    )
    assert out["planned_configurations"] == base["planned_configurations"]
    assert out["estimated_runtime_s"] == base["estimated_runtime_s"]
    assert "skip_estimate" not in out


def test_decoder_reuse_keeps_decoder_floor_and_does_not_hide():
    base = _base_workload(planned=80, seconds=200.0)
    out = apply_remaining_work_to_workload(
        base,
        n_wanted=10,
        n_covered=10,
        n_to_compute=0,
        fully_covered=False,
        hide_when_complete=False,
        scale_floor=1.0 - DECODER_TRANSFORM_TIME_SHARE,
        scale_planned=False,
        remaining_detail="0 transform(s) to fit · 10 reusable",
    )
    assert out.get("skip_estimate") is not True
    assert out["planned_configurations"] == 80
    assert out["estimated_runtime_s"] == pytest.approx(200.0 * (1.0 - DECODER_TRANSFORM_TIME_SHARE))
    assert "0 transform(s) to fit · 10 reusable" in out["detail_label"]


def test_decoder_all_transforms_missing_keeps_full_eta():
    base = _base_workload(planned=80, seconds=200.0)
    out = apply_remaining_work_to_workload(
        base,
        n_wanted=10,
        n_covered=0,
        n_to_compute=10,
        hide_when_complete=False,
        scale_floor=1.0 - DECODER_TRANSFORM_TIME_SHARE,
        scale_planned=False,
    )
    assert out["estimated_runtime_s"] == 200.0
    assert out["planned_configurations"] == 80


def test_estimate_workload_shape_unchanged():
    out = estimate_workload(
        feature_sets=["counts"],
        manifolds=["global_pca"],
        decode_windows=[0.25],
        n_decoders_hint=3,
    )
    assert out["planned_configurations"] > 0
    assert "estimated_runtime_s" in out
    assert "skip_estimate" not in out


def test_estimate_workload_targets_hint_scales_planned():
    base = estimate_workload(
        feature_sets=["counts"],
        manifolds=["global_pca"],
        decode_windows=[0.25],
        n_decoders_hint=3,
    )
    scaled = estimate_workload(
        feature_sets=["counts"],
        manifolds=["global_pca"],
        decode_windows=[0.25],
        n_decoders_hint=3,
        n_targets_hint=5,
    )
    assert scaled["planned_configurations"] == base["planned_configurations"] * 5
    assert scaled["estimated_runtime_s"] == pytest.approx(base["estimated_runtime_s"] * 5)


def test_feature_coverage_reads_f_cache_not_run_json(tmp_path, monkeypatch):
    from realtime.transform_cache import window_ms

    exp = tmp_path / "exp"
    root = exp / "decoder_comparison" / "sorted" / "models" / "feature_transforms"
    for fs, w in (("counts", 0.25), ("counts", 0.5), ("global_pca", 0.25), ("global_pca", 0.5)):
        name = f"{fs}__counts_w{window_ms(w):04d}ms"
        d = root / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text(
            f'{{"feature_set": "{fs}", "feature_type_eff": "counts", "decode_window_s": {w}}}'
        )

    cov = selection_already_covered(
        exp,
        feature_sets=["counts", "global_pca", "region_pca"],
        decode_windows=[0.25, 0.5],
        spike_source="sorted",
    )
    assert cov["n_wanted"] == 6
    assert cov["n_covered"] == 4
    assert cov["n_to_compute"] == 2
    assert cov["fully_covered"] is False

    full = selection_already_covered(
        exp,
        feature_sets=["counts"],
        decode_windows=[0.25],
        spike_source="sorted",
    )
    assert full["fully_covered"] is True
    assert full["n_to_compute"] == 0


def test_run_feature_analysis_skips_only_when_f_cache_complete(tmp_path):
    from realtime.transform_cache import window_ms

    exp = tmp_path / "exp"
    fig = exp / "figures" / "features"
    fig.mkdir(parents=True)
    for kind in ("variance", "traces", "correlation"):
        (fig / f"fig_feature_panel_{kind}.png").write_bytes(b"png")
    d = (
        exp / "decoder_comparison" / "sorted" / "models" / "feature_transforms"
        / f"counts__counts_w{window_ms(0.25):04d}ms"
    )
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        '{"feature_set": "counts", "feature_type_eff": "counts", "decode_window_s": 0.25}'
    )

    req = FeatureAnalysisRequest(
        input_dir=exp,
        feature_sets=("counts",),
        decode_windows=(0.25,),
        write_panel_pages=True,
        regenerate_simulation_figures=False,
        write_per_window_diagnostics=False,
    )
    out = run_feature_analysis(req)
    assert out["skipped"] is True
    assert out["reason"] == "feature_transforms_on_disk"

    fake_run = exp / "features" / "old"
    fake_run.mkdir(parents=True)
    (fake_run / "analysis_summary.json").write_text(
        '{"request": {"feature_sets": ["counts"], "decode_windows": [0.25, 1.0]}}'
    )
    cov = selection_already_covered(
        exp,
        feature_sets=["counts"],
        decode_windows=[0.25, 1.0],
        spike_source="sorted",
    )
    assert cov["fully_covered"] is False
    assert cov["n_to_compute"] == 1


def test_checkpoint_failure_fails_the_job(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ui.services.feature_analysis.selection_already_covered",
        lambda *a, **k: {
            "fully_covered": False,
            "n_wanted": 1,
            "n_covered": 0,
            "n_to_compute": 1,
            "covered": set(),
            "missing": {("counts", 1.0)},
            "matching_run_ids": [],
        },
    )
    monkeypatch.setattr(
        "ui.services.feature_analysis.resolve_feature_panel_windows",
        lambda *a, **k: {},
    )
    monkeypatch.setattr(
        "ui.services.feature_analysis.checkpoint_feature_transform",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom at 1s")),
    )
    req = FeatureAnalysisRequest(
        input_dir=tmp_path,
        feature_sets=("counts",),
        decode_windows=(1.0,),
        write_panel_pages=False,
        regenerate_simulation_figures=False,
        write_per_window_diagnostics=False,
    )
    with pytest.raises(RuntimeError, match="did not write F caches"):
        run_feature_analysis(req)


def test_run_fails_only_windows_that_could_not_be_written(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ui.services.feature_analysis.selection_already_covered",
        lambda *a, **k: {
            "fully_covered": False,
            "n_wanted": 2,
            "n_covered": 1,
            "n_to_compute": 1,
            "covered": {("counts", 0.25)},
            "missing": {("counts", 1.0)},
            "matching_run_ids": [],
        },
    )
    monkeypatch.setattr(
        "ui.services.feature_analysis.resolve_feature_panel_windows",
        lambda *a, **k: {},
    )

    def _ckpt(_dir, fs, *, decode_window=0.25, **_kw):
        w = float(decode_window)
        if w == 1.0:
            raise RuntimeError("1s boom")
        return {
            "feature_set": fs,
            "from_cache": True,
            "persisted": False,
            "decode_window_s": w,
        }

    monkeypatch.setattr(
        "ui.services.feature_analysis.checkpoint_feature_transform",
        _ckpt,
    )
    req = FeatureAnalysisRequest(
        input_dir=tmp_path,
        feature_sets=("counts",),
        decode_windows=(0.25, 1.0),
        write_panel_pages=False,
        regenerate_simulation_figures=False,
        write_per_window_diagnostics=False,
    )
    with pytest.raises(RuntimeError, match=r"counts @ 1s") as err:
        run_feature_analysis(req)
    assert "0.25" not in str(err.value)
