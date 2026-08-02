"""Tests for research-style figure captions in the compiled PDF."""

from pathlib import Path

from visualization.compile_figures_pdf import collect_pngs_by_section
from visualization.figure_captions import caption_for


def test_caption_for_known_stem_is_numbered_and_descriptive():
    path = Path("figures/behavior/behavior_trajectory_xy.png")
    text = caption_for(path, figure_number=3)
    assert text.startswith("Figure 3. ")
    assert "trajectory" in text.lower()
    assert "arena" in text.lower()


def test_caption_for_cell_class_pattern():
    path = Path("figures/neural/fig_population_tuning.png")
    text = caption_for(path, figure_number=1)
    assert "cell class" in text.lower()
    assert "heatmap" in text.lower() or "rate" in text.lower()


def test_caption_includes_sorted_source_note_for_decoder_path():
    path = Path("figures/decoder_comparison/sorted/position_error_vs_window.png")
    text = caption_for(
        path,
        figure_number=10,
        figures_dir=Path("figures"),
    )
    assert text.startswith("Figure 10. ")
    assert "sorted" in text.lower() or "Neuropixels" in text


def test_section_key_collapses_nested_realtime_dirs(tmp_path: Path):
    figures = tmp_path / "figures"
    nested = figures / "realtime_decoding" / "sorted" / "spatial_context_best"
    nested.mkdir(parents=True)
    png = nested / "closed_loop_events_over_time.png"
    png.write_bytes(b"")  # empty; collect only needs path existence
    # collect_pngs_by_section uses rglob; empty file is fine
    groups = collect_pngs_by_section(figures)
    assert "realtime_decoding/sorted" in groups
    assert groups["realtime_decoding/sorted"] == [png]


def test_simulation_pngs_group_by_category_subdir(tmp_path: Path):
    figures = tmp_path / "figures"
    for subdir, name in (
        ("behavior", "behavior_trajectory_xy.png"),
        ("features", "behavior_features_over_time.png"),
        ("neural", "ground_truth_spike_raster_all_cell_classes.png"),
        ("sorting", "sorting_loss_by_cell_class.png"),
    ):
        folder = figures / subdir
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b"")

    groups = collect_pngs_by_section(figures)
    # features/ collapses into the behavior section.
    assert set(groups) == {"behavior", "neural", "sorting"}
    assert len(groups["behavior"]) == 2
    assert len(groups["neural"]) == 1
    assert "simulation_report_summary" not in {
        p.stem for paths in groups.values() for p in paths
    }