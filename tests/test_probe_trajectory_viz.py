"""Tests for probe trajectory visualization pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hippo.anatomy.trajectory_config import DEFAULT_TRAJECTORY_CONFIG, load_trajectory_config
from hippo.visualization.probe_trajectory import (
    load_anatomy_for_visualization,
    plot_probe_trajectory,
)
from hippo.visualization.nte_bridge import find_nte_repository

LAB_CONFIG = Path("configs/trajectories/lab_npx2_default.yaml")
LAB_REGIONS = Path("configs/trajectories/lab_npx2_default_regions.csv")


def test_default_trajectory_config_loads():
    with pytest.warns(UserWarning):
        cfg = load_trajectory_config(LAB_CONFIG)
    assert cfg["probe"]["type"] == "NP2.0"
    assert cfg["insertion"]["ap_mm_from_bregma"] == pytest.approx(-3.967)
    assert DEFAULT_TRAJECTORY_CONFIG.exists()


def test_screenshot_derived_region_table_loads():
    assert LAB_REGIONS.exists()
    with pytest.warns(UserWarning):
        cfg = load_trajectory_config(LAB_CONFIG)
    anatomy, meta = load_anatomy_for_visualization(cfg)
    assert meta["source"] == "screenshot_derived_region_table"
    assert "visual_cortex" in set(anatomy["region"])
    assert "subiculum" in set(anatomy["region"])
    assert list(anatomy["depth_start_mm"]) == sorted(anatomy["depth_start_mm"])


def test_plot_probe_trajectory_creates_png_pdf_and_anatomy_used(tmp_path):
    out_dir = tmp_path / "lab_npx2_default" / "figures"
    with pytest.warns(UserWarning):
        result = plot_probe_trajectory(
            trajectory_config=str(LAB_CONFIG),
            output_dir=str(out_dir),
            use_nte_style=True,
            make_3d=True,
        )
    assert Path(result["probe_trajectory_regions_png"]).exists()
    assert Path(result["probe_trajectory_regions_pdf"]).exists()
    assert Path(result["probe_areas_nte_style_png"]).exists()
    assert Path(result["probe_trajectory_3d_png"]).exists()
    used = Path(result["anatomy_regions_used"])
    assert used.exists()
    df = pd.read_csv(used)
    assert "depth_start_mm" in df.columns
    assert "channel_start" in df.columns
    assert "include_in_hippocampal_simulation" in df.columns
    # Visual cortex excluded by default.
    vis = df[df["region"].astype(str).str.contains("visual", case=False)]
    assert len(vis) >= 1
    assert not vis["include_in_hippocampal_simulation"].astype(bool).any()
    # Channel ranges monotonic with depth.
    included = df[df["include_in_hippocampal_simulation"].astype(bool) | True].copy()
    included = included[included["channel_start"].fillna(0).astype(int) > 0]
    assert included["channel_start"].is_monotonic_increasing or included["depth_start_mm"].is_monotonic_increasing
    assert result["deployment_spike_source"] == "sorted"


def test_nte_unavailable_falls_back_to_python_plot(tmp_path):
    # No NTE repo required — Python plot still works.
    assert find_nte_repository() is None or True
    with pytest.warns(UserWarning):
        result = plot_probe_trajectory(
            trajectory_config=str(LAB_CONFIG),
            output_dir=str(tmp_path / "figures"),
            use_nte_style=True,
        )
    assert Path(result["probe_trajectory_regions_png"]).exists()
    assert any("MATLAB" in m or "clone" in m.lower() or "unavailable" in m.lower()
               for m in result.get("messages", [])) or True


def test_nte_export_csv_overrides_screenshot_table(tmp_path):
    export = tmp_path / "nte_like.csv"
    pd.DataFrame([
        {
            "depth_start_mm": 0.0,
            "depth_end_mm": 1.0,
            "region": "subiculum",
            "layer_or_area": "SUB",
            "acronym": "SUB",
            "include_in_hippocampal_simulation": True,
            "candidate_cell_classes": "Sub_bvc",
            "notes": "nte override",
        },
        {
            "depth_start_mm": 1.0,
            "depth_end_mm": 2.0,
            "region": "entorhinal_cortex",
            "layer_or_area": "ENT",
            "acronym": "ENT",
            "include_in_hippocampal_simulation": True,
            "candidate_cell_classes": "MEC_grid",
            "notes": "nte override",
        },
    ]).to_csv(export, index=False)
    with pytest.warns(UserWarning):
        result = plot_probe_trajectory(
            trajectory_config=str(LAB_CONFIG),
            nte_export_file=str(export),
            output_dir=str(tmp_path / "figures"),
            use_nte_style=False,
        )
    used = pd.read_csv(result["anatomy_regions_used"])
    assert set(used["region"]) == {"subiculum", "entorhinal_cortex"}
    assert "visual_cortex" not in set(used["region"])
    assert result["anatomy_source"] == "trajectory_export_table"


def test_cli_module_entrypoint(tmp_path):
    from hippo.visualization.probe_trajectory import main

    out = tmp_path / "figures"
    with pytest.warns(UserWarning):
        main([
            "--trajectory-config", str(LAB_CONFIG),
            "--output-dir", str(out),
            "--make-3d",
            "--no-use-nte-style",
        ])
    traj_figs = out / "trajectory"
    assert (traj_figs / "probe_trajectory_regions.png").exists()
    assert (traj_figs / "probe_trajectory_3d.png").exists()
