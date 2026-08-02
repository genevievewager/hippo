"""Tests for lab trajectory config, import, and cell capture."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from hippo.anatomy.cell_capture import (
    apply_cell_capture_to_ratinabox_params,
    crossed_regions,
    load_cell_capture_config,
)
from hippo.anatomy.trajectory_config import (
    DEFAULT_TRAJECTORY_CONFIG,
    load_trajectory_config,
    resolve_anatomy_regions_file,
    resolve_cell_capture_file,
    validate_trajectory_config,
)
from hippo.anatomy.trajectory_import import (
    ANATOMY_REGION_COLUMNS,
    anatomy_table_to_region_segments,
    assign_channels_from_depth,
    import_trajectory,
    load_lab_anatomy_regions_csv,
    schematic_anatomy_table,
    validate_channel_assignment,
    write_anatomy_regions_csv,
)
from hippo_sim.config import RATINABOX_PARAMS, REGION_SEGMENTS, SimConfig
from hippo_sim.pipeline import apply_trajectory_to_config, run_pipeline
from realtime.deployment_selection import DEPLOYMENT_SPIKE_SOURCE

LAB_CONFIG = Path("configs/trajectories/lab_npx2_default.yaml")
LAB_REGIONS = Path("configs/trajectories/lab_npx2_default_regions.csv")
LAB_CAPTURE = Path("configs/trajectories/lab_npx2_default_cell_capture.yaml")
HPC_OPTIMAL_CONFIG = Path("configs/trajectories/hpc_optimal.yaml")
HPC_OPTIMAL_REGIONS = Path("configs/trajectories/hpc_optimal_regions.csv")
HPC_OPTIMAL_CAPTURE = Path("configs/trajectories/hpc_optimal_cell_capture.yaml")
EXAMPLE_EXPORT = Path("data/probe_trajectories/lab_insertion_001.csv")


def test_default_trajectory_config_path_exists():
    assert DEFAULT_TRAJECTORY_CONFIG.exists()
    assert LAB_CONFIG.exists()
    assert LAB_REGIONS.exists()
    assert LAB_CAPTURE.exists()
    assert HPC_OPTIMAL_CONFIG.exists()
    assert HPC_OPTIMAL_REGIONS.exists()
    assert HPC_OPTIMAL_CAPTURE.exists()


def test_hpc_optimal_trajectory_captures_all_cell_types(tmp_path):
    """Optimal dorsal HPC stack must keep every allowlisted RatInABox type."""
    from hippo.anatomy.cell_capture import CELL_TYPE_TO_N_KEY

    config = SimConfig(output_dir=tmp_path / "hpc_opt", seed=1, session_duration_s=2.0)
    with pytest.warns(UserWarning):
        apply_trajectory_to_config(config, trajectory_config=HPC_OPTIMAL_CONFIG)
    regions = {s["region"] for s in config.region_segments}
    assert {"CA1", "CA2", "CA3", "DG", "Subiculum", "MEC"} <= regions
    missing = [
        ct for ct, key in CELL_TYPE_TO_N_KEY.items()
        if int(config.ratinabox_params.get(key, 0)) <= 0
    ]
    assert missing == []
    regions_path = resolve_anatomy_regions_file(load_trajectory_config(HPC_OPTIMAL_CONFIG))
    assert Path(regions_path).resolve() == HPC_OPTIMAL_REGIONS.resolve()
    assert resolve_cell_capture_file(
        load_trajectory_config(HPC_OPTIMAL_CONFIG)
    ).name == "hpc_optimal_cell_capture.yaml"


def test_load_lab_trajectory_config_warns_on_uncertain_fields():
    with pytest.warns(UserWarning):
        cfg = load_trajectory_config(LAB_CONFIG)
    assert cfg["probe"]["type"] == "NP2.0"
    assert cfg["insertion"]["ap_mm_from_bregma"] == pytest.approx(-3.967)
    assert cfg["insertion"]["ml_mm_from_bregma"] == pytest.approx(3.758)
    assert cfg["insertion"]["dv_uncertain"] is True
    assert cfg["decoder"]["deployment_spike_source"] == "sorted"
    assert cfg["decoder"]["use_ground_truth_for_model_selection"] is False
    assert resolve_anatomy_regions_file(cfg) == LAB_REGIONS
    assert resolve_cell_capture_file(cfg) == LAB_CAPTURE


def test_lab_regions_assign_channels_and_exclude_visual():
    raw = load_lab_anatomy_regions_csv(LAB_REGIONS)
    probe = {
        "type": "NP2.0",
        "site_pitch_um": 15,
        "n_channels": 384,
        "confirm_from_channel_map": True,
    }
    with pytest.warns(UserWarning, match="site_pitch"):
        df = assign_channels_from_depth(raw, probe)
    assert "depth_start_um" in df.columns
    assert df["depth_start_um"].iloc[0] == pytest.approx(0.0)
    assert df["depth_end_um"].iloc[-1] == pytest.approx(3550.0)
    segments = anatomy_table_to_region_segments(df)
    regions = {s["region"] for s in segments}
    assert "visual_cortex" not in regions
    assert "subiculum" in regions
    assert "entorhinal_cortex" in regions
    result = validate_channel_assignment(df, n_channels=384)
    assert result["ok_zero_or_one"]


def test_resolve_trajectory_by_name():
    from hippo.anatomy.trajectory_config import (
        list_trajectory_configs,
        resolve_trajectory_config,
    )

    names = {r["name"] for r in list_trajectory_configs()}
    assert "lab_npx2_default" in names
    assert resolve_trajectory_config("lab_npx2_default") == LAB_CONFIG.resolve()
    assert resolve_trajectory_config(LAB_CONFIG) == LAB_CONFIG.resolve()


def test_lab_default_apply_trajectory_zeros_ca2_ca3(tmp_path):
    config = SimConfig(output_dir=tmp_path / "lab", seed=1, session_duration_s=2.0)
    with pytest.warns(UserWarning):
        apply_trajectory_to_config(config, trajectory_config=LAB_CONFIG)
    assert config.probe_type == "NP2.0"
    assert config.site_pitch_um == 15.0
    # Lab path is Sub/ENT/DG-heavy — CA2/CA3 not crossed.
    assert config.ratinabox_params["n_ca2_place_cells"] == 0
    assert config.ratinabox_params["n_ca3_place_cells"] == 0
    assert config.ratinabox_params["n_sub_bvc_cells"] > 0
    assert config.ratinabox_params["n_mec_grid_cells"] > 0
    anatomy = pd.read_csv(config.output_dir / "anatomy_regions.csv")
    assert "subiculum" in set(anatomy["region"])
    assert config.trajectory_meta["visual_cortex_excluded"] is True
    assert config.trajectory_meta["deployment_spike_source"] == "sorted"
    # Active coords live inside the trial folder.
    active = json.loads((config.output_dir / "trajectory" / "active.json").read_text())
    assert active["active_trajectory_name"] == "lab_npx2_default"
    assert (config.output_dir / "trajectory" / "active_trajectory.yaml").exists()
    assert (config.output_dir / "trajectory" / "anatomy_regions_used.csv").exists()
    assert (config.output_dir / "trajectory" / "cell_capture.yaml").exists()
    assert config.trajectory_meta["active_trajectory_name"] == "lab_npx2_default"


def test_apply_trajectory_by_name_string(tmp_path):
    config = SimConfig(output_dir=tmp_path / "named", seed=1, session_duration_s=2.0)
    with pytest.warns(UserWarning):
        apply_trajectory_to_config(config, trajectory_config="lab_npx2_default")
    assert config.trajectory_meta["active_trajectory_name"] == "lab_npx2_default"
    assert config.trajectory_meta["ap_mm_from_bregma"] == pytest.approx(-3.967)


def test_lab_default_pipeline_short_run(tmp_path):
    config = SimConfig(output_dir=tmp_path / "run", seed=1, session_duration_s=2.0)
    with pytest.warns(UserWarning):
        apply_trajectory_to_config(config, trajectory_config=LAB_CONFIG)
    pytest.importorskip("ratinabox")
    summary = run_pipeline(config)
    units = pd.read_csv(config.output_dir / "units.csv")
    assert "visual_cortex" not in set(units["region"])
    assert set(units["region"]).issubset({
        "HPF_ProS_transition", "subiculum", "dentate_gyrus",
        "entorhinal_cortex", "deep_entorhinal_HATA",
    })
    assert (config.output_dir / "trajectory_metadata.json").exists()
    meta = json.loads((config.output_dir / "trajectory_metadata.json").read_text())
    assert meta["ap_mm_from_bregma"] == pytest.approx(-3.967)
    assert meta["probe_type"] == "NP2.0"
    assert meta["schematic_fallback_used"] is False
    assert (config.output_dir / "figures" / "trajectory" / "fig_probe_trajectory.png").exists()
    assert (config.output_dir / "trajectory" / "anatomy_regions_used.csv").exists()
    assert summary["deployment_spike_source"] == "sorted"


def test_include_non_hippocampal_keeps_visual_in_table_but_can_enable_capture(tmp_path):
    config = SimConfig(output_dir=tmp_path / "vis", seed=1, session_duration_s=2.0)
    with pytest.warns(UserWarning):
        apply_trajectory_to_config(
            config,
            trajectory_config=LAB_CONFIG,
            include_non_hippocampal_regions=True,
        )
    anatomy = pd.DataFrame(config.anatomy_table)
    assert "visual_cortex" in set(anatomy["region"])
    # Segments may still exclude visual unless include flags flipped in CSV;
    # include_non_hippocampal currently only flips the exclude default for
    # capture filtering — anatomy table still lists VIS for the figure.


def test_future_trajectory_config_can_be_swapped(tmp_path):
    """Different --trajectory-config without editing code."""
    new_cfg = tmp_path / "my_insertion.yaml"
    new_regions = tmp_path / "regions.csv"
    pd.DataFrame([
        {
            "depth_start_mm": 0.0,
            "depth_end_mm": 1.0,
            "region": "subiculum",
            "layer_or_area": "SUB",
            "acronym": "SUB",
            "include_in_hippocampal_simulation": True,
            "candidate_cell_classes": "Sub_bvc",
            "notes": "test",
        }
    ]).to_csv(new_regions, index=False)
    new_capture = tmp_path / "capture.yaml"
    new_capture.write_text(yaml.safe_dump({
        "cell_capture": {
            "subiculum": {
                "include": True,
                "unit_density_per_mm": 40,
                "cell_type_probabilities": {"Sub_bvc": 1.0},
            }
        },
        "recording_capture": {"detection_radius_um": 50},
    }))
    payload = {
        "animal": {"strain": "C57_WT", "species": "mouse", "atlas": "Allen_CCF",
                   "hemisphere": "right", "coordinate_status": "approximate"},
        "probe": {"type": "NP2.0", "site_pitch_um": 15, "n_channels": 384},
        "insertion": {
            "bregma_to_lambda_mm": 3.8,
            "ap_mm_from_bregma": -2.0,
            "ml_mm_from_bregma": 2.0,
            "dv_mm_from_brain_surface": 2.5,
            "dv_uncertain": False,
            "horizontal_angle_deg": 0,
            "vertical_angle_deg": 90,
            "angle_convention_uncertain": False,
        },
        "simulation": {
            "anatomy_regions_file": str(new_regions),
            "cell_capture_file": str(new_capture),
            "fallback_to_schematic_hippocampus": False,
            "exclude_non_hippocampal_regions_by_default": True,
        },
        "decoder": {
            "deployment_spike_source": "sorted",
            "use_ground_truth_for_model_selection": False,
        },
    }
    new_cfg.write_text(yaml.safe_dump(payload))
    config = SimConfig(output_dir=tmp_path / "swap", seed=2, session_duration_s=2.0)
    apply_trajectory_to_config(config, trajectory_config=new_cfg)
    assert config.trajectory_meta["ap_mm_from_bregma"] == pytest.approx(-2.0)
    assert set(s["region"] for s in config.region_segments) == {"subiculum"}
    assert config.ratinabox_params["n_sub_bvc_cells"] > 0
    assert config.ratinabox_params["n_mec_grid_cells"] == 0


def test_legacy_export_still_works():
    df, meta = import_trajectory(
        EXAMPLE_EXPORT,
        trajectory_config="configs/probe_trajectory.yaml",
        fallback_schematic=True,
    )
    assert not meta["used_fallback"]
    for col in ("region", "depth_start_um", "depth_end_um"):
        assert col in df.columns


def test_schematic_fallback_via_no_trajectory(tmp_path):
    config = SimConfig(output_dir=tmp_path / "schematic", seed=2, session_duration_s=2.0)
    # Do not call apply_trajectory_to_config — schematic path.
    pytest.importorskip("ratinabox")
    summary = run_pipeline(config)
    units = pd.read_csv(config.output_dir / "units.csv")
    assert len(units) > 0
    assert summary["trajectory_source"] == "schematic_fallback"


def test_sorted_spike_deployment_remains_default():
    assert DEPLOYMENT_SPIKE_SOURCE == "sorted"
    with pytest.warns(UserWarning):
        cfg = load_trajectory_config(LAB_CONFIG)
    assert cfg["decoder"]["use_ground_truth_for_model_selection"] is False
    assert cfg["decoder"]["deployment_spike_source"] == "sorted"


def test_validate_fails_without_anatomy_when_no_fallback(tmp_path):
    bad = {
        "animal": {"coordinate_status": "approximate"},
        "probe": {"type": "NP2.0"},
        "insertion": {
            "bregma_to_lambda_mm": 3.8,
            "ap_mm_from_bregma": -1.0,
            "ml_mm_from_bregma": 1.0,
            "dv_mm_from_brain_surface": 2.0,
        },
        "simulation": {
            "anatomy_regions_file": str(tmp_path / "missing.csv"),
            "fallback_to_schematic_hippocampus": False,
        },
        "decoder": {"deployment_spike_source": "sorted"},
    }
    with pytest.raises(ValueError, match="neither trajectory_export_file"):
        validate_trajectory_config(bad)
