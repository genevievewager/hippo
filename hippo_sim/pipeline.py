"""End-to-end simulation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hippo_sim.anatomy import AnatomyMap
from hippo_sim.behavior import simulate_behavior
from hippo_sim.config import SimConfig
from hippo_sim.neural_backend import simulate_neural_activity
from hippo_sim.recording import build_unit_templates, simulate_recording
from hippo_sim.sorting import kilosort_like_sort
from hippo_sim.spikes import generate_spike_trains


def _build_region_table(config: SimConfig) -> list[dict]:
    """Legacy schematic table builder (used when no anatomy_table is preloaded)."""
    from hippo_sim.config import REGION_TO_CELL_TYPE, channels_for_segment

    pitch = float(getattr(config, "site_pitch_um", 20.0))
    n_channels = int(config.n_channels)
    table = []
    for seg in config.region_segments:
        channels = channels_for_segment(
            seg["z_start"], seg["z_end"], pitch_um=pitch, n_channels=n_channels,
        )
        n_ch = len(channels)
        n_units = max(1, int(seg["density"] * n_ch)) if n_ch else 0
        ch_start = int(channels[0]) + 1 if len(channels) else 0
        ch_end = int(channels[-1]) + 1 if len(channels) else 0
        cell_type = REGION_TO_CELL_TYPE.get((seg["region"], seg["layer"]), "")
        table.append({
            "region": seg["region"],
            "layer": seg["layer"],
            "acronym": seg["region"],
            "depth_start_um": seg["z_start"],
            "depth_end_um": seg["z_end"],
            "channel_start": ch_start,
            "channel_end": ch_end,
            "channels": f"{ch_start}-{ch_end}" if n_ch else "",
            "n_channels": n_ch,
            "n_units": n_units,
            "cell_types": cell_type,
            "trajectory_source": "schematic_fallback",
        })
    return table


def _units_to_dataframe(units: list, config: SimConfig | None = None) -> pd.DataFrame:
    from hippo.anatomy.hippocampal_system import (
        annotate_units_for_analysis,
        canonicalize_region,
        unit_include_in_decoder,
    )

    traj_name = ""
    capture_source = "schematic"
    include_non_hpc = False
    if config is not None:
        meta = config.trajectory_meta or {}
        traj_name = str(Path(str(meta.get("trajectory_config_path") or "")).name)
        if meta.get("schematic_fallback_used"):
            capture_source = "schematic_fallback"
        elif meta.get("anatomy_source"):
            capture_source = "trajectory_config"
        include_non_hpc = bool(meta.get("include_non_hippocampal_regions", False))

    rows = []
    for u in units:
        include = unit_include_in_decoder(
            cell_type=u.cell_type,
            region=u.region,
            include_non_hippocampal=include_non_hpc,
        )
        row = {
            "unit_id": u.unit_id,
            "cell_type": u.cell_type,
            "region": u.region,
            "region_canonical": canonicalize_region(u.region),
            "layer": u.layer,
            "layer_or_area": u.layer,
            "acronym": getattr(u, "acronym", "") or "",
            "channel": u.channel + 1,
            "depth_um": u.depth_um,
            "place_x_cm": u.place_center_cm[0],
            "place_y_cm": u.place_center_cm[1],
            "hd_pref_rad": u.hd_pref_rad,
            "rate_model": u.rate_model or f"ratinabox_{u.cell_type}",
            "capture_source": capture_source,
            "trajectory_config_name": traj_name,
            "include_in_decoder": include,
        }
        if u.ratinabox_class is not None:
            row["ratinabox_class"] = u.ratinabox_class
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = annotate_units_for_analysis(df, include_non_hippocampal=include_non_hpc)
    return df


def _spikes_to_dataframe(
    spike_trains,
    units_df: pd.DataFrame,
) -> pd.DataFrame:
    unit_lookup = units_df.set_index("unit_id")
    rows = []
    for train in spike_trains:
        uid = train.unit_id
        meta = unit_lookup.loc[uid]
        for t in train.spike_times_s:
            row = {
                "unit_id": uid,
                "spike_time_s": t,
                "cell_type": meta["cell_type"],
                "region": meta["region"],
                "rate_model": meta.get("rate_model", meta["cell_type"]),
            }
            if "ratinabox_class" in meta.index:
                row["ratinabox_class"] = meta["ratinabox_class"]
            rows.append(row)
    return pd.DataFrame(rows)


def apply_trajectory_to_config(
    config: SimConfig,
    *,
    trajectory_config: Path | str | None = None,
    trajectory_export: Path | str | None = None,
    anatomy_regions_file: Path | str | None = None,
    cell_capture_config: Path | str | None = None,
    fallback_schematic: bool | None = None,
    include_non_hippocampal_regions: bool = False,
) -> SimConfig:
    """Load trajectory + cell-capture configs into ``SimConfig`` in place.

    Resolution order for anatomy:
      A. CLI ``trajectory_export`` or config ``insertion.trajectory_export_file``
      B. CLI ``anatomy_regions_file`` or config ``simulation.anatomy_regions_file``
      C. Schematic fallback (only if enabled)
    """
    from hippo.anatomy.cell_capture import (
        annotate_anatomy_cell_types,
        apply_cell_capture_to_ratinabox_params,
        load_cell_capture_config,
    )
    from hippo.anatomy.trajectory_config import (
        DEFAULT_TRAJECTORY_CONFIG,
        build_trajectory_metadata,
        load_trajectory_config,
        prefer_trajectory_export,
        resolve_anatomy_regions_file,
        resolve_cell_capture_file,
        resolve_trajectory_config,
        write_trial_trajectory_bundle,
    )
    from hippo.anatomy.trajectory_import import (
        anatomy_table_to_region_segments,
        assign_channels_from_depth,
        import_trajectory,
        load_lab_anatomy_regions_csv,
        schematic_anatomy_table,
        write_anatomy_regions_csv,
    )

    # Prefer the new configs/trajectories/*.yaml schema when that path is used
    # or when defaulting to the lab NP2.0 insertion.
    traj_path: Path | None = None
    if trajectory_config is not None:
        traj_path = resolve_trajectory_config(trajectory_config)
    use_lab_schema = False
    if traj_path is not None and traj_path.exists():
        text = traj_path.read_text()
        use_lab_schema = "ap_mm_from_bregma" in text or "anatomy_regions_file" in text
    elif traj_path is None and DEFAULT_TRAJECTORY_CONFIG.exists():
        traj_path = resolve_trajectory_config(None)
        use_lab_schema = True

    extra_warnings: list[str] = []
    schematic_fallback_used = False
    anatomy_source = "unknown"
    anatomy_used_path: str | None = None
    visual_excluded = False

    if use_lab_schema and traj_path is not None:
        traj_cfg = load_trajectory_config(traj_path)
        probe_cfg = traj_cfg.get("probe") or {}
        sim_cfg = traj_cfg.get("simulation") or {}
        insertion = traj_cfg.get("insertion") or {}

        if fallback_schematic is None:
            fallback_schematic = bool(sim_cfg.get("fallback_to_schematic_hippocampus", False))

        exclude_non_hpc = bool(sim_cfg.get("exclude_non_hippocampal_regions_by_default", True))
        if include_non_hippocampal_regions:
            exclude_non_hpc = False
        visual_excluded = exclude_non_hpc

        # Capture config.
        capture_path = cell_capture_config or resolve_cell_capture_file(traj_cfg)
        capture_cfg = load_cell_capture_config(capture_path)

        # Anatomy source resolution.
        export_override = trajectory_export
        if export_override is None and prefer_trajectory_export(traj_cfg):
            export_override = insertion.get("trajectory_export_file")

        if export_override is not None and Path(str(export_override)).exists():
            anatomy_df, import_meta = import_trajectory(
                export_override,
                fallback_schematic=False,
                region_segments=list(config.region_segments),
            )
            anatomy_source = "neuropixels_trajectory_explorer_export"
            anatomy_used_path = str(Path(str(export_override)).resolve())
            anatomy_df = assign_channels_from_depth(anatomy_df, probe_cfg)
        else:
            regions_path = anatomy_regions_file or resolve_anatomy_regions_file(traj_cfg)
            if regions_path is not None and Path(regions_path).exists():
                anatomy_df = load_lab_anatomy_regions_csv(
                    regions_path, exclude_non_hippocampal=exclude_non_hpc,
                )
                anatomy_df = assign_channels_from_depth(anatomy_df, probe_cfg)
                # Stamp insertion coords onto table.
                anatomy_df["insertion_ap_mm"] = insertion.get("ap_mm_from_bregma")
                anatomy_df["insertion_ml_mm"] = insertion.get("ml_mm_from_bregma")
                anatomy_df["insertion_depth_um"] = (
                    float(insertion["dv_mm_from_brain_surface"]) * 1000.0
                    if insertion.get("dv_mm_from_brain_surface") is not None
                    else pd.NA
                )
                anatomy_df["insertion_angle_ap_deg"] = insertion.get("horizontal_angle_deg")
                anatomy_df["insertion_angle_ml_deg"] = insertion.get("vertical_angle_deg")
                anatomy_source = "lab_anatomy_regions_file"
                anatomy_used_path = str(Path(regions_path).resolve())
            elif fallback_schematic:
                anatomy_df = schematic_anatomy_table(
                    list(config.region_segments),
                    n_channels=int(probe_cfg.get("n_channels") or config.n_channels),
                    pitch_um=float(probe_cfg.get("site_pitch_um") or config.site_pitch_um),
                    probe_type=str(probe_cfg.get("type") or "schematic"),
                )
                schematic_fallback_used = True
                anatomy_source = "schematic_fallback"
                extra_warnings.append("Fell back to schematic hippocampal anatomy.")
            else:
                raise FileNotFoundError(
                    "No trajectory export or anatomy_regions_file available, "
                    "and schematic fallback is disabled."
                )

        # Optionally force-include non-hippocampal bands for capture/segments.
        if include_non_hippocampal_regions and "include_in_hippocampal_simulation" in anatomy_df.columns:
            anatomy_df = anatomy_df.copy()
            anatomy_df["include_in_hippocampal_simulation"] = True
            visual_excluded = False

        anatomy_df = annotate_anatomy_cell_types(anatomy_df, capture_cfg)

        # Probe geometry onto SimConfig.
        if probe_cfg.get("n_channels") is not None:
            config.n_channels = int(probe_cfg["n_channels"])
        if probe_cfg.get("n_channels_recorded") is not None:
            config.n_channels = int(probe_cfg["n_channels_recorded"])
        if probe_cfg.get("site_pitch_um") is not None:
            config.site_pitch_um = float(probe_cfg["site_pitch_um"])
        if probe_cfg.get("type") is not None:
            config.probe_type = str(probe_cfg["type"])

        segments = anatomy_table_to_region_segments(anatomy_df)
        # When excluding non-hippocampal, segments already skip include=false rows.
        if segments:
            config.region_segments = segments

        baseline_rp = dict(config.ratinabox_params)
        config.ratinabox_params = apply_cell_capture_to_ratinabox_params(
            anatomy_df,
            config.ratinabox_params,
            capture_cfg,
            baseline_params=baseline_rp,
            scale_populations=not schematic_fallback_used,
        )
        # Apply recording_capture sorting hints when present.
        rec = capture_cfg.get("recording_capture") or {}
        if rec.get("sorting_miss_rate_default") is not None:
            config.sorting_params["miss_rate"] = float(rec["sorting_miss_rate_default"])
        if rec.get("jitter_ms_default") is not None:
            config.sorting_params["jitter_ms"] = float(rec["jitter_ms_default"])
        if rec.get("contamination_rate_default") is not None:
            config.sorting_params["contamination_rate"] = float(rec["contamination_rate_default"])

        config.cell_capture_config = capture_cfg
        bundle = write_trial_trajectory_bundle(
            config.output_dir,
            traj_cfg,
            anatomy_df,
            cell_capture_src=capture_path,
            anatomy_source=anatomy_source,
        )
        anatomy_used_path = bundle["anatomy_regions_used"]
        meta = build_trajectory_metadata(
            traj_cfg,
            anatomy_source=anatomy_source,
            anatomy_regions_used=anatomy_used_path,
            cell_capture_used=bundle.get("cell_capture_file") or (
                str(Path(str(capture_path)).resolve()) if capture_path else None
            ),
            visual_cortex_excluded=visual_excluded,
            schematic_fallback_used=schematic_fallback_used,
            extra_warnings=extra_warnings,
        )
        meta["source"] = anatomy_source
        meta["used_fallback"] = schematic_fallback_used
        meta["export_file"] = (
            str(export_override) if export_override else insertion.get("trajectory_export_file")
        )
        meta["include_non_hippocampal_regions"] = include_non_hippocampal_regions
        meta["trial_trajectory"] = bundle
        meta["active_trajectory_yaml"] = bundle.get("active_trajectory_yaml")
        config.trajectory_meta = meta
        config.anatomy_table = anatomy_df.to_dict(orient="records")
        return config

    # ---- Legacy probe_trajectory.yaml / export-only path ----
    if fallback_schematic is None:
        fallback_schematic = True
    capture_cfg = load_cell_capture_config(cell_capture_config)
    anatomy_df, traj_meta = import_trajectory(
        trajectory_export or anatomy_regions_file,
        trajectory_config=trajectory_config,
        fallback_schematic=fallback_schematic,
        region_segments=list(config.region_segments),
    )
    anatomy_df = annotate_anatomy_cell_types(anatomy_df, capture_cfg)

    probe_cfg = (traj_meta.get("config") or {}).get("probe") or {}
    if probe_cfg.get("n_channels") is not None:
        config.n_channels = int(probe_cfg["n_channels"])
    if probe_cfg.get("site_pitch_um") is not None:
        config.site_pitch_um = float(probe_cfg["site_pitch_um"])
    if probe_cfg.get("type") is not None:
        config.probe_type = str(probe_cfg["type"])

    segments = anatomy_table_to_region_segments(anatomy_df)
    if segments:
        config.region_segments = segments

    baseline_rp = dict(config.ratinabox_params)
    scale_populations = (
        not bool(traj_meta.get("used_fallback", False))
        and str(traj_meta.get("source", "")) != "schematic_fallback"
    )
    config.ratinabox_params = apply_cell_capture_to_ratinabox_params(
        anatomy_df,
        config.ratinabox_params,
        capture_cfg,
        baseline_params=baseline_rp,
        scale_populations=scale_populations,
    )
    config.cell_capture_config = capture_cfg
    config.trajectory_meta = traj_meta
    config.anatomy_table = anatomy_df.to_dict(orient="records")
    write_anatomy_regions_csv(anatomy_df, Path(config.output_dir) / "anatomy_regions.csv")
    return config


def run_pipeline(config: SimConfig) -> dict:
    """Run full simulation and save outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    print("[1/7] Simulating behavior (RatInABox)...", flush=True)
    behavior_result = simulate_behavior(config)
    behavior = behavior_result.trace
    behavior_df = pd.DataFrame({
        "time_s": behavior.time_s,
        "x_cm": behavior.position_cm[:, 0],
        "y_cm": behavior.position_cm[:, 1],
        "speed_cm_s": behavior.speed_cm_s,
        "head_direction_rad": behavior.head_direction_rad,
        "distance_to_wall_cm": behavior.distance_to_wall_cm,
        "acceleration_cm_s2": behavior.acceleration_cm_s2,
    })
    behavior_df.to_csv(config.output_dir / "behavior.csv", index=False)

    print("[2/7] Building anatomy and unit assignments...", flush=True)
    if config.anatomy_table:
        anatomy_df = pd.DataFrame(config.anatomy_table)
    else:
        from hippo.anatomy.trajectory_import import schematic_anatomy_table

        anatomy_df = schematic_anatomy_table(
            config.region_segments,
            n_channels=config.n_channels,
            pitch_um=float(getattr(config, "site_pitch_um", 20.0)),
            probe_type=getattr(config, "probe_type", "neuropixels_1.0"),
        )
        config.trajectory_meta = {
            "source": "schematic_fallback",
            "used_fallback": True,
            "schematic_fallback_used": True,
            "export_file": None,
            "deployment_spike_source": "sorted",
        }

    anatomy_df.to_csv(config.output_dir / "anatomy_regions.csv", index=False, na_rep="NA")
    region_table = anatomy_df.to_dict(orient="records")

    print("[3/7] Generating neural activity (RatInABox)...", flush=True)
    units, rates, neural_metadata = simulate_neural_activity(
        config=config,
        behavior=behavior,
        rng=rng,
        behavior_result=behavior_result,
    )

    np.save(config.output_dir / "rates.npy", rates)
    with open(config.output_dir / "neural_backend_metadata.json", "w") as f:
        json.dump(neural_metadata, f, indent=2)

    _save_ratinabox_group_outputs(config, units, rates, neural_metadata)

    units_df = _units_to_dataframe(units, config)
    # Attach template amplitudes from capture model if available.
    if "template_amplitude_uv" not in units_df.columns:
        rec = (config.cell_capture_config or {}).get("recording_capture") or {}
        amp_min = float(rec.get("min_template_amplitude_uv", 20.0))
        amp_max = float(rec.get("max_template_amplitude_uv", 200.0))
        rng_amp = np.random.default_rng(config.seed + 17)
        units_df["template_amplitude_uv"] = rng_amp.uniform(amp_min, amp_max, size=len(units_df))
    units_df.to_csv(config.output_dir / "units.csv", index=False)

    n_excluded = int((~units_df["include_in_decoder"].astype(bool)).sum()) if len(units_df) else 0

    anatomy = AnatomyMap(units=units, region_table=region_table, channel_to_region={})

    print("[4/7] Generating ground-truth spike trains...", flush=True)
    spike_trains = generate_spike_trains(rates, config, rng, time_axis=behavior.time_s)
    gt_df = _spikes_to_dataframe(spike_trains, units_df)
    gt_df.to_csv(config.output_dir / "spikes_ground_truth.csv", index=False)

    print("[5/7] Building templates and simulating Neuropixels recording...", flush=True)
    templates = build_unit_templates(anatomy, config, rng)
    events = simulate_recording(spike_trains, templates, config, rng)

    print("[6/7] Kilosort-like re-extraction...", flush=True)
    sorted_spikes = kilosort_like_sort(events, templates, spike_trains, config, rng)

    sorted_rows = [{
        "unit_id": s.unit_id,
        "spike_time_s": s.time_s,
        "channel": s.channel + 1,
        "confidence": s.confidence,
    } for s in sorted_spikes]
    pd.DataFrame(sorted_rows).to_csv(config.output_dir / "spikes_sorted.csv", index=False)

    print("[7/7] Saving summary + trajectory figures...", flush=True)
    # Keep anatomy_regions_used.csv identical to the table driving the simulation.
    traj_dir = config.output_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    anatomy_used = traj_dir / "anatomy_regions_used.csv"
    anatomy_df.to_csv(anatomy_used, index=False, na_rep="NA")

    fig_paths: dict = {}
    try:
        from visualization.constants import FIGURE_SUBDIR_TRAJECTORY

        fig_traj_dir = config.output_dir / "figures" / FIGURE_SUBDIR_TRAJECTORY
        traj_cfg_path = (config.trajectory_meta or {}).get("trajectory_config_path")
        if traj_cfg_path and Path(str(traj_cfg_path)).exists():
            from hippo.visualization.probe_trajectory import plot_probe_trajectory

            viz = plot_probe_trajectory(
                trajectory_config=str(traj_cfg_path),
                anatomy_regions_file=str(anatomy_used),
                nte_export_file=None,
                output_dir=str(fig_traj_dir),
                include_non_hippocampal_regions=bool(
                    (config.trajectory_meta or {}).get("include_non_hippocampal_regions", False)
                ),
                use_nte_style=False,
                make_3d=False,
                run_root=config.output_dir,
                prefer_explicit_anatomy_file=True,
            )
            # Compact publication page (replaces loose probe/unit-count PNGs).
            from visualization.publication_trajectory_plots import (
                generate_publication_trajectory_figures,
            )

            fig_paths = {
                "fig_probe_trajectory": generate_publication_trajectory_figures(
                    anatomy_df,
                    units_df,
                    config.output_dir / "figures",
                    n_channels=config.n_channels,
                    meta=config.trajectory_meta,
                ),
            }
            # Keep any remaining png/pdf paths from viz metadata that survive cleanup.
            for k, v in viz.items():
                if isinstance(v, str) and (v.endswith(".png") or v.endswith(".pdf")):
                    p = Path(v)
                    if p.exists():
                        fig_paths[k] = p
        else:
            from hippo.anatomy.trajectory_plots import generate_trajectory_validation_figures

            fig_paths = generate_trajectory_validation_figures(
                anatomy_df,
                units_df,
                config.output_dir / "figures",
                meta=config.trajectory_meta,
                n_channels=config.n_channels,
            )
    except Exception as exc:  # pragma: no cover
        fig_paths = {}
        print(f"Warning: trajectory figures failed: {exc}", flush=True)

    deployment_source = (config.trajectory_meta or {}).get(
        "deployment_spike_source", "sorted",
    )
    from hippo.anatomy.hippocampal_system import geometry_summary

    geom = geometry_summary(units_df)
    summary = {
        "neural_backend": "ratinabox_neurons",
        "n_units": len(units),
        "n_units_include_in_decoder": int(units_df["include_in_decoder"].astype(bool).sum()) if len(units_df) else 0,
        "n_units_excluded_from_decoder": n_excluded,
        "analysis_cell_types": "ratinabox_hippocampal_system_only",
        "n_behavior_steps": config.n_behavior_steps,
        "behavior_dt": config.behavior_dt,
        "rates_shape": list(rates.shape),
        "n_ground_truth_spikes": len(gt_df),
        "n_sorted_spikes": len(sorted_rows),
        "session_duration_s": config.session_duration_s,
        "n_channels": config.n_channels,
        "seed": config.seed,
        "arena_size_cm": config.arena_size_cm,
        "trajectory_source": (config.trajectory_meta or {}).get("source", "schematic_fallback"),
        "trajectory_used_fallback": bool(
            (config.trajectory_meta or {}).get(
                "schematic_fallback_used",
                (config.trajectory_meta or {}).get("used_fallback", True),
            )
        ),
        "trajectory_export_file": (config.trajectory_meta or {}).get("export_file"),
        "deployment_spike_source": deployment_source,
        "geometry": geom,
        "circuit_profile": geom.get("circuit_profile"),
        "present_regions": geom.get("present_regions"),
        "active_trajectory_name": (config.trajectory_meta or {}).get("active_trajectory_name"),
        "active_trajectory_yaml": (config.trajectory_meta or {}).get("active_trajectory_yaml"),
        "notes": (
            "Ground-truth spikes are for simulation diagnostics only. "
            "Deployable decoder selection must use sorted spikes."
        ),
    }
    if "ratinabox_cell_groups" in neural_metadata:
        summary["ratinabox_cell_groups"] = neural_metadata["ratinabox_cell_groups"]
    if fig_paths:
        summary["trajectory_figures"] = {k: str(v) for k, v in fig_paths.items()}

    with open(config.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Preferred name per acceptance criteria; keep legacy alias too.
    meta_path = config.output_dir / "trajectory_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(config.trajectory_meta or {}, f, indent=2, default=str)
    with open(config.output_dir / "trajectory_meta.json", "w") as f:
        json.dump(config.trajectory_meta or {}, f, indent=2, default=str)

    print("Done.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _save_ratinabox_group_outputs(
    config: SimConfig,
    units: list,
    rates: np.ndarray,
    metadata: dict,
) -> None:
    """Save per-group RatInABox rate arrays and metadata."""
    groups = metadata.get("ratinabox_cell_groups", {})
    if not groups:
        return

    group_meta_rows = []
    offset = 0
    arrays = {}
    for rb_class, n_group in groups.items():
        sl = rates[offset:offset + n_group]
        arrays[rb_class] = sl
        sample_unit = units[offset]
        group_meta_rows.append({
            "group_name": rb_class,
            "ratinabox_class": rb_class,
            "cell_type": sample_unit.cell_type,
            "rate_model": sample_unit.rate_model,
            "n_units": n_group,
        })
        offset += n_group

    np.savez(config.output_dir / "ratinabox_rates_by_group.npz", **arrays)
    pd.DataFrame(group_meta_rows).to_csv(
        config.output_dir / "ratinabox_group_metadata.csv", index=False,
    )
