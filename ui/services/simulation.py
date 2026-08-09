"""Simulation configuration and dataset generation for the Streamlit UI.

Calls the same ``generate_dataset`` / ``SimConfig`` backend as the CLI.
"""

from __future__ import annotations

import re
import subprocess
from copy import deepcopy
from dataclasses import asdict,dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from hippo.anatomy.trajectory_config import (
    DEFAULT_TRAJECTORY_CONFIG,
    list_trajectory_configs,
    resolve_trajectory_config,
)
from hippo_sim.config import (
    ARENA_SIZE_CM,
    BEHAVIOR_DT,
    RECORDING_PARAMS,
    SORTING_PARAMS,
    SimConfig,
)
from hippo_sim.pipeline import apply_trajectory_to_config
from run_simulation import generate_dataset


ProgressCallback = Callable[[str, int, int], None]


@dataclass
class UISimulationConfig:
    """UI-facing simulation form values mapped onto :class:`SimConfig`."""

    dataset_name: str = "ratinabox_ui_001"
    output_root: Path = field(default_factory=lambda: Path("outputs"))
    seed: int = 42
    duration_s: float = 600.0
    # Behavior — locked to project convention (20 Hz)
    behavior_dt: float = BEHAVIOR_DT
    arena_size_cm: float = ARENA_SIZE_CM
    thigmotaxis: float = 0.8
    # Trajectory / population
    trajectory: str | None = None
    no_trajectory: bool = False
    include_non_hippocampal_regions: bool = False
    fallback_schematic_anatomy: bool | None = None
    # Population size overrides (applied into ratinabox_params when set)
    population_overrides: dict[str, int] = field(default_factory=dict)
    # Recording / sorting (existing backend keys)
    sorting_overrides: dict[str, float] = field(default_factory=dict)
    recording_overrides: dict[str, float] = field(default_factory=dict)
    # Post-generation
    generate_diagnostic_figures: bool = True


def sanitize_dataset_name(name: str) -> str:
    """Safe folder name: alphanumeric, dash, underscore."""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name).strip())
    cleaned = cleaned.strip("_") or "dataset"
    return cleaned[:120]


def available_trajectories() -> list[dict[str, Any]]:
    return list_trajectory_configs()


def behavior_sampling_hz(behavior_dt: float = BEHAVIOR_DT) -> float:
    return 1.0 / float(behavior_dt)


def validate_ui_sim_config(cfg: UISimulationConfig) -> list[str]:
    errors: list[str] = []
    name = sanitize_dataset_name(cfg.dataset_name)
    if not name:
        errors.append("Dataset name is required.")
    if cfg.duration_s <= 0:
        errors.append("Duration must be positive.")
    if abs(float(cfg.behavior_dt) - BEHAVIOR_DT) > 1e-9:
        errors.append(
            f"behavior_dt must be {BEHAVIOR_DT} s (20 Hz alignment). "
            "This field is locked by the scientific pipeline."
        )
    if cfg.arena_size_cm <= 0:
        errors.append("Arena size must be positive.")
    out = Path(cfg.output_root) / name
    if out.exists() and any(out.iterdir()):
        errors.append(
            f"Dataset already exists: {out}. Choose a different name "
            "(refusing to overwrite)."
        )
    return errors


def output_dir_for(cfg: UISimulationConfig) -> Path:
    return Path(cfg.output_root) / sanitize_dataset_name(cfg.dataset_name)


def build_sim_config(cfg: UISimulationConfig) -> SimConfig:
    """Translate UI form values into the scientific SimConfig."""
    out = output_dir_for(cfg)
    config = SimConfig(
        output_dir=out,
        seed=int(cfg.seed),
        session_duration_s=float(cfg.duration_s),
        behavior_dt=BEHAVIOR_DT,
        arena_size_cm=float(cfg.arena_size_cm),
        thigmotaxis=float(cfg.thigmotaxis),
    )
    # Apply sorting / recording overlays
    sorting = deepcopy(config.sorting_params)
    sorting.update({k: float(v) for k, v in cfg.sorting_overrides.items()})
    config.sorting_params = sorting
    recording = deepcopy(config.recording_params)
    for k, v in cfg.recording_overrides.items():
        if isinstance(RECORDING_PARAMS.get(k), tuple):
            continue
        recording[k] = float(v)
    config.recording_params = recording

    if not cfg.no_trajectory:
        traj = cfg.trajectory
        if traj is None and DEFAULT_TRAJECTORY_CONFIG.exists():
            traj = str(DEFAULT_TRAJECTORY_CONFIG)
        if traj is not None:
            resolved = resolve_trajectory_config(traj)
            apply_trajectory_to_config(
                config,
                trajectory_config=resolved,
                fallback_schematic=cfg.fallback_schematic_anatomy,
                include_non_hippocampal_regions=cfg.include_non_hippocampal_regions,
            )

    # Apply after trajectory so cell-capture defaults can be overridden from the UI.
    if cfg.population_overrides:
        rb = deepcopy(config.ratinabox_params)
        for k, v in cfg.population_overrides.items():
            if k in rb:
                rb[k] = int(v)
        config.ratinabox_params = rb

    return config


def try_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_simulation_config_yaml(cfg: UISimulationConfig, sim: SimConfig) -> Path:
    """Persist full provenance next to the dataset (does not alter science)."""
    path = Path(sim.output_dir) / "simulation_config.yaml"
    payload = {
        "dataset_name": sanitize_dataset_name(cfg.dataset_name),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": try_git_commit(),
        "ui": {
            "seed": cfg.seed,
            "duration_s": cfg.duration_s,
            "behavior_dt": BEHAVIOR_DT,
            "behavior_sampling_hz": behavior_sampling_hz(),
            "arena_size_cm": cfg.arena_size_cm,
            "thigmotaxis": cfg.thigmotaxis,
            "trajectory": cfg.trajectory,
            "no_trajectory": cfg.no_trajectory,
            "include_non_hippocampal_regions": cfg.include_non_hippocampal_regions,
            "population_overrides": cfg.population_overrides,
            "sorting_overrides": cfg.sorting_overrides,
            "recording_overrides": cfg.recording_overrides,
        },
        "sim_config": {
            "output_dir": str(sim.output_dir),
            "seed": sim.seed,
            "session_duration_s": sim.session_duration_s,
            "behavior_dt": sim.behavior_dt,
            "arena_size_cm": sim.arena_size_cm,
            "thigmotaxis": sim.thigmotaxis,
            "n_channels": sim.n_channels,
            "sorting_params": sim.sorting_params,
            "recording_params": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in sim.recording_params.items()
            },
            "ratinabox_params": {
                k: v for k, v in sim.ratinabox_params.items()
                if k != "feedforward"
            },
            "trajectory_meta": sim.trajectory_meta,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def generate_ui_dataset(
    cfg: UISimulationConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Validate, generate, save provenance, optionally build diagnostic figures."""
    errors = validate_ui_sim_config(cfg)
    if errors:
        raise ValueError("; ".join(errors))

    sim = build_sim_config(cfg)
    summary = generate_dataset(sim, progress_callback=progress_callback)
    save_simulation_config_yaml(cfg, sim)

    if cfg.generate_diagnostic_figures:
        if progress_callback:
            progress_callback("Generating diagnostic figures...", 8, 8)
        try:
            from visualization.experiment_viz import generate_experiment_figures

            generate_experiment_figures(
                sim.output_dir,
                include_simulation=True,
                include_comparison=False,
                include_realtime=False,
                compile_pdf=False,
            )
        except Exception as exc:  # noqa: BLE001 — figures are best-effort after sim
            summary = dict(summary)
            summary["figure_generation_warning"] = str(exc)

    summary = dict(summary)
    summary["output_dir"] = str(sim.output_dir)
    summary["dataset_name"] = sanitize_dataset_name(cfg.dataset_name)
    return summary


# Backward-compatible thin wrappers used by older UI pages
@dataclass
class SimulationRequest:
    output_dir: Path
    seed: int = 42
    duration_s: float = 600.0
    trajectory: str | None = None
    no_trajectory: bool = False


def build_sim_config_from_request(req: SimulationRequest) -> SimConfig:
    ui = UISimulationConfig(
        dataset_name=Path(req.output_dir).name,
        output_root=Path(req.output_dir).parent,
        seed=req.seed,
        duration_s=req.duration_s,
        trajectory=req.trajectory,
        no_trajectory=req.no_trajectory,
    )
    return build_sim_config(ui)


def run_simulation(req: SimulationRequest) -> Path:
    ui = UISimulationConfig(
        dataset_name=Path(req.output_dir).name,
        output_root=Path(req.output_dir).parent,
        seed=req.seed,
        duration_s=req.duration_s,
        trajectory=req.trajectory,
        no_trajectory=req.no_trajectory,
        generate_diagnostic_figures=False,
    )
    # Allow overwrite when called from legacy path that already chose the dir
    out = Path(req.output_dir)
    if out.exists() and any(out.iterdir()):
        # Legacy explicit path: still refuse silent clobber unless empty
        raise FileExistsError(f"Dataset already exists: {out}")
    generate_ui_dataset(ui)
    return out


def default_sorting_sliders() -> dict[str, tuple[float, float, float]]:
    """key → (min, max, default) for Streamlit sliders."""
    return {
        "miss_rate": (0.0, 0.5, float(SORTING_PARAMS["miss_rate"])),
        "contamination_rate": (0.0, 0.5, float(SORTING_PARAMS["contamination_rate"])),
        "jitter_ms": (0.0, 2.0, float(SORTING_PARAMS["jitter_ms"])),
        "merge_prob": (0.0, 0.3, float(SORTING_PARAMS["merge_prob"])),
        "false_positive_rate": (0.0, 0.05, float(SORTING_PARAMS["false_positive_rate"])),
    }


def default_recording_sliders() -> dict[str, tuple[float, float, float]]:
    return {
        "noise_std_uv": (0.0, 50.0, float(RECORDING_PARAMS["noise_std_uv"])),
        "overlap_collision_prob": (0.0, 0.3, float(RECORDING_PARAMS["overlap_collision_prob"])),
        "motion_amplitude_drift_per_min": (
            0.0, 1.0, float(RECORDING_PARAMS["motion_amplitude_drift_per_min"]),
        ),
    }


def population_count_keys() -> list[str]:
    return [
        "n_ca1_place_cells",
        "n_ca3_place_cells",
        "n_dg_place_cells",
        "n_ca2_place_cells",
        "n_mec_grid_cells",
        "n_mec_hd_cells",
        "n_sub_bvc_cells",
        "n_mec_speed_cells",
        "n_int_ca1",
        "n_int_ca3",
        "n_int_dg",
        "n_int_ca2",
        "n_int_sub",
    ]
