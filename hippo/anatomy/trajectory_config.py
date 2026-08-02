"""Load and validate reusable trajectory configuration YAMLs.

Lab insertion metadata lives under ``configs/trajectories/`` so future animals
can swap AP/ML/DV/probe/region tables without editing simulation code.

Each simulation trial snapshots the *active* trajectory into
``<output>/trajectory/`` so coordinates travel with the run.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TRAJECTORY_CONFIG = Path("configs/trajectories/lab_npx2_default.yaml")
TRAJECTORY_CONFIG_DIR = Path("configs/trajectories")

REQUIRED_SECTIONS = ("animal", "probe", "insertion", "simulation", "decoder")

_TEMPLATE_NAMES = frozenset({"example_new_insertion.yaml"})


def list_trajectory_configs(
    config_dir: str | Path | None = None,
    *,
    include_templates: bool = False,
) -> list[dict[str, Any]]:
    """List selectable insertion YAMLs under ``configs/trajectories/``."""
    root = Path(config_dir) if config_dir is not None else TRAJECTORY_CONFIG_DIR
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.yaml")):
        name = path.name
        if name.endswith("_cell_capture.yaml"):
            continue
        if name in _TEMPLATE_NAMES and not include_templates:
            continue
        rows.append({
            "name": path.stem,
            "path": str(path.resolve()),
            "is_default": (
                path.resolve() == DEFAULT_TRAJECTORY_CONFIG.resolve()
                if DEFAULT_TRAJECTORY_CONFIG.exists()
                else False
            ),
            "is_template": name in _TEMPLATE_NAMES,
        })
    return rows


def resolve_trajectory_config(
    name_or_path: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
) -> Path:
    """Resolve a trajectory name (``lab_npx2_default``) or filesystem path.

    ``None`` → default lab NP2.0 config.
    """
    root = Path(config_dir) if config_dir is not None else TRAJECTORY_CONFIG_DIR
    if name_or_path is None:
        if DEFAULT_TRAJECTORY_CONFIG.exists():
            return DEFAULT_TRAJECTORY_CONFIG.resolve()
        raise FileNotFoundError(
            f"Default trajectory config not found: {DEFAULT_TRAJECTORY_CONFIG}"
        )

    raw = Path(str(name_or_path))
    if raw.exists():
        return raw.resolve()

    stem = raw.name
    if stem.endswith(".yaml") or stem.endswith(".yml"):
        candidate = root / stem
    else:
        candidate = root / f"{stem}.yaml"
    if candidate.exists():
        return candidate.resolve()

    available = ", ".join(r["name"] for r in list_trajectory_configs(root)) or "(none)"
    raise FileNotFoundError(
        f"Trajectory config not found: {name_or_path!r}. "
        f"Pass a path or a name under {root}/. Available: {available}"
    )


def load_trajectory_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a trajectory YAML; default is the lab NP2.0 initial insertion."""
    cfg_path = resolve_trajectory_config(path)
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Trajectory config must be a mapping: {cfg_path}")
    config = {section: dict(raw.get(section) or {}) for section in REQUIRED_SECTIONS}
    for key, value in raw.items():
        if key not in config:
            config[key] = value
    config["_config_path"] = str(cfg_path.resolve())
    config["_trajectory_name"] = cfg_path.stem
    validate_trajectory_config(config)
    return config


def validate_trajectory_config(config: dict[str, Any]) -> None:
    """Validate required fields; emit warnings for uncertain DV/angles."""
    warnings_out: list[str] = []
    probe = config.get("probe") or {}
    insertion = config.get("insertion") or {}
    simulation = config.get("simulation") or {}

    if not probe.get("type"):
        raise ValueError("trajectory config: probe.type is required")

    ap = insertion.get("ap_mm_from_bregma")
    ml = insertion.get("ml_mm_from_bregma")
    dv = insertion.get("dv_mm_from_brain_surface")
    missing = [
        name
        for name, val in (
            ("ap_mm_from_bregma", ap),
            ("ml_mm_from_bregma", ml),
            ("dv_mm_from_brain_surface", dv),
        )
        if val is None
    ]
    if missing:
        raise ValueError(
            "trajectory config: insertion requires AP, ML, and DV "
            f"(missing: {', '.join(missing)})"
        )

    if insertion.get("bregma_to_lambda_mm") is None:
        raise ValueError(
            "trajectory config: insertion.bregma_to_lambda_mm is required "
            "for bregma-relative coordinates"
        )

    if insertion.get("dv_uncertain"):
        msg = (
            "DV depth is marked uncertain "
            f"(dv_mm_from_brain_surface={dv}); confirm before claiming CCF registration."
        )
        warnings_out.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if insertion.get("angle_convention_uncertain"):
        msg = (
            "Angle convention is marked uncertain "
            f"(horizontal={insertion.get('horizontal_angle_deg')}, "
            f"vertical={insertion.get('vertical_angle_deg')}); "
            "confirm Neuropixels Trajectory Explorer convention."
        )
        warnings_out.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if probe.get("confirm_from_channel_map"):
        msg = (
            f"probe.site_pitch_um={probe.get('site_pitch_um')} should be confirmed "
            "from the actual channel map / Open Ephys recording settings."
        )
        warnings_out.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    export = insertion.get("trajectory_export_file")
    anatomy = simulation.get("anatomy_regions_file")
    fallback = bool(simulation.get("fallback_to_schematic_hippocampus", False))
    export_ok = bool(export) and Path(str(export)).exists()
    anatomy_ok = bool(anatomy) and Path(str(anatomy)).exists()
    if not export_ok and not anatomy_ok and not fallback:
        raise ValueError(
            "trajectory config: neither trajectory_export_file nor "
            "anatomy_regions_file exists, and fallback_to_schematic_hippocampus "
            "is false"
        )

    config.setdefault("_validation_warnings", [])
    config["_validation_warnings"] = list(
        dict.fromkeys(list(config.get("_validation_warnings") or []) + warnings_out)
    )


def resolve_anatomy_regions_file(config: dict[str, Any]) -> Path | None:
    """Prefer Trajectory Explorer export over approximate region table."""
    insertion = config.get("insertion") or {}
    simulation = config.get("simulation") or {}
    export = insertion.get("trajectory_export_file")
    if export:
        path = Path(str(export))
        if path.exists():
            return path
    anatomy = simulation.get("anatomy_regions_file")
    if anatomy:
        path = Path(str(anatomy))
        if path.exists():
            return path
    return None


def resolve_cell_capture_file(config: dict[str, Any]) -> Path | None:
    simulation = config.get("simulation") or {}
    path = simulation.get("cell_capture_file")
    if not path:
        return None
    p = Path(str(path))
    return p if p.exists() else None


def prefer_trajectory_export(config: dict[str, Any]) -> bool:
    """True when a usable NTE export should override the approximate CSV."""
    insertion = config.get("insertion") or {}
    export = insertion.get("trajectory_export_file")
    return bool(export) and Path(str(export)).exists()


def build_trajectory_metadata(
    config: dict[str, Any],
    *,
    anatomy_source: str,
    anatomy_regions_used: str | None,
    cell_capture_used: str | None,
    visual_cortex_excluded: bool,
    schematic_fallback_used: bool,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble ``trajectory_metadata.json`` payload."""
    insertion = config.get("insertion") or {}
    probe = config.get("probe") or {}
    animal = config.get("animal") or {}
    decoder = config.get("decoder") or {}
    warns = list(config.get("_validation_warnings") or [])
    if extra_warnings:
        warns.extend(extra_warnings)
    source_path = config.get("_config_path")
    return {
        "active_trajectory_name": config.get("_trajectory_name")
        or (Path(str(source_path)).stem if source_path else None),
        "trajectory_config_path": source_path,
        "animal": animal,
        "probe_type": probe.get("type"),
        "site_pitch_um": probe.get("site_pitch_um"),
        "confirm_from_channel_map": probe.get("confirm_from_channel_map"),
        "shank_count": probe.get("shank_count"),
        "selected_shank": probe.get("selected_shank"),
        "ap_mm_from_bregma": insertion.get("ap_mm_from_bregma"),
        "ml_mm_from_bregma": insertion.get("ml_mm_from_bregma"),
        "dv_mm_from_brain_surface": insertion.get("dv_mm_from_brain_surface"),
        "dv_uncertain": bool(insertion.get("dv_uncertain")),
        "horizontal_angle_deg": insertion.get("horizontal_angle_deg"),
        "vertical_angle_deg": insertion.get("vertical_angle_deg"),
        "angle_convention_uncertain": bool(insertion.get("angle_convention_uncertain")),
        "bregma_to_lambda_mm": insertion.get("bregma_to_lambda_mm"),
        "coordinate_source": insertion.get("coordinate_source"),
        "coordinate_status": animal.get("coordinate_status"),
        "anatomy_source": anatomy_source,
        "anatomy_regions_file": anatomy_regions_used,
        "cell_capture_file": cell_capture_used,
        "visual_cortex_excluded": visual_cortex_excluded,
        "schematic_fallback_used": schematic_fallback_used,
        "deployment_spike_source": decoder.get("deployment_spike_source", "sorted"),
        "use_ground_truth_for_model_selection": bool(
            decoder.get("use_ground_truth_for_model_selection", False)
        ),
        "warnings": warns,
        "notes": insertion.get("notes"),
    }


def write_trial_trajectory_bundle(
    output_dir: str | Path,
    traj_cfg: dict[str, Any],
    anatomy_df: Any,
    *,
    cell_capture_src: str | Path | None = None,
    anatomy_source: str | None = None,
) -> dict[str, Any]:
    """Snapshot the active NPX coordinates into ``<trial>/trajectory/``.

    Makes each trial self-contained: active YAML, region table, optional cell
    capture copy, and ``active.json`` summarizing which lab coords were used.
    """
    from hippo.anatomy.trajectory_import import write_anatomy_regions_csv

    out = Path(output_dir)
    traj_dir = out / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)

    source_path = traj_cfg.get("_config_path")
    name = traj_cfg.get("_trajectory_name") or (
        Path(str(source_path)).stem if source_path else "trajectory"
    )

    regions_path = traj_dir / "anatomy_regions_used.csv"
    write_anatomy_regions_csv(anatomy_df, regions_path)
    write_anatomy_regions_csv(anatomy_df, out / "anatomy_regions.csv")

    capture_dest: Path | None = None
    if cell_capture_src is not None and Path(cell_capture_src).exists():
        capture_dest = traj_dir / "cell_capture.yaml"
        shutil.copy2(Path(cell_capture_src), capture_dest)

    active: dict[str, Any] = {}
    for key in REQUIRED_SECTIONS:
        section = traj_cfg.get(key)
        active[key] = dict(section) if isinstance(section, dict) else (section or {})
    for key, value in traj_cfg.items():
        if key.startswith("_") or key in active:
            continue
        active[key] = value

    active.setdefault("simulation", {})
    active["simulation"]["anatomy_regions_file"] = str(regions_path.resolve())
    if capture_dest is not None:
        active["simulation"]["cell_capture_file"] = str(capture_dest.resolve())

    active_yaml = traj_dir / "active_trajectory.yaml"
    with open(active_yaml, "w") as f:
        yaml.safe_dump(active, f, sort_keys=False, default_flow_style=False)

    insertion = active.get("insertion") or {}
    probe = active.get("probe") or {}
    active_meta = {
        "active_trajectory_name": name,
        "active_trajectory_yaml": str(active_yaml.resolve()),
        "source_config_path": str(source_path) if source_path else None,
        "anatomy_regions_used": str(regions_path.resolve()),
        "cell_capture_file": str(capture_dest.resolve()) if capture_dest else None,
        "anatomy_source": anatomy_source,
        "probe_type": probe.get("type"),
        "ap_mm_from_bregma": insertion.get("ap_mm_from_bregma"),
        "ml_mm_from_bregma": insertion.get("ml_mm_from_bregma"),
        "dv_mm_from_brain_surface": insertion.get("dv_mm_from_brain_surface"),
        "horizontal_angle_deg": insertion.get("horizontal_angle_deg"),
        "vertical_angle_deg": insertion.get("vertical_angle_deg"),
        "bregma_to_lambda_mm": insertion.get("bregma_to_lambda_mm"),
    }
    (traj_dir / "active.json").write_text(
        json.dumps(active_meta, indent=2, default=str) + "\n"
    )
    return active_meta
