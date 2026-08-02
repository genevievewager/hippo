"""Probe trajectory import and cell-type capture for hippocampal simulation.

Converts Neuropixels Trajectory Explorer (or manually defined) probe paths
into internal anatomy tables and captured unit populations.
"""

from hippo.anatomy.cell_capture import (
    apply_cell_capture_to_ratinabox_params,
    load_cell_capture_config,
    sample_units_from_regions,
)
from hippo.anatomy.hippocampal_system import (
    ALLOWED_CELL_TYPES,
    annotate_units_for_analysis,
    canonicalize_region,
    filter_units_for_analysis,
    geometry_summary,
    infer_circuit_profile,
    recommended_partitions,
)
from hippo.anatomy.trajectory_config import (
    DEFAULT_TRAJECTORY_CONFIG,
    list_trajectory_configs,
    load_trajectory_config,
    resolve_anatomy_regions_file,
    resolve_cell_capture_file,
    resolve_trajectory_config,
    validate_trajectory_config,
    write_trial_trajectory_bundle,
)
from hippo.anatomy.trajectory_import import (
    ANATOMY_REGION_COLUMNS,
    anatomy_table_to_region_segments,
    assign_channels_from_depth,
    import_trajectory,
    load_lab_anatomy_regions_csv,
    load_probe_trajectory_config,
    schematic_anatomy_table,
    write_anatomy_regions_csv,
)

__all__ = [
    "ALLOWED_CELL_TYPES",
    "ANATOMY_REGION_COLUMNS",
    "DEFAULT_TRAJECTORY_CONFIG",
    "anatomy_table_to_region_segments",
    "annotate_units_for_analysis",
    "apply_cell_capture_to_ratinabox_params",
    "assign_channels_from_depth",
    "canonicalize_region",
    "filter_units_for_analysis",
    "geometry_summary",
    "import_trajectory",
    "infer_circuit_profile",
    "list_trajectory_configs",
    "load_cell_capture_config",
    "load_lab_anatomy_regions_csv",
    "load_probe_trajectory_config",
    "load_trajectory_config",
    "recommended_partitions",
    "resolve_anatomy_regions_file",
    "resolve_cell_capture_file",
    "resolve_trajectory_config",
    "sample_units_from_regions",
    "schematic_anatomy_table",
    "validate_trajectory_config",
    "write_anatomy_regions_csv",
    "write_trial_trajectory_bundle",
]
