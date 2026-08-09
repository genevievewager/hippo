"""Dataset discovery and metadata inspection (no Streamlit imports)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from realtime.output_contracts import validate_simulation_outputs


@dataclass
class DatasetInfo:
    """Lightweight descriptor for a simulation / experiment folder."""

    name: str
    path: Path
    summary: dict[str, Any] = field(default_factory=dict)
    has_sorted_spikes: bool = False
    has_ground_truth_spikes: bool = False
    has_decoder_comparison: bool = False
    has_realtime: bool = False
    validation_errors: list[str] = field(default_factory=list)


def default_outputs_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / "outputs"


def list_datasets(outputs_root: Path | None = None) -> list[DatasetInfo]:
    """Scan ``outputs/`` for directories that look like experiment runs."""
    root = Path(outputs_root) if outputs_root else default_outputs_root()
    if not root.exists():
        return []
    datasets: list[DatasetInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "summary.json").exists() and not (child / "behavior.csv").exists():
            continue
        datasets.append(inspect_dataset(child))
    return datasets


def inspect_dataset(path: Path) -> DatasetInfo:
    """Load summary / availability flags for one experiment directory."""
    path = Path(path)
    summary: dict[str, Any] = {}
    errors: list[str] = []
    summary_path = path / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Failed to parse summary.json: {exc}")
    else:
        errors.append("summary.json missing")

    missing = validate_simulation_outputs(path)
    if missing:
        errors.append(f"Missing simulation artifacts: {', '.join(missing)}")

    return DatasetInfo(
        name=path.name,
        path=path,
        summary=summary,
        has_sorted_spikes=(path / "spikes_sorted.csv").exists(),
        has_ground_truth_spikes=(path / "spikes_ground_truth.csv").exists(),
        has_decoder_comparison=_has_comparison(path),
        has_realtime=_has_realtime(path),
        validation_errors=errors,
    )


def _has_comparison(path: Path) -> bool:
    candidates = [
        path / "decoder_comparison" / "sorted" / "decoder_comparison_metrics.csv",
        path / "decoder_comparison" / "decoder_comparison_metrics.csv",
        path / "decoder_comparison_metrics.csv",
    ]
    return any(p.exists() for p in candidates)


def _has_realtime(path: Path) -> bool:
    rt = path / "realtime_decoding"
    if not rt.exists():
        return False
    return any(rt.rglob("decoded_realtime.csv"))


def load_summary(path: Path) -> dict[str, Any]:
    with open(Path(path) / "summary.json") as f:
        return json.load(f)


def load_units_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(Path(path) / "units.csv")


def load_behavior_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(Path(path) / "behavior.csv")


def region_unit_counts(units_df: pd.DataFrame) -> pd.DataFrame:
    """Return unit counts by anatomical region when columns exist."""
    region_col = None
    for candidate in ("region", "subfield", "anatomy_region", "ratinabox_group"):
        if candidate in units_df.columns:
            region_col = candidate
            break
    if region_col is None:
        return pd.DataFrame({"region": ["unknown"], "n_units": [len(units_df)]})
    counts = (
        units_df.groupby(region_col, dropna=False)
        .size()
        .reset_index(name="n_units")
        .rename(columns={region_col: "region"})
    )
    return counts.sort_values("n_units", ascending=False)


def cell_type_counts(units_df: pd.DataFrame) -> pd.DataFrame:
    for candidate in ("cell_type", "ratinabox_cell_type", "cell_class"):
        if candidate in units_df.columns:
            return (
                units_df.groupby(candidate, dropna=False)
                .size()
                .reset_index(name="n_units")
                .rename(columns={candidate: "cell_type"})
                .sort_values("n_units", ascending=False)
            )
    return pd.DataFrame()


def sorting_degradation_summary(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    """Collect sorting / recording degradation diagnostics if present."""
    out: dict[str, Any] = {}
    for key in (
        "n_ground_truth_spikes",
        "n_sorted_spikes",
        "n_units",
        "n_units_include_in_decoder",
        "deployment_spike_source",
        "notes",
    ):
        if key in summary:
            out[key] = summary[key]
    gt_n = summary.get("n_ground_truth_spikes")
    sorted_n = summary.get("n_sorted_spikes")
    if isinstance(gt_n, (int, float)) and isinstance(sorted_n, (int, float)) and gt_n > 0:
        out["spike_recovery_fraction"] = float(sorted_n) / float(gt_n)
        out["spike_loss_fraction"] = 1.0 - out["spike_recovery_fraction"]
    meta = path / "neural_backend_metadata.json"
    if meta.exists():
        try:
            with open(meta) as f:
                out["neural_backend_metadata"] = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return out


def available_spike_sources(path: Path) -> list[str]:
    sources: list[str] = []
    if (Path(path) / "spikes_sorted.csv").exists():
        sources.append("sorted")
    if (Path(path) / "spikes_ground_truth.csv").exists():
        sources.append("ground_truth")
    return sources
