"""Validate README output contracts for the public workflow."""

from __future__ import annotations

from pathlib import Path

SIMULATION_REQUIRED = (
    "behavior.csv",
    "units.csv",
    "spikes_ground_truth.csv",
    "spikes_sorted.csv",
    "summary.json",
    "rates.npy",
    "anatomy_regions.csv",
    "ratinabox_group_metadata.csv",
)

DECODE_REQUIRED = (
    "decoder_comparison/sorted/decoder_comparison_metrics.csv",
    "models/best_realtime_decoders.json",
    "deployment_decoder_selection/all_sorted_window_scores.csv",
    "deployment_decoder_selection/best_decoder_by_target_sorted.csv",
)

VIZ_REQUIRED_DIRS = (
    "figures/decoder_comparison",
    "figures/realtime_decoding",
    "figures/deployment_decoder_selection",
    "figures/latency",
)


class OutputContractError(FileNotFoundError):
    """Raised when a public-workflow output contract is violated."""


def _missing(root: Path, rels: tuple[str, ...]) -> list[str]:
    return [r for r in rels if not (root / r).exists()]


def validate_simulation_outputs(experiment_dir: Path) -> list[str]:
    """Return missing simulation artifacts (empty if OK)."""
    return _missing(Path(experiment_dir), SIMULATION_REQUIRED)


def validate_decode_outputs(experiment_dir: Path) -> list[str]:
    """Return missing decode/deployment artifacts (empty if OK)."""
    root = Path(experiment_dir)
    missing = _missing(root, DECODE_REQUIRED)
    rt = root / "realtime_decoding" / "sorted"
    if not rt.exists() or not any(rt.rglob("decoded_realtime.csv")):
        missing.append("realtime_decoding/sorted/**/decoded_realtime.csv")
    return missing


def validate_visualization_outputs(experiment_dir: Path) -> list[str]:
    """Return missing visualization artifacts (empty if OK)."""
    root = Path(experiment_dir)
    missing: list[str] = []
    if not (root / "figures" / "output.pdf").exists():
        missing.append("figures/output.pdf")
    for d in VIZ_REQUIRED_DIRS:
        path = root / d
        if not path.exists() or not any(path.glob("*.png")):
            missing.append(f"{d}/*.png")
    return missing


def assert_simulation_outputs(experiment_dir: Path) -> None:
    missing = validate_simulation_outputs(experiment_dir)
    if missing:
        raise OutputContractError(
            f"Simulation outputs incomplete under {experiment_dir}: missing {missing}"
        )


def assert_decode_outputs(experiment_dir: Path) -> None:
    missing = validate_decode_outputs(experiment_dir)
    if missing:
        raise OutputContractError(
            f"Decode/deployment outputs incomplete under {experiment_dir}: missing {missing}"
        )


def assert_visualization_outputs(experiment_dir: Path) -> None:
    missing = validate_visualization_outputs(experiment_dir)
    if missing:
        raise OutputContractError(
            f"Visualization outputs incomplete under {experiment_dir}: missing {missing}"
        )
