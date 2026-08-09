"""Realtime replay helpers for the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from run_realtime_decoding import RealtimeReplayConfig, run_realtime_decoding


@dataclass
class ReplayArtifacts:
    root: Path
    decoded: pd.DataFrame
    metrics: dict[str, Any]
    selected_config: dict[str, Any] | None = None
    closed_loop: pd.DataFrame | None = None


def find_realtime_runs(experiment_dir: Path) -> list[Path]:
    """Directories that contain ``decoded_realtime.csv``."""
    root = Path(experiment_dir) / "realtime_decoding"
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("decoded_realtime.csv")})


def load_replay_artifacts(run_dir: Path) -> ReplayArtifacts:
    run_dir = Path(run_dir)
    decoded_path = run_dir / "decoded_realtime.csv"
    if not decoded_path.exists():
        matches = list(run_dir.rglob("decoded_realtime.csv"))
        if not matches:
            raise FileNotFoundError(f"No decoded_realtime.csv under {run_dir}")
        decoded_path = matches[0]
        run_dir = decoded_path.parent

    decoded = pd.read_csv(decoded_path)
    metrics: dict[str, Any] = {}
    metrics_path = run_dir / "realtime_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    selected = None
    cfg_path = run_dir / "selected_realtime_decoder_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            selected = json.load(f)
    closed = None
    cl_path = run_dir / "closed_loop_events.csv"
    if cl_path.exists():
        closed = pd.read_csv(cl_path)

    return ReplayArtifacts(
        root=run_dir,
        decoded=decoded,
        metrics=metrics,
        selected_config=selected,
        closed_loop=closed,
    )


def default_comparison_dir(experiment_dir: Path) -> Path | None:
    cand = Path(experiment_dir) / "decoder_comparison" / "sorted"
    if (cand / "decoder_comparison_metrics.csv").exists():
        return cand
    cand2 = Path(experiment_dir) / "decoder_comparison"
    if (cand2 / "decoder_comparison_metrics.csv").exists():
        return cand2
    return None


def build_replay_config(
    *,
    input_dir: Path,
    output_dir: Path,
    spike_source: str = "sorted",
    update_dt: float = 0.025,
    decode_window: float = 0.250,
    use_best_decoder: bool = True,
    closed_loop_target: str = "position",
    comparison_dir: Path | None = None,
    feature_type: str = "counts",
    manifold_n_components: int = 3,
) -> RealtimeReplayConfig:
    return RealtimeReplayConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        spike_source=spike_source,
        update_dt=float(update_dt),
        decode_window=float(decode_window),
        use_best_decoder=use_best_decoder,
        comparison_dir=comparison_dir or default_comparison_dir(input_dir),
        closed_loop_target=closed_loop_target,
        feature_type=feature_type,
        manifold_n_components=int(manifold_n_components),
    )


def execute_replay(config: RealtimeReplayConfig):
    """Call the shared realtime decoding backend."""
    return run_realtime_decoding(config)
