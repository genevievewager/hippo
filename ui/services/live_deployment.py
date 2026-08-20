"""Service helpers for the Live Deployment UI tab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realtime.decoder_comparison import ALL_TARGETS, PRIMARY_METRIC
from realtime.deployment_bundle import (
    best_deployable,
    load_deployment_bundle,
    pack_deployment_bundle,
)
from realtime.deployment_selection import DEPLOYMENT_SPIKE_SOURCE
from realtime.live.config import DeployableConfiguration
from realtime.live.registry import DeploymentRegistry
from realtime.live.session_logger import SessionLogger
from realtime.live.spike_stream import OpenEphysSpikeStream, ReplaySpikeStream, SpikeStream
from realtime.live_decoder import DeploymentMode, LiveDecoder


def available_targets(experiment_dir: Path | None = None) -> list[str]:
    """Behavioral targets known to the project (prefer registry keys when present)."""
    if experiment_dir is not None:
        try:
            from realtime.deployment_selection import load_best_realtime_decoders

            payload = load_best_realtime_decoders(experiment_dir)
            keys = sorted((payload.get("targets") or {}).keys())
            if keys:
                return keys
        except Exception:  # noqa: BLE001
            pass
    return list(ALL_TARGETS)


def primary_metric_info(target: str) -> tuple[str | None, str | None]:
    name, direction = PRIMARY_METRIC.get(target, (None, None))
    return name, direction


def select_best_config(
    experiment_dir: Path,
    target: str,
    *,
    selection_policy: str = "shortest_near_optimal",
) -> DeployableConfiguration:
    return DeploymentRegistry(experiment_dir).best(
        target,
        spike_source=DEPLOYMENT_SPIKE_SOURCE,
        deployable_only=True,
        selection_policy=selection_policy,
    )


def ensure_bundle(
    experiment_dir: Path,
    target: str,
    *,
    force_rebuild: bool = False,
    selection_policy: str = "shortest_near_optimal",
) -> Path:
    """Return path to a packed bundle, building it if needed."""
    cfg = select_best_config(experiment_dir, target, selection_policy=selection_policy)
    out = (
        experiment_dir
        / "deployment_bundles"
        / f"{target}__{cfg.D}__w{int(round(cfg.W * 1000)):04d}ms"
    )
    if out.exists() and not force_rebuild and (out / "decoder.joblib").exists():
        return out
    return pack_deployment_bundle(
        experiment_dir, target, output_dir=out, selection_policy=selection_policy,
    )


def make_stream(
    source: str,
    experiment_dir: Path,
    *,
    endpoint: str | None = None,
) -> SpikeStream:
    if source == "replay":
        return ReplaySpikeStream(experiment_dir=experiment_dir, spike_source="sorted")
    if source in ("open_ephys", "live", "live_open_ephys"):
        return OpenEphysSpikeStream(endpoint=endpoint)
    raise ValueError(f"Unknown input source {source!r}")


def build_live_decoder_from_experiment(
    experiment_dir: Path,
    target: str,
    *,
    force_rebuild: bool = False,
) -> tuple[LiveDecoder, Path]:
    bundle_path = ensure_bundle(experiment_dir, target, force_rebuild=force_rebuild)
    return LiveDecoder.from_bundle(bundle_path), bundle_path


def create_session_logger(experiment_dir: Path) -> SessionLogger:
    return SessionLogger.create(experiment_dir, prefix="session")


def history_to_frame(decoder: LiveDecoder):
    import pandas as pd

    rows = []
    for rec in decoder.history:
        row = {
            "time": rec.timestamp,
            "inference_latency_ms": rec.inference_latency_ms,
            "loop_latency_ms": rec.loop_latency_ms,
            "overrun": rec.overrun,
            "n_spikes_in_window": rec.n_spikes_in_window,
        }
        pred = rec.prediction
        if isinstance(pred, dict):
            row.update(pred)
            if "x" in pred and "y" in pred:
                row["decoded_x"] = pred["x"]
                row["decoded_y"] = pred["y"]
        elif decoder.target == "speed":
            row["decoded_speed"] = pred
        elif decoder.target == "acceleration":
            row["decoded_acceleration"] = pred
        elif decoder.target == "head_direction":
            row["decoded_head_direction_deg"] = pred
        else:
            row["prediction"] = pred
        # Flatten common flags
        for k, v in rec.flags.items():
            if k.startswith("decoded_") or k in ("confidence", "predicted_class"):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


PIPELINE_TEST_BANNER = (
    "**PIPELINE TEST MODE**\n\n"
    "This model was trained on simulated/synthetic neural data. "
    "Live predictions are intended to validate acquisition and inference "
    "infrastructure and should not be interpreted as validated behavioral decoding."
)


def mode_banner(decoder: LiveDecoder) -> str | None:
    if decoder.mode == DeploymentMode.PIPELINE_TEST or decoder.is_simulation_trained:
        return PIPELINE_TEST_BANNER
    return None


def config_display_dict(cfg: DeployableConfiguration) -> dict[str, Any]:
    return {
        "F": cfg.F,
        "E": cfg.E,
        "D": cfg.D,
        "W_s": cfg.W,
        "C": cfg.C,
        "metric": cfg.metric_name,
        "metric_value": cfg.metric_value,
        "direction": cfg.metric_direction,
        "deployable": cfg.deployable,
        "spike_source": cfg.spike_source,
    }


def list_existing_bundles(experiment_dir: Path) -> list[Path]:
    root = Path(experiment_dir) / "deployment_bundles"
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "decoder.joblib").exists()]
    )


def load_bundle_summary(bundle_dir: Path) -> dict[str, Any]:
    cfg = json.loads((bundle_dir / "config.json").read_text())
    meta = json.loads((bundle_dir / "metadata.json").read_text())
    return {"path": str(bundle_dir), "config": cfg, "metadata": meta}
