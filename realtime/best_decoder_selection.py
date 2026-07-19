"""Select best decoder/window from decoder comparison outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from realtime.decoder_comparison import windowed_model_path
from realtime.train_decoder import TrainedDecoders

# Core suite used by RealTimeDecoder for multi-target decoded traces.
SUITE_TARGETS = ("position", "spatial_context", "movement_state", "speed")


def resolve_best_decoder_table(
    comparison_dir: Path,
    spike_source: str,
) -> tuple[pd.DataFrame, Path]:
    """Load best_decoder_by_target.csv for a spike source."""
    comparison_dir = Path(comparison_dir)
    candidates = [
        comparison_dir / spike_source / "best_decoder_by_target.csv",
        comparison_dir / "best_decoder_by_target.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "spike_source" in df.columns:
                filtered = df[df["spike_source"] == spike_source]
                if not filtered.empty:
                    return filtered.reset_index(drop=True), path
            return df.reset_index(drop=True), path
    raise FileNotFoundError(
        f"Could not find best_decoder_by_target.csv under {comparison_dir} "
        f"(looked for {spike_source}/best_decoder_by_target.csv and best_decoder_by_target.csv)"
    )


def resolve_comparison_models_dir(comparison_dir: Path, spike_source: str) -> Path:
    """Return models/ directory for a spike source under comparison outputs."""
    comparison_dir = Path(comparison_dir)
    nested = comparison_dir / spike_source / "models"
    if nested.exists():
        return nested
    flat = comparison_dir / "models"
    if flat.exists():
        return flat
    raise FileNotFoundError(
        f"Could not find comparison models under {comparison_dir} "
        f"(looked for {spike_source}/models and models)"
    )


def select_best_decoder_row(
    comparison_dir: Path,
    spike_source: str,
    closed_loop_target: str,
    selection_policy: str = "shortest_near_optimal",
) -> tuple[dict[str, Any], Path]:
    """Return the selected best-decoder row and source file path."""
    df, path = resolve_best_decoder_table(comparison_dir, spike_source)
    target_df = df[df["target_name"] == closed_loop_target]
    if target_df.empty:
        available = sorted(df["target_name"].unique().tolist())
        raise ValueError(
            f"Target {closed_loop_target!r} not found in {path}. "
            f"Available targets: {available}"
        )
    row = target_df.iloc[0].to_dict()
    models_dir = resolve_comparison_models_dir(comparison_dir, spike_source)
    feature_type = str(row.get("best_feature_type", "counts"))

    if selection_policy == "best_accuracy":
        selected_window = float(row["best_decode_window_s"])
        selected_decoder = str(row["best_decoder_name"])
        model_path = row.get("best_window_model_path") or row.get("model_path") or ""
        decoder_config_raw = row.get("decoder_config_json", "{}")
        n_comp = row.get("best_manifold_n_components")
        # Best-row column matches the best-W model / transform.
        transform_path = row.get("manifold_transform_path")
    elif selection_policy == "shortest_near_optimal":
        selected_window = float(row["recommended_realtime_window_s"])
        selected_decoder = str(
            row.get("recommended_realtime_decoder_name", row["best_decoder_name"])
        )
        model_path = row.get("realtime_model_path") or ""
        decoder_config_raw = row.get(
            "realtime_decoder_config_json", row.get("decoder_config_json", "{}")
        )
        n_comp = row.get("best_manifold_n_components")
        # Do not use the top-level manifold_transform_path here: that column
        # stores the *best-W* transform. Prefer the path from the realtime
        # config (matched to recommended_realtime_window_s) below.
        transform_path = None
    else:
        raise ValueError(
            f"Unknown selection_policy {selection_policy!r}; "
            "use 'best_accuracy' or 'shortest_near_optimal'"
        )

    decoder_config: Any = decoder_config_raw
    if isinstance(decoder_config, str):
        try:
            decoder_config = json.loads(decoder_config)
        except json.JSONDecodeError:
            decoder_config = {}

    if n_comp is None and isinstance(decoder_config, dict):
        n_comp = decoder_config.get("manifold_n_components")
    if isinstance(decoder_config, dict):
        cfg_transform = decoder_config.get("manifold_transform_path")
        if cfg_transform:
            # Window-matched transform from the selected config wins.
            transform_path = cfg_transform
    if not transform_path:
        transform_path = row.get("manifold_transform_path")
    if isinstance(decoder_config, dict) and decoder_config.get("feature_type"):
        # Prefer feature type from the selected realtime config when present.
        if selection_policy == "shortest_near_optimal":
            feature_type = str(decoder_config.get("feature_type", feature_type))

    if not model_path:
        model_path = str(
            windowed_model_path(
                models_dir, closed_loop_target, selected_window, feature_type, n_comp,
            )
        )

    selected = {
        **row,
        "selection_policy": selection_policy,
        "selected_decoder_name": selected_decoder,
        "selected_decode_window_s": selected_window,
        "selected_feature_type": feature_type,
        "selected_manifold_n_components": n_comp,
        "selected_manifold_transform_path": transform_path,
        "selected_model_path": str(model_path),
        "comparison_models_dir": str(models_dir),
        "decoder_config": decoder_config,
        "from_file": str(path),
    }
    return selected, path


def _classifier_classes(pipeline: Any) -> list[str]:
    if hasattr(pipeline, "classes_"):
        return [str(c) for c in pipeline.classes_]
    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "classes_"):
                return [str(c) for c in step.classes_]
    return []


def load_windowed_model(
    models_dir: Path,
    target: str,
    decode_window: float,
    feature_type: str = "counts",
    n_components: Any = None,
) -> Any:
    """Load a comparison model trained at a specific window."""
    path = windowed_model_path(
        models_dir, target, decode_window, feature_type, n_components,
    )
    if not path.exists():
        # Backward-compatible path without k/ component nesting
        legacy = (
            Path(models_dir) / "by_window" / str(feature_type)
            / f"w{float(decode_window):.3f}s" / f"{target}.joblib"
        )
        if legacy.exists():
            return joblib.load(legacy)
        raise FileNotFoundError(
            f"Missing comparison model for target={target!r} "
            f"window={decode_window} feature={feature_type} k={n_components}: {path}"
        )
    return joblib.load(path)


def load_pretrained_suite(
    models_dir: Path,
    decode_window: float,
    feature_type: str = "counts",
    n_components: Any = None,
) -> TrainedDecoders:
    """Load position/context/movement/speed models trained at decode_window."""
    models_dir = Path(models_dir)
    loaded = {
        target: load_windowed_model(
            models_dir, target, decode_window, feature_type, n_components,
        )
        for target in SUITE_TARGETS
    }
    return TrainedDecoders(
        position=loaded["position"],
        spatial_context=loaded["spatial_context"],
        movement_state=loaded["movement_state"],
        speed=loaded["speed"],
        spatial_context_classes=_classifier_classes(loaded["spatial_context"]),
        movement_state_classes=_classifier_classes(loaded["movement_state"]),
    )


def copy_loaded_models_to_output(
    models_dir: Path,
    output_models_dir: Path,
    decode_window: float,
    feature_type: str,
    closed_loop_target: str,
    primary_model_path: Path | None = None,
    n_components: Any = None,
    manifold_transform_path: Path | None = None,
) -> None:
    """Copy reused comparison artifacts into the realtime run directory."""
    output_models_dir = Path(output_models_dir)
    output_models_dir.mkdir(parents=True, exist_ok=True)
    for target in SUITE_TARGETS:
        src = windowed_model_path(
            models_dir, target, decode_window, feature_type, n_components,
        )
        if src.exists():
            shutil.copy2(src, output_models_dir / f"{target}_decoder.joblib")
            meta = src.with_suffix(".json")
            if meta.exists():
                shutil.copy2(meta, output_models_dir / f"{target}_decoder.json")
    if primary_model_path is not None and Path(primary_model_path).exists():
        shutil.copy2(
            primary_model_path,
            output_models_dir / f"primary_{closed_loop_target}_decoder.joblib",
        )
    if manifold_transform_path is not None and Path(manifold_transform_path).exists():
        dest = output_models_dir / "feature_transformer"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(manifold_transform_path, dest)
