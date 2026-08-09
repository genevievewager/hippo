"""Feature-set diagnostics wrappers (calls realtime extractors on demand)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.neural_features import (
    ALL_FEATURE_SETS,
    FEATURE_SET_DEFINITIONS,
    NeuralFeatureExtractor,
)
from realtime.timing import extract_behavior_times, resolve_update_dt_s


@dataclass
class FeatureDiagnostics:
    feature_set: str
    families: tuple[str, ...]
    n_features: int
    n_samples: int
    feature_names: list[str]
    variance: np.ndarray
    example_traces: pd.DataFrame
    correlation_subset: pd.DataFrame | None
    region_blocks: dict[str, Any]


def list_feature_sets() -> tuple[str, ...]:
    return tuple(ALL_FEATURE_SETS)


def feature_set_families(name: str) -> tuple[str, ...]:
    return FEATURE_SET_DEFINITIONS.get(name, ())


def compute_feature_diagnostics(
    input_dir,
    feature_set: str,
    *,
    spike_source: str = "sorted",
    decode_window: float = 0.250,
    max_features_for_corr: int = 40,
    max_trace_features: int = 8,
    max_samples: int = 400,
) -> FeatureDiagnostics:
    """Extract one feature set and return compact diagnostics.

    Intentionally capped for UI responsiveness; does not write experiment
    outputs.
    """
    if feature_set not in ALL_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    data = load_simulation_data(input_dir, spike_source)
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"],
        derive_from_behavior=True,
        update_dt_s=0.050,
        behavior_times=behavior_times,
    )
    decode_times = make_decode_times(
        data["session_duration"],
        decode_window,
        update_dt,
        behavior_times=behavior_times,
    )
    if len(decode_times) > max_samples:
        idx = np.linspace(0, len(decode_times) - 1, max_samples).astype(int)
        decode_times = decode_times[idx]

    extractor = NeuralFeatureExtractor.from_feature_set(
        feature_set,
        units_df=data["units_df"],
        unit_ids=data["unit_ids"],
        decode_window=decode_window,
        update_dt=update_dt,
    )
    result = extractor.extract_matrix(data["spikes_df"], decode_times)
    X = np.asarray(result.feature_vector, dtype=float)
    names = list(result.feature_names)

    var = np.nanvar(X, axis=0) if X.size else np.asarray([])
    n_trace = min(max_trace_features, X.shape[1]) if X.ndim == 2 and X.shape[1] else 0
    top_idx = np.argsort(var)[::-1][:n_trace] if n_trace else []
    traces = pd.DataFrame(
        {names[i] if i < len(names) else f"f{i}": X[:, i] for i in top_idx}
    )
    traces.insert(0, "time_s", decode_times)

    corr_df = None
    n_corr = min(max_features_for_corr, X.shape[1]) if X.ndim == 2 else 0
    if n_corr >= 2:
        sub = X[:, :n_corr]
        corr = np.corrcoef(sub, rowvar=False)
        corr_names = names[:n_corr]
        corr_df = pd.DataFrame(corr, index=corr_names, columns=corr_names)

    region_blocks: dict[str, Any] = {
        "n_units": len(data["unit_ids"]),
        "families": list(feature_set_families(feature_set)),
        "feature_dimension": int(getattr(result, "n_features", X.shape[1])),
    }

    return FeatureDiagnostics(
        feature_set=feature_set,
        families=feature_set_families(feature_set),
        n_features=int(X.shape[1]) if X.ndim == 2 else 0,
        n_samples=int(X.shape[0]) if X.ndim == 2 else 0,
        feature_names=names,
        variance=var,
        example_traces=traces,
        correlation_subset=corr_df,
        region_blocks=region_blocks,
    )
