"""Feature-set diagnostics wrappers (calls realtime extractors on demand)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.neural_features import (
    ALL_FEATURE_SETS,
    FEATURE_SET_DEFINITIONS,
    NeuralFeatureExtractor,
)
from realtime.neural_features.comparison import effective_spike_feature_type
from realtime.timing import extract_behavior_times, resolve_update_dt_s
from realtime.train_decoder import causal_train_test_split


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
    extras: dict[str, Any] | None = None


def list_feature_sets() -> tuple[str, ...]:
    return tuple(ALL_FEATURE_SETS)


def feature_set_families(name: str) -> tuple[str, ...]:
    return FEATURE_SET_DEFINITIONS.get(name, ())


def checkpoint_feature_transform(
    input_dir,
    feature_set: str,
    *,
    spike_source: str = "sorted",
    decode_window: float = 0.250,
    feature_type: str = "counts",
    train_frac: float = 0.70,
    force: bool = False,
) -> dict[str, Any]:
    """Fit SpikeFeatureTransformer F on train split and save to shared cache.

    Matches Decoder Benchmark's F key: ``{feature_set}__{f_eff}_w####ms``.
    """
    from realtime.transform_cache import (
        comparison_roots_for_feature_cache,
        ensure_comparison_root,
        find_feature_transform_in_roots,
        save_feature_transform_checkpoint,
    )

    if feature_set not in ALL_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    write_root = ensure_comparison_root(Path(input_dir), spike_source=spike_source)
    f_eff = effective_spike_feature_type(feature_set, feature_type)
    if not force:
        roots = comparison_roots_for_feature_cache(Path(input_dir), spike_source)
        search_roots: list[Path] = []
        seen: set[Path] = set()
        for root in (write_root, *roots):
            key = root.resolve() if root.exists() else root
            if key in seen:
                continue
            seen.add(key)
            search_roots.append(root)
        existing = find_feature_transform_in_roots(
            search_roots,
            feature_set=feature_set,
            feature_type_eff=f_eff,
            decode_window=float(decode_window),
        )
        if existing is not None:
            return {
                "feature_set": feature_set,
                "feature_type_eff": f_eff,
                "decode_window_s": float(decode_window),
                "saved_path": str(existing),
                "from_cache": True,
                "persisted": False,
            }

    from realtime.feature_representations import make_spike_feature_transformer

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
    extractor = NeuralFeatureExtractor.from_feature_set(
        feature_set,
        units_df=data["units_df"],
        unit_ids=data["unit_ids"],
        decode_window=decode_window,
        update_dt=update_dt,
    )
    extracted = extractor.extract_matrix(data["spikes_df"], decode_times)
    X = np.asarray(extracted.feature_vector, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise RuntimeError(f"Empty feature matrix for {feature_set!r}")

    train_mask, _ = causal_train_test_split(decode_times, train_frac)
    if int(train_mask.sum()) < 2:
        train_mask = np.ones(len(decode_times), dtype=bool)

    f_transform = make_spike_feature_transformer(
        f_eff,
        decode_window=decode_window,
        units_df=data["units_df"],
        unit_ids=data["unit_ids"],
    )
    f_transform.fit(X[train_mask])
    saved = save_feature_transform_checkpoint(
        f_transform,
        write_root,
        feature_set=feature_set,
        feature_type_eff=f_eff,
        decode_window=float(decode_window),
        extra_meta={
            "fit_scope": "train_split",
            "train_frac": float(train_frac),
            "source": "feature_explorer",
            "spike_source": spike_source,
            "n_train_samples": int(train_mask.sum()),
            "n_features_in": int(X.shape[1]),
        },
    )
    # Also persist the neural extractor (Benchmark always saves these).
    try:
        from realtime.transform_cache import comparison_model_roots, neural_extractor_dirname

        neural_base = comparison_model_roots(write_root)["neural"]
        neural_base.mkdir(parents=True, exist_ok=True)
        extractor.save(neural_base / neural_extractor_dirname(feature_set, decode_window))
    except Exception:
        pass

    return {
        "feature_set": feature_set,
        "feature_type_eff": f_eff,
        "decode_window_s": float(decode_window),
        "saved_path": str(saved),
        "from_cache": False,
        "persisted": True,
        "comparison_root": str(write_root),
    }


def compute_feature_diagnostics(
    input_dir,
    feature_set: str,
    *,
    spike_source: str = "sorted",
    decode_window: float = 0.250,
    max_features_for_corr: int = 40,
    max_trace_features: int = 8,
    max_samples: int = 400,
    persist: bool = False,
    train_frac: float = 0.70,
) -> FeatureDiagnostics:
    """Extract one feature set and return compact diagnostics.

    Intentionally capped for UI responsiveness. When ``persist`` is True, also
    checkpoint SpikeFeatureTransformer F into the shared decoder_comparison store.
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

    extras: dict[str, Any] = {}
    if persist and X.ndim == 2 and X.shape[0] > 0:
        try:
            extras["checkpoint"] = checkpoint_feature_transform(
                input_dir,
                feature_set,
                spike_source=spike_source,
                decode_window=decode_window,
                train_frac=train_frac,
            )
        except Exception as exc:  # noqa: BLE001
            extras["checkpoint_error"] = str(exc)

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
        extras=extras or None,
    )
