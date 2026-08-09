"""Causal NeuralFeatureExtractor: spike stream → compact population-state features.

Pipeline position
-----------------
    Spike events → NeuralFeatureExtractor → FeatureVector(t)
                 → scaling (train-fit) → manifold → decoder

All windows are half-open ``[t - W, t)`` matching ``realtime.spike_binner``.
No future spikes are used. History-dependent features use only previous
decoder updates or internal bins that still lie before ``t``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from realtime.neural_features.binning import (
    build_causal_unit_bin_tensor,
    counts_from_bin_tensor,
    n_coactivity_bins,
    regional_population_traces,
)
from realtime.neural_features.families import (
    append_count_dynamics,
    append_counts,
    append_cross_region_coactivity,
    append_lagged_coupling,
    append_population_statistics,
    append_regional_statistics,
    append_within_region_coactivity,
)
from realtime.neural_features.feature_sets import (
    ALL_FEATURE_SETS,
    COACTIVITY_FAMILIES,
    FEATURE_SET_DEFINITIONS,
    build_ablation_definitions,
    default_extractor_config,
    families_for_feature_set,
)
from realtime.neural_features.types import FeatureExtractionResult, FeatureSpec
from realtime.spike_binner import build_causal_spike_matrix, count_spikes_in_window


def _resolve_region_labels(
    units_df: pd.DataFrame | None,
    unit_ids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Infer per-unit region labels; fall back to ``unknown`` when missing."""
    labels = np.array(["unknown"] * len(unit_ids), dtype=object)
    if units_df is None or "unit_id" not in getattr(units_df, "columns", []):
        return labels, ["unknown"]
    col = "region" if "region" in units_df.columns else None
    if col is None:
        return labels, ["unknown"]
    indexed = units_df.set_index("unit_id")
    out: list[str] = []
    for uid in unit_ids:
        if int(uid) in indexed.index:
            val = indexed.loc[int(uid), col]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            out.append("unknown" if pd.isna(val) else str(val))
        else:
            out.append("unknown")
    labels = np.asarray(out, dtype=object)
    order = sorted({str(r) for r in labels.tolist()})
    return labels, order


class NeuralFeatureExtractor:
    """Extract a configured feature set from causal spike windows.

    Parameters
    ----------
    feature_set :
        Named feature-set id (see ``FEATURE_SET_DEFINITIONS``) or a custom
        family tuple via ``families=``.
    families :
        Explicit family list; overrides ``feature_set`` composition when given.
    decode_window :
        Causal window length W (seconds).
    update_dt :
        Decoder update interval (for Δcount / derivative).
    coactivity_bin_dt :
        Internal bin width for coactivity / lagged features.
    include_count_derivative :
        Also emit Δcount / update_dt when dynamics are enabled.
    allow_full_pairwise :
        Emit O(N²) pairwise correlations (off by default; not realtime-friendly).
    lagged_coupling_lags :
        Positive internal-bin lags for directed A→B coupling.
    """

    def __init__(
        self,
        feature_set: str = "counts",
        *,
        families: tuple[str, ...] | list[str] | None = None,
        decode_window: float = 0.250,
        update_dt: float = 0.050,
        coactivity_bin_dt: float | None = None,
        include_count_derivative: bool = False,
        allow_full_pairwise: bool = False,
        lagged_coupling_lags: tuple[int, ...] = (1, 2),
        units_df: pd.DataFrame | None = None,
        unit_ids: list[int] | np.ndarray | None = None,
    ):
        self.feature_set = str(feature_set)
        if families is not None:
            fams = tuple(families)
        else:
            ablations = build_ablation_definitions()
            if feature_set in FEATURE_SET_DEFINITIONS:
                fams = families_for_feature_set(feature_set)
            elif feature_set in ablations:
                fams = ablations[feature_set]
            else:
                raise ValueError(
                    f"Unknown feature_set {feature_set!r}; "
                    f"expected one of {ALL_FEATURE_SETS} or an ablation name"
                )

        # Deduplicate while preserving order.
        self.families: tuple[str, ...] = tuple(dict.fromkeys(fams))
        self.decode_window = float(decode_window)
        self.update_dt = float(update_dt)
        if coactivity_bin_dt is None:
            coactivity_bin_dt = max(0.010, self.decode_window / 10.0)
        self.coactivity_bin_dt = float(coactivity_bin_dt)
        self.include_count_derivative = bool(include_count_derivative)
        self.allow_full_pairwise = bool(allow_full_pairwise)
        self.lagged_coupling_lags = tuple(int(x) for x in lagged_coupling_lags)

        self.unit_ids = (
            np.asarray(unit_ids, dtype=int)
            if unit_ids is not None
            else np.asarray([], dtype=int)
        )
        self.units_df = units_df
        self.region_labels_, self.region_order_ = _resolve_region_labels(
            units_df, self.unit_ids,
        )

        self.feature_names_: list[str] = []
        self.feature_metadata_: list[FeatureSpec] = []
        self.n_features_: int | None = None
        self._prev_counts: np.ndarray | None = None  # for online extract_at

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_feature_set(
        cls,
        feature_set: str,
        *,
        units_df: pd.DataFrame | None,
        unit_ids: list[int] | np.ndarray,
        decode_window: float,
        update_dt: float = 0.050,
        **kwargs: Any,
    ) -> "NeuralFeatureExtractor":
        cfg = default_extractor_config(
            feature_set,
            decode_window=decode_window,
            update_dt=update_dt,
            **{k: v for k, v in kwargs.items() if k in {
                "coactivity_bin_dt",
                "include_count_derivative",
                "allow_full_pairwise",
                "lagged_coupling_lags",
            }},
        )
        ablations = build_ablation_definitions()
        families = None
        if feature_set in ablations and feature_set not in FEATURE_SET_DEFINITIONS:
            families = ablations[feature_set]
        return cls(
            feature_set=feature_set,
            families=families,
            units_df=units_df,
            unit_ids=unit_ids,
            **{k: cfg[k] for k in (
                "decode_window",
                "update_dt",
                "coactivity_bin_dt",
                "include_count_derivative",
                "allow_full_pairwise",
                "lagged_coupling_lags",
            )},
        )

    def needs_coactivity_bins(self) -> bool:
        return bool(set(self.families) & COACTIVITY_FAMILIES)

    def n_coactivity_bins(self) -> int:
        return n_coactivity_bins(self.decode_window, self.coactivity_bin_dt)

    # ------------------------------------------------------------------
    # Core assembly
    # ------------------------------------------------------------------
    def _assemble(
        self,
        counts: np.ndarray,
        *,
        X_bins: np.ndarray | None = None,
        prev_counts: np.ndarray | None = None,
    ) -> FeatureExtractionResult:
        counts = np.asarray(counts, dtype=float)
        if counts.ndim == 1:
            counts = counts.reshape(1, -1)
        parts: list[np.ndarray] = []
        specs: list[FeatureSpec] = []
        extras: dict[str, Any] = {"validity": {}}

        for family in self.families:
            if family == "counts":
                append_counts(parts, specs, counts, self.unit_ids)
            elif family == "count_dynamics":
                append_count_dynamics(
                    parts,
                    specs,
                    counts,
                    prev_counts,
                    self.unit_ids,
                    update_dt=self.update_dt,
                    include_derivative=self.include_count_derivative,
                )
            elif family == "population_statistics":
                append_population_statistics(
                    parts, specs, counts, self.decode_window,
                )
            elif family == "regional_statistics":
                append_regional_statistics(
                    parts,
                    specs,
                    counts,
                    self.region_labels_,
                    self.region_order_,
                    self.decode_window,
                )
            elif family == "within_region_coactivity":
                if X_bins is None:
                    raise RuntimeError("within_region_coactivity requires X_bins")
                extras["validity"]["within_region"] = append_within_region_coactivity(
                    parts,
                    specs,
                    X_bins,
                    self.region_labels_,
                    self.region_order_,
                    allow_full_pairwise=self.allow_full_pairwise,
                )
            elif family == "cross_region_coactivity":
                if X_bins is None:
                    raise RuntimeError("cross_region_coactivity requires X_bins")
                traces, order = regional_population_traces(
                    X_bins, self.region_labels_, self.region_order_,
                )
                extras["validity"]["cross_region"] = append_cross_region_coactivity(
                    parts, specs, traces, order,
                )
            elif family == "lagged_coupling":
                if X_bins is None:
                    raise RuntimeError("lagged_coupling requires X_bins")
                traces, order = regional_population_traces(
                    X_bins, self.region_labels_, self.region_order_,
                )
                extras["validity"]["lagged"] = append_lagged_coupling(
                    parts,
                    specs,
                    traces,
                    order,
                    self.lagged_coupling_lags,
                    coactivity_bin_dt=self.coactivity_bin_dt,
                )
            else:
                raise ValueError(f"Unknown feature family: {family}")

        if not parts:
            X = np.zeros((counts.shape[0], 0), dtype=float)
        else:
            X = np.concatenate(parts, axis=1)

        names = [s.name for s in specs]
        if len(names) != X.shape[1]:
            raise RuntimeError(
                f"Feature name/column mismatch: {len(names)} names vs {X.shape[1]} cols"
            )
        if np.any(~np.isfinite(X)):
            # Harden against NaN leakage from numerical edge cases.
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.feature_names_ = names
        self.feature_metadata_ = specs
        self.n_features_ = int(X.shape[1])
        return FeatureExtractionResult(
            feature_vector=X,
            feature_names=names,
            feature_metadata=specs,
            extras=extras,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_matrix(
        self,
        spikes_df: pd.DataFrame,
        decode_times: np.ndarray,
        *,
        counts: np.ndarray | None = None,
    ) -> FeatureExtractionResult:
        """Offline batch extraction aligned to ``decode_times``."""
        decode_times = np.asarray(decode_times, dtype=float)
        t0 = time.perf_counter()
        X_bins = None
        if self.needs_coactivity_bins():
            X_bins = build_causal_unit_bin_tensor(
                spikes_df,
                self.unit_ids,
                decode_times,
                self.decode_window,
                self.coactivity_bin_dt,
            )
            if counts is None:
                counts = counts_from_bin_tensor(X_bins)
        if counts is None:
            counts = build_causal_spike_matrix(
                spikes_df,
                self.unit_ids,
                decode_times,
                self.decode_window,
            )
        # Causal Δcount uses previous decode sample (aligned by index).
        # Sample 0 has no history → Δ = 0 (deterministic).
        prev = None
        if "count_dynamics" in self.families:
            prev = np.zeros_like(counts)
            if len(counts) > 1:
                prev[1:] = counts[:-1]
            # Force first-sample delta to zero by matching prev[0] to counts[0]
            # after assembly alternative: set prev[0] = counts[0].
            prev[0] = counts[0]
        result = self._assemble(counts, X_bins=X_bins, prev_counts=prev)
        result.extras["feature_extract_ms"] = (time.perf_counter() - t0) * 1000.0
        result.extras["feature_dimension"] = result.n_features
        return result

    def extract_at(
        self,
        spikes_df: pd.DataFrame,
        t: float,
        *,
        prev_counts: np.ndarray | None = None,
    ) -> FeatureExtractionResult:
        """Single-timestamp causal extraction (realtime path)."""
        t = float(t)
        X_bins = None
        if self.needs_coactivity_bins():
            X_bins = build_causal_unit_bin_tensor(
                spikes_df,
                self.unit_ids,
                np.asarray([t]),
                self.decode_window,
                self.coactivity_bin_dt,
            )
            counts = counts_from_bin_tensor(X_bins)
        else:
            counts = count_spikes_in_window(
                spikes_df,
                self.unit_ids,
                t - self.decode_window,
                t,
            ).reshape(1, -1)

        if prev_counts is None:
            prev_counts = self._prev_counts
        if prev_counts is not None:
            prev_counts = np.asarray(prev_counts, dtype=float).reshape(1, -1)

        result = self._assemble(counts, X_bins=X_bins, prev_counts=prev_counts)
        # Store for next online step.
        self._prev_counts = counts.copy()
        return result

    def reset_history(self) -> None:
        self._prev_counts = None

    def get_metadata(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "families": list(self.families),
            "decode_window_s": self.decode_window,
            "update_dt_s": self.update_dt,
            "coactivity_bin_dt_s": self.coactivity_bin_dt,
            "n_coactivity_bins": self.n_coactivity_bins() if self.needs_coactivity_bins() else None,
            "include_count_derivative": self.include_count_derivative,
            "allow_full_pairwise": self.allow_full_pairwise,
            "lagged_coupling_lags": list(self.lagged_coupling_lags),
            "n_units": int(len(self.unit_ids)),
            "n_features": self.n_features_,
            "feature_names": list(self.feature_names_),
            "feature_metadata": [s.to_dict() for s in self.feature_metadata_],
            "region_order": list(self.region_order_),
            "realtime_safe": all(
                s.realtime_safe for s in self.feature_metadata_
            ) if self.feature_metadata_ else True,
        }

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "neural_feature_extractor.joblib")
        with open(output_dir / "meta.json", "w") as f:
            json.dump(self.get_metadata(), f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> "NeuralFeatureExtractor":
        return joblib.load(Path(input_dir) / "neural_feature_extractor.joblib")
