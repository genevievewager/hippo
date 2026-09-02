"""Unified neural observation transformer: spikes → x_t^(W).

Training
    raw spikes → ObservationTransformer.fit_transform() → FeatureDataset
Realtime
    new spikes → ObservationTransformer.transform_one() → x_t

The same fitted objects (NeuralFeatureExtractor + SpikeFeatureTransformer)
are used on both paths. ``W`` and ``update_dt`` are owned by this object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from realtime.feature_representations import (
    SpikeFeatureTransformer,
    make_spike_feature_transformer,
)
from realtime.neural_features import NeuralFeatureExtractor
from realtime.neural_features.comparison import effective_spike_feature_type
from realtime.pipeline_artifacts import (
    ArtifactOrigin,
    ArtifactProvenance,
    ArtifactStatus,
    FeatureDataset,
    ObservationConfig,
    write_provenance,
)
from realtime.pipeline_invariants import (
    PipelineInvariantError,
    assert_spike_source_explicit,
    assert_train_only_fit,
    assert_valid_timing,
)
from realtime.spike_binner import build_causal_spike_matrix, count_spikes_in_window


@dataclass
class ObservationFitStats:
    n_train: int
    n_test: int
    n_features: int
    origin: ArtifactOrigin


class ObservationTransformer:
    """Causal observation O(W, F) with train-only F fitting."""

    def __init__(
        self,
        observation: ObservationConfig,
        *,
        units_df: pd.DataFrame | None = None,
        unit_ids: list[int] | np.ndarray | None = None,
        extractor: NeuralFeatureExtractor | None = None,
        f_transform: SpikeFeatureTransformer | None = None,
        coactivity_bin_dt: float | None = None,
        include_count_derivative: bool = False,
        allow_full_pairwise: bool = False,
        lagged_coupling_lags: tuple[int, ...] = (1, 2),
    ):
        assert_valid_timing(observation.window_s, observation.update_dt)
        assert_spike_source_explicit(observation.source_spikes)
        self.observation = observation
        self.units_df = units_df
        self.unit_ids = (
            np.asarray(unit_ids, dtype=int)
            if unit_ids is not None
            else np.asarray([], dtype=int)
        )
        f_eff = effective_spike_feature_type(
            observation.feature_set, observation.feature_type,
        )
        self.feature_type_eff = f_eff
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = NeuralFeatureExtractor.from_feature_set(
                observation.feature_set,
                units_df=units_df,
                unit_ids=self.unit_ids,
                decode_window=observation.window_s,
                update_dt=observation.update_dt,
                coactivity_bin_dt=coactivity_bin_dt,
                include_count_derivative=include_count_derivative,
                allow_full_pairwise=allow_full_pairwise,
                lagged_coupling_lags=lagged_coupling_lags,
            )
        if f_transform is not None:
            self.f_transform = f_transform
        else:
            self.f_transform = make_spike_feature_transformer(
                f_eff,
                decode_window=observation.window_s,
                units_df=units_df,
                unit_ids=self.unit_ids,
            )
        self._fitted = getattr(self.f_transform, "n_features_in_", None) is not None
        self._prev_counts: np.ndarray | None = None
        self.last_stats: ObservationFitStats | None = None

    @property
    def window_s(self) -> float:
        return float(self.observation.window_s)

    @property
    def update_dt(self) -> float:
        return float(self.observation.update_dt)

    def extract_raw(
        self,
        spikes_df: pd.DataFrame,
        decode_times: np.ndarray,
        *,
        counts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Target-independent extraction (no F fit)."""
        if counts is None and not self.extractor.needs_coactivity_bins():
            counts = build_causal_spike_matrix(
                spikes_df,
                self.unit_ids,
                np.asarray(decode_times, dtype=float),
                self.window_s,
            )
        result = self.extractor.extract_matrix(
            spikes_df,
            np.asarray(decode_times, dtype=float),
            counts=None if self.extractor.needs_coactivity_bins() else counts,
        )
        X = np.asarray(result.feature_vector, dtype=float)
        names = list(result.feature_names)
        return X, names

    def fit(
        self,
        X_raw: np.ndarray,
        *,
        train_mask: np.ndarray | None = None,
    ) -> "ObservationTransformer":
        """Fit F on the training partition only."""
        X = np.asarray(X_raw, dtype=float)
        if train_mask is not None:
            mask = np.asarray(train_mask, dtype=bool).reshape(-1)
            assert_train_only_fit(mask, context="ObservationTransformer.fit")
            if mask.shape[0] != X.shape[0]:
                raise PipelineInvariantError(
                    "train_mask length does not match feature matrix rows."
                )
            X_fit = X[mask]
        else:
            X_fit = X
        self.f_transform.fit(X_fit)
        self._fitted = True
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ObservationTransformer must be fit before transform")
        return np.asarray(self.f_transform.transform(np.asarray(X_raw, dtype=float)))

    def fit_transform(
        self,
        spikes_df: pd.DataFrame,
        decode_times: np.ndarray,
        *,
        train_mask: np.ndarray | None = None,
        test_mask: np.ndarray | None = None,
        counts: np.ndarray | None = None,
        provenance: ArtifactProvenance | None = None,
    ) -> FeatureDataset:
        X_raw, names = self.extract_raw(spikes_df, decode_times, counts=counts)
        self.fit(X_raw, train_mask=train_mask)
        X = self.transform(X_raw)
        n_train = int(np.asarray(train_mask).sum()) if train_mask is not None else int(X.shape[0])
        n_test = int(np.asarray(test_mask).sum()) if test_mask is not None else 0
        self.last_stats = ObservationFitStats(
            n_train=n_train,
            n_test=n_test,
            n_features=int(X.shape[1]),
            origin=ArtifactOrigin.COMPUTED,
        )
        return FeatureDataset(
            observation=self.observation,
            timestamps=np.asarray(decode_times, dtype=float),
            X=X,
            feature_names=names,
            train_mask=train_mask,
            test_mask=test_mask,
            provenance=provenance or ArtifactProvenance(
                run_id=self.observation.hash(),
                config_hash=self.observation.fit_hash(),
                origin=ArtifactOrigin.COMPUTED,
                status=ArtifactStatus.FRESH,
            ),
        )

    def transform_one(
        self,
        spikes_df: pd.DataFrame | None = None,
        t: float | None = None,
        *,
        counts: np.ndarray | None = None,
        prev_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Single-timestep observation for realtime replay."""
        if not self._fitted:
            raise RuntimeError("ObservationTransformer must be fit before transform_one")
        if t is None:
            raise PipelineInvariantError("transform_one requires a timestamp t")
        t = float(t)
        if spikes_df is not None:
            result = self.extractor.extract_at(
                spikes_df, t, prev_counts=prev_counts if prev_counts is not None else self._prev_counts,
            )
            raw = np.asarray(result.feature_vector, dtype=float).reshape(1, -1)
            if counts is not None:
                self._prev_counts = np.asarray(counts, dtype=float).reshape(1, -1)
            else:
                self._prev_counts = raw
        elif counts is not None:
            counts_2d = np.asarray(counts, dtype=float).reshape(1, -1)
            result = self.extractor._assemble(
                counts_2d, prev_counts=prev_counts if prev_counts is not None else self._prev_counts,
            )
            raw = np.asarray(result.feature_vector, dtype=float).reshape(1, -1)
            self._prev_counts = counts_2d
        else:
            raise PipelineInvariantError("transform_one requires spikes_df or counts")
        if hasattr(self.f_transform, "transform_one"):
            z = self.f_transform.transform_one(raw.ravel())
            return np.asarray(z, dtype=float).reshape(1, -1)
        return np.asarray(self.f_transform.transform(raw), dtype=float)

    def reset_history(self) -> None:
        self._prev_counts = None
        if hasattr(self.extractor, "reset_history"):
            self.extractor.reset_history()

    def get_metadata(self) -> dict[str, Any]:
        meta = {
            "kind": "ObservationTransformer",
            "observation": self.observation.to_dict(),
            "config_hash": self.observation.hash(),
            "fit_hash": self.observation.fit_hash(),
            "feature_type_eff": self.feature_type_eff,
            "fitted": bool(self._fitted),
        }
        if hasattr(self.extractor, "get_metadata"):
            meta["extractor"] = self.extractor.get_metadata()
        if hasattr(self.f_transform, "get_metadata"):
            meta["f_transform"] = self.f_transform.get_metadata()
        return meta

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor.save(output_dir / "neural_extractor")
        self.f_transform.save(output_dir / "feature_transform")
        joblib.dump(self.observation.to_dict(), output_dir / "observation.joblib")
        meta = self.get_metadata()
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        write_provenance(output_dir, {
            "kind": "ObservationTransformer",
            "config_hash": self.observation.hash(),
            "fit_hash": self.observation.fit_hash(),
            "observation": self.observation.to_dict(),
            "origin": ArtifactOrigin.COMPUTED.value,
        })
        return output_dir

    @classmethod
    def load(cls, input_dir: Path) -> "ObservationTransformer":
        input_dir = Path(input_dir)
        obs_path = input_dir / "observation.joblib"
        meta_path = input_dir / "meta.json"
        if obs_path.exists():
            raw = joblib.load(obs_path)
            observation = ObservationConfig.from_dict(raw if isinstance(raw, dict) else {})
        elif meta_path.exists():
            meta = json.loads(meta_path.read_text())
            observation = ObservationConfig.from_dict(meta.get("observation") or meta)
        else:
            raise FileNotFoundError(f"No observation metadata in {input_dir}")
        extractor = NeuralFeatureExtractor.load(input_dir / "neural_extractor")
        f_transform = SpikeFeatureTransformer.load(input_dir / "feature_transform")
        inst = cls(
            observation,
            units_df=getattr(extractor, "units_df", None),
            unit_ids=getattr(extractor, "unit_ids", None),
            extractor=extractor,
            f_transform=f_transform,
        )
        inst._fitted = True
        return inst
