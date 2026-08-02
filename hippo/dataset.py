"""Shared ManifoldDataset: one load/align/preprocess path for manifold methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hippo.unit_metadata import normalize_unit_metadata


@dataclass
class ManifoldDataset:
    """Causal neural observations aligned to behavioral timestamps.

    Parameters
    ----------
    activity
        Shape ``[n_times, n_units]`` — spike counts or transformed features.
    timestamps_s
        Prediction times (behavioral frames), length ``n_times``.
    behavior
        Behavior / latent labels aligned to ``timestamps_s``.
    annotations
        Extra per-time annotations (network state, etc.); may be empty.
    unit_ids
        Column order of ``activity``.
    unit_metadata
        Normalized unit table (one row per unit_id).
    """

    activity: np.ndarray
    timestamps_s: np.ndarray
    behavior: pd.DataFrame
    annotations: pd.DataFrame
    unit_ids: np.ndarray
    unit_metadata: pd.DataFrame
    spike_source: str
    integration_window_s: float
    update_interval_s: float
    activity_representation: str
    timing_validation: dict[str, Any] = field(default_factory=dict)
    train_mask: np.ndarray | None = None
    test_mask: np.ndarray | None = None

    @property
    def n_times(self) -> int:
        return int(self.activity.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.activity.shape[1])

    def subset_units(self, unit_ids: np.ndarray | list[int]) -> ManifoldDataset:
        """Return a view with a subset of units (same times)."""
        unit_ids = np.asarray(unit_ids, dtype=int)
        index = {int(u): i for i, u in enumerate(self.unit_ids)}
        cols = [index[int(u)] for u in unit_ids if int(u) in index]
        keep_ids = np.asarray([self.unit_ids[i] for i in cols], dtype=int)
        meta = self.unit_metadata.set_index("unit_id").loc[keep_ids].reset_index()
        return ManifoldDataset(
            activity=self.activity[:, cols],
            timestamps_s=self.timestamps_s,
            behavior=self.behavior,
            annotations=self.annotations,
            unit_ids=keep_ids,
            unit_metadata=meta,
            spike_source=self.spike_source,
            integration_window_s=self.integration_window_s,
            update_interval_s=self.update_interval_s,
            activity_representation=self.activity_representation,
            timing_validation=self.timing_validation,
            train_mask=self.train_mask,
            test_mask=self.test_mask,
        )


def load_manifold_dataset(
    input_dir: Path | str,
    *,
    spike_source: str = "sorted",
    integration_window_s: float = 0.250,
    activity_representation: str = "counts",
    train_frac: float = 0.70,
    align_to_behavior: bool = True,
    alignment_tolerance_s: float = 0.005,
) -> ManifoldDataset:
    """
    Load a session once and build causal activity aligned to behavior frames.

    ``activity_representation``:
      - ``counts``: raw spike counts in ``[t-W, t)``
      - ``rates``: counts / W
    """
    # Deferred imports avoid cycle: realtime.data_loading → hippo → dataset → data_loading
    from realtime.data_loading import load_simulation_data, make_decode_times
    from realtime.decoding_targets import align_extended_behavior_to_decoder_times
    from realtime.spike_features import apply_feature_mode, build_causal_spike_matrix
    from realtime.timing import (
        assert_alignment,
        extract_behavior_times,
        resolve_update_dt_s,
        validate_behavior_timestamps,
    )
    from realtime.train_decoder import causal_train_test_split

    input_dir = Path(input_dir)
    data = load_simulation_data(input_dir, spike_source)
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"],
        derive_from_behavior=align_to_behavior,
        behavior_times=behavior_times,
    )
    validation = validate_behavior_timestamps(
        behavior_times, expected_dt_s=update_dt, alignment_tolerance_s=alignment_tolerance_s,
    )
    decode_times = make_decode_times(
        data["session_duration"],
        integration_window_s,
        update_dt,
        behavior_times=behavior_times if align_to_behavior else None,
    )
    max_err = assert_alignment(
        decode_times, behavior_times, alignment_tolerance_s=alignment_tolerance_s,
    )
    behavior = align_extended_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"]
    )
    X = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, integration_window_s,
    )
    if activity_representation not in ("counts", "rates"):
        raise ValueError(
            f"Phase-1 activity_representation must be 'counts' or 'rates'; "
            f"got {activity_representation!r}"
        )
    X = apply_feature_mode(X, activity_representation, integration_window_s)
    train_mask, test_mask = causal_train_test_split(decode_times, train_frac)
    unit_meta = normalize_unit_metadata(data["units_df"])
    # Ensure unit_metadata row order matches unit_ids
    unit_meta = unit_meta.set_index("unit_id").loc[list(data["unit_ids"])].reset_index()

    return ManifoldDataset(
        activity=X,
        timestamps_s=np.asarray(decode_times, dtype=float),
        behavior=behavior,
        annotations=pd.DataFrame({"time": decode_times}),
        unit_ids=np.asarray(data["unit_ids"], dtype=int),
        unit_metadata=unit_meta,
        spike_source=spike_source,
        integration_window_s=float(integration_window_s),
        update_interval_s=float(update_dt),
        activity_representation=activity_representation,
        timing_validation={
            **validation.__dict__,
            "max_alignment_error_s": max_err,
        },
        train_mask=train_mask,
        test_mask=test_mask,
    )
