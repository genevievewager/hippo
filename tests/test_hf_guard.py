"""HF feature-width guard: mismatch must RAISE (never skip).

Call sites that build counts/features for Report 2 v2 (grep audit):

  analysis/real_quadrant/adapter.py:494
      load_units_and_spikes(..., regions=regions)  — HF when regions set
  analysis/real_quadrant/adapter.py:591–599
      causal_count_matrix on filtered unit_ids
  analysis/real_quadrant/run_m3_v2.py:~179
      build_segment_bundle(..., regions=HF_REGION_ALLOWLIST) + assert
  analysis/real_quadrant/run_m3_v2.py:~121
      rebuild for gpfa_causal cache with regions=HF_REGION_ALLOWLIST + assert
  analysis/real_quadrant/run_m3.py:~612
      _score_gpfa_causal rebuild with region_allowlist from source_summary + assert
  realtime/quadrant_n5_run.py:~1144–1160
      analyze_source loads bundle X_counts; assert vs len(unit_ids)
      (covers raw, raw_lag, pca, dm, lds, isomap, gpfa, kNN, A5/A13, contract)

Non-v2 (intentionally no HF allow-list; not used by run_m3_v2):
  analysis/real_quadrant/run_m3.py:687          — v1 worker bundle (no regions)
  analysis/real_quadrant/run_m1.py / run_m2.py / run_m3_extend_d.py
  analysis/real_quadrant/posthoc_smooth.py:372  — M2-only gpfa_causal helper
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.real_quadrant.hf_guard import (
    HFFeatureWidthError,
    assert_model_obs_eq_n_units,
    assert_n_features_eq_n_units,
)


def test_assert_n_features_matches():
    X = np.zeros((10, 40))
    assert_n_features_eq_n_units(X, 40, context="ok")


def test_all_region_features_vs_hf_model_raises():
    """All-region feature matrix (269) fed where HF expects 256 → RAISE."""
    X_all = np.zeros((100, 269))
    n_hf = 256
    with pytest.raises(HFFeatureWidthError, match="n_features=269 != n_units=256"):
        assert_n_features_eq_n_units(X_all, n_hf, context="synthetic.gpfa_causal")


def test_model_obs_mismatch_raises():
    class _M:
        C = np.zeros((269, 8))  # all-region obs dim

    with pytest.raises(HFFeatureWidthError, match="model_n_obs=269 != n_units=256"):
        assert_model_obs_eq_n_units(_M(), 256, context="synthetic.model")
