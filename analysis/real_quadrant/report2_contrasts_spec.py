"""Report-2 planned contrasts (logical method names).

grid_mode:
  final  — both arms from final-grid rows (pca/lds extended)
  grid20 — both arms locked to d≤20 era (dm / isomap / offline gpfa)
  either — unchanged methods (raw family, gpfa_causal)
"""

from __future__ import annotations

# (a, b, short_label, grid_mode)
CONTRASTS_R2 = (
    ("lds", "raw_smooth", "Dynamics vs EMA", "final"),
    ("gpfa_causal", "raw_smooth", "Causal GPFA vs EMA", "either"),
    ("lds_smooth", "raw_smooth", "LDS+EMA vs EMA", "final"),
    ("pca_smooth", "raw_smooth", "PCA+EMA vs EMA", "final"),
    ("dm_smooth", "pca_smooth", "DM+EMA vs PCA+EMA", "grid20"),
    ("raw_smooth", "raw", "EMA alone", "either"),
)

# Secondary / mechanism (Fig4c)
CONTRASTS_MECH = (
    ("lds", "raw_lag", "Dynamics beyond history", "final"),
)
