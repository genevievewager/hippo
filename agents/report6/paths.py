"""Absolute read-only sources and local output root (gitignored)."""

from __future__ import annotations

from pathlib import Path

R2_ROOM_A = Path(
    "/home/gmwager/projects/hippo-r2v2/outputs/real_quadrant/m3_v2/room_A"
)
R2_COUNT_CACHE = Path(
    "/home/gmwager/projects/hippo-r2v2/outputs/real_quadrant/cache"
)
R1_MATCHED_SIM = Path(
    "/home/gmwager/projects/hippo-r1/outputs/report1_revision/matched_sim"
)

# hippo-r6 local
R6_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = R6_ROOT / "outputs" / "report6"
ALIAS_PATH = OUT_ROOT / "alias_map.json"  # gitignored; may hold real IDs
BETA_DIR = OUT_ROOT / "beta_animal"

DT_S = 0.050  # update_dt used by Report 2 / 1 causal grids
WINDOW_S = 0.250
