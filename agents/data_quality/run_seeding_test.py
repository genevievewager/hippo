#!/usr/bin/env python3
"""Test whether global numpy seeding makes simulate_behavior deterministic.

No monkeypatching. Compares position hashes of two runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from hippo_sim.behavior import simulate_behavior
from hippo_sim.config import SimConfig


def _pos_hash(pos: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pos, dtype=np.float64).tobytes()).hexdigest()


def run_once(seed: int, duration_s: float = 60.0) -> dict:
    np.random.seed(seed)
    # Also seed the Generator API in case RiaB uses it elsewhere
    rng = np.random.default_rng(seed)
    _ = rng  # constructed; Agent may still use np.random module state
    cfg = SimConfig(
        output_dir=Path("/tmp/dq_seed_unused"),
        seed=seed,
        session_duration_s=duration_s,
        behavior_dt=0.05,
        arena_size_cm=95.64160313187207,
        thigmotaxis=0.8,
    )
    result = simulate_behavior(cfg)
    pos = result.trace.position_cm
    return {
        "seed": seed,
        "pos_sha256": _pos_hash(pos),
        "speed_mean": float(np.mean(result.trace.speed_cm_s)),
        "n_steps": int(pos.shape[0]),
    }


def main() -> None:
    a1 = run_once(20261005)
    a2 = run_once(20261005)
    b1 = run_once(999)
    out = {
        "same_seed_run1": a1,
        "same_seed_run2": a2,
        "other_seed": b1,
        "same_seed_positions_identical": a1["pos_sha256"] == a2["pos_sha256"],
        "different_seed_positions_identical": a1["pos_sha256"] == b1["pos_sha256"],
        "conclusion": None,
    }
    if out["same_seed_positions_identical"]:
        out["conclusion"] = (
            "Global np.random.seed before simulate_behavior yields identical "
            "positions across two runs (Agent appears to use np.random state)."
        )
    else:
        out["conclusion"] = (
            "Global np.random.seed does NOT make behaviour deterministic; "
            "Agent RNG is independent. DF5 stands."
        )
    path = assert_dq_path(DATA_QUALITY_ROOT / "matched" / "seeding_test.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
